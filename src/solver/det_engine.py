"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
import sys
from typing import Iterable

import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tools.visualize_image_annotation import visualize_detection
from tools.concatenate_images import concatenate_images
import os
import concurrent.futures
import time

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA, Warmup

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get('SAVE_INTERMEDIATE_VISUALIZE_RESULT', 'False') == 'True'

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        no_gt = False
        num_gts = [len(t["labels"]) for t in targets]
        max_gt_num = max(num_gts)
        if max_gt_num == 0: # no gt for denoising will cause error in model forward
            no_gt = True
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            
            for b, target in enumerate(targets):
                image = samples[b].cpu()
                _, H, W = image.shape
                target_cpu = {}
                for k, v in target.items():
                    if k == 'boxes':
                        target_cpu[k] = v.cpu().detach().clone() * torch.tensor([W, H, W, H])
                    else:
                        target_cpu[k] = v.cpu().detach().clone()
                visualize_detection(image, target_cpu, f"sample_gt", return_image=False, type="xywh")

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
):
    SAVE_TEST_VISUALIZE_RESULT = os.environ.get('SAVE_TEST_VISUALIZE_RESULT', 'False') == 'True'
    if SAVE_TEST_VISUALIZE_RESULT:
        os.makedirs("visualize_all", exist_ok=True)
        print("Saving visualize results to visualize_all/")
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    # For defe Accuracy calculation
    if model.encoder.use_defe:
        total_defe_samples = 0
        ample_defe_predictions = 0
        total_anchor_num = 0

    MAX_PENDING_TASKS = 256
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        pending_futures = []
        
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            image_ids = [t["image_id"].item() for t in targets]
            coco = data_loader.dataset.coco
            file_names = [coco.loadImgs(id)[0]['file_name'] for id in image_ids]

            outputs = model(samples, targets=targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            if SAVE_TEST_VISUALIZE_RESULT:
                process_args = []
                scale_factor = float(samples[0].shape[1] / orig_target_sizes[0][0])
                for i in range(len(targets)):
                    sample_cpu = samples[i].cpu()
                    target_cpu = {k: v.cpu() for k, v in targets[i].items()}
                    result_cpu = {k: v.cpu() for k, v in results[i].items()}
                    process_args.append((
                        sample_cpu,
                        target_cpu,
                        result_cpu,
                        file_names[i],
                        scale_factor
                    ))
                
                if len(pending_futures) >= MAX_PENDING_TASKS:
                    while len(pending_futures) > 0:
                        done_futures = []
                        for future in pending_futures:
                            if future.done():
                                done_futures.append(future)
                        
                        for future in done_futures:
                            pending_futures.remove(future)
                        
                        if not done_futures:
                            time.sleep(0.1)

                for args in process_args:
                    future = executor.submit(process_image_pair, args)
                    pending_futures.append(future)

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if model.encoder.use_defe:
                # For defe Ample Rate calculation
                pred_defe = outputs['batch_queries_num'][0]
                if pred_defe >= targets[0]['labels'].shape[0]:
                    ample_defe_predictions += 1
                total_defe_samples += 1
                total_anchor_num += outputs['batch_queries_num'][0]

        concurrent.futures.wait(pending_futures)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if model.encoder.use_defe:
        print("defe Ample Rate:", ample_defe_predictions / total_defe_samples)
        print("defe Average Anchor Number:", total_anchor_num / total_defe_samples)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator


def process_image_pair(args):
    sample, target, result, filename, scale_factor = args
    sample_img = visualize_detection(sample, target, f"sample_{filename}", return_image=True)
    result_img = visualize_detection(sample, result, f"result_{filename}", 
                                   scale_factor=scale_factor, return_image=True)
    concatenate_images(sample_img, result_img, output_path=f"visualize_all/{filename}")
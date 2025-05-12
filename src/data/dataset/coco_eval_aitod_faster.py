'''
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
'''

import contextlib
import copy
import os

import faster_coco_eval.core.mask as mask_util
import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster

from ...core import register
from ...misc import dist_utils

__all__ = [
    "AitodCocoFasterEvaluator",
]


class AitodCOCOeval_faster(COCOeval_faster):

    def __init__(self, coco_gt, iou_type, print_function=print, separate_eval=True):
        super(AitodCOCOeval_faster, self).__init__(coco_gt, iouType=iou_type, print_function=print_function, separate_eval=separate_eval)
        self.maxDets = [1, 100, 1500]
        self.areaRng = [[0**2, 1e5**2], [0**2, 8**2], [8**2, 16**2], [16**2, 32**2],
                        [32**2, 1e5**2]]
        self.areaRngLbl = ['all', 'verytiny', 'tiny', 'small', 'medium']

    def summarize(self):
        """Compute and display summary metrics for evaluation results.

        Note this functin can *only* be applied on the default parameter
        setting
        """

        def _summarizeDets():
            _count = 18 if self.lvis_style else 15
            stats = np.zeros((_count,))

            stats[0] = self._summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = self._summarize(1, iouThr=.25, maxDets=self.params.maxDets[2])
            stats[2] = self._summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[3] = self._summarize(1,
                                  iouThr=.75,
                                  maxDets=self.params.maxDets[2])
            stats[4] = self._summarize(1,
                                  areaRng='verytiny',
                                  maxDets=self.params.maxDets[2])
            stats[5] = self._summarize(1,
                                  areaRng='tiny',
                                  maxDets=self.params.maxDets[2])
            stats[6] = self._summarize(1,
                                  areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[7] = self._summarize(1,
                                  areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[8] = self._summarize(0, maxDets=self.params.maxDets[0])
            stats[9] = self._summarize(0, maxDets=self.params.maxDets[1])
            stats[10] = self._summarize(0, maxDets=self.params.maxDets[2])
            stats[11] = self._summarize(0,
                                  areaRng='verytiny',
                                  maxDets=self.params.maxDets[2])
            stats[12] = self._summarize(0,
                                   areaRng='tiny',
                                   maxDets=self.params.maxDets[2])
            stats[13] = self._summarize(0,
                                   areaRng='small',
                                   maxDets=self.params.maxDets[2])
            stats[14] = self._summarize(0,
                                   areaRng='medium',
                                   maxDets=self.params.maxDets[2])

            if self.lvis_style:
                stats[15] = self._summarize(1, maxDets=self.params.maxDets[-1], freq_group_idx=0)  # APr
                stats[16] = self._summarize(1, maxDets=self.params.maxDets[-1], freq_group_idx=1)  # APc
                stats[17] = self._summarize(1, maxDets=self.params.maxDets[-1], freq_group_idx=2)  # APf

            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])  # AP_all
            stats[1] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AP_50
            stats[2] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AP_75
            stats[3] = self._summarize(1, maxDets=self.params.maxDets[-1], areaRng="medium")  # AP_medium
            stats[4] = self._summarize(1, maxDets=self.params.maxDets[-1], areaRng="large")  # AP_large
            stats[5] = self._summarize(0, maxDets=self.params.maxDets[-1])  # AR_all
            stats[6] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AR_50
            stats[7] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AR_75
            stats[8] = self._summarize(0, maxDets=self.params.maxDets[-1], areaRng="medium")  # AR_medium
            stats[9] = self._summarize(0, maxDets=self.params.maxDets[-1], areaRng="large")  # AR_large
            return stats

        def _summarizeKps_crowd():
            stats = np.zeros((9,))
            stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])  # AP_all
            stats[1] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AP_50
            stats[2] = self._summarize(1, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AP_75
            stats[3] = self._summarize(0, maxDets=self.params.maxDets[-1])  # AR_all
            stats[4] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.5)  # AR_50
            stats[5] = self._summarize(0, maxDets=self.params.maxDets[-1], iouThr=0.75)  # AR_75
            type_result = self.get_type_result(first=0.2, second=0.8)

            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | type={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision"
            typeStr = "(AP)"
            iouStr = f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}"
            self.print_function(iStr.format(titleStr, typeStr, iouStr, "easy", self.params.maxDets[-1], type_result[0]))
            self.print_function(
                iStr.format(titleStr, typeStr, iouStr, "medium", self.params.maxDets[-1], type_result[1])
            )
            self.print_function(iStr.format(titleStr, typeStr, iouStr, "hard", self.params.maxDets[-1], type_result[2]))
            stats[6] = type_result[0]  # AP_easy
            stats[7] = type_result[1]  # AP_medium
            stats[8] = type_result[2]  # AP_hard

            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")

        iouType = self.params.iouType

        if iouType in set(["segm", "bbox", "boundary"]):
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        elif iouType == "keypoints_crowd":
            summarize = _summarizeKps_crowd
        else:
            ValueError(f"iouType must be bbox, segm, boundary or keypoints or keypoints_crowd. Get {iouType}")

        self.all_stats = summarize()
        self.stats = self.all_stats[:12]



@register()
class AitodCocoFasterEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt: COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = AitodCOCOeval_faster(
                coco_gt, iou_type=iou_type, print_function=print, separate_eval=True
            )

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = AitodCOCOeval_faster(
                self.coco_gt, iou_type=iou_type, print_function=print, separate_eval=True
            )
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds),
                )
            )

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()

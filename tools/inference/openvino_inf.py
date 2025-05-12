"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import cv2
import numpy as np
import openvino
from openvino.runtime import Core


class OvInfer:
    def __init__(self, model_path, device_name="AUTO"):
        self.resized_image = None
        self.ratio = None
        self.resize_image = None
        self.ori_image = None
        self.device = device_name
        self.model_path = model_path
        self.core = Core()
        self.available_device = self.core.available_devices
        self.compile_model = self.core.compile_model(self.model_path, device_name)
        self.target_size = [
            self.compile_model.inputs[0].get_partial_shape()[2].get_length(),
            self.compile_model.inputs[0].get_partial_shape()[3].get_length(),
        ]
        self.query_num = self.compile_model.outputs[0].get_partial_shape()[1].get_length()

    def infer(self, inputs: dict):
        infer_request = self.compile_model.create_infer_request()
        for input_name, input_data in inputs.items():
            input_tensor = openvino.Tensor(input_data)
            infer_request.set_tensor(input_name, input_tensor)
        infer_request.infer()
        outputs = {
            "labels": infer_request.get_tensor("labels").data,
            "boxes": infer_request.get_tensor("boxes").data,
            "scores": infer_request.get_tensor("scores").data,
        }
        return outputs

    def process_image(self, ori_image, keep_ratio: bool):
        self.ori_image = ori_image
        h, w = ori_image.shape[:2]
        if keep_ratio:
            r = min(self.target_size[0] / h, self.target_size[1] / w)
            self.ratio = r
            new_w = int(w * r)
            new_h = int(h * r)
            temp_image = cv2.resize(ori_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_image = np.full(
                (self.target_size[0], self.target_size[1], 3), 114, dtype=temp_image.dtype
            )
            resized_image[:new_h, :new_w, :] = temp_image
            self.resized_image = resized_image
        else:
            self.resized_image = cv2.resize(
                ori_image, self.target_size, interpolation=cv2.INTER_LINEAR
            )
        blob_image = cv2.dnn.blobFromImage(self.resized_image, 1.0 / 255.0)
        orig_size = np.array([self.resized_image.shape[0], self.resized_image.shape[1]]).reshape(
            1, 2
        )
        inputs = {
            "images": blob_image,
            "orig_target_sizes": orig_size,
        }
        return inputs

    def get_available_device(self):
        return self.available_device

    def draw_and_save_image(self, infer_result, image_path, score_threshold=0.6):
        draw_image = self.ori_image
        scores = infer_result["scores"]
        labels = infer_result["labels"]
        boxes = infer_result["boxes"]
        for i in range(self.query_num):
            if scores[0, i] > score_threshold:
                cx = boxes[0, i, 0] * self.ratio
                cy = boxes[0, i, 1] * self.ratio
                bx = boxes[0, i, 2] * self.ratio
                by = boxes[0, i, 3] * self.ratio
                cv2.rectangle(
                    draw_image, (int(cx), int(cy), int(bx - cx), int(by - cy)), (255, 0, 0), 1
                )
        cv2.imwrite(image_path, draw_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-image", "--image", type=str, required=True)
    parser.add_argument("-ov_model", "--ov_model", type=str, required=True)
    args = parser.parse_args()
    img = cv2.imread(args.image)
    mOvInfer = OvInfer(args.ov_model)
    inputs = mOvInfer.process_image(img, True)
    outputs = mOvInfer.infer(inputs)
    mOvInfer.draw_and_save_image(outputs, "openvino_result.jpg")

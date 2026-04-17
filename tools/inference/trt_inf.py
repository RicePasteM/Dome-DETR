"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import collections
import contextlib
import os
import time
from collections import OrderedDict

import cv2  # Added for video processing
import numpy as np
import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TorchOutputAllocator(trt.IOutputAllocator):
    def __init__(self, torch_dtype, device):
        trt.IOutputAllocator.__init__(self)
        self.torch_dtype = torch_dtype
        self.device = device
        self.tensor = None
        self.shape = None
        self.size_bytes = 0

    def _ensure_tensor(self, size):
        size = int(size)
        itemsize = torch.empty((), dtype=self.torch_dtype).element_size()
        numel = max(1, (size + itemsize - 1) // itemsize)
        if self.tensor is None or self.size_bytes < size:
            self.tensor = torch.empty(numel, dtype=self.torch_dtype, device=self.device)
            self.size_bytes = size
        return int(self.tensor.data_ptr())

    def reallocate_output(self, tensor_name, memory, size, alignment):
        return self._ensure_tensor(size)

    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
        return self._ensure_tensor(size)

    def notify_shape(self, tensor_name, shape):
        self.shape = tuple(int(dim) for dim in shape)


class TRTInference(object):
    def __init__(
        self, engine_path, device="cuda:0", backend="torch", max_batch_size=32, verbose=False
    ):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.output_allocators = self.get_output_allocators()
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.time_profile = TimeProfiler()

    @staticmethod
    def _normalize_shape(shape):
        return tuple(int(dim) for dim in shape)

    @staticmethod
    def _allocate_tensor(shape, dtype, device):
        return torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)

    @staticmethod
    def _numel(shape):
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)

    @staticmethod
    def _numpy_to_torch_dtype(dtype):
        sample = np.empty((1,), dtype=dtype)
        return torch.from_numpy(sample).dtype

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = list(engine.get_tensor_shape(name))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if shape[0] == -1:
                    shape[0] = max_batch_size
                    context.set_input_shape(name, shape)
                shape = self._normalize_shape(shape)
                data = self._allocate_tensor(shape, dtype, device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
            else:
                shape = self._normalize_shape(context.get_tensor_shape(name))
                torch_dtype = self._numpy_to_torch_dtype(dtype)
                data = torch.empty(0, dtype=torch_dtype, device=device)
                bindings[name] = Binding(name, dtype, shape, data, 0)

        return bindings

    def get_output_allocators(self):
        allocators = {}
        for n in self.output_names:
            torch_dtype = self._numpy_to_torch_dtype(self.bindings[n].dtype)
            allocator = TorchOutputAllocator(torch_dtype, self.device)
            allocators[n] = allocator
            self.context.set_output_allocator(n, allocator)
            self.context.set_tensor_address(n, 0)
        return allocators

    def _refresh_output_bindings(self):
        for n in self.output_names:
            shape = self._normalize_shape(self.context.get_tensor_shape(n))
            binding = self.bindings[n]
            if any(dim < 0 for dim in shape):
                max_bytes = int(self.context.get_max_output_size(n))
                itemsize = np.dtype(binding.dtype).itemsize
                shape = (max(1, (max_bytes + itemsize - 1) // itemsize),)
            if self._numel(binding.shape) < self._numel(shape):
                data = self._allocate_tensor(shape, binding.dtype, self.device)
                self.bindings[n] = binding._replace(shape=shape, data=data, ptr=data.data_ptr())

    def run_torch(self, blob):
        for n in self.input_names:
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=tuple(blob[n].shape))

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)

        outputs = {}
        for n in self.input_names:
            self.context.set_tensor_address(n, int(blob[n].data_ptr()))
        for n in self.output_names:
            allocator = self.output_allocators[n]
            allocator.shape = None
            self.context.set_output_allocator(n, allocator)
            self.context.set_tensor_address(n, 0)

        stream = torch.cuda.current_stream(device=self.device)
        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        stream.synchronize()

        for n in self.output_names:
            allocator = self.output_allocators[n]
            if allocator.tensor is None or allocator.shape is None:
                raise ValueError(f"TensorRT output allocator did not materialize {n}.")
            numel = self._numel(allocator.shape)
            outputs[n] = allocator.tensor[:numel].reshape(allocator.shape)

        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def resolve_input_size(self, override=None):
        if override is not None:
            return override
        if "images" in self.bindings:
            shape = self.bindings["images"].shape
            if len(shape) >= 4 and shape[-1] == shape[-2]:
                return int(shape[-1])
        return 640


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

    return images


def process_image(m, file_path, device, input_size):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil)[None]

    blob = {
        "images": im_data.to(device),
        "orig_target_sizes": orig_size.to(device),
    }

    output = m(blob)
    result_images = draw([im_pil], output["labels"], output["boxes"], output["scores"])
    result_images[0].save("trt_result.jpg")
    print("Image processing complete. Result saved as 'result.jpg'.")


def process_video(m, file_path, device, input_size):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("trt_result.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)

        im_data = transforms(frame_pil)[None]

        blob = {
            "images": im_data.to(device),
            "orig_target_sizes": orig_size.to(device),
        }

        output = m(blob)

        # Draw detections on the frame
        result_images = draw([frame_pil], output["labels"], output["boxes"], output["scores"])

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result_video.mp4'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-trt", "--trt", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Square inference size. Defaults to the TensorRT engine input shape.",
    )

    args = parser.parse_args()

    m = TRTInference(args.trt, device=args.device)
    input_size = m.resolve_input_size(args.size)
    print(f"Using input size: {input_size}")

    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(m, file_path, args.device, input_size)
    else:
        # Process as video
        process_video(m, file_path, args.device, input_size)

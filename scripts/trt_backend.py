"""TensorRT forward for sequential two-option Laya questions."""

from pathlib import Path

import torch


INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
OUTPUTS = ("logits", "act_logits")


class TensorRTForward:
    def __init__(self, path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int64: torch.int64,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        self.dtypes = {
            name: dtype_map[self.engine.get_tensor_dtype(name)]
            for name in (*INPUTS, *OUTPUTS)
        }
        self.calls = 0

    def __call__(self, batch):
        if "option_ids" in batch or batch["marker_pos"].shape[1] != 2:
            raise ValueError("This engine requires sequential two-option questions")
        with torch.cuda.stream(self.stream):
            inputs = {
                name: batch[name].to(device="cuda", dtype=self.dtypes[name]).contiguous()
                for name in INPUTS
            }
            for name, tensor in inputs.items():
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"Input {name} is outside the TensorRT optimization profile")
                self.context.set_tensor_address(name, tensor.data_ptr())
            outputs = {
                name: torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    dtype=self.dtypes[name], device="cuda",
                )
                for name in OUTPUTS
            }
            for name, tensor in outputs.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT engine execution failed")
        # Complete work before releasing local buffers or decoding on another stream.
        self.stream.synchronize()
        if not all(torch.isfinite(tensor).all().item() for tensor in outputs.values()):
            raise RuntimeError("TensorRT returned non-finite logits")
        self.calls += 1
        return tuple(outputs[name] for name in OUTPUTS)

"""Export the pinned Laya model and run its TensorRT Noul benchmark."""

import argparse
import json
import os
from pathlib import Path

import torch
from laya.serve import build_router, create_app

from bench_noul import request_body


INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
OUTPUTS = ("logits", "act_logits")


def export_onnx(path):
    os.environ["LAYA_DEVICE"] = "cpu"
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    model = build_router().load("multilingual").model.eval()
    mask = torch.ones((2, 96), dtype=torch.long)
    mask[1, 80:] = 0
    example = (
        torch.randint(5, 1000, (2, 96)), mask, torch.tensor([[5, 20], [6, 21]]),
        torch.ones((2, 2), dtype=torch.bool), torch.tensor([1, 1]),
    )
    dynamic_axes = {name: {0: "batch"} for name in (*INPUTS, *OUTPUTS)}
    for name in ("input_ids", "attention_mask"):
        dynamic_axes[name][1] = "sequence"
    torch.onnx.export(
        model, example, str(path), opset_version=18, dynamo=False,
        input_names=list(INPUTS), output_names=list(OUTPUTS), dynamic_axes=dynamic_axes,
    )
    print(f"Exported {path}", flush=True)


class TensorRTForward:
    def __init__(self, path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with path.open("rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        dtype_map = {
            trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool,
        }
        self.dtypes = {name: dtype_map[self.engine.get_tensor_dtype(name)]
                       for name in (*INPUTS, *OUTPUTS)}
        self.calls = 0

    def __call__(self, batch):
        if "option_ids" in batch or batch["marker_pos"].shape[1] != 2:
            raise ValueError("This engine profile expects two-option sequential Laya questions")
        inputs = {name: batch[name].to(device="cuda", dtype=self.dtypes[name]).contiguous()
                  for name in INPUTS}
        for name, tensor in inputs.items():
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"{name} shape {tuple(tensor.shape)} is outside the TensorRT profile")
            self.context.set_tensor_address(name, tensor.data_ptr())
        outputs = {
            name: torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=self.dtypes[name], device="cuda")
            for name in OUTPUTS
        }
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT failed to execute the Laya engine")
        self.calls += 1
        return tuple(outputs.values())


def serve(path, port):
    import tensorrt as trt
    import uvicorn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TensorRT benchmark")
    os.environ["LAYA_DEVICE"] = "cpu"
    router = build_router()
    agent = router.load("multilingual")
    forward = TensorRTForward(path)
    # The host Agent retains tokenization and decoding. TensorRT owns the CUDA forward.
    agent.device = torch.device("cuda")
    agent._infer = forward
    long_state = Path("thor-long-state.txt").read_text(encoding="utf-8").strip()
    body = json.loads(request_body(5, long_state))
    result = router.predict(body["state"], body["questions"], model=body["model"])
    if forward.calls != 1 or result["answers"].keys() != body["questions"].keys():
        raise RuntimeError("TensorRT warm-up did not complete through the engine")
    print(json.dumps({
        "engine": str(path), "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(),
        "calls": forward.calls, "warmup_usage": result["usage"],
    }), flush=True)
    uvicorn.run(create_app(router), host="127.0.0.1", port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "serve"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--port", type=int, default=18049)
    args = parser.parse_args()
    if args.mode == "export":
        export_onnx(args.path)
    else:
        serve(args.path, args.port)


if __name__ == "__main__":
    main()

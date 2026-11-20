"""Serve one pinned Laya checkpoint through its upstream HTTP API."""

import argparse
import hmac
import json
import os
from importlib.metadata import version
from pathlib import Path

import torch
import uvicorn

from bench_noul import request_body
from runtime import build_router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdpa", "fa2", "trt"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fa2-source", choices=("standalone", "vllm"), default="standalone")
    parser.add_argument("--vllm-fa2-dir", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18047)
    parser.add_argument("--proof-only", action="store_true")
    args = parser.parse_args()
    if args.backend == "trt" and not args.engine:
        parser.error("--engine is required for TensorRT")
    if args.backend == "fa2" and args.fa2_source == "vllm" and not args.vllm_fa2_dir:
        parser.error("--vllm-fa2-dir is required with --fa2-source vllm")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and a usable GPU are required")

    router = build_router(args.model_dir, "cpu" if args.backend == "trt" else "cuda")
    agent = router.load("multilingual")
    counters = {}
    trt_forward = None
    if args.backend == "fa2":
        from fa2_backend import install_fa2
        counters = install_fa2(agent.model.encoder, args.fa2_source, args.vllm_fa2_dir)
    if args.backend == "trt":
        from trt_backend import TensorRTForward
        trt_forward = TensorRTForward(args.engine)
        agent.device = torch.device("cuda")
        agent._infer = trt_forward

    long_state = (Path(__file__).resolve().parents[1] / "fixtures/recordings-long.txt").read_text()
    body = json.loads(request_body(5, long_state))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = router.predict(body["state"], body["questions"], model="multilingual")
        torch.cuda.synchronize()
    events = {
        event.key: event.count
        for event in profile.key_averages()
        if "scaled_dot_product" in event.key or "_vllm_fa2_C" in event.key
    }
    if agent.device.type != "cuda" or agent.cpu_fallback_count:
        raise RuntimeError("Warm-up used CPU fallback")
    if result["answers"].keys() != body["questions"].keys():
        raise RuntimeError("Warm-up returned incomplete answers")
    if not all(0 <= row["noul"] <= 1 for row in result["answers"].values()):
        raise RuntimeError("Warm-up returned invalid probabilities")
    if args.backend == "fa2" and sum(counters.values()) != len(agent.model.encoder.layers):
        raise RuntimeError("FA2 did not execute every encoder attention layer")
    if args.backend == "sdpa" and not any("scaled_dot_product" in key for key in events):
        raise RuntimeError("SDPA was not observed in warm-up")
    if args.backend == "trt" and trt_forward.calls == 0:
        raise RuntimeError("TensorRT did not execute")

    def status():
        return {
            "backend": args.backend,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "device": str(agent.device),
            "weight_dtype": str(next(agent.model.parameters()).dtype),
            "compute_policy": "TensorRT mixed FP16" if trt_forward else str(agent.dtype_for(5)),
            "encoder_attention": agent.model.encoder.config._attn_implementation,
            "fa2_calls": dict(counters),
            "trt_calls": trt_forward.calls if trt_forward else 0,
            "cpu_fallback_count": agent.cpu_fallback_count,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
            "laya": version("laya"),
            "transformers": version("transformers"),
            "warmup_operator_events": events,
        }

    print(json.dumps(status(), ensure_ascii=False), flush=True)
    if args.proof_only:
        return
    from laya.serve import create_app
    app = create_app(router)
    from fastapi import Header, HTTPException

    def protected_status(authorization: str | None = Header(default=None)):
        api_key = os.environ.get("LAYA_API_KEY")
        if api_key and not hmac.compare_digest(
            (authorization or "").encode(), f"Bearer {api_key}".encode(),
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return status()

    app.add_api_route("/jetson/status", protected_status, methods=["GET"])
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

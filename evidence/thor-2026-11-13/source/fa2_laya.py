"""Run the Laya benchmark with the FA2 extension borrowed from vLLM."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import uvicorn
from borrowed_fa2.flash_attn_interface import (
    FA2_AVAILABLE,
    FA2_UNAVAILABLE_REASON,
    flash_attn_varlen_func,
)
from laya.serve import build_router, create_app
from transformers.models.modernbert import modeling_modernbert

from bench_noul import request_body


def pack_mask(attention_mask, output_attentions=False):
    valid = attention_mask.bool()
    indices = valid.flatten().nonzero().flatten()
    lengths = valid.sum(dim=1, dtype=torch.int32)
    offsets = torch.nn.functional.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
    return (indices, offsets, attention_mask.shape[1]), None


def install_fa2(encoder):
    calls = Counter()

    def attention(module, qkv, attention_mask, position_ids, local_attention, bs, dim, **kwargs):
        indices, offsets, length = attention_mask
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = modeling_modernbert.apply_rotary_pos_emb(query, key, cos, sin)
        if query.device.type != "cuda" or query.dtype != torch.float16:
            raise ValueError("FA2 benchmark requires CUDA FP16")
        padded = indices.numel() != bs * length
        tensors = [x.transpose(1, 2).reshape(bs * length, module.num_heads, module.head_dim)
                   for x in (query, key, value)]
        if padded:
            tensors = [x.index_select(0, indices) for x in tensors]
        output = flash_attn_varlen_func(
            *tensors, length, offsets, length, offsets,
            causal=False, window_size=local_attention, fa_version=2,
        ).reshape(-1, dim)
        if padded:
            output = output.new_zeros((bs * length, dim)).index_copy_(0, indices, output)
        calls["global" if local_attention == (-1, -1) else "sliding"] += 1
        return (output.reshape(bs, length, dim),)

    modeling_modernbert.MODERNBERT_ATTENTION_FUNCTION["vllm_fa2"] = attention
    encoder.config._attn_implementation = "vllm_fa2"
    encoder._update_attention_mask = pack_mask
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18048)
    args = parser.parse_args()
    if not FA2_AVAILABLE:
        raise RuntimeError(FA2_UNAVAILABLE_REASON)
    router = build_router()
    agent = router.load("multilingual")
    calls = install_fa2(agent.model.encoder)
    long_state = Path("thor-long-state.txt").read_text(encoding="utf-8").strip()
    body = json.loads(request_body(5, long_state))
    result = router.predict(body["state"], body["questions"], model=body["model"])
    if sum(calls.values()) != len(agent.model.encoder.layers) or agent.cpu_fallback_count:
        raise RuntimeError(f"Unexpected FA2 path: {calls}, CPU fallbacks {agent.cpu_fallback_count}")
    if result["answers"].keys() != body["questions"].keys():
        raise RuntimeError("The FA2 warm-up returned an incomplete response")
    print(json.dumps({"attention": "vllm_fa2", "calls": calls, "device": str(agent.device),
                      "dtype": str(agent.dtype_for(5)), "warmup_usage": result["usage"]}), flush=True)
    uvicorn.run(create_app(router), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

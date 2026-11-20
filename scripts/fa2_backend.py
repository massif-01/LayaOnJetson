"""Adapt standalone or vLLM FA2 to the pinned ModernBERT encoder."""

import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

import torch
from transformers.models.modernbert import modeling_modernbert as modernbert


def load_vllm_fa2(directory):
    directory = directory.resolve()
    package_name = "_laya_vllm_fa2"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.flash_attn_interface", directory / "flash_attn_interface.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not module.FA2_AVAILABLE:
        raise RuntimeError(module.FA2_UNAVAILABLE_REASON)

    def forward(q, k, v, offsets, length, window):
        return module.flash_attn_varlen_func(
            q, k, v, length, offsets, length, offsets,
            causal=False, window_size=window, fa_version=2,
        )

    return forward


def load_standalone_fa2():
    from flash_attn import flash_attn_varlen_func

    def forward(q, k, v, offsets, length, window):
        return flash_attn_varlen_func(
            q, k, v, offsets, offsets, length, length,
            causal=False, window_size=window,
        )

    return forward


def install_fa2(encoder, source, vllm_directory=None):
    if encoder.config.model_type != "modernbert":
        raise ValueError("This adapter requires a ModernBERT-family encoder")
    forward = load_vllm_fa2(Path(vllm_directory)) if source == "vllm" else load_standalone_fa2()
    calls = Counter()

    def pack_mask(attention_mask, output_attentions=False):
        if output_attentions or attention_mask.ndim != 2:
            raise ValueError("FA2 adapter requires a 2D padding mask and output_attentions=False")
        indices = attention_mask.bool().flatten().nonzero().flatten()
        lengths = attention_mask.sum(dim=1, dtype=torch.int32)
        offsets = torch.nn.functional.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
        return (indices, offsets, attention_mask.shape[1]), None

    def attention(module, qkv, attention_mask, position_ids, local_attention, bs, dim, **kwargs):
        indices, offsets, length = attention_mask
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = modernbert.apply_rotary_pos_emb(query, key, cos, sin)
        if query.device.type != "cuda" or query.dtype != torch.float16:
            raise ValueError("FA2 adapter requires CUDA FP16 tensors")
        padded = indices.numel() != bs * length
        tensors = [
            tensor.transpose(1, 2).reshape(bs * length, module.num_heads, module.head_dim)
            for tensor in (query, key, value)
        ]
        if padded:
            tensors = [tensor.index_select(0, indices) for tensor in tensors]
        output = forward(*tensors, offsets, length, local_attention).reshape(-1, dim)
        if padded:
            output = output.new_zeros((bs * length, dim)).index_copy_(0, indices, output)
        calls["global" if local_attention == (-1, -1) else "sliding"] += 1
        return (output.reshape(bs, length, dim),)

    modernbert.MODERNBERT_ATTENTION_FUNCTION["jetson_fa2"] = attention
    encoder.config._attn_implementation = "jetson_fa2"
    encoder._update_attention_mask = pack_mask
    return calls

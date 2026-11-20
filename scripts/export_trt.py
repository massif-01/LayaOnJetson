"""Export the pinned Laya model for the measured 1–5-question TensorRT profile."""

import argparse
from pathlib import Path

import torch

from runtime import build_router
from trt_backend import INPUTS, OUTPUTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    model = build_router(args.model_dir, "cpu").load("multilingual").model.eval()
    mask = torch.ones((2, 96), dtype=torch.long)
    mask[1, 80:] = 0
    example = (
        torch.randint(5, 1000, (2, 96)),
        mask,
        torch.tensor([[5, 20], [6, 21]]),
        torch.ones((2, 2), dtype=torch.bool),
        torch.tensor([1, 1]),
    )
    axes = {name: {0: "batch"} for name in (*INPUTS, *OUTPUTS)}
    for name in ("input_ids", "attention_mask"):
        axes[name][1] = "sequence"
    torch.onnx.export(
        model, example, str(args.output), opset_version=18, dynamo=False,
        input_names=list(INPUTS), output_names=list(OUTPUTS), dynamic_axes=axes,
    )
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()

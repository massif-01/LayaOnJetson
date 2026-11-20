"""Exercise CUDA FP16 and record the exact installed software and Laya files."""

import hashlib
import json
import platform
from importlib.metadata import distribution, version
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    tensor = torch.ones((64, 64), dtype=torch.float16, device="cuda")
    if (tensor @ tensor)[0, 0].item() != 64:
        raise RuntimeError("CUDA FP16 matrix multiplication failed")
    laya_path = Path(distribution("laya").locate_file("laya"))
    versions = {name: version(name) for name in ("torch", "laya", "transformers")}
    print(json.dumps({
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch_arches": torch.cuda.get_arch_list(),
        "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
        "laya_source_sha256": {
            name: hashlib.sha256((laya_path / name).read_bytes()).hexdigest()
            for name in ("agent.py", "common.py", "serve.py")
        },
    }, indent=2))


if __name__ == "__main__":
    main()

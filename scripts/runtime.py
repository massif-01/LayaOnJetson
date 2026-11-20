"""Shared model registration for the command-line entry points."""

import os

from download_model import verify_model


def build_router(model_directory, device):
    import json
    from laya.serve import build_router as upstream_router

    verify_model(model_directory)
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LAYA_DEVICE": device,
        "LAYA_CUDA_AMP": "fp16",
        "LAYA_MODELS": "multilingual",
        "LAYA_PRELOAD": "1",
        "LAYA_AUTO_TASK": "0",
        "LAYA_IDLE_UNLOAD_SECONDS": "0",
        "LAYA_TOURNAMENT_GROUP": "0",
        "LAYA_EXTRA_MODELS": json.dumps({"multilingual": str(model_directory.resolve())}),
    })
    return upstream_router()

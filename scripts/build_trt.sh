#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash scripts/build_trt.sh INPUT.onnx OUTPUT.engine\n' >&2
    exit 2
fi
trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$(dirname "$2")"
"$trtexec" --onnx="$1" --fp16 --skipInference --builderOptimizationLevel=3 \
    --minShapes=input_ids:1x8,attention_mask:1x8,marker_pos:1x2,marker_mask:1x2,qtype:1 \
    --optShapes=input_ids:5x512,attention_mask:5x512,marker_pos:5x2,marker_mask:5x2,qtype:5 \
    --maxShapes=input_ids:5x1024,attention_mask:5x1024,marker_pos:5x2,marker_mask:5x2,qtype:5 \
    --memPoolSize=workspace:2048 --saveEngine="$2" \
    --timingCacheFile="$2.timing.cache" --profilingVerbosity=detailed

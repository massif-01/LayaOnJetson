#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    printf 'Usage: bash scripts/create_env.sh ENV PYTHON_VERSION TORCH_WHEEL sdpa|trt\n' >&2
    exit 2
fi
env_name="$1"
python_version="$2"
torch_wheel="$3"
mode="$4"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$mode" in
    sdpa) requirements="$root/requirements/common.txt" ;;
    trt) requirements="$root/requirements/trt.txt" ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac

conda create -y -n "$env_name" "python=$python_version" pip 'setuptools<82'
conda run -n "$env_name" python -m pip install "$torch_wheel"
torch_pin="$(conda run -n "$env_name" python -c 'from importlib.metadata import version; print("torch==" + version("torch"))')"
constraint="$(mktemp)"
trap 'rm -f "$constraint"' EXIT
printf '%s\n' "$torch_pin" > "$constraint"
conda run -n "$env_name" python -m pip install -c "$constraint" -r "$requirements"
conda run -n "$env_name" python -m pip check
printf 'Environment ready. Activate with: conda activate %s\n' "$env_name"

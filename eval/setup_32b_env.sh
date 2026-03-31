#!/bin/bash
set -euo pipefail

current_file_path=$(dirname "$(realpath "$0")")

python -m pip install -r "$current_file_path/requirements_32b.txt"
python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY

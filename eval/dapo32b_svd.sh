#!/bin/bash
set -euo pipefail

current_file_path=$(dirname "$(realpath "$0")")

python "$current_file_path/dapo32b_svd.py" \
  --base_model_path "$current_file_path/models/dapo32b/base" \
  --target_model_path "$current_file_path/models/dapo32b/full" \
  --output_dir "$current_file_path/models/dapo32b" \
  --rank 1 \
  --param_ratio 0.01

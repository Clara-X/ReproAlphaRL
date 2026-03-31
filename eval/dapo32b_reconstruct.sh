#!/bin/bash
set -euo pipefail

current_file_path=$(dirname "$(realpath "$0")")

python "$current_file_path/dapo32b_reconstruct.py" \
  --base_model_path "$current_file_path/models/dapo32b/base" \
  --svd_path "$current_file_path/models/dapo32b/svd_components.pt" \
  --output_dir "$current_file_path/models/dapo32b/rank_1pct" \
  --rank -1 \
  --alpha 1.0

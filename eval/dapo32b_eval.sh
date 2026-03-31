#!/bin/bash
set -euo pipefail

current_file_path=$(dirname "$(realpath "$0")")

python "$current_file_path/dapo32b_eval.py" \
  --full_model_path "$current_file_path/models/dapo32b/full" \
  --rank1_model_path "$current_file_path/models/dapo32b/rank_1" \
  --rank1pct_model_path "$current_file_path/models/dapo32b/rank_1pct" \
  --output_dir "$current_file_path/output/dapo32b_runthrough" \
  --cuda_visible_devices "0,1,2,3" \
  --temperature 0.6 \
  --n_sampling 4 \
  --k 1 \
  --split test \
  --max_tokens 30000 \
  --seed 0 \
  --model_keys full,rank_1pct \
  --skip_existing \
  "$@"

#!/bin/bash
set -euo pipefail

current_file_path=$(dirname "$(realpath "$0")")

python "$current_file_path/dapo32b_prepare.py" \
  --base_dir "$current_file_path/models/dapo32b/base" \
  --full_dir "$current_file_path/models/dapo32b/full" \
  --existing_8b_path "$current_file_path/models/DAPO-step-0" \
  --smoke-test

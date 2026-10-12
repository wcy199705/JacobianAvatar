#!/bin/bash
# Mesh training only (Pre → L0 → L1 → All). Gaussian training is skipped. On the first training, frame validation runs automatically first.
# Usage: ./03_skip_gs_mesh_only.sh <dataset_yaml> <gpu_id>
set -eu
exec bash "$(dirname "$0")/01_train_custom.sh" "${1:?Usage: $0 <dataset_yaml> <gpu_id>}" "${2:?Usage: $0 <dataset_yaml> <gpu_id>}" --skip_gs

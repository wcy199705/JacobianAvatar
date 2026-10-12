#!/bin/bash
# Load and check the config only, without training (pre-check of the extends chain / path errors).
# Usage: ./05_dry_run_config_check.sh <dataset_yaml>
#   e.g.: ./05_dry_run_config_check.sh configs/datasets/Custom-my_scene.yaml
set -eu
YAML=${1:?Usage: $0 <dataset_yaml>}
source "$(dirname "$0")/_env.sh"
cd "$REPO"
$PY main.py --config "$YAML" --dry_run

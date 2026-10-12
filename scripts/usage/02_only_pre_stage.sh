#!/bin/bash
# Run frame validation only (initial body-estimation Pre stage -> creates <training_folder>/joint_valid_frames.txt).
# Same step that 01_train_custom.sh runs automatically on the first training. Use it to inspect the list first.
# To regenerate the list, delete <training_folder>/joint_valid_frames.txt and rerun.
# Usage: ./02_only_pre_stage.sh <dataset_yaml> <gpu_id>
set -eu
YAML=${1:?Usage: $0 <dataset_yaml> <gpu_id>}
GPU=${2:?Usage: $0 <dataset_yaml> <gpu_id>}
source "$(dirname "$0")/_env.sh"
cd "$REPO"
apply_recipe_env "$YAML"
TRAIN_FOLDER=$($PY - "$YAML" <<'PY' 2>/dev/null | tail -1
import sys, yaml
sys.path.insert(0, '.')
from src._bootstrap import install_path; install_path()
from src.config import load_config
cfg = load_config(sys.argv[1])
print(yaml.safe_load(open(f'configs/legacy/SMPLX_dataloader/CUSTOM/{cfg.scene_name}.yaml'))['Path']['training_folder'])
PY
)
[ -d "$TRAIN_FOLDER" ] || { echo "[FATAL] training_folder not found: '$TRAIN_FOLDER'"; exit 1; }
[ -f "$TRAIN_FOLDER/joint_valid_frames.txt" ] && { echo "Already exists: $TRAIN_FOLDER/joint_valid_frames.txt (delete it and rerun to regenerate)"; exit 0; }
TMP_YAML="$(dirname "$YAML")/_frame_validation_$(basename "$YAML")"
printf 'extends: ./%s\nsave_root_override: %s/_frame_validation/run\n' "$(basename "$YAML")" "$TRAIN_FOLDER" > "$TMP_YAML"
trap 'rm -f "$TMP_YAML"' EXIT
env CUDA_VISIBLE_DEVICES=$GPU $TRAIN_ENV $CUSTOM_ENV $PY scripts/main_hard_zero_headhand.py --config "$TMP_YAML" --only_pre
echo "Valid frames: $(grep -c . "$TRAIN_FOLDER/joint_valid_frames.txt") / excluded: $(grep -c . "$TRAIN_FOLDER/joint_invalid_frames.txt" 2>/dev/null || echo 0)"

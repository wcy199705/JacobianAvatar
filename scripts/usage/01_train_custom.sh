#!/bin/bash
# Custom scene training: Pre → L0 → L1 → All → GS#1.
#
# On the first training of a scene, frame validation runs first to create <training_folder>/joint_valid_frames.txt
# (excluding frames where the body estimate does not match the mask). All later training uses the frames filtered by this list.
# Validation runs in a separate location inside the data folder (_frame_validation/), so it does not mix with the main training output folder.
#
# Usage: ./01_train_custom.sh <dataset_yaml> <gpu_id> [extra flags]
#   e.g.: ./01_train_custom.sh configs/datasets/Custom-my_scene.yaml 2
#       ./01_train_custom.sh configs/datasets/Custom-my_scene.yaml 2 --skip_gs    # mesh stages only
set -eu
YAML=${1:?Usage: $0 <dataset_yaml> <gpu_id> [flags]}
GPU=${2:?Usage: $0 <dataset_yaml> <gpu_id> [flags]}
shift 2
source "$(dirname "$0")/_env.sh"
cd "$REPO"
apply_recipe_env "$YAML"

TRAIN_FOLDER=$($PY - "$YAML" <<'PY' 2>/dev/null | tail -1
import sys, yaml
sys.path.insert(0, '.')
from src._bootstrap import install_path; install_path()
from src.config import load_config
cfg = load_config(sys.argv[1])
scene_yaml = f'configs/legacy/SMPLX_dataloader/CUSTOM/{cfg.scene_name}.yaml'
print(yaml.safe_load(open(scene_yaml))['Path']['training_folder'])
PY
)
[ -d "$TRAIN_FOLDER" ] || { echo "[FATAL] training_folder not found: '$TRAIN_FOLDER' (check the scene yaml)"; exit 1; }

if [ ! -f "$TRAIN_FOLDER/joint_valid_frames.txt" ]; then
  echo "=== [1/2] frame validation (Pre stage only, output: $TRAIN_FOLDER/_frame_validation) ==="
  TMP_YAML="$(dirname "$YAML")/_frame_validation_$(basename "$YAML")"
  printf 'extends: ./%s\nsave_root_override: %s/_frame_validation/run\n' "$(basename "$YAML")" "$TRAIN_FOLDER" > "$TMP_YAML"
  trap 'rm -f "$TMP_YAML"' EXIT
  env CUDA_VISIBLE_DEVICES=$GPU $TRAIN_ENV $CUSTOM_ENV $PY scripts/main_hard_zero_headhand.py --config "$TMP_YAML" --only_pre
  rm -f "$TMP_YAML"; trap - EXIT
  [ -f "$TRAIN_FOLDER/joint_valid_frames.txt" ] || { echo "[FATAL] frame validation list was not created"; exit 1; }
  echo "Valid frames: $(grep -c . "$TRAIN_FOLDER/joint_valid_frames.txt") / excluded: $(grep -c . "$TRAIN_FOLDER/joint_invalid_frames.txt" 2>/dev/null || echo 0)"
  echo "=== [2/2] main training ==="
fi

env CUDA_VISIBLE_DEVICES=$GPU $TRAIN_ENV $CUSTOM_ENV $PY scripts/main_hard_zero_headhand.py --config "$YAML" --skip_gs2 "$@"

#!/bin/bash
# Visualize training results (360-degree orbit render + video).
#   Mesh_RGB / P_normal → All stage ckpt,  GS_RGB → GS#1 ckpt  (rendered twice, then merged)
# Usage: ./04_render_viz.sh <dataset_yaml> <out_dir> <gpu_id> [render_orbit.py options]
#   e.g.: ./04_render_viz.sh configs/datasets/Custom-my_scene.yaml outputs/my_scene/viz 2 --mode canon --num_angles 72
#   The ckpt location is found automatically from save_root_override in the dataset yaml.
set -eu
YAML=${1:?Usage: $0 <dataset_yaml> <out_dir> <gpu_id> [options]}
OUT_DIR=${2:?}
GPU=${3:?}
shift 3
source "$(dirname "$0")/_env.sh"
cd "$REPO"
SAVE_ROOT=$($PY -c "
import sys; sys.path.insert(0,'.')
from src._bootstrap import install_path; install_path()
from src.config import load_config
print(load_config('$YAML').save_root_override)" 2>/dev/null | tail -1)
CKPT_DIR=$(ls -d "$SAVE_ROOT"/3DGS_SMPLX_Version_poisson/AFTER_ECCV_*/ 2>/dev/null | head -1)
ALL_CKPT=$(ls "$CKPT_DIR"/All_iter_*_Avatar.pth 2>/dev/null | head -1)
GS_CKPT=$(ls "$CKPT_DIR"/GS_iter_*_Avatar.pth 2>/dev/null | head -1)
[ -f "$ALL_CKPT" ] || { echo "All ckpt not found: $CKPT_DIR"; exit 1; }
[ -f "$GS_CKPT" ]  || { echo "GS#1 ckpt not found: $CKPT_DIR"; exit 1; }
echo "=== 1/2: All ckpt → Mesh_RGB, P_normal ==="
env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rendering/render_orbit.py --config "$YAML" --ckpt_path "$ALL_CKPT" --out_dir "$OUT_DIR/allckpt" "$@"
echo "=== 2/2: GS#1 ckpt → GS_RGB ==="
env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rendering/render_orbit.py --config "$YAML" --ckpt_path "$GS_CKPT" --out_dir "$OUT_DIR/gs1" "$@"
echo "=== Merge + video ==="
for sub in $(cd "$OUT_DIR/allckpt" && ls -d */ | tr -d /); do
  M="$OUT_DIR/merged/$sub"; mkdir -p "$M"
  ln -sfn "$(cd "$OUT_DIR/allckpt/$sub" && pwd)/Mesh_RGB_rgba" "$M/Mesh_RGB_rgba"
  ln -sfn "$(cd "$OUT_DIR/allckpt/$sub" && pwd)/P_normal_rgba" "$M/P_normal_rgba"
  ln -sfn "$(cd "$OUT_DIR/gs1/$sub" && pwd)/GS_RGB_rgba" "$M/GS_RGB_rgba"
  $PY scripts/rendering/build_videos.py "$M" --fps 10 || true
done
echo "Output: $OUT_DIR/merged"

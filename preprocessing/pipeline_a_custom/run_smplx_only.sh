#!/usr/bin/env bash
# SMPL-X pose estimation only (for generating Novel Pose drivers).
# Use when only a pose JSON export is needed, not training data.
#
# skips vs full pipeline:
#   step 4 (Sapiens2 normals)  — not used for pose optimization
#   step 7 (head crop)         — for the GS#1/GS#2 head loss, not needed for pose
#
# Remaining steps: 1a → 2 → mv → 1b → 3 → 5 → 6
#
# Usage:
#   ./run_smplx_only.sh <scene_basename> <gpu_id>
#   e.g.  bash run_smplx_only.sh my_scene 2
#
# After completion, export to novel-pose driver format:
#   python export_to_novel_pose.py --scene_root <path> --out_root <dst>
#
set -u
SCENE="${1:?usage: $0 <scene_basename> <gpu_id>}"
GPU="${2:?usage: $0 <scene_basename> <gpu_id>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_scene.sh" "$SCENE" "$GPU" "1a,2,mv,1b,3,5,6"

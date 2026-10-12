#!/bin/bash
# Preprocess captured video (frame images) into training data (7 steps + finalize).
# Prerequisites: <scene>_training/images/*.png (+ optional K.npy), configs/legacy/SMPLX_dataloader/CUSTOM/<scene>.yaml
# Usage: ./00_preprocess_custom.sh <scene> <gpu_id> [steps]
#   e.g.: ./00_preprocess_custom.sh my_scene 2
#       ./00_preprocess_custom.sh my_scene 2 5,6,7,finalize   # rerun selected steps only
set -eu
SCENE=${1:?Usage: $0 <scene> <gpu_id> [steps]}
GPU=${2:?Usage: $0 <scene> <gpu_id> [steps]}
source "$(dirname "$0")/_env.sh"
bash "$REPO/preprocessing/pipeline_a_custom/run_scene.sh" "$SCENE" "$GPU" ${3:+"$3"}
echo "Log: ${LOG_DIR:-$REPO/logs/preprocess}/$SCENE.log"

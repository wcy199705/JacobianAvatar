#!/usr/bin/env bash
# Run full 7-step preprocessing for ONE CUSTOM scene.
# Steps 1,2,3,5,6,7 use the jacobian_avatar_prep env via the meta wrapper.
# Step 4 (Sapiens2 normals) uses the sapiens2 env via a dedicated CUSTOM wrapper.
# Between step1 and step2, first_bbox.txt is auto-created from DWPose output
# (mmpose_result_mmpose/bboxes/<first_frame>.txt — already in the x1,y1,x2,y2
#  format SAMURAI expects).
#
# Usage:
#   bash preprocessing/pipeline_a_custom/run_scene.sh <scene_yaml_basename> <gpu_id> [steps]
#   (scene yaml: configs/legacy/SMPLX_dataloader/CUSTOM/<scene_yaml_basename>.yaml)
#   e.g.  bash preprocessing/pipeline_a_custom/run_scene.sh my_scene 2
#         bash preprocessing/pipeline_a_custom/run_scene.sh my_scene 2 3,4,5
set -u
set -o pipefail

SCENE="${1:?usage: $0 <scene_basename> <gpu_id> [steps]}"
GPU="${2:?usage: $0 <scene_basename> <gpu_id> [steps]}"
# Default to mask-first pipeline:
#   1a → 2 (SAMURAI) → mv (masked debug video) → 1b → 3,4,5,6,7 → finalize
#   finalize: create link normals -> crop_normals_sapiens2 (the training loader reads normals/)
STEPS="${3:-1a,2,mv,1b,3,4,5,6,7,finalize}"

# Repo root = two levels above this script (preprocessing/pipeline_a_custom/).
JA_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Legacy preprocessing modules (scene/, render/, utils/) bundled under preprocessing/legacy.
LEGACY=$JA_ROOT/preprocessing/legacy
# Conda envs (override via env vars if installed elsewhere).
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null)}
PREP_PY=${PREP_PY:-$CONDA_BASE/envs/jacobian_avatar_prep/bin/python}
PREP_PATH=$(dirname "$PREP_PY")
SAPIENS_PY=${SAPIENS_PY:-$CONDA_BASE/envs/sapiens2/bin/python}
EGL_LIB=$JA_ROOT/libs/MHR/.pixi/envs/default/lib

# Thread cap — avoid CPU thrashing when 4 scenes run in parallel + other
# training jobs are active. PyTorch/numpy/MKL default to nproc, which thrashes.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
# CUDA allocator: reduce fragmentation for long per-frame loops in step6.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WRAPPER=$JA_ROOT/preprocessing/pipeline_a_custom/run_step.py
STEP4=$JA_ROOT/preprocessing/pipeline_a_custom/steps/step4_sapiens2_normals.py
MAKE_K=$JA_ROOT/preprocessing/pipeline_a_custom/utils/make_K_npy.py
MAKE_MASKED_VIDEO=$JA_ROOT/preprocessing/pipeline_a_custom/utils/make_masked_video.py

YAML=$JA_ROOT/configs/legacy/SMPLX_dataloader/CUSTOM/${SCENE}.yaml
if [[ ! -f $YAML ]]; then
  echo "[FATAL] YAML not found: $YAML" >&2
  exit 1
fi

# Resolve training_folder from YAML
SCENE_ROOT=$(/usr/bin/env python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['Path']['training_folder'])")
echo "[run_custom_scene] scene=$SCENE  gpu=$GPU  steps=$STEPS  root=$SCENE_ROOT"

LOG_DIR=${LOG_DIR:-$JA_ROOT/logs/preprocess}
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/${SCENE}.log
echo "[$(date +%H:%M:%S)] === START scene=$SCENE gpu=$GPU steps=$STEPS ===" >> "$LOG"

# -------- helpers --------
prep_env_step() {
  local step=$1
  echo "[$(date +%H:%M:%S)] --- step $step (prep env) ---" >> "$LOG"
  ( cd "$JA_ROOT" && \
    CUDA_VISIBLE_DEVICES=$GPU \
    PATH=$PREP_PATH:$PATH \
    PYTHONPATH=.:$LEGACY \
    $PREP_PY "$WRAPPER" --step "$step" --scene "$SCENE" ) >> "$LOG" 2>&1
}

step5_with_egl() {
  echo "[$(date +%H:%M:%S)] --- step 5 (prep env + EGL) ---" >> "$LOG"
  ( cd "$JA_ROOT" && \
    CUDA_VISIBLE_DEVICES=$GPU \
    PATH=$PREP_PATH:$PATH \
    PYTHONPATH=.:$LEGACY \
    LD_LIBRARY_PATH=$EGL_LIB:${LD_LIBRARY_PATH:-} \
    $PREP_PY "$WRAPPER" --step 5 --scene "$SCENE" ) >> "$LOG" 2>&1
}

step4_sapiens2() {
  echo "[$(date +%H:%M:%S)] --- step 4 (sapiens2 env) ---" >> "$LOG"
  ( cd "$JA_ROOT" && \
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONPATH=$JA_ROOT/libs/sapiens2:. \
    $SAPIENS_PY "$STEP4" --scene "$SCENE" ) >> "$LOG" 2>&1
}

copy_first_bbox() {
  # DWPose writes bboxes/00000.txt etc. (x1,y1,x2,y2). SAMURAI needs first_bbox.txt
  # in the same format at the scene root.
  local src first
  src="$SCENE_ROOT/mmpose_result_mmpose/bboxes"
  if [[ ! -d $src ]]; then
    echo "[FATAL] bbox dir missing — step1 did not run? $src" >> "$LOG"
    return 1
  fi
  first=$(ls "$src"/*.txt 2>/dev/null | head -1)
  if [[ -z $first ]]; then
    echo "[FATAL] no bbox txt files under $src" >> "$LOG"
    return 1
  fi
  cp "$first" "$SCENE_ROOT/first_bbox.txt"
  echo "[$(date +%H:%M:%S)] first_bbox.txt <- $(basename $first): $(cat $SCENE_ROOT/first_bbox.txt)" >> "$LOG"
}

# -------- auto-generate K.npy (idempotent: skips if already present) --------
echo "[$(date +%H:%M:%S)] --- ensure K.npy ---" >> "$LOG"
$PREP_PY "$MAKE_K" --scene "$SCENE" >> "$LOG" 2>&1 || {
  echo "[$(date +%H:%M:%S)] !!! make_K_npy failed — aborting" >> "$LOG"
  exit 1
}

# -------- main loop --------
IFS=',' read -ra STEP_ARR <<< "$STEPS"
for s in "${STEP_ARR[@]}"; do
  case "$s" in
    1)  prep_env_step 1 ;;
    1a) prep_env_step 1a ;;
    1b) prep_env_step 1b ;;
    2)
      [[ -f $SCENE_ROOT/first_bbox.txt ]] || copy_first_bbox
      prep_env_step 2 ;;
    mv)
      echo "[$(date +%H:%M:%S)] --- masked debug video ---" >> "$LOG"
      $PREP_PY "$MAKE_MASKED_VIDEO" --scene "$SCENE" >> "$LOG" 2>&1 || \
        echo "[$(date +%H:%M:%S)] !!! make_masked_video failed (non-fatal)" >> "$LOG"
      ;;
    3) prep_env_step 3 ;;
    4) step4_sapiens2 ;;
    5) step5_with_egl ;;
    6) prep_env_step 6 ;;
    7) prep_env_step 7 ;;
    finalize)
      ln -sfn crop_normals_sapiens2 "$SCENE_ROOT/normals"
      echo "[$(date +%H:%M:%S)] finalize: normals -> crop_normals_sapiens2" >> "$LOG"
      case "$(basename "$SCENE_ROOT")" in
        *training*) ;;
        *) echo "[WARN] scene folder name has no 'training' — training loader needs '<scene>_training'" | tee -a "$LOG" ;;
      esac
      ;;
    *)
      echo "[FATAL] unknown step: $s" >> "$LOG"
      exit 1 ;;
  esac
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] !!! step $s FAILED (rc=$rc) — aborting scene $SCENE" >> "$LOG"
    exit $rc
  fi
done

echo "[$(date +%H:%M:%S)] === END scene=$SCENE OK ===" >> "$LOG"

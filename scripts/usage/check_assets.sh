#!/bin/bash
# Checks that weights/resources not included in the repository are in place (download: fetch_assets.sh).
# (files needed for preprocessing + training + rendering)
# Usage: bash scripts/usage/check_assets.sh
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
REQUIRED=(
  # External library sources (fetch_libs.sh)
  libs/MHR/mhr/mhr.py
  libs/WAFT/model/backbone/dinov3.py
  libs/WAFT/thirdparty/dinov3/hubconf.py
  libs/samurai/sam2/sam2/build_sam.py
  libs/PromptHMR/prompt_hmr/__init__.py
  libs/sapiens2/sapiens
  libs/mmpose/configs/_base_/default_runtime.py
  # MHR (SMPL-X → MHR mesh conversion)
  libs/MHR/assets/compact_v6_1.model
  libs/MHR/assets/corrective_activation.npz
  libs/MHR/assets/corrective_blendshapes_lod1.npz
  libs/MHR/assets/corrective_blendshapes_lod3.npz
  libs/MHR/assets/lod1.fbx
  libs/MHR/assets/lod3.fbx
  libs/MHR/.pixi/envs/default/bin/python
  libs/MHR/.pixi/envs/default/lib/libEGL.so
  # PromptHMR (preprocessing 5)
  libs/PromptHMR/data/pretrain/phmr/config.yaml
  libs/PromptHMR/data/pretrain/phmr/checkpoint.ckpt
  # Sapiens2 (preprocessing 4)
  libs/sapiens2/checkpoints/normal/sapiens2_1b_normal.safetensors
  # WAFT (training optical flow)
  libs/WAFT/waftv2-ckpts/dinov3/spring.pth
  # SAMURAI / SAM2.1 (preprocessing 2)
  libs/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt
  # DWPose / RTMDet (preprocessing 1a/1b)
  libs/mmpose/dw-ll_ucoco_384.pth
  libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
  # Body models (download after accepting the license)
  resource/smplx/SMPLX_NEUTRAL.pkl
  resource/smplx/SMPLX_NEUTRAL.npz
  resource/smpl/SMPL_NEUTRAL.pkl
  resource/model_transfer/smplx2smpl.pkl
  resource/smplx2smpl_joints.npy
  resource/body_segmentation/smplx/smplx_vert_segmentation.json
)
missing=0
for f in "${REQUIRED[@]}"; do
  if [ -e "$f" ]; then printf "  OK       %s\n" "$f"; else printf "  MISSING  %s\n" "$f"; missing=$((missing+1)); fi
done
if env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/python -c "import mhr, smplx, sklearn" 2>/dev/null; then
  echo "  OK       MHR pixi env: import mhr"
else
  echo "  MISSING  MHR pixi env: import mhr failed (see ASSETS.md, section 3 (manual install), item 2)"; missing=$((missing+1))
fi
if [ -e resource/smplx/SMPLX_NEUTRAL.npz ]; then
  PYN=libs/MHR/.pixi/envs/default/bin/python; [ -x "$PYN" ] || PYN=python3
  n=$(env -u PYTHONPATH "$PYN" -c "import numpy as np; print(np.load('resource/smplx/SMPLX_NEUTRAL.npz')['shapedirs'].shape[-1])" 2>/dev/null)
  if [ "$n" = "20" ]; then
    echo "  OK       SMPL-X v1.0 (shape 10 + expression 10)"
  else
    echo "  WRONG    resource/smplx/SMPLX_NEUTRAL.npz has ${n:-?} components (v1.0 = 20 required; re-download with fetch_assets.sh or see ASSETS.md)"; missing=$((missing+1))
  fi
fi
echo; [ $missing -eq 0 ] && echo "All required files found." || { echo "$missing item(s) missing — see scripts/usage/ASSETS.md"; exit 1; }

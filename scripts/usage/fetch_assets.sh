#!/bin/bash
# Downloads weights and body models not included in the repository and places them where expected (details: scripts/usage/ASSETS.md).
# Existing items are skipped, so it is safe to rerun after a failure.
#
# Usage: bash scripts/usage/fetch_assets.sh
#
# First fetches external library sources from GitHub via fetch_libs.sh.
# SMPL-X / SMPLify (SMPL) require registration on the official sites; username and password are prompted during the run.
#   https://smpl-x.is.tue.mpg.de   https://smplify.is.tue.mpg.de
# If you already have the zip files, point to them with environment variables to use them without logging in:
#   SMPLX_ZIP=models_smplx_v1_1.zip  SMPLIFY_ZIP=mpips_smplify_public_v2.zip
#
# Required tools: git, curl, unzip, pixi (MHR runtime env), python from the training conda env (with gdown, huggingface_hub)
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source scripts/usage/_env.sh
# Preflight: system tools and the training env (gdown, huggingface_hub) must be ready before downloading anything.
for t in git curl unzip; do
  command -v $t >/dev/null || { echo "[FATAL] '$t' not found — install it first (INSTALL.md, section 1)"; exit 1; }
done
"$PY" -c "import gdown, huggingface_hub" 2>/dev/null || {
  echo "[FATAL] '$PY' cannot import gdown / huggingface_hub — create the Jacobian_avatar env first (INSTALL.md, section 2) or set PY=/path/to/python"
  exit 1; }
bash scripts/usage/fetch_libs.sh || { echo "[FATAL] failed to fetch library sources"; exit 1; }
TMP=$REPO/.fetch_tmp
mkdir -p "$TMP"
FAILED=()

step() { echo; echo "=== $1 ==="; }
have() { [ -e "$1" ] && echo "  already exists: $1"; }
fail() { echo "  [FAILED] $1"; FAILED+=("$1"); }
gdown_() { $PY -m gdown "$@"; }

# Login download from the official site (download.is.tue.mpg.de). $1=domain $2=file name $3=output path $4=site name
tue_download() {
  local domain=$1 sfile=$2 out=$3 site=$4 user pass
  echo "  $site account ($domain) — register at: https://$domain.is.tue.mpg.de"
  read -r -p "  Username: " user
  read -r -s -p "  Password: " pass; echo
  curl -L --fail -o "$out" --data-urlencode "username=$user" --data-urlencode "password=$pass" \
    "https://download.is.tue.mpg.de/download.php?domain=$domain&sfile=$sfile&resume=1"
  # On login failure, an HTML page is returned instead of the zip — keep it (renamed) so the cause can be checked
  if ! unzip -tq "$out" >/dev/null 2>&1; then
    if [ -s "$out" ]; then
      mv -f "$out" "$out.html"
      echo "  Not a valid zip (wrong username/password or license not accepted?). Server response kept at: $out.html"
    fi
    return 1
  fi
}

# ---------------------------------------------------------------- 1. MHR model files
step "1. MHR assets (libs/MHR/assets)"
MHR_FILES="compact_v6_1.model corrective_activation.npz corrective_blendshapes_lod1.npz corrective_blendshapes_lod3.npz lod1.fbx lod3.fbx"
if ! have libs/MHR/assets/lod3.fbx; then
  curl -L --fail -o "$TMP/mhr_assets.zip" https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip \
    && (cd libs/MHR && unzip -oq "$TMP/mhr_assets.zip" $(for f in $MHR_FILES; do echo "assets/$f"; done)) \
    && rm -f "$TMP/mhr_assets.zip" || fail "MHR assets"
fi

# ---------------------------------------------------------------- 2. MHR runtime env
step "2. MHR runtime env (libs/MHR/.pixi)"
if ! have libs/MHR/.pixi/envs/default/bin/python; then
  # The pixi installer puts the binary in ~/.pixi/bin, which is not on PATH until a new shell is opened.
  PIXI=$(command -v pixi || ls "${PIXI_HOME:-$HOME/.pixi}/bin/pixi" 2>/dev/null)
  if [ -n "$PIXI" ]; then
    (cd libs/MHR && "$PIXI" install) || fail "pixi install"
  else
    fail "pixi not found — install from https://pixi.sh and rerun (curl -fsSL https://pixi.sh/install.sh | bash)"
  fi
fi
if [ -d libs/MHR/.pixi/envs/default/lib ]; then
  # Extra packages for MHR's SMPL(-X) conversion tool (tools/mhr_smpl_conversion/README.md), installed into the env only
  env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/python -c "import smplx, sklearn" 2>/dev/null || \
    env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/uv pip install --python libs/MHR/.pixi/envs/default/bin/python --no-deps \
      smplx==0.1.28 scikit-learn==1.8.0 scipy==1.17.1 tqdm==4.67.3 joblib==1.5.3 threadpoolctl==3.6.0 || fail "MHR conversion-tool packages"
  [ -e libs/MHR/.pixi/envs/default/lib/libEGL.so ] || ln -sf libEGL.so.1 libs/MHR/.pixi/envs/default/lib/libEGL.so
  if ! env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/python -c "import mhr" 2>/dev/null; then
    echo "../../../../../.." > libs/MHR/.pixi/envs/default/lib/python3.12/site-packages/_mhr.pth
  fi
fi

# ---------------------------------------------------------------- 3. PromptHMR
step "3. PromptHMR (libs/PromptHMR/data/pretrain/phmr)"
if ! have libs/PromptHMR/data/pretrain/phmr/checkpoint.ckpt; then
  mkdir -p libs/PromptHMR/data/pretrain
  gdown_ --folder -O libs/PromptHMR/data/pretrain/ https://drive.google.com/drive/folders/1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y \
    || fail "PromptHMR phmr"
fi

# ---------------------------------------------------------------- 4. SMPL-X <-> SMPL conversion (PromptHMR body_models)
step "4. smplx2smpl.pkl, smplx2smpl_joints.npy (resource/)"
if ! have resource/model_transfer/smplx2smpl.pkl || ! have resource/smplx2smpl_joints.npy; then
  gdown_ --folder -O "$TMP/body_models" https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7 \
    && mkdir -p resource/model_transfer \
    && cp "$(find "$TMP/body_models" -name smplx2smpl.pkl | head -1)" resource/model_transfer/ \
    && cp "$(find "$TMP/body_models" -name smplx2smpl_joints.npy | head -1)" resource/ \
    || fail "PromptHMR body_models"
fi

# ---------------------------------------------------------------- 5. Sapiens2 normal
step "5. Sapiens2 normal 1B (libs/sapiens2/checkpoints/normal)"
if ! have libs/sapiens2/checkpoints/normal/sapiens2_1b_normal.safetensors; then
  $PY -c "from huggingface_hub import hf_hub_download as d; d('facebook/sapiens2-normal-1b', 'sapiens2_1b_normal.safetensors', local_dir='libs/sapiens2/checkpoints/normal')" \
    || fail "Sapiens2 (if HuggingFace terms acceptance is required, run huggingface-cli login and rerun)"
fi

# ---------------------------------------------------------------- 6. WAFT
step "6. WAFT (libs/WAFT/waftv2-ckpts/dinov3/spring.pth)"
if ! have libs/WAFT/waftv2-ckpts/dinov3/spring.pth; then
  gdown_ -O "$TMP/a2.zip" 1RA4j0GlNCy7flmc3VIY71ZV-Kz-RYDox \
    && (cd libs/WAFT && unzip -oq "$TMP/a2.zip" 'waftv2-ckpts/dinov3/spring.pth') \
    && rm -f "$TMP/a2.zip" || fail "WAFT"
fi

# ---------------------------------------------------------------- 7. SAM 2.1
step "7. SAM 2.1 base_plus (libs/samurai/sam2/checkpoints)"
if ! have libs/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt; then
  curl -L --fail -o libs/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt || fail "SAM 2.1"
fi

# ---------------------------------------------------------------- 8. DWPose / RTMDet
step "8. DWPose · RTMDet (libs/mmpose)"
if ! have libs/mmpose/dw-ll_ucoco_384.pth; then
  curl -L --fail -o libs/mmpose/dw-ll_ucoco_384.pth https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth || fail "DWPose"
fi
if ! have libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth; then
  curl -L --fail -o libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth || fail "RTMDet"
fi

# ---------------------------------------------------------------- 9. body segmentation (Meshcapade)
step "9. SMPL-X body segmentation (resource/body_segmentation)"
SEG=resource/body_segmentation/smplx/smplx_vert_segmentation.json
if ! have $SEG; then
  mkdir -p "$(dirname $SEG)"
  curl -L --fail -o $SEG https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json \
    || fail "body segmentation"
fi

# ---------------------------------------------------------------- 10. SMPL-X (registration)
step "10. SMPL-X (resource/smplx) — registration required"
if ! have resource/smplx/SMPLX_NEUTRAL.npz || ! have resource/smplx/SMPLX_NEUTRAL.pkl; then
  Z=${SMPLX_ZIP:-$TMP/models_smplx_v1_1.zip}
  if [ -f "$Z" ] || tue_download smplx models_smplx_v1_1.zip "$Z" SMPL-X; then
    rm -rf "$TMP/smplx" && mkdir -p "$TMP/smplx" resource/smplx \
      && unzip -oq "$Z" 'models/smplx/SMPLX_NEUTRAL.npz' 'models/smplx/SMPLX_NEUTRAL.pkl' -d "$TMP/smplx" \
      && $PY - "$TMP/smplx/models/smplx" resource/smplx <<'PY' || fail "SMPL-X conversion"
# This repository uses a model with 10 shape + 10 expression components.
# From v1.1 (shape 300 + expression 100), keep only shape components 0~9 and expression components 300~309, and drop UV (vt, ft).
import sys, os, pickle, numpy as np
src, dst = sys.argv[1], sys.argv[2]
def slim(d):
    d = dict(d)
    d.pop('vt', None); d.pop('ft', None)
    sd = np.asarray(d['shapedirs'])
    d['shapedirs'] = np.concatenate([sd[..., :10], sd[..., 300:310]], axis=-1)
    return d
npz = slim(np.load(os.path.join(src, 'SMPLX_NEUTRAL.npz'), allow_pickle=True))
np.savez(os.path.join(dst, 'SMPLX_NEUTRAL.npz'), **npz)
pkl = slim(pickle.load(open(os.path.join(src, 'SMPLX_NEUTRAL.pkl'), 'rb'), encoding='latin1'))
pickle.dump(pkl, open(os.path.join(dst, 'SMPLX_NEUTRAL.pkl'), 'wb'), protocol=2)
print('  SMPL-X (shape 10 + expression 10) saved:', dst)
PY
  else
    fail "SMPL-X download (check username/password, or set SMPLX_ZIP=<models_smplx_v1_1.zip>)"
  fi
fi

# ---------------------------------------------------------------- 11. SMPL (SMPLify, registration)
step "11. SMPL neutral (resource/smpl) — registration required"
if ! have resource/smpl/SMPL_NEUTRAL.pkl; then
  Z=${SMPLIFY_ZIP:-$TMP/mpips_smplify_public_v2.zip}
  if [ -f "$Z" ] || tue_download smplify mpips_smplify_public_v2.zip "$Z" SMPLify; then
    mkdir -p resource/smpl \
      && unzip -p "$Z" 'smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' > resource/smpl/SMPL_NEUTRAL.pkl \
      && [ -s resource/smpl/SMPL_NEUTRAL.pkl ] || { rm -f resource/smpl/SMPL_NEUTRAL.pkl; fail "SMPL unzip"; }
  else
    fail "SMPLify download (check username/password, or set SMPLIFY_ZIP=<mpips_smplify_public_v2.zip>)"
  fi
fi

# ----------------------------------------------------------------
echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed items (see the manual install instructions in scripts/usage/ASSETS.md):"
  printf '  - %s\n' "${FAILED[@]}"
else
  rm -rf "$TMP"
fi
bash scripts/usage/check_assets.sh

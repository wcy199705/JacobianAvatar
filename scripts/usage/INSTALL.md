# Environment Setup

## 1. System requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 20.04+) |
| GPU | NVIDIA, 24GB VRAM recommended (tested on RTX 3090) |
| CUDA driver | A driver supporting CUDA 12.6 or later (the Sapiens2 environment uses cu126) |
| Disk | ~0.3GB external library sources + ~20GB weights, body models, and MHR runtime environment + 10–30GB preprocessing per scene + 10–30GB training outputs |
| Build tools | `git`, `gcc`, `ld`, CUDA toolkit 12.1 (`nvcc`, for building the CUDA extensions in requirements from source), [pixi](https://pixi.sh) (MHR runtime environment) |
| System packages | `curl`, `unzip` (used by `fetch_assets.sh`), `libgl1`, `libglib2.0-0` (needed by OpenCV / mmcv) — on Ubuntu: `sudo apt install curl unzip libgl1 libglib2.0-0` |

## 2. Three conda environments

Some preprocessing models require different PyTorch/Python versions, so the setup is split into three environments.

| Environment | Purpose | Python / PyTorch | Package list |
|---|---|---|---|
| `Jacobian_avatar` | Training, visualization | 3.9 / 2.5.1+cu121 | `requirements.txt` |
| `jacobian_avatar_prep` | Preprocessing steps 1, 2, 3, 5, 6, 7 | 3.9 / 2.5.1+cu121 | `requirements_prep.txt` |
| `sapiens2` | Preprocessing step 4 (normal estimation) | 3.12 / 2.11+cu126 | `requirements_sapiens2.txt` |

```bash
# Training environment
conda create -n Jacobian_avatar python=3.9 -y && conda activate Jacobian_avatar
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps -r requirements.txt
pip install --no-deps --no-build-isolation -r requirements_build.txt   # source builds against the installed torch (chumpy, mmcv, CUDA extensions)

# Preprocessing environment (includes mmpose 1.3.2 / mmdet 3.3.0 / mmcv 2.1.0)
conda create -n jacobian_avatar_prep python=3.9 -y && conda activate jacobian_avatar_prep
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps -r requirements_prep.txt
pip install --no-deps --no-build-isolation -r requirements_build.txt   # source builds against the installed torch (chumpy, mmcv, CUDA extensions)

# Sapiens2 environment
conda create -n sapiens2 python=3.12 -y && conda activate sapiens2
pip install -r requirements_sapiens2.txt
```

- `requirements_build.txt` lists `chumpy`, `mmcv` and the CUDA extensions (`diff_gauss`, `nvdiffrast`, `pytorch3d`, `simple_knn`), built from source at pinned commits
  against the already installed torch (`--no-build-isolation`; requires the CUDA toolkit / `nvcc` and `gcc`). `libigl` is also built from a pinned
  GitHub commit as part of `requirements*.txt`. `torch_scatter` and `torch_sparse` come from the PyG wheel index (first lines of the requirements files).
- `mmcv` needs a build matching your PyTorch and CUDA versions. Keep the versions in `requirements*.txt` exactly as given.
- By default the scripts use the environments under `$(conda info --base)/envs/`. If yours are elsewhere, set the Python paths before running:
  ```bash
  export PY=/path/to/envs/Jacobian_avatar/bin/python
  export PREP_PY=/path/to/envs/jacobian_avatar_prep/bin/python
  export SAPIENS_PY=/path/to/envs/sapiens2/bin/python
  ```

## 3. Libraries and weights

The **source code** of external libraries is fetched from each original GitHub repository at a pinned commit (unmodified), and the **weights, body models, and MHR runtime environment**
are also fetched separately and placed in the designated locations. Fetch them with `bash scripts/usage/fetch_assets.sh`; for sources and manual installation, see [ASSETS.md](ASSETS.md).

| Location | Contents | Used in | How to obtain |
|---|---|---|---|
| `libs/MHR` (+ `.pixi` environment) | SMPL-X → MHR mesh conversion, EGL library | Training setup, preprocessing 5 | Original GitHub + weights |
| `libs/WAFT` | Optical flow | Training | Original GitHub + weights |
| `libs/mmpose` | DWPose / RTMDet person and keypoint detection | Preprocessing 1a/1b | Original GitHub + weights |
| `libs/samurai` | SAMURAI (SAM2.1) person tracking | Preprocessing 2 | Original GitHub + weights |
| `libs/sapiens2` | Sapiens2 1B normal model | Preprocessing 4 | Original GitHub + weights |
| `libs/PromptHMR` | PromptHMR global pose estimation | Preprocessing 5 | Original GitHub + weights |
| `preprocessing/legacy` | Modules for preprocessing (`scene/`, `render/`, `utils/`) | All preprocessing | — |
| `resource/` | SMPL-X/SMPL body models, body segmentation, conversion files | All stages | Installed separately |

The SMPL-X / SMPL models are not included due to their licenses; download them from their official websites after agreeing to the license terms.

## 4. Sanity check

```bash
# Training environment + config loading
bash scripts/usage/05_dry_run_config_check.sh configs/datasets/Custom-05-v14_optim_v6_freeze100_scaleL2x100.yaml
# → [jacobian_avatar] dataset=Custom scene=05

# MHR runtime environment (created by fetch_assets.sh via pixi install)
cd <repo> && env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/python -c "import mhr; print(mhr.__file__)"
# → <repo>/libs/MHR/mhr/__init__.py
```

## 5. Runtime environment variables

`scripts/usage/*.sh` sets these automatically, so you do not need to set them yourself. Set the following only when running without the scripts
(the three per-region Gaussian settings are read only by `scripts/main_hard_zero_headhand.py`, so you must use this entry point).

```bash
export LR_SCHEDULER_ADAPTIVE=1             # cosine learning-rate schedule
export SCHEDULER_RESET_PER_STAGE=1         # reset the schedule at each stage
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GS_ZERO_HEAD_HAND=1                 # fix head / hand Gaussian offsets
export MESH2GS_HEAD_SIGMA_MULT=0.3         # per-region initial Gaussian size
export MESH2GS_HAND_SIGMA_MULT=0.3
export MESH2GS_BODY_SIGMA_MULT=0.6
python scripts/main_hard_zero_headhand.py --config configs/datasets/Custom-my_scene.yaml
```

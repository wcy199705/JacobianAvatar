# Installing External Libraries, Weights, and Body Models

This repository contains **only the source code we wrote**. External library sources, model weights, body models,
and the MHR runtime environment (about 20GB in total) are not included for licensing and size reasons; the scripts below fetch them from their original sources.

## 0. External library sources

`scripts/usage/fetch_libs.sh` (called automatically by fetch_assets.sh) fetches each GitHub repository at the commit below into `libs/`.
**The originals are not modified.** All glue code adapting them to this repository (path setup, model construction, etc.) lives on our side
(`core/data/waft_avatar.py`, `core/io/_mhr_bridge.py`, `preprocessing/pipeline_a_custom/utils/{prompthmr_setup,run_samurai}.py`).

| Location | GitHub | Commit | License | Used in |
|---|---|---|---|---|
| `libs/MHR` | [facebookresearch/MHR](https://github.com/facebookresearch/MHR) | `98d520f` | Apache-2.0 | Training setup |
| `libs/WAFT` (+ `thirdparty/dinov3` submodule) | [princeton-vl/WAFT](https://github.com/princeton-vl/WAFT) | `b152ff1` | BSD-3 (DINOv3: DINOv3 License) | Training (optical flow) |
| `libs/samurai` | [yangchris11/samurai](https://github.com/yangchris11/samurai) | `76ba195` | Apache-2.0 | Preprocessing 2, 5 |
| `libs/PromptHMR` | [yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR) | `2a1c013` | Non-commercial research use (PromptHMR License) | Preprocessing 5 |
| `libs/sapiens2` | [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2) | `5c7391a` | Sapiens2 License | Preprocessing 4 |
| `libs/mmpose` | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | `71ec36e` | Apache-2.0 | Preprocessing 1a, 1b (config files) |

Each library is subject to its own license. The optional CUDA extension of SAM 2 (`_C`) is not built (same setting as in the paper's experiments —
the "Skipping the post-processing step" warning in preprocessing step 2 is expected).

## 1. Automatic installation (recommended)

SMPL-X and SMPL require agreeing to their licenses, so first register on the two websites below.

- SMPL-X: https://smpl-x.is.tue.mpg.de
- SMPLify (SMPL neutral model): https://smplify.is.tue.mpg.de

After installing [pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | bash`), run from the repository root:

```bash
bash scripts/usage/fetch_assets.sh
```

- Library sources (GitHub) and public weights are downloaded automatically; for SMPL-X / SMPLify, the script prompts for your username and password.
- PromptHMR's MCS viewer output (`gloss`, `smpl_rs`) and the DROID-SLAM backend are not installed (not needed for fixed-camera preprocessing).
- Existing items are skipped, so if some items fail, fix the cause and simply run it again.
- If you already have the zip files, you can pass their paths instead of logging in:
  ```bash
  SMPLX_ZIP=/path/models_smplx_v1_1.zip SMPLIFY_ZIP=/path/mpips_smplify_public_v2.zip \
      bash scripts/usage/fetch_assets.sh
  ```
- At the end, `scripts/usage/check_assets.sh` runs and shows whether every item is `OK`.
- Google Drive items may fail when the download quota is exceeded. Re-run after a while or follow the manual installation below.
- On the first training run, torchvision's VGG16 and AlexNet weights (for LPIPS, about 0.8GB) are downloaded automatically from `download.pytorch.org`
  to `~/.cache/torch/hub/checkpoints/`. On a server without internet access, copy these two files
  (`vgg16-397923af.pth`, `alexnet-owt-7be5be79.pth`) to the same location in advance.

## 2. Installed weights and body models

| # | Location | Files | Size | Source |
|---|---|---|---|---|
| 1 | `libs/MHR/assets/` | `compact_v6_1.model`, `corrective_activation.npz`, `corrective_blendshapes_lod{1,3}.npz`, `lod{1,3}.fbx` | 0.8G | [MHR GitHub release v1.0.1](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1) `assets.zip` |
| 2 | `libs/MHR/.pixi/` | MHR runtime environment (Python 3.12 + pymomentum) | 7.9G | `pixi install` (`libs/MHR/pixi.lock`) |
| 3 | `libs/PromptHMR/data/pretrain/phmr/` | `config.yaml`, `checkpoint.ckpt` | 2.2G | [PromptHMR Google Drive `phmr`](https://drive.google.com/drive/folders/1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y) |
| 4 | `resource/model_transfer/` | `smplx2smpl.pkl` | 0.6G | [PromptHMR Google Drive `body_models`](https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7) |
| 4 | `resource/` | `smplx2smpl_joints.npy` | 1MB | 〃 |
| 5 | `libs/sapiens2/checkpoints/normal/` | `sapiens2_1b_normal.safetensors` | 5.9G | [HuggingFace `facebook/sapiens2-normal-1b`](https://huggingface.co/facebook/sapiens2-normal-1b) |
| 6 | `libs/WAFT/waftv2-ckpts/dinov3/` | `spring.pth` | 0.2G | `a2.zip` from [WAFT Google Drive](https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_) |
| 7 | `libs/samurai/sam2/checkpoints/` | `sam2.1_hiera_base_plus.pt` | 0.3G | [Meta SAM 2.1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| 8 | `libs/mmpose/` | `dw-ll_ucoco_384.pth` | 0.4G | [HuggingFace `yzd-v/DWPose`](https://huggingface.co/yzd-v/DWPose) |
| 8 | `libs/mmpose/` | `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` | 0.1G | [OpenMMLab (RTMPose)](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth) |
| 9 | `resource/body_segmentation/smplx/` | `smplx_vert_segmentation.json` | 0.2MB | [Meshcapade wiki](https://github.com/Meshcapade/wiki/tree/main/assets/SMPL_body_segmentation/smplx) |
| 10 | `resource/smplx/` | `SMPLX_NEUTRAL.npz`, `SMPLX_NEUTRAL.pkl` | 0.3G | [SMPL-X](https://smpl-x.is.tue.mpg.de) `models_smplx_v1_1.zip` → converted (see below) |
| 11 | `resource/smpl/` | `SMPL_NEUTRAL.pkl` | 40MB | [SMPLify](https://smplify.is.tue.mpg.de) `mpips_smplify_public_v2.zip` |

## 3. Manual installation

Fetch only the items the script failed on, as below. Commands are run from the repository root
(Google Drive requires `pip install gdown`, HuggingFace requires `pip install huggingface_hub`).

**1. MHR model files**
```bash
curl -L -o /tmp/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
cd libs/MHR && unzip /tmp/mhr_assets.zip assets/compact_v6_1.model assets/corrective_activation.npz \
    assets/corrective_blendshapes_lod1.npz assets/corrective_blendshapes_lod3.npz assets/lod1.fbx assets/lod3.fbx && cd ../..
```

**2. MHR runtime environment** (if `pixi` is not found right after installing it, open a new shell or use `~/.pixi/bin/pixi`)
```bash
cd libs/MHR && pixi install && ln -sf libEGL.so.1 .pixi/envs/default/lib/libEGL.so && cd ../..
# packages for the SMPL(-X) conversion tool, installed into the MHR env (no change to MHR files)
env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/uv pip install --python libs/MHR/.pixi/envs/default/bin/python --no-deps \
    smplx==0.1.28 scikit-learn==1.8.0 scipy==1.17.1 tqdm==4.67.3 joblib==1.5.3 threadpoolctl==3.6.0
env -u PYTHONPATH libs/MHR/.pixi/envs/default/bin/python -c "import mhr; print(mhr.__file__)"   # → <repo>/libs/MHR/mhr/__init__.py
```
If `import mhr` fails, make the content of `libs/MHR/.pixi/envs/default/lib/python3.12/site-packages/_mhr.pth` the single line `../../../../../..`.

**3. PromptHMR**
```bash
mkdir -p libs/PromptHMR/data/pretrain
gdown --folder -O libs/PromptHMR/data/pretrain/ https://drive.google.com/drive/folders/1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y
```

**4. SMPL-X ↔ SMPL conversion files**
```bash
gdown --folder -O /tmp/phmr_body https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7
mkdir -p resource/model_transfer
cp /tmp/phmr_body/smplx2smpl.pkl resource/model_transfer/
cp /tmp/phmr_body/smplx2smpl_joints.npy resource/
```

**5. Sapiens2** (run `huggingface-cli login` if the model page requires accepting its terms)
```bash
huggingface-cli download facebook/sapiens2-normal-1b sapiens2_1b_normal.safetensors --local-dir libs/sapiens2/checkpoints/normal
```

**6. WAFT** — download `a2.zip` (3.7G) from the Google Drive folder, then:
```bash
cd libs/WAFT && unzip /path/a2.zip waftv2-ckpts/dinov3/spring.pth && cd ../..
```

**7. SAM 2.1**
```bash
curl -L -o libs/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

**8. DWPose · RTMDet**
```bash
curl -L -o libs/mmpose/dw-ll_ucoco_384.pth https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth
curl -L -o libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```

**9. body segmentation**
```bash
mkdir -p resource/body_segmentation/smplx
curl -L -o resource/body_segmentation/smplx/smplx_vert_segmentation.json \
    https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json
```

**10. SMPL-X** — download **SMPL-X v1.1 (`models_smplx_v1_1.zip`)** from the website's Download page, then pass the zip path to the script, which also performs the conversion.
```bash
SMPLX_ZIP=/path/models_smplx_v1_1.zip bash scripts/usage/fetch_assets.sh
```
This repository uses a model with 10 shape + 10 expression components. The script keeps only shape components 0–9 and expression components 300–309 of v1.1
and saves them as `resource/smplx/SMPLX_NEUTRAL.{npz,pkl}` (`check_assets.sh` verifies the component count of 20).

**11. SMPL** — download `mpips_smplify_public_v2.zip` from the SMPLify website, then:
```bash
mkdir -p resource/smpl
unzip -p mpips_smplify_public_v2.zip smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl > resource/smpl/SMPL_NEUTRAL.pkl
```

Once every item is `OK`, follow the steps in `scripts/usage/USAGE.md`.

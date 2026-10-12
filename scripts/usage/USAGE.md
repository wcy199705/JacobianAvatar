# JacobianAvatar (Custom) Usage

This guide walks through the full process of building an animatable 3D avatar from a single-camera video you record yourself.
First create the conda environments with [INSTALL.md](INSTALL.md), then fetch the weights and body models with `bash scripts/usage/fetch_assets.sh` ([ASSETS.md](ASSETS.md)).

```
captured video → (1) prepare scene folder → (2) write scene yaml → (3) preprocessing → (4) write training recipe yaml → (5) training → (6) visualization
```

### Related files

| What to look up | File |
|---|---|
| Overview, quick start | `README.md` |
| Installation (3 conda environments, build tools, environment variables) | `scripts/usage/INSTALL.md` |
| Fetching / checking weights and body models | `scripts/usage/fetch_assets.sh`, `scripts/usage/check_assets.sh`, `scripts/usage/ASSETS.md` |
| List of run scripts | `scripts/usage/README.md` |
| Preprocessing steps | `preprocessing/README.md` |
| Training flow by stage | `docs/PIPELINE.md` |
| Rendering options | `docs/VISUALIZATION.md` |
| Scene folder layout | `docs/DATASETS.md` |
| Example configs (`my_scene`) | `configs/legacy/SMPLX_dataloader/CUSTOM/my_scene.yaml` (scene), `configs/datasets/Custom-my_scene.yaml` (training recipe) |
| Run scripts | `scripts/usage/00_preprocess_custom.sh` → `01_train_custom.sh` → `04_render_viz.sh` |

---

## 1. Prepare the scene folder

Split the video into frame images and arrange them as below. **The folder name must end with `_training`**
(the training loader uses this name to recognize a training folder and load the crop cameras, normals, and keypoints).

```
/path/to/my_scene_training/
├── images/          00000.png, 00001.png, ...   (consecutive numbering from 0, PNG)
└── K.npy            3x3 camera intrinsics (optional — estimated from the image size if missing)
```

Recommended capture conditions:
- A single person with the full body visible, 50–600 frames
- Camera motion and rotation are fine, but **zoom changes are not supported** (K is fixed)
- 1080p or higher recommended (face and hand quality)

Example of extracting frames from a video: `ffmpeg -i video.mp4 -start_number 0 /path/to/my_scene_training/images/%05d.png`

---

## 2. Write the scene yaml (shared by preprocessing and training)

The example scene config `configs/legacy/SMPLX_dataloader/CUSTOM/my_scene.yaml` is already in place.
Edit only its `Path` section (for another scene, copy it to `<scene>.yaml`):

```yaml
Path:
  save_root: 'outputs/my_scene'                       # default output location (the recipe yaml takes precedence)
  training_folder: '/path/to/my_scene_training'       # the scene folder from step 1
  novel_pose_folder: '/path/to/my_scene_training'     # for Custom, same as training_folder
  exavatar_pose_folder: ''                            # (optional) pose sequence for extra_pose rendering during training
```

- The file name (`my_scene`) becomes the scene name, used as-is in the preprocessing command and in the training recipe's `scene_name`.
- `use_flat_hand`, `canon_leg_angle`, and `canon_sholder_angle` under `SMPL` configure the canonical pose and are usually left unchanged.
- `exavatar_pose_folder` can be left empty. Set it to an ExAvatar-format sequence folder (`cam_params/*.json`, `smplx_params_smoothed/*.json`) only if you want to inspect intermediate training results in other poses. Such a folder can be created by preprocessing another video and running `preprocessing/pipeline_a_custom/export/export_to_novel_pose.py`.

---

## 3. Preprocessing

```bash
bash scripts/usage/00_preprocess_custom.sh my_scene 2      # scene name, GPU id
```

The following steps run automatically in order. The log is written to `logs/preprocess/my_scene.log`.

| Step | Description | Output |
|---|---|---|
| (auto) | Estimate K.npy if missing | `K.npy` |
| 1a | Keypoint detection on the first frame (DWPose) → person bbox | `first_bbox.txt` |
| 2 | Person tracking (SAMURAI) | `masks/` |
| mv | Video for checking masks | `debug_masked.mp4`, `masked_images/` |
| 1b | Keypoint detection on all frames (selects the person inside the mask) | `mmpose_result_mmpose/` |
| 3 | Person-centered crop | `image_crop/`, `mask_crop/` |
| 4 | Surface normal estimation (Sapiens2, separate conda env) | `crop_normals_sapiens2/` |
| 5 | Global pose estimation (PromptHMR, fixed camera) | `global_smpl/`, `global_smoothed/`, etc. |
| 6 | Per-frame SMPL-X fitting | `global_smpl_optimized/` |
| 7 | Face region crop | `Head_img/`, `Head_mask/`, `Head_aabb.npy` |
| finalize | Link `normals → crop_normals_sapiens2` | `normals` |

- To re-run only some steps: `bash scripts/usage/00_preprocess_custom.sh my_scene 2 5,6,7,finalize`
- After completion, check `debug_masked.mp4` to make sure the person mask does not leak onto other people or the background.

---

## 4. Write the training recipe yaml

The example recipe `configs/datasets/Custom-my_scene.yaml` is already in place and needs no changes for `my_scene`
(for another scene, copy it to `Custom-<scene>.yaml` and change `scene_name` / `save_root_override`):

```yaml
extends: ./Custom-05-v14_optim_v6_freeze100_scaleL2x100.yaml   # inherit the recommended recipe
scene_name: 'my_scene'                     # same as the scene yaml file name from step 2
data_path: /path/to/my_scene_training      # for reference only (the actual path is training_folder in the scene yaml)
save_root_override: outputs/my_scene/default   # where training results are saved (may be relative to the repository)
```

#### Choosing a base recipe

| `extends:` | Recipe | Runtime env applied by the scripts |
|---|---|---|
| `./Custom-05-v14_optim_v6_freeze100_scaleL2x100.yaml` (default) | Custom recipe | head/hand Gaussian offset hard-zero, Gaussian size 0.3 / 0.3 / 0.6 |
| `../default2.yaml` | Base recipe 2 (v14_optim_v6) | hard-zero, Gaussian size 0.2 / 0.2 / 0.45, body-offset delay 100 |
| `../default2_loose.yaml` | Base recipe 2 for loose clothing / long hair (`loose_smpl`) | same as default2 |
| `../default3.yaml` | Base recipe 3 (v14_optim_v2) | none |

The runtime env comes from the recipe's `runtime_env` key (`configs/default2.yaml`, `default3.yaml`); `01_train_custom.sh` / `02_only_pre_stage.sh` apply it automatically.

### 4.1 Config inheritance (extends)

```
Custom-my_scene.yaml
 └─ Custom-05-v14_optim_v6_freeze100_scaleL2x100.yaml   (recommended recipe: per-stage iters, GS regularization)
     └─ Custom-05-sdf1recipe-noanchor-rgbRefineReg10x-v14_optim.yaml
         └─ ../default_v6_custom.yaml   (Custom defaults: per-stage iters, optical flow weights, etc.)
             └─ ./default_v6.yaml   (global defaults)
```

Read from bottom to top: values in a child file override those in its parent. For nested entries (e.g. `stages.gs.max_iters`), only the specified keys change and the rest keep the parent values.

### 4.2 Commonly changed settings

```yaml
stages:
  pre:
    max_iters: 150           # initial body estimation
    loose_smpl: true         # for loose clothing / long hair (pre_loose_smpl path)
    iso_sdf_vts_norm: 0.1    # weaken the SDF constraint that pulls the mesh to the body (loose clothing)
    iso_sdf_face_norm: 0.01
  l0:  {max_iters: 100}      # mesh level 0
  l1:  {max_iters: 50}       # mesh level 1
  all: {max_iters: 150}      # full mesh fine-tuning
  gs:  {max_iters: 150}      # Gaussian training
training:
  gs_freeze_rgb_iters: 100   # freeze colors for the first 100 GS iters
losses:
  gs:
    level_1:
      semantic_scale_l2: {default: 100, head: 100000, hand: 10000}   # per-region Gaussian size regularization
```

For scenes with loose clothing or long hair, adding just `iso_sdf_vts_norm: 0.1` and `iso_sdf_face_norm: 0.01` above (1/10 of the defaults) is enough.

### 4.3 Check the config

```bash
bash scripts/usage/05_dry_run_config_check.sh configs/datasets/Custom-my_scene.yaml
```

If everything is correct, `[jacobian_avatar] dataset=Custom scene=my_scene` is printed.

---

## 5. Training

```bash
bash scripts/usage/01_train_custom.sh configs/datasets/Custom-my_scene.yaml 2
```

```
(first run) frame validation → creates joint_valid_frames.txt (runs Pre only in _frame_validation/ inside the data folder)
Pre   → initial body estimation
L0    → mesh level 0
L1    → mesh level 1
All   → final mesh           ← ckpt for mesh visualization
GS#1  → 3D Gaussian training  ← ckpt for Gaussian visualization
```

- Training ends at GS#1 (Custom has no ground-truth test video, so there is no test-time pose refinement stage).
- The required environment variables (learning-rate schedule, per-region Gaussian settings) are set automatically by the script (`scripts/usage/_env.sh`).
- If a stage's checkpoint already exists, that stage is loaded instead of retrained. After an interruption, re-run the same command to resume.
- On the first training run for a scene, the script first runs frame validation and creates `<scene folder>/joint_valid_frames.txt`, excluding frames where the person mask and the body fit poorly. All subsequent training runs (even with a different recipe) use only the frames in this list. Delete this file to re-run validation.
- To run selected stages only: `02_only_pre_stage.sh` (Pre only), `03_skip_gs_mesh_only.sh` (up to the mesh).

Output location (`save_root_override`):

```
outputs/my_scene/default/3DGS_SMPLX_Version_poisson/
├── preprocessing/                Pre stage results
└── AFTER_ECCV_.../
    ├── All_iter_*_Avatar.pth     mesh checkpoint
    ├── GS_iter_*_Avatar.pth      Gaussian checkpoint
    └── {l0,L1,All,GS}_iter_*/    per-stage intermediate renders (train / canonical, plus extra_pose if exavatar_pose_folder is set)
```

---

## 6. Visualization

```bash
bash scripts/usage/04_render_viz.sh configs/datasets/Custom-my_scene.yaml outputs/my_scene/viz 2 --mode canon --num_angles 72
```

- Mesh and normals are rendered from the All checkpoint and Gaussians from the GS#1 checkpoint; the results are then merged into mp4/webm.
- `--mode posed --frame_idx 10` : orbit rendering in the pose of a specific frame of the captured video
- Other options: [docs/VISUALIZATION.md](../../docs/VISUALIZATION.md)

---

## 7. Troubleshooting

| Symptom | What to check |
|---|---|
| The step 2 mask follows a different person | Adjust the video start so the target person appears largest in the first frame |
| Errors about crop cameras / normals when training starts | Whether the scene folder name ends with `_training` and the `normals` link (finalize) exists |
| `FileNotFoundError: configs/legacy/.../CUSTOM/<scene>.yaml` | Whether the recipe's `scene_name` matches the scene yaml file name |
| CUDA out of memory | Check whether other processes are using the same GPU (`nvidia-smi`); a 24GB GPU is recommended |
| Error that a weight file cannot be found | Find missing items with `bash scripts/usage/check_assets.sh`, then re-run `fetch_assets.sh` |
| Google Drive items fail in `fetch_assets.sh` | Download quota limit — re-run after a while, or follow the manual installation in `ASSETS.md` |

# pipeline_a_custom/

Preprocessing of captured video frames into training data such as SMPL-X, normals, and masks (7 steps + finalize).
Usually run via `scripts/usage/00_preprocess_custom.sh` (Section 3 of [USAGE.md](../../scripts/usage/USAGE.md)).

## Run

Prerequisites:
- Scene folder `<scene>_training/images/00000.png ...` (+ `K.npy`, optional)
- `Path.training_folder` in the scene config `configs/legacy/SMPLX_dataloader/CUSTOM/<scene>.yaml` must point to the scene folder
- conda envs `jacobian_avatar_prep` and `sapiens2`, plus weights (`scripts/usage/fetch_assets.sh`)

```bash
# Full pipeline for one scene
bash preprocessing/pipeline_a_custom/run_scene.sh <scene> <gpu_id>

# Selected steps only (comma-separated)
bash preprocessing/pipeline_a_custom/run_scene.sh <scene> <gpu_id> 5,6,7,finalize

# Multiple scenes in parallel, one per GPU
bash preprocessing/pipeline_a_custom/run_batch.sh 2,3 sceneA sceneB
```

## Steps

| # | Script | env | Output |
|---|---|---|---|
| (auto) | `utils/make_K_npy.py` | prep | `K.npy` (estimated from the image size if missing) |
| 1a | `steps/step1_dwpose.py` (mode `1a`) | prep | `first_bbox.txt` (DWPose on the first frame) |
| 2 | `steps/step2_samurai.py` | prep | `masks/` (SAMURAI person tracking) |
| mv | `utils/make_masked_video.py` | prep | `debug_masked.mp4`, `masked_images/` (for checking) |
| 1b | `steps/step1_dwpose.py` (mode `1b`) | prep | `mmpose_result_mmpose/` (DWPose on all frames for the person inside the mask) |
| 3 | `steps/step3_crop_masks.py` | prep | `image_crop/`, `mask_crop/` |
| 4 | `steps/step4_sapiens2_normals.py` | **sapiens2** | `crop_normals_sapiens2/` |
| 5 | `steps/step5_global_optimize.py` | prep | `hmr_results/`, `global_smpl/`, `global_smoothed/`, etc. (PromptHMR) |
| 6 | `steps/step6_smplx_fit.py` | prep | `global_smpl_optimized/` (final per-frame SMPL-X) |
| 7 | `steps/step7_crop_head.py` | prep | `Head_img/`, `Head_mask/`, `Head_aabb.npy` |
| finalize | (run_scene.sh) | — | `normals -> crop_normals_sapiens2` |

`run_scene.sh` switches the conda env per step automatically. Steps 5–7 use the results of the preceding steps.

The external libraries (`libs/samurai`, `libs/PromptHMR`, `libs/sapiens2`, `libs/mmpose`) are used unmodified from their original GitHub sources;
the glue code for this repository is in `utils/run_samurai.py` (SAMURAI tracking) and `utils/prompthmr_setup.py` (PromptHMR paths and body model setup).
The SAM 2 "Skipping the post-processing step" warning in step 2 is expected (the optional CUDA extension is not used).

## Other tools

- `utils/video2images.py` — split a video file into frame images
- `utils/count_persons_per_frame.py`, `utils/find_isolated_frames.py` — check DWPose person counts and frames with missing SAMURAI masks
- `run_smplx_only.sh <scene> <gpu_id>` — run up to SMPL-X only, without normals (4) and face crop (7) (for extracting pose sequences)
- `export/export_to_novel_pose.py --scene_root <scene folder> --out_root <output folder>` — convert the SMPL-X of a preprocessed scene into
  an ExAvatar-format pose sequence (`cam_params/*.json`, `smplx_params_smoothed/*.json`).
  Setting it as `exavatar_pose_folder` in the scene yaml produces `extra_pose` renders in those poses during training.
- `viz/viz_novel_pose_mesh.py`, `viz/viz_hand_compare.py` — visualizations for checking pose sequences and hand poses

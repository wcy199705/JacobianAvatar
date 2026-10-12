# Visualization

Three visualization types + one video-encoder utility. All follow the **VIZ CKPT RULE**:

> Mesh_RGB_rgba, P_normal_rgba → **All ckpt** (`All_iter_0_sampling_5_Avatar.pth`)
> GS_RGB_rgba → **GS#1 ckpt** (`GS_iter_0_sampling_1_..._Avatar.pth`)

## Scripts (`scripts/rendering/`)
| Script | Purpose | Key CLI |
|---|---|---|
| `render_orbit.py` | 360° or partial-arc camera orbit around a fixed frame (canon or posed). | `--mode {canon,posed} --num_angles 72 --angle_start -30 --angle_end 60 --cropped` |
| `render_canon_seq.py` | Canonical rendering across all train + novel_pose frames from a fixed front camera. | (no extra args) |
| `render_per_frame_cropped.py` | Novel_pose sequence with per-frame DWPose-centered K at scene_max canvas. | `--train_folder <dir> --test_folder <dir>` (Custom: both are the scene folder) |
| `build_videos.py` | Encode PNG folders (`Mesh_RGB_rgba/`, `GS_RGB_rgba/`, `P_normal_rgba/`) into mp4/webm with auto-crop. | `<subdir> [<subdir> ...] --fps 10` |

All scripts share `_common.py` helpers: `setup_avatar_from_yaml`, `apply_pkl_patch`,
`extract_frame_params`, `load_bboxes`, `scene_max_wh`.

## Two-Pass Rendering (VIZ CKPT RULE)
`scripts/usage/04_render_viz.sh` automates the two passes + merge + video encoding.

For a full multi-output visualization (Mesh + GS + Normal), run each script **twice** per scene:

```bash
# Pass 1: All ckpt → keep Mesh_RGB_rgba, P_normal_rgba
python scripts/rendering/render_orbit.py \
  --config configs/datasets/Custom-my_scene.yaml \
  --ckpt_path <save_root>/AFTER_.../All_iter_0_sampling_5_Avatar.pth \
  --out_dir out/allckpt/my_scene --mode canon ...

# Pass 2: GS#1 ckpt → keep GS_RGB_rgba
python scripts/rendering/render_orbit.py \
  --config <same yaml> \
  --ckpt_path <save_root>/AFTER_.../GS_iter_0_sampling_1_..._Avatar.pth \
  --out_dir out/gs1/my_scene --mode canon ...

# Merge: symlink or copy Mesh_RGB_rgba + P_normal_rgba from allckpt tree
#        symlink or copy GS_RGB_rgba from gs1 tree → merged final tree.
```

`build_videos.py` operates on merged subdirs to produce the final mp4/webm.

## Auto-crop
`build_videos.py` samples ~20 frames per subdir, finds union of alpha bbox, adds 30px margin,
and rounds dims to multiples of 16 for codec compatibility.

## Camera Conventions (render_orbit.py)
- `--fov_scale > 1` → wider FOV → person appears smaller (adds margin, prevents cut-offs).
- `--resolution_scale 2.0` → 2× W and H (K scaled accordingly) — for higher-res output.
- `--look_at_shift_y -0.4` → shift look_at UP by 0.4m in world (world Y-down = image-down convention). Compensates for SMPL transl being at pelvis (torso center is above pelvis).
- `--auto_distance` → for posed mode, distance = ||look_at|| ≈ original training-camera distance.
- `--cropped` → for posed mode, use scene_max DWPose canvas (max across all train+test frames).

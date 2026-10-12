"""Validation-only helper: per-frame mesh silhouette vs GT mask comparison.

Called from src/stages/pre.py at the end of Pre stage (Pre_Step3 done).
Dumps a TSV (frame, outside_ratio, IoU) and an invalid-frames list, but does
NOT mutate the train_dataset (dry-run). Threshold filtering can be done later
based on the dumped distribution.
"""
from __future__ import annotations
import copy
import os
from typing import Any

import numpy as np
import torch

# Mesh class lives at core.mesh in the jacobian_avatar tree (lazy-imported below
# so the top-level import doesn't fail when this module is loaded standalone).


def _composite_viz(img_bgr, sil_bool, gt_bool, joints_uv_np, in_mask_per_joint):
    """Build a BGR composite: RGB + mesh silhouette overlay (semi-transparent
    green) + GT mask outline (blue) + projected joints (green=inside,
    red=outside)."""
    import cv2
    H, W = img_bgr.shape[:2]
    viz = img_bgr.copy()

    # Mesh silhouette overlay
    overlay = np.zeros_like(viz)
    overlay[sil_bool] = (0, 200, 0)
    viz = cv2.addWeighted(viz, 0.7, overlay, 0.3, 0)

    # GT mask outline
    contours, _ = cv2.findContours(gt_bool.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz, contours, -1, (255, 0, 0), 2)

    # Joints (in BGR: green=inside, red=outside)
    for (x, y), inside in zip(joints_uv_np.astype(int), in_mask_per_joint):
        if 0 <= x < W and 0 <= y < H:
            color = (0, 255, 0) if inside else (0, 0, 255)
            cv2.circle(viz, (int(x), int(y)), 4, color, -1)
    return viz


def dump_frame_validity(state, smpl_mesh, smpl_params_batch,
                        out_tsv_path: str, threshold: float = 0.3,
                        viz_dir: str = None) -> None:
    """For each sampled frame, render the posed SMPL mesh silhouette and
    compare to the SAMURAI GT mask. Write per-frame outside_ratio + IoU.
    """
    import sys
    print(f"[validity] enter; out={out_tsv_path}", flush=True)
    from core.mesh import Mesh  # lazy
    import cv2
    gpo = state.extras['gpo_optimizer']
    mesh_renderer = gpo.mesh_renderer
    print(f"[validity] mesh_renderer={type(mesh_renderer).__name__}  "
          f"sampling_cams={len(state.extras.get('sampling_cams', []))} "
          f"sampling_masks={len(state.extras.get('sampling_masks', []))}", flush=True)

    sampling_cams = state.extras['sampling_cams']
    sampling_masks = state.extras['sampling_masks']
    sampling_imgs = state.extras.get('sampling_imgs', None)
    file_names = getattr(state.train_dataset, 'file_names', None)

    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)
        viz_video_path = viz_dir.rstrip('/') + '.mp4'
        vw = None  # lazy init after first frame size known
    else:
        vw = None
        viz_video_path = None

    # canonical vertices (already includes Pre-stage disp; smpl_mesh keeps
    # v_canon as (1, V, 3)).
    vertices = smpl_mesh.v_canon
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)

    # Convert (B,...) batch params dict to per-frame list of dicts.
    batch_param_list = smpl_mesh.batch2list(smpl_params_batch)
    n = len(batch_param_list)

    os.makedirs(os.path.dirname(out_tsv_path), exist_ok=True)
    rows = []
    invalid = []
    n_render_fail = 0

    with torch.no_grad():
        for i in range(n):
            params_i = copy.deepcopy(batch_param_list[i])
            # full_pose in this code-base is sometimes 6d-rot; convert if so.
            if (params_i.get('full_pose') is not None
                    and params_i['full_pose'].shape[-1] not in (165, 156, 144, 72)):
                try:
                    params_i['full_pose'] = smpl_mesh.sixrot_to_full_pose(
                        params_i['full_pose'])
                except Exception:
                    pass

            try:
                posed_verts = smpl_mesh.FS_fullposedict(
                    params_i, v_canon=vertices, return_J=False)
                # Manual joint forward (FS_fullposedict's joint output is buggy)
                B = params_i['full_pose'].shape[0]
                rel_rot = smpl_mesh.rel_transform(
                    params_i['full_pose'].view(B, -1, 3), inverse=False)
                rel_rot_mat = rel_rot.view(B, -1, 4, 4)
                j_canon = smpl_mesh._j_canon[:, :rel_rot_mat.shape[1], :].expand(B, -1, -1)
                from core.body_model.smpl_utils import transform_homogeneous
                posed_joints = transform_homogeneous(rel_rot_mat, j_canon)  # (B, J, 3)
                posed_joints = posed_joints + params_i['transl'].view(B, -1, 3)
                posed_joints = posed_joints.squeeze(0)  # (J, 3)
            except Exception as e:
                if i < 3:
                    import traceback
                    print(f"[validity] fwd_fail frame {i}: {e}", flush=True)
                    traceback.print_exc()
                n_render_fail += 1
                fn = file_names[i] if file_names else f"frame_{i:05d}"
                rows.append((fn, float('nan'), float('nan'), float('nan'),
                             0, 0, 'fwd_fail'))
                continue

            mesh = Mesh(vertices=posed_verts.squeeze(0),
                        indices=smpl_mesh.face,
                        colors=smpl_mesh.rgb,
                        device=smpl_mesh.device)

            try:
                rendered = mesh_renderer.render_batch(mesh, [sampling_cams[i]])
                sil = rendered['mask']  # (1, H, W, 1) or similar
            except Exception as e:
                n_render_fail += 1
                fn = file_names[i] if file_names else f"frame_{i:05d}"
                rows.append((fn, float('nan'), float('nan'), 'render_fail'))
                continue

            sil = sil[0].squeeze()  # (H, W)
            sil_bool = (sil > 0.5).cpu().numpy()

            gt = sampling_masks[i]
            if isinstance(gt, torch.Tensor):
                gt_np = gt.detach().cpu().numpy()
            else:
                gt_np = np.asarray(gt)
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(-1) if gt_np.shape[-1] == 1 else gt_np.squeeze(0)
            gt_bool = gt_np > 0.5

            # Align shapes (silhouette may flip vs GT; we just need bools)
            if sil_bool.shape != gt_bool.shape:
                # Crude resize via numpy (preserve aspect issues unlikely here)
                import cv2
                gt_bool = cv2.resize(gt_bool.astype(np.uint8),
                                     (sil_bool.shape[1], sil_bool.shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)

            outside = sil_bool & (~gt_bool)
            sil_area = int(sil_bool.sum())
            outside_ratio = (outside.sum() / sil_area) if sil_area > 0 else 1.0

            inter = int((sil_bool & gt_bool).sum())
            union = int((sil_bool | gt_bool).sum())
            iou = inter / union if union > 0 else 0.0

            # --- Joint outside-mask analysis ---
            # SMPL-X joint ordering: 0-19 body, 20-21 wrist, 22-24 head/face,
            # 25-39 l_hand finger, 40-54 r_hand finger. Ignore finger joints (25+)
            # when deciding validity —
            # wrists (20, 21) and body/head/face still must be in mask.
            BODY_END = 25  # exclusive: keep joints [0, BODY_END)
            cam = sampling_cams[i]
            try:
                j3d_all = posed_joints if posed_joints.dim() == 2 else posed_joints.squeeze(0)
                j3d = j3d_all[:BODY_END]
                j_cam = (cam.R.to(j3d) @ j3d.T).T + cam.t.to(j3d)        # (J, 3)
                j_uv_h = (cam.K.to(j3d) @ j_cam.T).T                     # (J, 3)
                j_uv = j_uv_h[:, :2] / j_uv_h[:, 2:3].clamp(min=1e-6)    # (J, 2)
                in_front = (j_cam[:, 2] > 0).cpu().numpy()
                H, W = gt_bool.shape
                xs = j_uv[:, 0].long().cpu().numpy()
                ys = j_uv[:, 1].long().cpu().numpy()
                in_img = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                valid_proj = in_front & in_img
                xs_c = np.clip(xs, 0, W - 1)
                ys_c = np.clip(ys, 0, H - 1)
                in_mask = gt_bool[ys_c, xs_c]
                n_joints = int(valid_proj.sum())
                n_joints_outside = int(((~in_mask) & valid_proj).sum())
                joints_outside_ratio = (n_joints_outside / n_joints) if n_joints else 1.0
            except Exception as e:
                n_joints, n_joints_outside, joints_outside_ratio = 0, 0, float('nan')

            fn = file_names[i] if file_names else f"frame_{i:05d}"
            rows.append((fn, float(outside_ratio), float(iou),
                         float(joints_outside_ratio),
                         n_joints_outside, n_joints, 'ok'))
            if outside_ratio > threshold:
                invalid.append(fn)

            # --- visualization dump ---
            if viz_dir is not None and sampling_imgs is not None:
                try:
                    img_t = sampling_imgs[i]
                    img_np = (img_t.detach().cpu().numpy()
                              if isinstance(img_t, torch.Tensor) else np.asarray(img_t))
                    if img_np.dtype != np.uint8:
                        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    if img_np.shape[-1] == 3:
                        img_bgr = img_np[..., ::-1].copy()  # RGB → BGR
                    else:
                        img_bgr = img_np
                    # Resize to silhouette H,W if needed
                    if img_bgr.shape[:2] != sil_bool.shape:
                        img_bgr = cv2.resize(
                            img_bgr, (sil_bool.shape[1], sil_bool.shape[0]))

                    in_mask_per_joint = (in_mask & valid_proj)
                    joints_uv_np = j_uv.detach().cpu().numpy()
                    viz = _composite_viz(img_bgr, sil_bool, gt_bool,
                                         joints_uv_np, in_mask_per_joint)
                    base = os.path.basename(fn).replace('.png', '')
                    out_png = os.path.join(viz_dir, f"{base}.png")
                    cv2.imwrite(out_png, viz)
                    if vw is None:
                        H, W = viz.shape[:2]
                        # cap output size to 1280-long-side for mp4 footprint
                        scale = min(1.0, 1280.0 / max(H, W))
                        vw_W, vw_H = int(W * scale), int(H * scale)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        vw = cv2.VideoWriter(viz_video_path, fourcc, 30, (vw_W, vw_H))
                    viz_resized = (cv2.resize(viz, (vw_W, vw_H),
                                              interpolation=cv2.INTER_AREA)
                                   if (W, H) != (vw_W, vw_H) else viz)
                    vw.write(viz_resized)
                except Exception as e:
                    print(f"[viz] frame {i}: viz failed: {e}", flush=True)

    if vw is not None:
        vw.release()
        print(f"[viz] saved viz frames → {viz_dir}  mp4 → {viz_video_path}", flush=True)

    # ---- write outputs ----
    with open(out_tsv_path, 'w') as f:
        f.write("frame\toutside_ratio\tIoU\tjoints_outside_ratio\tn_joints_outside\tn_joints\tstatus\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]}\t{r[5]}\t{r[6]}\n")

    inv_path = out_tsv_path.replace('.tsv', '_invalid.txt')
    with open(inv_path, 'w') as f:
        f.write("\n".join(map(str, invalid)) + ("\n" if invalid else ""))

    # --- Auto-publish joint_valid_frames.txt to training_folder for the
    # downstream Custom dataset's convention-based auto-detect. Lines list
    # every frame with n_joints_outside==0 (body+wrist+face all in mask).
    try:
        _training_root = state.train_dataset.root_folder
        _valid_path = os.path.join(_training_root, 'joint_valid_frames.txt')
        _valid_invalid_path = os.path.join(_training_root, 'joint_invalid_frames.txt')
        _ok_rows = [r for r in rows if r[6] == 'ok' and r[4] == 0]
        _bad_rows = [r for r in rows if r[6] == 'ok' and r[4] > 0]
        with open(_valid_path, 'w') as _vf:
            for _r in _ok_rows:
                _vf.write(_r[0] + '\n')
        with open(_valid_invalid_path, 'w') as _vf:
            for _r in _bad_rows:
                _vf.write(_r[0] + '\n')
        print(f"[validity] auto-published valid list ({len(_ok_rows)}) → "
              f"{_valid_path}", flush=True)
    except Exception as _e:
        print(f"[validity] auto-publish failed: {_e}", flush=True)

    # ---- histogram ----
    ratios = np.array([r[1] for r in rows if not np.isnan(r[1])])
    j_ratios = np.array([r[3] for r in rows if not np.isnan(r[3])])
    j_outside_counts = np.array([r[4] for r in rows if r[6] == 'ok'])
    if len(ratios):
        print(f"[validity] dumped {len(rows)} frames to {out_tsv_path}", flush=True)
        print(f"[validity] render_fail: {n_render_fail}", flush=True)
        print(f"[validity] silhouette outside_ratio  min={ratios.min():.3f}  "
              f"mean={ratios.mean():.3f}  median={np.median(ratios):.3f}  "
              f"max={ratios.max():.3f}", flush=True)
        for t in (0.1, 0.2, 0.3, 0.5, 0.7):
            print(f"[validity]   silhouette > {t:.2f} : {(ratios > t).sum():4d} "
                  f"frames ({100*(ratios > t).mean():.1f}%)", flush=True)
        print(f"[validity] joints_outside_ratio  min={j_ratios.min():.3f}  "
              f"mean={j_ratios.mean():.3f}  median={np.median(j_ratios):.3f}  "
              f"max={j_ratios.max():.3f}", flush=True)
        print(f"[validity] joints_outside_count   min={j_outside_counts.min()}  "
              f"mean={j_outside_counts.mean():.2f}  max={j_outside_counts.max()}", flush=True)
        for t in (0.05, 0.1, 0.2, 0.3, 0.5):
            print(f"[validity]   joint_ratio > {t:.2f} : {(j_ratios > t).sum():4d} "
                  f"frames ({100*(j_ratios > t).mean():.1f}%)", flush=True)
        for k in (1, 3, 5, 10):
            print(f"[validity]   joints_outside_count >= {k} : "
                  f"{(j_outside_counts >= k).sum():4d} frames "
                  f"({100*(j_outside_counts >= k).mean():.1f}%)", flush=True)
        print(f"[validity] invalid (silhouette>{threshold}) list: {inv_path}", flush=True)

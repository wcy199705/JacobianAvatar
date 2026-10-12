"""Render 360° / partial-arc camera orbit around a fixed novel_pose frame.

Two modes:
  canon: mesh is at canonical (SMPL Y-up); requires R_front pre-multiply + rotation sign flip.
  posed: mesh already in world coords (world -Y up); no R_front needed.

CLI is intentionally minimal — most sweeps should call the render_orbit() function directly
from a python script (see scripts/rendering/render_orbit_batch.py for a batching example).

VIZ CKPT RULE: for a full multi-output visualization (Mesh + GS + Normal), call this twice:
  - Once with All ckpt → keep Mesh_RGB_rgba, P_normal_rgba outputs
  - Once with GS#1 ckpt → keep GS_RGB_rgba outputs
Do NOT use Opt_pose (GS#2) for GS renders.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.rendering._common import (
    apply_monkey_patch_if_needed,
    apply_pkl_patch,
    extract_frame_params,
    load_bboxes,
    scene_max_wh,
    setup_avatar_from_yaml,
)

apply_monkey_patch_if_needed()

import numpy as np
import torch

from core.camera import UnifiedCamera
from scripts.save_all_outputs_v2 import Save_all_outputs_v2


def _build_orbit_cameras(
    K: torch.Tensor,
    W: int,
    H: int,
    distance: float,
    num_angles: int,
    look_at: torch.Tensor,
    device: str,
    canon: bool = False,
    angle_start_deg: float = 0.0,
    angle_end_deg: float = 360.0,
) -> List:
    """Build cameras orbiting around look_at, world Y-axis.

    Uses UnifiedCamera's world_view convention: p_view = self.R @ p + gl_base @ self.t,
    with gl_base = diag(1, -1, -1).

    Args:
        canon: if True, mesh is SMPL Y-up (canonical) → prefix R_front to flip Y/Z so head
               appears up in the image. Also flips theta sign so orbit direction matches
               the posed-mode orbit (R_front changes handedness observed).
        angle_start_deg / angle_end_deg: angle range in degrees; num_angles frames are
               distributed linearly (exclusive end).
    """
    cams = []
    gl_base = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    R_front = torch.diag(torch.tensor([1.0, -1.0, -1.0])) if canon else torch.eye(3)
    target_view = torch.tensor([0.0, 0.0, distance], dtype=torch.float32)
    theta_sign = -1.0 if canon else 1.0
    start_rad = math.radians(angle_start_deg)
    end_rad = math.radians(angle_end_deg)
    for i in range(num_angles):
        frac = i / num_angles if num_angles > 1 else 0.0
        theta = (start_rad + (end_rad - start_rad) * frac) * theta_sign
        c = math.cos(-theta); s = math.sin(-theta)
        R_orbit = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
        R = R_front @ R_orbit
        T = gl_base @ (target_view - R @ look_at)
        cams.append(UnifiedCamera.perspective_camera_from_K(
            K=K, width=W, height=H, znear=0.01, zfar=1000.0, R=R, T=T, device=device))
    return cams


def render_orbit(
    config: str,
    ckpt_path: str,
    out_dir: str,
    mode: str = 'canon',            # 'canon' | 'posed'
    split: str = 'novel',           # 'train' | 'novel'
    frame_idx: int = 0,
    num_angles: int = 72,
    angle_start: float = 0.0,
    angle_end: float = 360.0,
    distance: float = 6.0,          # used for canon; posed uses auto if auto_distance=True
    auto_distance: bool = True,
    fov_scale: float = 1.0,
    resolution_scale: float = 2.0,
    look_at_shift_y: float = -0.4,
    cropped: bool = False,          # for posed: use scene_max canvas instead of full frame
    train_folder: str = None,
    test_folder: str = None,
    out_subname: str = None,
) -> int:
    """Render one orbit variant. See module docstring for viz ckpt rule."""
    state, avatar = setup_avatar_from_yaml(config, ckpt_path)
    apply_pkl_patch(state, ckpt_path)

    train_ds = state.train_dataset
    zju = state.extras['ZJU_Novel_pose_dataset']
    gpo = state.extras['gpo_optimizer']
    smpl_mesh = state.smpl_mesh

    sfd = train_ds.smpl_full_dict if split == 'train' else zju.smpl_full_dict
    frame_params = extract_frame_params(sfd, frame_idx, smpl_mesh)
    if frame_params is None:
        print(f'[render_orbit] frame_idx {frame_idx} out of range'); return 1

    canon_mode = (mode == 'canon')
    # Canvas: cropped scene_max (for posed) or full-frame (for canon or non-cropped posed)
    if cropped and not canon_mode:
        max_W, max_H = scene_max_wh([
            os.path.join(train_folder, 'mmpose_result_mmpose', 'bboxes'),
            os.path.join(test_folder, 'mmpose_result_mmpose', 'bboxes'),
        ])
        W, H = max_W, max_H
    else:
        W = int(train_ds.original_width); H = int(train_ds.original_height)
    K = torch.Tensor(train_ds.cam_int).clone()
    K[0, 2] = W / 2.0; K[1, 2] = H / 2.0
    if fov_scale != 1.0:
        K[0, 0] /= fov_scale; K[1, 1] /= fov_scale
    if resolution_scale != 1.0:
        s = float(resolution_scale)
        K[0, 0] *= s; K[1, 1] *= s; K[0, 2] *= s; K[1, 2] *= s
        W = int(round(W * s)); H = int(round(H * s))

    # look_at: origin for canon, pelvis (transl) for posed; then shift up by look_at_shift_y
    if canon_mode:
        look_at = torch.zeros(3)
    else:
        t = frame_params.get('transl')
        if hasattr(t, 'squeeze'): t = t.squeeze()
        if t.dim() > 1: t = t[0]
        look_at = t.detach().cpu().clone()
    if look_at_shift_y != 0.0:
        look_at = look_at.clone(); look_at[1] += look_at_shift_y

    # distance: use ||look_at|| (auto) for posed; else use --distance
    _dist = distance
    if auto_distance and not canon_mode:
        _dist = float(look_at.norm().item())

    cams = _build_orbit_cameras(K, W, H, _dist, num_angles, look_at, avatar.device,
                                canon=canon_mode,
                                angle_start_deg=angle_start, angle_end_deg=angle_end)
    params_list = [frame_params] * num_angles

    sub = out_subname or f'orbit_{mode}_{split}_f{frame_idx}'
    out_sub = Path(out_dir) / sub
    out_sub.mkdir(parents=True, exist_ok=True)
    print(f'[render_orbit] → {out_sub} ({num_angles} @ [{angle_start},{angle_end}] {W}x{H} dist={_dist:.2f} canon={canon_mode})')
    with torch.no_grad():
        Save_all_outputs_v2(
            avatar, gpo.mesh_renderer, str(out_sub),
            cameras=cams, smpl_params=params_list,
            canon=canon_mode, bg=[1.0, 1.0, 1.0])
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser(description='Render one orbit variant.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', choices=['canon', 'posed'], default='canon')
    ap.add_argument('--split', choices=['train', 'novel'], default='novel')
    ap.add_argument('--frame_idx', type=int, default=0)
    ap.add_argument('--num_angles', type=int, default=72)
    ap.add_argument('--angle_start', type=float, default=0.0)
    ap.add_argument('--angle_end', type=float, default=360.0)
    ap.add_argument('--distance', type=float, default=6.0)
    ap.add_argument('--auto_distance', action='store_true')
    ap.add_argument('--fov_scale', type=float, default=1.0)
    ap.add_argument('--resolution_scale', type=float, default=2.0)
    ap.add_argument('--look_at_shift_y', type=float, default=-0.4)
    ap.add_argument('--cropped', action='store_true')
    ap.add_argument('--train_folder', default=None)
    ap.add_argument('--test_folder', default=None)
    ap.add_argument('--out_subname', default=None)
    args = ap.parse_args()
    return render_orbit(**vars(args))


if __name__ == '__main__':
    sys.exit(_main())

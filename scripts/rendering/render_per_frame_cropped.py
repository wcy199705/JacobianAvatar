"""Render per-frame cropped novel_pose sequence using DWPose-centered scene_max K.

Each novel_pose frame gets its own K (cx/cy shifted so the DWPose bbox center lands at
scene_max canvas center). Output size = (scene_max_W, scene_max_H) where scene_max is
max over all train+test DWPose raw bboxes.

VIZ CKPT RULE: run twice (All ckpt for Mesh+Normal, GS#1 for GS).
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
from pathlib import Path

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


def render_per_frame_cropped(config: str, ckpt_path: str, out_dir: str,
                             train_folder: str, test_folder: str) -> int:
    """Render novel_pose frames with per-frame K + scene_max canvas."""
    state, avatar = setup_avatar_from_yaml(config, ckpt_path)
    apply_pkl_patch(state, ckpt_path)

    zju = state.extras['ZJU_Novel_pose_dataset']
    gpo = state.extras['gpo_optimizer']
    smpl_mesh = state.smpl_mesh

    train_bb_dir = os.path.join(train_folder, 'mmpose_result_mmpose', 'bboxes')
    test_bb_dir = os.path.join(test_folder, 'mmpose_result_mmpose', 'bboxes')
    max_W, max_H = scene_max_wh([train_bb_dir, test_bb_dir])
    test_bbs = load_bboxes(test_bb_dir)
    test_img_files = sorted(glob.glob(os.path.join(test_folder, 'images', '*.png')))
    print(f'[render_per_frame_cropped] scene_max WxH = {max_W} x {max_H}')

    full_h = int(zju.original_height); full_w = int(zju.original_width)
    base_K = torch.Tensor(np.load(os.path.join(zju.root_folder, 'K.npy'))).clone()

    new_cams = []
    for i in range(zju.num_frames):
        stem = os.path.basename(test_img_files[i]).rsplit('.', 1)[0] if i < len(test_img_files) else f'{i:06d}'
        bb = test_bbs.get(stem)
        if bb is None or bb[2] == 0 or bb[3] == 0:
            cx = full_w / 2.0; cy = full_h / 2.0
        else:
            cx = bb[0] + bb[2] / 2.0; cy = bb[1] + bb[3] / 2.0
        x0 = max(0, int(round(cx - max_W / 2.0)))
        y0 = max(0, int(round(cy - max_H / 2.0)))
        if x0 + max_W > full_w: x0 = max(0, full_w - max_W)
        if y0 + max_H > full_h: y0 = max(0, full_h - max_H)
        new_K = base_K.clone()
        new_K[0, 2] = base_K[0, 2] - x0
        new_K[1, 2] = base_K[1, 2] - y0
        cam = UnifiedCamera.perspective_camera_from_K(
            K=new_K, width=max_W, height=max_H,
            znear=0.01, zfar=1000.0, device=avatar.device)
        new_cams.append(cam)
    zju.cameras = new_cams
    zju.cam_int = base_K
    zju.h = max_H; zju.w = max_W

    sfd = zju.smpl_full_dict
    B = len(new_cams)
    params_list = [extract_frame_params(sfd, i, smpl_mesh) for i in range(B)]

    out_sub = Path(out_dir) / 'novel_pose_perframe_v2'
    out_sub.mkdir(parents=True, exist_ok=True)
    print(f'[render_per_frame_cropped] → {out_sub} ({B} frames @ {max_W}x{max_H})')
    with torch.no_grad():
        Save_all_outputs_v2(
            avatar, gpo.mesh_renderer, str(out_sub),
            cameras=new_cams, smpl_params=params_list,
            canon=False, bg=[1.0, 1.0, 1.0])
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--train_folder', required=True)
    ap.add_argument('--test_folder', required=True)
    args = ap.parse_args()
    return render_per_frame_cropped(args.config, args.ckpt_path, args.out_dir,
                                     args.train_folder, args.test_folder)


if __name__ == '__main__':
    sys.exit(_main())

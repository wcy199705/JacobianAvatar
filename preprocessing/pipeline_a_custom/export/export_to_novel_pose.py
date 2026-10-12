"""Convert preprocessed CUSTOM scene's per-frame SMPL-X + K.npy into the
`Novel_Poses/Exavatar/<name>/` JSON format used for animation pose driving.

Input  (from CUSTOM preprocessing):
  <scene_root>/K.npy                              shape (3,3)
  <scene_root>/global_smpl_optimized/<idx>.pt     dict with SMPL-X tensors

Output:
  <out_root>/cam_params/<idx>.json    {R, t, focal, princpt}
  <out_root>/smplx_params_smoothed/<idx>.json
      {root_pose, body_pose, jaw_pose, leye_pose, reye_pose,
       lhand_pose, rhand_pose, expr, trans}

The CUSTOM training/global_smpl_optimized convention uses these keys
(observed via subject 10):
    global_orient   →  root_pose
    body_pose       →  body_pose
    jaw_pose        →  jaw_pose
    leye_pose       →  leye_pose       (may be absent — emit zeros)
    reye_pose       →  reye_pose       (may be absent — emit zeros)
    left_hand_pose  →  lhand_pose
    right_hand_pose →  rhand_pose
    expression      →  expr
    transl          →  trans

R/t are identity/zero (canonical camera). The .pt's global_orient + transl
already absorb camera-relative motion, matching the jungkook convention.

Usage:
  python export_to_novel_pose.py \
    --scene_root /path/to/my_scene_training \
    --out_root /path/to/novel_pose_out
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch


KEY_MAP = {
    'global_orient':   'root_pose',
    'body_pose':       'body_pose',
    'jaw_pose':        'jaw_pose',
    'leye_pose':       'leye_pose',
    'reye_pose':       'reye_pose',
    'left_hand_pose':  'lhand_pose',
    'right_hand_pose': 'rhand_pose',
    'expression':      'expr',
    'transl':          'trans',
}

ZERO_LEN = {  # SMPL-X canonical sizes (axis-angle / flat)
    'jaw_pose': 3,
    'leye_pose': 3,
    'reye_pose': 3,
    'expression': 10,
    'left_hand_pose': 45,
    'right_hand_pose': 45,
}

# hand pose estimation from HaMeR can be noisy → use the
# SMPL-X default (flat) hand pose (all zeros). Keys here are SOURCE key names.
FORCE_ZERO_KEYS = {'left_hand_pose', 'right_hand_pose'}


def _tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().squeeze().numpy().astype(float).flatten().tolist()
    if isinstance(x, np.ndarray):
        return x.squeeze().astype(float).flatten().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [float(x)]


def export_one(idx: int, smplx_pt_path: str, K: np.ndarray, W: int, H: int,
               cam_dir: str, smplx_dir: str) -> None:
    d = torch.load(smplx_pt_path, map_location='cpu', weights_only=False)

    out_smplx = {}
    for src_key, dst_key in KEY_MAP.items():
        if src_key in FORCE_ZERO_KEYS:
            out_smplx[dst_key] = [0.0] * ZERO_LEN[src_key]
            continue
        v = d.get(src_key, None)
        if v is None:
            out_smplx[dst_key] = [0.0] * ZERO_LEN.get(src_key, 3)
        else:
            out_smplx[dst_key] = _tensor_to_list(v)

    cam = {
        'R': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        't': [0.0, 0.0, 0.0],
        'focal': [float(K[0, 0]), float(K[1, 1])],
        'princpt': [float(K[0, 2]), float(K[1, 2])],
    }

    with open(os.path.join(cam_dir, f'{idx}.json'), 'w') as f:
        json.dump(cam, f)
    with open(os.path.join(smplx_dir, f'{idx}.json'), 'w') as f:
        json.dump(out_smplx, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_root', required=True,
                    help='Path to preprocessed scene (must have K.npy + global_smpl_optimized/)')
    ap.add_argument('--out_root', required=True,
                    help='Output dir (will create cam_params/ and smplx_params_smoothed/)')
    ap.add_argument('--keep_hands', action='store_true',
                    help='Keep estimated hand pose (HaMeR result). '
                         'Default: force zero (flat) hands.')
    args = ap.parse_args()

    if args.keep_hands:
        FORCE_ZERO_KEYS.clear()
        print('[export] keeping estimated hand pose (no zero-out)')

    K_path = os.path.join(args.scene_root, 'K.npy')
    smplx_dir_src = os.path.join(args.scene_root, 'global_smpl_optimized')
    assert os.path.isfile(K_path), f'K.npy not found: {K_path}'
    assert os.path.isdir(smplx_dir_src), f'global_smpl_optimized not found: {smplx_dir_src}'

    K = np.load(K_path)

    images_dir = os.path.join(args.scene_root, 'images')
    pngs = sorted(f for f in os.listdir(images_dir) if f.endswith('.png'))
    if pngs:
        from PIL import Image
        with Image.open(os.path.join(images_dir, pngs[0])) as im:
            W, H = im.size
    else:
        W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)

    cam_dir = os.path.join(args.out_root, 'cam_params')
    smplx_dst = os.path.join(args.out_root, 'smplx_params_smoothed')
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(smplx_dst, exist_ok=True)

    pts = sorted(f for f in os.listdir(smplx_dir_src) if f.endswith('.pt'))
    print(f'[export] {len(pts)} frames; K={K.tolist()}; res={W}x{H}; → {args.out_root}')
    for i, fname in enumerate(pts):
        # Output as 0-indexed sequential (jungkook convention: 0.json, 1.json, ...)
        # regardless of source filename (00000.pt / img_0001.pt / etc.)
        export_one(i, os.path.join(smplx_dir_src, fname), K, W, H,
                   cam_dir, smplx_dst)
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(pts)}]')

    print(f'[export] done — wrote {len(pts)} cam_params + {len(pts)} smplx_params_smoothed')


if __name__ == '__main__':
    main()

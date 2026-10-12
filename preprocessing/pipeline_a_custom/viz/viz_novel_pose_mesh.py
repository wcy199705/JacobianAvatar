"""Render Parapara-style novel pose JSONs as SMPL-X meshes (CPU painter, no EGL)."""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import cv2
import smplx
from natsort import natsorted

SMPL_ROOT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), 'resource')


def project_and_shade(verts: np.ndarray, faces: np.ndarray, K: np.ndarray,
                      out_size: tuple[int, int]) -> np.ndarray:
    """CPU painter: project triangles, depth-sort, fill with normal shading."""
    H, W = out_size
    img_W, img_H = int(K[0, 2] * 2), int(K[1, 2] * 2)
    sx, sy = W / img_W, H / img_H
    K2 = K.copy()
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy

    # verts in camera frame already (identity extrinsic). Project.
    # If z<=0, behind camera — skip those.
    z = verts[:, 2]
    valid_z = z > 1e-3
    if not valid_z.all():
        z = np.where(valid_z, z, 1e-3)
    u = K2[0, 0] * verts[:, 0] / z + K2[0, 2]
    v = K2[1, 1] * verts[:, 1] / z + K2[1, 2]

    # Face vertices in 3D + 2D
    fv3 = verts[faces]              # (F, 3, 3)
    fu  = u[faces]                  # (F, 3)
    fv  = v[faces]                  # (F, 3)
    fz  = z[faces].mean(axis=1)     # (F,) depth proxy for sort

    # Backface culling + lighting
    e1 = fv3[:, 1] - fv3[:, 0]
    e2 = fv3[:, 2] - fv3[:, 0]
    n = np.cross(e1, e2)
    nrm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n /= nrm
    # camera is at origin looking +z; ray to face is fv3.mean (centroid) → cam (0)
    # but simpler: shade by |dot(n, view_dir)| where view_dir = (0,0,-1)
    light = np.array([0.3, -0.7, -0.7])
    light /= np.linalg.norm(light)
    intensity = np.clip(n @ light, 0.15, 1.0)  # 0.15 = ambient

    # Painter's: back to front
    order = np.argsort(-fz)
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    base_color = np.array([200, 200, 215], dtype=np.float32)  # B G R
    for k in order:
        pts = np.stack([fu[k], fv[k]], axis=1)
        if (pts[:, 0] < -5).all() or (pts[:, 0] > W + 5).all(): continue
        if (pts[:, 1] < -5).all() or (pts[:, 1] > H + 5).all(): continue
        col = tuple(int(c) for c in (base_color * intensity[k]).clip(0, 255))
        cv2.fillConvexPoly(img, pts.astype(np.int32), col, lineType=cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--novel_pose_root', required=True)
    ap.add_argument('--n_samples', type=int, default=12)
    ap.add_argument('--cell', type=int, default=320)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model = smplx.SMPLX(
        model_path=os.path.join(SMPL_ROOT, 'smplx'),
        gender='neutral', num_betas=10, num_expression_coeffs=10,
        use_face_contour=True, use_pca=False, flat_hand_mean=True,
        ext='pkl', batch_size=1,
    )
    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(_dev)
    faces = model.faces.astype(np.int32)

    cam_files = natsorted(glob.glob(os.path.join(args.novel_pose_root, 'cam_params', '*.json')))
    smplx_files = natsorted(glob.glob(os.path.join(args.novel_pose_root, 'smplx_params_smoothed', '*.json')))
    N = len(smplx_files)
    print(f'[viz] {N} frames')
    idxs = np.linspace(0, N - 1, args.n_samples, dtype=int).tolist()

    cell = args.cell
    cols = 4
    rows = (len(idxs) + cols - 1) // cols
    canvas = np.full((rows * (cell + 4) + 4, cols * (cell + 4) + 4, 3), 255, dtype=np.uint8)

    with torch.no_grad():
        for k, i in enumerate(idxs):
            with open(smplx_files[i]) as f: p = json.load(f)
            with open(cam_files[i]) as f: cam = json.load(f)
            K = np.eye(3); K[0, 0], K[1, 1] = cam['focal']; K[0, 2], K[1, 2] = cam['princpt']

            def t(x): return torch.tensor(x, dtype=torch.float32, device=_dev).view(1, -1)
            out = model(
                global_orient=t(p['root_pose']), body_pose=t(p['body_pose']),
                jaw_pose=t(p['jaw_pose']),
                leye_pose=t(p['leye_pose']), reye_pose=t(p['reye_pose']),
                left_hand_pose=t(p['lhand_pose']), right_hand_pose=t(p['rhand_pose']),
                expression=t(p['expr']), transl=t(p['trans']),
                betas=torch.zeros((1, 10), device=_dev),
            )
            v = out.vertices[0].cpu().numpy()
            img = project_and_shade(v, faces, K, (cell, cell))
            r = k // cols; c = k % cols
            y = r * (cell + 4) + 4; x = c * (cell + 4) + 4
            canvas[y:y + cell, x:x + cell] = img
            cv2.putText(canvas, f'#{i}', (x + 6, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)
            print(f'  [{k+1}/{len(idxs)}] frame {i}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f'[viz] wrote {args.out}')


if __name__ == '__main__':
    main()

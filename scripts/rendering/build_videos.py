"""Encode PNG-frame folders into mp4/webm videos with optional auto-crop by alpha.

Reads rendered subdirs (e.g., Mesh_RGB_rgba/*.png, GS_RGB_rgba/*.png, P_normal_rgba/*.png)
and produces matching {folder}.mp4 and {folder}.webm at the parent level.

Auto-crop: samples ~20 frames, unions alpha bbox, adds fixed margin, rounds to 16-divisible.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import imageio.v2 as imageio

DEFAULT_FPS = 10
KINDS = ('Mesh_RGB_rgba', 'GS_RGB_rgba', 'P_normal_rgba')


def _sorted(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, '*.png')))


def _load(p: str):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 2: img = np.stack([img] * 3, -1)
    if img.shape[-1] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def auto_crop_bbox(paths: List[str], margin: int = 30, div: int = 16
                   ) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return (x0, y0, x1, y1, W_orig, H_orig) after margin + div-multiple rounding."""
    if not paths: return None
    xs, ys = [], []
    step = max(1, len(paths) // 20)
    for p in paths[::step][:20]:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        if img.shape[-1] == 4:
            alpha = img[..., 3]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = 255 - gray
        ys_, xs_ = np.where(alpha > 5)
        if len(xs_):
            xs.append(xs_.min()); xs.append(xs_.max())
            ys.append(ys_.min()); ys.append(ys_.max())
    if not xs: return None
    H, W = img.shape[:2]
    x0 = max(0, min(xs) - margin); x1 = min(W, max(xs) + margin)
    y0 = max(0, min(ys) - margin); y1 = min(H, max(ys) + margin)
    w = ((x1 - x0 + div - 1) // div) * div
    h = ((y1 - y0 + div - 1) // div) * div
    cx = (x0 + x1) // 2; cy = (y0 + y1) // 2
    x0 = max(0, cx - w // 2); x1 = min(W, x0 + w); x0 = max(0, x1 - w)
    y0 = max(0, cy - h // 2); y1 = min(H, y0 + h); y0 = max(0, y1 - h)
    return (x0, y0, x1, y1, W, H)


def _write_mp4(frames: List, path: str, fps: int):
    imageio.mimsave(path, [f[..., :3] for f in frames], fps=fps, quality=8)


def _write_webm(frames: List, path: str, fps: int):
    try:
        w = imageio.get_writer(path, fps=fps, codec='libvpx-vp9',
                               ffmpeg_params=['-pix_fmt', 'yuva420p', '-b:v', '2M'])
        for f in frames:
            if f.shape[-1] == 3:
                f = np.concatenate([f, 255 * np.ones(f.shape[:2] + (1,), np.uint8)], -1)
            w.append_data(f)
        w.close()
    except Exception as e:
        print(f'  webm fail {os.path.basename(path)}: {e}')


def build_videos_from_pngs(subdir: str, fps: int = DEFAULT_FPS,
                            kinds: Tuple[str, ...] = KINDS) -> None:
    """For a scene-orbit subdir containing PNG folders, build matching mp4/webm."""
    print(f'=== {subdir} ===')
    # Auto-crop from union of all present kind alphas
    all_paths = []
    for kind in kinds:
        all_paths.extend(_sorted(os.path.join(subdir, kind)))
    if not all_paths:
        print('  no frames, skip'); return
    bbox = auto_crop_bbox(all_paths)
    if bbox is None:
        print('  no bbox, skip'); return
    x0, y0, x1, y1, W, H = bbox
    print(f'  crop: x[{x0},{x1}] y[{y0},{y1}] → {x1 - x0}x{y1 - y0} (orig {W}x{H})')
    for kind in kinds:
        paths = _sorted(os.path.join(subdir, kind))
        if not paths: continue
        frames = []
        for p in paths:
            img = _load(p)
            if img is not None:
                frames.append(img[y0:y1, x0:x1])
        if not frames: continue
        _write_mp4(frames, os.path.join(subdir, f'{kind}.mp4'), fps)
        _write_webm(frames, os.path.join(subdir, f'{kind}.webm'), fps)


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='One or more scene-orbit subdirs to process.')
    ap.add_argument('--fps', type=int, default=DEFAULT_FPS)
    args = ap.parse_args()
    for p in args.paths:
        if os.path.isdir(p):
            build_videos_from_pngs(p, fps=args.fps)
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(_main())

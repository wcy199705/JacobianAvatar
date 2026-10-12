"""
Auto-generate K.npy (3x3 camera intrinsic matrix) for a CUSTOM scene.

Convention (matches CUSTOM_FRAMES_GUIDE.md Option B + actual pipeline behavior):
  - f_px = max(W, H)         # ~50mm fov default for unknown camera
  - cx, cy = W/2, H/2        # principal point at image center

Reads scene root from the YAML's Path.training_folder. Picks the first PNG in
images/ to read W,H. Skips silently if K.npy already exists (so users who hand-
authored an EXIF-based K aren't overwritten).

Usage:
  python make_K_npy.py --scene <yaml_basename>
"""
import argparse
import os
import numpy as np
import yaml
from PIL import Image

JACOBIAN_AVATAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True,
                   help="YAML basename (e.g. my_scene)")
    args = p.parse_args()

    yaml_path = os.path.join(
        JACOBIAN_AVATAR_ROOT, "configs", "legacy", "SMPLX_dataloader", "CUSTOM",
        f"{args.scene}.yaml")
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    scene_root = params["Path"]["training_folder"]

    out_path = os.path.join(scene_root, "K.npy")
    if os.path.exists(out_path):
        print(f"[make_K_npy] {out_path} already exists — skipping.")
        return

    images_dir = os.path.join(scene_root, "images")
    pngs = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))
    if not pngs:
        raise RuntimeError(f"No PNGs under {images_dir}")
    with Image.open(os.path.join(images_dir, pngs[0])) as im:
        W, H = im.size

    f_px = float(max(W, H))
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[f_px, 0,    cx],
                  [0,    f_px, cy],
                  [0,    0,    1]], dtype=np.float64)
    np.save(out_path, K)
    print(f"[make_K_npy] {args.scene}: W={W} H={H} f_px={f_px} cx={cx} cy={cy} "
          f"→ saved {out_path}")


if __name__ == "__main__":
    main()

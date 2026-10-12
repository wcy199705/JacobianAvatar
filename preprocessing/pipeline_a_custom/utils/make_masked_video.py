"""
Debug helper: mask SAMURAI segmentation against original frames and write
both per-frame PNGs and an mp4 video for visual inspection.

For each frame f:
  masked_img = image  (where mask>0)
  masked_img = 0      (where mask==0)

Outputs:
  <scene_root>/masked_images/<frame>.png         per-frame masked PNG
  <scene_root>/debug_masked.mp4                  30 fps preview video

Usage:
  python make_masked_video.py --scene <yaml_basename> [--fps 30] [--max-side 1280]
"""
import argparse
import os

import cv2
import numpy as np
import yaml

JACOBIAN_AVATAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True,
                   help="YAML basename (e.g. ECCV_demo_05)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-side", type=int, default=1280,
                   help="Downscale frames so the longest side <= this (mp4 size)")
    args = p.parse_args()

    yaml_path = os.path.join(
        JACOBIAN_AVATAR_ROOT, "configs", "legacy", "SMPLX_dataloader", "CUSTOM",
        f"{args.scene}.yaml")
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    scene_root = params["Path"]["training_folder"]

    images_dir = os.path.join(scene_root, "images")
    masks_dir = os.path.join(scene_root, "masks")
    out_dir = os.path.join(scene_root, "masked_images")
    out_video = os.path.join(scene_root, "debug_masked.mp4")
    os.makedirs(out_dir, exist_ok=True)

    img_names = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    if not img_names:
        raise RuntimeError(f"No PNGs under {images_dir}")
    if not os.path.isdir(masks_dir):
        raise RuntimeError(f"masks/ missing: {masks_dir} — run step2 first")

    # Probe size from first frame
    first = cv2.imread(os.path.join(images_dir, img_names[0]))
    H, W = first.shape[:2]
    scale = min(1.0, args.max_side / max(H, W))
    out_W, out_H = int(W * scale), int(H * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, args.fps, (out_W, out_H))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_video}")

    written = 0
    missing_mask = 0
    for name in img_names:
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        mask_path = os.path.join(masks_dir, name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape[:2] != img.shape[:2]:
                masked = img  # corrupt mask → show raw image
                missing_mask += 1
            else:
                m3 = (mask > 0)[..., None]
                masked = (img * m3).astype(img.dtype)
        else:
            masked = img
            missing_mask += 1

        cv2.imwrite(os.path.join(out_dir, name), masked)
        if scale < 1.0:
            masked = cv2.resize(masked, (out_W, out_H), interpolation=cv2.INTER_AREA)
        vw.write(masked)
        written += 1

    vw.release()
    print(f"[make_masked_video] {args.scene}: wrote {written} masked PNGs to "
          f"{out_dir} (missing-mask fallback: {missing_mask}), video: {out_video}")


if __name__ == "__main__":
    main()

"""
Sapiens2 Normal Inference (1B) for CUSTOM dataset scenes.

  - Reads scene root from the CUSTOM YAML's Path.training_folder
  - Uses local sapiens2 weights at
    <repo>/libs/sapiens2/checkpoints/normal/

Outputs match the Sapiens 1b convention used by the rest of the pipeline:
  - <scene_root>/crop_normals_sapiens2/NNNNN.npy : float32 (H, W, 3) unit normals
  - <scene_root>/crop_normals_sapiens2/NNNNN.png : uint8 (H, 2W, 3) [img | normal_vis]

Usage:
  CUDA_VISIBLE_DEVICES=N \
  $SAPIENS_PY \
  preprocessing/pipeline_a_custom/steps/step4_sapiens2_normals.py \
    --scene my_scene
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from sapiens.dense.models import init_model


JACOBIAN_AVATAR_V2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root (libs/sapiens2 bundled)
SAPIENS2_REPO_ROOT = f"{JACOBIAN_AVATAR_V2}/libs/sapiens2"
CKPT_ROOT = f"{SAPIENS2_REPO_ROOT}/checkpoints"


def build_model(model_size: str, device: str):
    cfg_path = (
        f"{SAPIENS2_REPO_ROOT}/sapiens/dense/configs/normal/metasim_render_people/"
        f"sapiens2_{model_size}_normal_metasim_render_people-1024x768.py"
    )
    ckpt_path = f"{CKPT_ROOT}/normal/sapiens2_{model_size}_normal.safetensors"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[sapiens2] cfg:  {cfg_path}\n[sapiens2] ckpt: {ckpt_path}")
    return init_model(cfg_path, ckpt_path, device=device)


@torch.inference_mode()
def infer_one(model, image_bgr: np.ndarray) -> np.ndarray:
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs, data_samples = data["inputs"], data["data_samples"]

    normal = model(inputs)
    normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)

    pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"]["padding_size"]
    normal = normal[:, :,
                    pad_top : inputs.shape[2] - pad_bottom,
                    pad_left : inputs.shape[3] - pad_right]
    normal = F.interpolate(
        normal,
        size=(image_bgr.shape[0], image_bgr.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return normal.squeeze(0).cpu().float().numpy().transpose(1, 2, 0)


def save_outputs(out_dir, frame_id, image_bgr, normal, mask):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{frame_id}.npy"), normal.astype(np.float32))
    normal_vis = normal.copy()
    if mask is not None:
        normal_vis[mask == 0] = -1
    normal_vis_rgb = ((normal_vis + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    normal_vis_bgr = normal_vis_rgb[:, :, ::-1]
    side_by_side = np.concatenate([image_bgr, normal_vis_bgr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"{frame_id}.png"), side_by_side)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True,
                   help="YAML basename (e.g. my_scene)")
    p.add_argument("--model_size", default="1b", choices=["0.4b", "0.8b", "1b", "5b"])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    yaml_path = os.path.join(
        JACOBIAN_AVATAR_V2, "configs", "legacy", "SMPLX_dataloader", "CUSTOM", f"{args.scene}.yaml"
    )
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    scene_root = params["Path"]["training_folder"]

    image_dir = os.path.join(scene_root, "image_crop")
    mask_dir = os.path.join(scene_root, "mask_crop")
    out_dir = os.path.join(scene_root, "crop_normals_sapiens2")
    if not os.path.isdir(image_dir):
        print(f"[ERROR] image_crop not found at {image_dir} — did step3 run?",
              file=sys.stderr)
        sys.exit(1)

    frames = sorted(f[:-4] for f in os.listdir(image_dir) if f.endswith(".png"))
    print(f"[sapiens2] scene={args.scene}  #frames={len(frames)}  out={out_dir}")

    model = build_model(args.model_size, args.device)

    t0 = time.time()
    fail = 0
    consec_fail = 0
    per_frame_times = []
    for fid in tqdm(frames, desc="sapiens2 normal"):
        img_path = os.path.join(image_dir, f"{fid}.png")
        mask_path = os.path.join(mask_dir, f"{fid}.png")
        image = cv2.imread(img_path)
        if image is None:
            fail += 1
            consec_fail += 1
            if consec_fail >= 3:
                print(f"[ERROR] {consec_fail} consecutive read failures.", file=sys.stderr)
                sys.exit(2)
            continue

        mask = None
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = (m > 0)

        t_start = time.time()
        try:
            normal = infer_one(model, image)
        except Exception as e:
            print(f"[ERROR] frame {fid}: {e}", file=sys.stderr)
            fail += 1
            consec_fail += 1
            if consec_fail >= 3:
                print(f"[ERROR] {consec_fail} consecutive inference failures.",
                      file=sys.stderr)
                sys.exit(2)
            continue
        torch.cuda.synchronize()
        per_frame_times.append(time.time() - t_start)
        consec_fail = 0
        save_outputs(out_dir, fid, image, normal, mask)

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    mean_t = float(np.mean(per_frame_times)) if per_frame_times else float("nan")
    print(f"\n[sapiens2] DONE {elapsed:.1f}s, mean {mean_t*1000:.0f} ms/frame, "
          f"peak VRAM {vram_mb:.0f} MiB, ok={len(frames) - fail}/{len(frames)}")


if __name__ == "__main__":
    main()

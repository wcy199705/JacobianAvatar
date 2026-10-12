"""SAMURAI (SAM 2.1 + motion-aware memory) person mask tracking from the first-frame bbox.

Uses the unmodified upstream repo in libs/samurai (fetched by scripts/usage/fetch_libs.sh).
"""
import gc
import os
import os.path as osp
import sys
import tempfile

import cv2
import numpy as np
import torch

_JA_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
_SAM2_ROOT = osp.join(_JA_ROOT, 'libs', 'samurai', 'sam2')
if _SAM2_ROOT not in sys.path:
    sys.path.append(_SAM2_ROOT)
from sam2.build_sam import build_sam2_video_predictor

_IMG_EXT = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')


def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def _as_numbered_jpg_folder(frames, tmp_dir):
    """SAM 2 only lists `<int>.jpg` files. Expose the frames under that naming via symlinks
    (pixels are decoded by content, so PNG frames are read unchanged)."""
    for i, frame_path in enumerate(frames):
        os.symlink(osp.abspath(frame_path), osp.join(tmp_dir, f'{i:05d}.jpg'))
    return tmp_dir


def run_samurai(save_folder, bbox_path, image_folder,
                model_path=osp.join(_SAM2_ROOT, 'checkpoints', 'sam2.1_hiera_base_plus.pt'), device="cuda:0"):
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)

    prompts = load_txt(bbox_path)
    frames = sorted([osp.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(_IMG_EXT)])
    loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
    height, width = loaded_frames[0].shape[:2]

    os.makedirs(save_folder, exist_ok=True)
    color = [(255, 0, 0)]
    with tempfile.TemporaryDirectory() as tmp_dir, \
            torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(_as_numbered_jpg_folder(frames, tmp_dir), offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask_to_vis[obj_id] = mask > 0.0

            image_name = os.path.basename(frames[frame_idx])
            for obj_id, mask in mask_to_vis.items():
                img = loaded_frames[frame_idx]
                mask_img = np.zeros((height, width, 3), np.uint8)
                mask_img[mask] = color[(obj_id + 1) % len(color)]
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)
            save_mask = 255 * mask
            cv2.imwrite(os.path.join(save_folder, image_name), save_mask)

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    return masks

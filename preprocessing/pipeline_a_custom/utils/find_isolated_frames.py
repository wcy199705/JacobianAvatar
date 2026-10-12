"""
For each frame, detect all persons and decide whether the step1b-selected
person ("target") is spatially isolated from every other detection (no bbox
overlap with anyone).

Output:
  <scene_root>/isolated_frames.txt          frame names where target bbox
                                            does NOT overlap any other
                                            detection (single-person frames
                                            also count as isolated).
  <scene_root>/isolated_frames_idx.txt      same, 5-digit indices only.

Target bbox is read from mmpose_result_mmpose/bboxes/<frame>.txt
(format: x1,y1,x2,y2 — produced by step1b).

Usage:
  CUDA_VISIBLE_DEVICES=N \
  $PREP_PY \
  preprocessing/pipeline_a_custom/utils/find_isolated_frames.py --scene ECCV_demo_06
"""
import argparse
import os
from collections import Counter

import numpy as np
import yaml
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms

JACOBIAN_AVATAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
DET_CONFIG = "libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT = "libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
BBOX_THR = 0.3
NMS_THR = 0.3
DET_CAT_ID = 0


def bbox_touches(a, b):
    """True iff the two axis-aligned bboxes share any pixel."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    args = p.parse_args()

    os.chdir(JACOBIAN_AVATAR_ROOT)
    yaml_path = os.path.join(
        "configs", "legacy", "SMPLX_dataloader", "CUSTOM", f"{args.scene}.yaml")
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    scene_root = params["Path"]["training_folder"]
    images_dir = os.path.join(scene_root, "images")
    target_bbox_dir = os.path.join(scene_root, "mmpose_result_mmpose", "bboxes")

    pngs = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    print(f"[isolated] {args.scene}: {len(pngs)} frames")

    detector = init_detector(DET_CONFIG, DET_CKPT, device="cuda:0")

    isolated_names = []
    reason = Counter()
    n_target_missing = 0

    for name in tqdm(pngs, desc="detect+iou"):
        stem = name.split(".")[0]
        img_path = os.path.join(images_dir, name)
        det = inference_detector(detector, img_path)
        pi = det.pred_instances.cpu().numpy()
        mask = (pi.labels == DET_CAT_ID) & (pi.scores > BBOX_THR)
        bbs = np.concatenate(
            (pi.bboxes[mask], pi.scores[mask, None]), axis=1)
        if len(bbs) > 0:
            bbs = bbs[nms(bbs, NMS_THR), :4]
        n = len(bbs)

        if n == 0:
            reason["no_detection"] += 1
            continue
        if n == 1:
            isolated_names.append(name)
            reason["single_detection"] += 1
            continue

        # Load target bbox produced by step1b
        target_path = os.path.join(target_bbox_dir, stem + ".txt")
        if not os.path.exists(target_path):
            n_target_missing += 1
            reason["target_missing"] += 1
            continue
        with open(target_path) as f:
            t = [float(x) for x in f.read().strip().split(",")]
        if len(t) != 4:
            reason["target_bad_format"] += 1
            continue
        target = t  # [x1, y1, x2, y2]

        # Identify which detection is the target (max IoU)
        ious_to_target = [iou(target, b.tolist()) for b in bbs]
        tgt_idx = int(np.argmax(ious_to_target))
        others = [bbs[i].tolist() for i in range(len(bbs)) if i != tgt_idx]

        touches_any = any(bbox_touches(target, o) for o in others)
        if not touches_any:
            isolated_names.append(name)
            reason["multi_but_isolated"] += 1
        else:
            reason["overlaps_other"] += 1

    with open(os.path.join(scene_root, "isolated_frames.txt"), "w") as f:
        for n in isolated_names: f.write(n + "\n")
    with open(os.path.join(scene_root, "isolated_frames_idx.txt"), "w") as f:
        for n in isolated_names: f.write(n.split(".")[0] + "\n")

    print("\n[isolated] breakdown:")
    for k in ("no_detection", "single_detection", "multi_but_isolated",
              "overlaps_other", "target_missing", "target_bad_format"):
        print(f"  {k:25s}: {reason.get(k,0):5d}")
    print(f"\n[isolated] TOTAL isolated frames (single OR multi-but-non-overlapping): "
          f"{len(isolated_names)} ({100*len(isolated_names)/len(pngs):.1f}% of {len(pngs)})")
    print(f"  list: {scene_root}/isolated_frames.txt")


if __name__ == "__main__":
    main()

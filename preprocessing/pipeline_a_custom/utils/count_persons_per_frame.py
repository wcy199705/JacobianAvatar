"""
Run DWPose person-detection only (no pose, no visualization) on every frame
of a CUSTOM scene and report how many persons were detected per frame.

Outputs:
  <scene_root>/person_count_per_frame.tsv     frame<TAB>count
  <scene_root>/single_person_frames.txt       frames with exactly 1 detection
  <scene_root>/single_person_frames_idx.txt   same, just the 5-digit index
  prints histogram summary

Usage:
  CUDA_VISIBLE_DEVICES=N \
  $PREP_PY \
  preprocessing/pipeline_a_custom/utils/count_persons_per_frame.py --scene ECCV_demo_06

Must be run from the repo root (PYTHONPATH=. plus
JacobianAvatar_v2 root for utils.io.mesh resolution).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import yaml
from tqdm import tqdm

# mmdet / mmpose
from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms

JACOBIAN_AVATAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root

DET_CONFIG = "libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT = "libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
BBOX_THR = 0.3
NMS_THR = 0.3
DET_CAT_ID = 0  # person


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

    pngs = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    print(f"[count_persons] {args.scene}: {len(pngs)} frames")

    detector = init_detector(DET_CONFIG, DET_CKPT, device="cuda:0")

    counts = []
    with open(os.path.join(scene_root, "person_count_per_frame.tsv"), "w") as out:
        out.write("frame\tcount\n")
        for name in tqdm(pngs, desc="detect"):
            img_path = os.path.join(images_dir, name)
            det = inference_detector(detector, img_path)
            pi = det.pred_instances.cpu().numpy()
            mask = (pi.labels == DET_CAT_ID) & (pi.scores > BBOX_THR)
            bboxes = np.concatenate(
                (pi.bboxes[mask], pi.scores[mask, None]), axis=1)
            if len(bboxes) > 0:
                bboxes = bboxes[nms(bboxes, NMS_THR), :]
            n = int(len(bboxes))
            counts.append((name, n))
            out.write(f"{name}\t{n}\n")

    hist = Counter(n for _, n in counts)
    print("\n[count_persons] histogram (count -> frames):")
    for n in sorted(hist.keys()):
        print(f"  {n} person(s): {hist[n]} frames")

    single = [name for name, n in counts if n == 1]
    with open(os.path.join(scene_root, "single_person_frames.txt"), "w") as f:
        for n in single: f.write(n + "\n")
    with open(os.path.join(scene_root, "single_person_frames_idx.txt"), "w") as f:
        for n in single: f.write(n.split(".")[0] + "\n")

    print(f"\n[count_persons] single-person frames: {len(single)} "
          f"({100*len(single)/len(pngs):.1f}% of {len(pngs)})")
    print(f"  list: {scene_root}/single_person_frames.txt")
    print(f"  idx : {scene_root}/single_person_frames_idx.txt")
    print(f"  tsv : {scene_root}/person_count_per_frame.tsv")


if __name__ == "__main__":
    main()

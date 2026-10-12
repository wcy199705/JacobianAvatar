"""
Meta wrapper for CUSTOM preprocessing steps.

Reads an existing step script (which hardcodes
`configs/legacy/SMPLX_dataloader/CUSTOM/scarf.yaml`), patches the literal in memory
to point at the requested per-scene YAML, then execs it as __main__.

This lets us run scenes in parallel without touching any existing code on
disk — each process has its own patched source in memory.

Steps:
  1   — legacy: full DWPose, single-person filter by SUM(keypoint_scores).
        DEPRECATED for multi-person scenes — use 1a + 1b instead.
  1a  — DWPose on FIRST FRAME ONLY, produces first_bbox.txt for SAMURAI.
  2   — SAMURAI single-person mask tracking (uses first_bbox.txt).
  1b  — DWPose on ALL frames, but picks the person whose visible keypoints
        fall most inside the SAMURAI mask. Replaces all step1-style outputs.
  3-7 — unchanged (mask_crop, sapiens normals, global SMPL, per-frame fit,
        head crop).

Usage:
  python run_custom_step.py --step <id> --scene <yaml_basename>
  where <id> ∈ {1a, 1b, 1, 2, 3, 5, 6, 7}
"""
import argparse
import os
import sys

JACOBIAN_AVATAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root

# Every entry maps step id -> (relative path, patch mode)
STEP_PATHS = {
    "1":  ("preprocessing/pipeline_a_custom/steps/step1_dwpose.py", "step1_sum"),
    "1a": ("preprocessing/pipeline_a_custom/steps/step1_dwpose.py", "step1a"),
    "1b": ("preprocessing/pipeline_a_custom/steps/step1_dwpose.py", "step1b"),
    "2":  ("preprocessing/pipeline_a_custom/steps/step2_samurai.py", None),
    "3":  ("preprocessing/pipeline_a_custom/steps/step3_crop_masks.py", None),
    "5":  ("preprocessing/pipeline_a_custom/steps/step5_global_optimize.py", None),
    "6":  ("preprocessing/pipeline_a_custom/steps/step6_smplx_fit.py", None),
    "7":  ("preprocessing/pipeline_a_custom/steps/step7_crop_head.py", "step7_safe_crop"),
}

# YAML hardcoded literals across all step scripts.
OLD_LITERALS = [
    "os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','scarf.yaml')",
    "os.path.join('configs', 'legacy', 'SMPLX_dataloader', 'CUSTOM', f'{scene}.yaml')",
]

# ------------------------------------------------------------------
# step1 source patches. All three modes (1, 1a, 1b) start from the same
# step1_dwpose.py and apply mode-specific replacements.
# ------------------------------------------------------------------

STEP1_FILTER_ANCHOR = """        bbox_list.append(bboxes)
        pose_results = inference_topdown(pose_estimator, img_path, bboxes)
        data_samples = merge_data_samples(pose_results)"""

# Mode "step1_sum" (and "step1a"): pick person by sum(keypoint_scores).
STEP1_FILTER_SUM = """        pose_results = inference_topdown(pose_estimator, img_path, bboxes)
        if len(pose_results) > 1:
            _sums = [float(_r.pred_instances.keypoint_scores.sum()) for _r in pose_results]
            _best = int(np.argmax(_sums))
            pose_results = [pose_results[_best]]
            bboxes = bboxes[_best:_best+1]
        bbox_list.append(bboxes)
        data_samples = merge_data_samples(pose_results)"""

# Mode "step1b": prefer the person whose visible keypoints fall most inside the
# SAMURAI mask, then among such candidates pick the one with highest sum of
# keypoint scores (= the well-visible, large-enough person). Falls back to
# sum-of-scores if SAMURAI mask is missing or empty.
#
# Tightening rules: ignore tiny detections (false-positive
# hand/face fragments), and synthesize a default full-image bbox when no
# person is detected at all so the downstream second loop doesn't crash on an
# empty array.
STEP1_FILTER_MASK = """        pose_results = inference_topdown(pose_estimator, img_path, bboxes)
        if len(pose_results) == 0 or len(bboxes) == 0:
            _img_for_shape = cv2.imread(img_path)
            _h_full, _w_full = _img_for_shape.shape[:2]
            bboxes = np.array([[0.0, 0.0, float(_w_full), float(_h_full)]], dtype=np.float32)
            print(f'[step1b] {os.path.basename(img_path)}: no detection — full-image fallback bbox')
        elif len(pose_results) >= 1:
            _mask_path = os.path.join(save_root, 'masks', os.path.basename(img_path))
            _mask = cv2.imread(_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(_mask_path) else None
            _sums = [float(_r.pred_instances.keypoint_scores.sum()) for _r in pose_results]
            _bbox_w = bboxes[:, 2] - bboxes[:, 0]
            _bbox_h = bboxes[:, 3] - bboxes[:, 1]
            _size_ok = [(_bbox_w[i] > 50 and _bbox_h[i] > 100) for i in range(len(pose_results))]
            if _mask is None:
                _best = int(np.argmax(_sums))
            else:
                _h, _w = _mask.shape[:2]
                def _mask_overlap(_r):
                    _kps = _r.pred_instances.keypoints[0]
                    _scs = _r.pred_instances.keypoint_scores[0]
                    _vis = _scs > 0.3
                    if _vis.sum() == 0:
                        return 0.0
                    _xs = np.clip(_kps[_vis, 0].astype(int), 0, _w - 1)
                    _ys = np.clip(_kps[_vis, 1].astype(int), 0, _h - 1)
                    return float((_mask[_ys, _xs] > 0).mean())
                _overs = [_mask_overlap(_r) for _r in pose_results]
                _valid = [i for i in range(len(pose_results)) if _overs[i] >= 0.5 and _size_ok[i]]
                if _valid:
                    _best = max(_valid, key=lambda i: _sums[i])
                elif max(_overs) > 0:
                    _best = int(np.argmax(_overs))
                else:
                    _best = int(np.argmax(_sums))
            pose_results = [pose_results[_best]]
            bboxes = bboxes[_best:_best+1]
        bbox_list.append(bboxes)
        data_samples = merge_data_samples(pose_results)"""

# Per-frame keypoint saver — keep sum-of-scores criterion so it stays in
# sync with the bbox-side filter.
STEP1_KPT_OLD = "if (kpt_save is None) or (kpt_save[:,2].mean() < kpt[:,2].mean()):"
STEP1_KPT_NEW = "if (kpt_save is None) or (kpt_save[:,2].sum() < kpt[:,2].sum()):"

# image_crop size cap: when a single detection bbox is larger than the frame
# (mmpose can output bboxes that extend slightly outside the image), the
# common-window crop computed below would use width > img_width, which then
# triggers the `img_width - width` fallback in step1 and yields negative
# offsets / 2-pixel-wide crops. Cap width/height to the actual frame size.
STEP1_SIZE_OLD = """    width = math.ceil(max(widths))
    height = math.ceil(max(heights))"""
STEP1_SIZE_NEW = """    width = math.ceil(max(widths))
    height = math.ceil(max(heights))
    _probe = cv2.imread(img_paths[0])
    if _probe is not None:
        width = min(width, _probe.shape[1])
        height = min(height, _probe.shape[0])"""

# Mode "step1a": slice img_list to first frame only.
STEP1A_ANCHOR = "images, img_list = data_loader.load_images(image_root, return_mask=False)"
STEP1A_NEW = """images, img_list = data_loader.load_images(image_root, return_mask=False)
    # === STEP1A: process only first frame to generate first_bbox.txt ===
    images = images[:1]
    img_list = img_list[:1]"""

# step7 safety: when head bbox clips to a 0-pixel slice (head off-image), the
# cv2.resize call on the empty crop raises (-215:ssize.empty()). Replace the
# empty crop with a fallback centered region of the same target size.
STEP7_OLD = """        croped_img = img[int(y1):int(y2), int(x1):int(x2), :]
        cropped_mask = mask[int(y1):int(y2), int(x1):int(x2),]
        croped_img = cv2.resize(croped_img,(resize_size_W,resize_size_H),cv2.INTER_LANCZOS4)"""
STEP7_NEW = """        croped_img = img[int(y1):int(y2), int(x1):int(x2), :]
        cropped_mask = mask[int(y1):int(y2), int(x1):int(x2),]
        if croped_img.size == 0:
            croped_img = np.zeros((resize_size_H, resize_size_W, 3), dtype=img.dtype)
            print(f'[step7] frame {img_name}: head bbox empty → zeros image fallback')
        if cropped_mask.size == 0:
            cropped_mask = np.zeros((resize_size_H, resize_size_W), dtype=mask.dtype)
            print(f'[step7] frame {img_name}: head bbox empty → zeros mask fallback')
        croped_img = cv2.resize(croped_img,(resize_size_W,resize_size_H),cv2.INTER_LANCZOS4)"""


def patch_source(src: str, yaml_basename: str, patch_mode):
    target = f"os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','{yaml_basename}.yaml')"
    patched = src
    n_replacements = 0
    for old in OLD_LITERALS:
        if old in patched:
            patched = patched.replace(old, target)
            n_replacements += 1
    if "scenes=['scarf']" in patched:
        patched = patched.replace("scenes=['scarf']", f"scenes=['{yaml_basename}']")
        n_replacements += 1
    if n_replacements == 0:
        raise RuntimeError("No yaml literal found to patch in source.")

    if patch_mode in ("step1_sum", "step1a", "step1b"):
        if STEP1_FILTER_ANCHOR not in patched:
            raise RuntimeError(f"step1 filter anchor missing for mode={patch_mode}")
        new_filter = STEP1_FILTER_MASK if patch_mode == "step1b" else STEP1_FILTER_SUM
        patched = patched.replace(STEP1_FILTER_ANCHOR, new_filter)

        if STEP1_KPT_OLD not in patched:
            raise RuntimeError("step1 keypoint-save anchor missing.")
        patched = patched.replace(STEP1_KPT_OLD, STEP1_KPT_NEW)

        if STEP1_SIZE_OLD not in patched:
            raise RuntimeError("step1 size-cap anchor missing.")
        patched = patched.replace(STEP1_SIZE_OLD, STEP1_SIZE_NEW)

        if patch_mode == "step1a":
            if STEP1A_ANCHOR not in patched:
                raise RuntimeError("step1a img_list anchor missing.")
            patched = patched.replace(STEP1A_ANCHOR, STEP1A_NEW)

    if patch_mode == "step7_safe_crop":
        if STEP7_OLD not in patched:
            raise RuntimeError("step7 crop anchor missing.")
        patched = patched.replace(STEP7_OLD, STEP7_NEW)

    return patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=str, required=True, choices=list(STEP_PATHS.keys()))
    p.add_argument("--scene", type=str, required=True,
                   help="YAML basename without .yaml")
    args = p.parse_args()

    # Drop this script's own dir from sys.path: its utils/ package would shadow
    # the legacy utils/ (preprocessing/legacy) that the step scripts import.
    _own = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [q for q in sys.path if os.path.abspath(q or '.') != _own]
    os.chdir(JACOBIAN_AVATAR_ROOT)
    step_rel, patch_mode = STEP_PATHS[args.step]
    if not os.path.exists(step_rel):
        raise FileNotFoundError(f"Step script not found: {step_rel}")

    src = open(step_rel).read()
    patched = patch_source(src, args.scene, patch_mode)

    code = compile(patched, step_rel, "exec")
    g = {"__name__": "__main__", "__file__": os.path.abspath(step_rel)}
    sys.argv = [step_rel]
    exec(code, g)


if __name__ == "__main__":
    main()

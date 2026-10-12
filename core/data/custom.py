"""CUSTOM dataset class — SceneDataset + optional valid-frames filter.

Inherits SceneDataset (which loads image_crop/mask_crop/cameras/smplx),
then optionally drops frames listed in a `valid_frames_file`. Auto-detect
convention path is `<training_folder>/joint_valid_frames.txt` (written by
Pre-stage validation dump).

Use from yaml:
  dataset_name: Custom
  valid_frames_file: /path/to/list.txt   # optional, overrides auto-detect

If the file is absent, behavior is identical to plain SceneDataset loading.
"""
from __future__ import annotations
import glob
import os

import torch

from core.data.scene import SceneDataset


class Custom(SceneDataset):
    def __init__(self, root_folder, smpl_config, no_crop=False, image_size=None,
                 use_head_data=False, img_scale_factor=1.0, device='cuda',
                 valid_frames_file=None, anchor_frame_file=None):
        # Auto-detect convention path if the caller didn't pass one.
        if not valid_frames_file:
            _auto = os.path.join(root_folder, 'joint_valid_frames.txt')
            if os.path.exists(_auto):
                valid_frames_file = _auto
                print(f"[Custom] auto-detected valid_frames_file → {_auto}", flush=True)

        # Build the underlying SceneDataset (all frames loaded).
        super().__init__(root_folder=root_folder,
                         smpl_config=smpl_config,
                         no_crop=no_crop,
                         image_size=image_size,
                         use_head_data=use_head_data,
                         img_scale_factor=img_scale_factor,
                         device=device)

        # Apply filter post-init so we don't have to touch SceneDataset.
        if valid_frames_file and os.path.exists(valid_frames_file):
            self._apply_valid_filter(valid_frames_file)

        # Resolve anchor frame index (file may contain abs path or basename).
        self.anchor_frame_idx = None
        if anchor_frame_file and os.path.exists(anchor_frame_file):
            with open(anchor_frame_file) as _f:
                _anchor_lines = [ln.strip() for ln in _f if ln.strip()]
            if _anchor_lines:
                _want = _anchor_lines[0]
                _want_base = os.path.basename(_want)
                for _i, _p in enumerate(self.file_names):
                    if _p == _want or os.path.basename(_p) == _want_base:
                        self.anchor_frame_idx = _i
                        break
                print(f"[Custom] anchor_frame_idx={self.anchor_frame_idx} "
                      f"({_want})", flush=True)

    # ------------------------------------------------------------------
    def _apply_valid_filter(self, valid_frames_file: str) -> None:
        """Drop frames not listed in valid_frames_file from every per-frame
        attribute. Covers list, torch.Tensor, np.ndarray, dict-of-tensors,
        and the lazy loaders (`_LazyPairedImageList.path_pairs`,
        `_LazyNormalList.file_paths`)."""
        import numpy as _np  # local
        with open(valid_frames_file) as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        basenames = {os.path.basename(p) for p in wanted}
        all_files = list(self.file_names)
        n_full = len(all_files)
        keep_idx = [i for i, p in enumerate(all_files)
                    if (p in wanted) or (os.path.basename(p) in basenames)]
        if len(keep_idx) == n_full:
            return  # nothing dropped
        self.file_names = [all_files[i] for i in keep_idx]
        print(f"[Custom] valid_frames_file filter: {n_full} -> "
              f"{len(self.file_names)} ({valid_frames_file})", flush=True)

        def _subsample(obj):
            if obj is None:
                return obj
            # list (cameras, smpl_list, images/masks fallback)
            if isinstance(obj, list) and len(obj) == n_full:
                return [obj[i] for i in keep_idx]
            # torch.Tensor (target_kpts, smpl_params values)
            if isinstance(obj, torch.Tensor) and obj.dim() >= 1 and obj.shape[0] == n_full:
                return obj[keep_idx]
            # np.ndarray (head_aabb, eager normals fallback)
            if isinstance(obj, _np.ndarray) and obj.ndim >= 1 and obj.shape[0] == n_full:
                return obj[keep_idx]
            # Lazy paired image/mask (path_pairs of length n_full)
            if hasattr(obj, 'path_pairs') and isinstance(obj.path_pairs, list) \
                    and len(obj.path_pairs) == n_full:
                obj.path_pairs = [obj.path_pairs[i] for i in keep_idx]
                return obj
            # Lazy normal loader (file_paths of length n_full)
            if hasattr(obj, 'file_paths') and isinstance(obj.file_paths, list) \
                    and len(obj.file_paths) == n_full:
                obj.file_paths = [obj.file_paths[i] for i in keep_idx]
                return obj
            # dict of per-frame tensors (smpl_params, smpl_full_dict, smpl_batch)
            if isinstance(obj, dict):
                changed = {}
                for k, v in obj.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == n_full:
                        changed[k] = v[keep_idx]
                    elif isinstance(v, _np.ndarray) and v.ndim >= 1 and v.shape[0] == n_full:
                        changed[k] = v[keep_idx]
                    elif isinstance(v, list) and len(v) == n_full:
                        changed[k] = [v[i] for i in keep_idx]
                    else:
                        changed[k] = v
                return changed
            return obj  # leave anything else alone

        # Apply to every known per-frame attribute. Covers SceneDataset + BaseDataset.
        for attr in ('images', 'masks', 'normals', 'cameras', 'target_kpts',
                     'crop_bboxes', 'head_aabb', 'head_images', 'head_masks',
                     'smpl_params', 'smpl_list', 'smpl_full_dict', 'smpl_batch'):
            if hasattr(self, attr):
                setattr(self, attr, _subsample(getattr(self, attr)))

        # num_frames mirror — many downstream callers rely on this.
        if hasattr(self, 'num_frames'):
            self.num_frames = len(self.file_names)

"""Subset head_imgs / head_masks / head_aabb to match the sampled cameras.

Bug being fixed:
  The legacy stage code passes the FULL `train_dataset.head_images` (size N)
  into `ECCVavatar_w_split_datasets_w_split_level(...)`, even when `cameras`
  / `images` are a triplet-sampled SUBSET (size K). Inside the optimizer,
  `all_batch_idxs` is then used to index BOTH:

      tmp_images      = images[all_batch_idxs]      # frame triplet_idxs[i]
      tmp_head_images = head_imgs[all_batch_idxs]   # frame i (≠ triplet_idxs[i])

  → mesh renders pose A while head loss compares against GT of frame B.

Fix:
  Detect by camera identity. `get_sampled_data` keeps the same Camera
  objects, so `id(c)` of the cameras passed into the optimizer matches
  one of `train_dataset.cameras`. Resolve the indices through that
  identity map and slice head_imgs / head_masks / head_aabb to align
  with the sampled cameras.

Side effect:
  Replaces `SceneDataset.__init__` to cache the latest constructed dataset
  in a module-level holder, so the wrapped optimizer can find it.
"""
from __future__ import annotations

from core.data.scene import SceneDataset
from core.optim import avatar_optimizer as opt_mod


_train_dataset_for_head_fix = [None]


def register_train_dataset(ds) -> None:
    """Populate the train-dataset cache directly.

    The `SceneDataset.__init__` wrapper installed by `apply()` populates this same
    cache on construction — but `apply()` may run AFTER datasets are built
    in the v5_release pipeline (mesh_setup runs after setup_trainer). Call
    this from `setup_trainer` once `train_dataset` exists so the wrapped
    optimizer can find it.
    """
    _train_dataset_for_head_fix[0] = ds
    if hasattr(ds, 'head_aabb'):
        print(f'[HEAD_IDX_FIX] registered train_dataset '
              f'({len(ds.cameras)} frames, head_aabb={ds.head_aabb.shape})')


def apply() -> None:
    orig_init = SceneDataset.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if hasattr(self, 'head_aabb'):
            _train_dataset_for_head_fix[0] = self
            print(f'[HEAD_IDX_FIX] cached train_dataset '
                  f'({len(self.cameras)} frames, head_aabb={self.head_aabb.shape})')

    SceneDataset.__init__ = patched_init

    orig_fn = opt_mod.AvatarOptimizer.ECCVavatar_w_split_datasets_w_split_level

    def wrapped(self, *args, **kwargs):
        ds = _train_dataset_for_head_fix[0]
        cameras = kwargs.get('cameras')
        head_imgs = kwargs.get('head_imgs')
        head_masks = kwargs.get('head_masks')
        head_bboxs = kwargs.get('head_bboxs')

        if ds is None or not cameras or head_imgs is None or head_bboxs is None:
            return orig_fn(self, *args, **kwargs)

        n_cam = len(cameras)
        try:
            n_head = len(head_imgs)
        except Exception:
            return orig_fn(self, *args, **kwargs)

        if n_head == n_cam:
            return orig_fn(self, *args, **kwargs)

        id_to_idx = {id(c): i for i, c in enumerate(ds.cameras)}
        indices = [id_to_idx.get(id(c), -1) for c in cameras]
        if any(i < 0 for i in indices):
            print(f'[HEAD_IDX_FIX] WARN: {sum(1 for i in indices if i<0)}/'
                  f'{len(indices)} cameras NOT in train_dataset — passthrough')
            return orig_fn(self, *args, **kwargs)

        import numpy as np
        idx_arr = np.asarray(indices)

        if isinstance(head_imgs, (list, tuple)):
            kwargs['head_imgs'] = [head_imgs[i] for i in indices]
        else:
            kwargs['head_imgs'] = head_imgs[idx_arr]
        if head_masks is not None:
            if isinstance(head_masks, (list, tuple)):
                kwargs['head_masks'] = [head_masks[i] for i in indices]
            else:
                kwargs['head_masks'] = head_masks[idx_arr]
        kwargs['head_bboxs'] = head_bboxs[idx_arr]

        # anchor_idx is supplied in train_dataset coord; remap to position
        # within the sampled cameras list (the coord ECCVavatar sees).
        # If anchor isn't in the sampled set (typical for All/GS stages),
        # build `_anchor_ext_data` from the full train_dataset so ECCVavatar
        # can still do a separate 1-frame forward for the anchor. This keeps
        # anchor_head_only active in All/GS.
        anchor_idx = kwargs.get('anchor_idx', None)
        if anchor_idx is not None:
            try:
                new_anchor = indices.index(int(anchor_idx))
                kwargs['anchor_idx'] = new_anchor
                print(f'[HEAD_IDX_FIX] anchor_idx remap {anchor_idx} → {new_anchor} '
                      f'(train_dataset → sampled-cameras coord)')
            except ValueError:
                a = int(anchor_idx)
                # Build ext data from full train_dataset. head_imgs/masks/bboxs
                # were the ORIGINAL kwargs (before subset above), so re-fetch
                # from ds directly to be safe.
                _head_img_a = ds.head_images[a:a + 1] if hasattr(ds.head_images, '__getitem__') else None
                if _head_img_a is None:
                    _head_img_a = [ds.head_images[a]]
                _head_mask_a = ds.head_masks[a:a + 1] if hasattr(ds.head_masks, '__getitem__') else None
                if _head_mask_a is None:
                    _head_mask_a = [ds.head_masks[a]]
                _head_bbox_a = ds.head_aabb[a:a + 1]
                _smpl_batch_a = {}
                for _k, _v in ds.smpl_params.items():
                    if _k == 'betas':
                        _smpl_batch_a[_k] = _v
                    else:
                        _smpl_batch_a[_k] = _v[a:a + 1]
                kwargs['_anchor_ext_data'] = {
                    'camera': ds.cameras[a],
                    'head_img': _head_img_a,
                    'head_mask': _head_mask_a,
                    'head_bbox': _head_bbox_a,
                    'smpl_batch': _smpl_batch_a,
                }
                # Keep anchor_head_only True — ECCVavatar will pick up ext data.
                # anchor_idx sentinel = -1 tells ECCVavatar to read ext data.
                kwargs['anchor_idx'] = -1
                print(f'[HEAD_IDX_FIX] anchor_idx={anchor_idx} not in sampled '
                      f'cameras — using full train_dataset for anchor (All/GS mode)')

        tail = '...' if len(indices) > 6 else ''
        print(f'[HEAD_IDX_FIX] subset head data: full {n_head} → matched {n_cam} '
              f'via camera identity (indices={indices[:6]}{tail})')
        return orig_fn(self, *args, **kwargs)

    opt_mod.AvatarOptimizer.ECCVavatar_w_split_datasets_w_split_level = wrapped

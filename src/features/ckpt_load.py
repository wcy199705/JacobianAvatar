"""Robust ckpt loading: handle shape mismatches per-kind.

When the current model's MLP dims differ from the saved ckpt (e.g.,
scale_offset output_dim 1→3 override), strict load_state_dict() would
crash. This patch:

  - For *parameters* with shape mismatch: drop the key (the affected layer
    re-inits from the model's constructor) — parameter shapes are typically
    wired to architecture choices that should not silently grow.
  - For *non-parameter buffers* (e.g. `avatar_indices.*.rhand`): RESIZE the
    in-model buffer to the saved shape and load. These buffers store mesh
    region indices that depend on Poisson remesh output; if the upstream
    mesh varies slightly between launches, the region size may differ —
    but the saved value is the canonical one we want to keep.

Wraps `Cascade_Avatar.load_model`. Idempotent (re-apply replaces wrapper).
"""
from __future__ import annotations

import torch

from core.avatar import cascade as cad


def _get_module_attr(root, dotted: str):
    """Resolve `a.b.c` against `root`; return (parent_module, leaf_name) or
    (None, None) if any segment is missing."""
    parts = dotted.split('.')
    obj = root
    for p in parts[:-1]:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        elif p.isdigit() and hasattr(obj, '__getitem__'):
            obj = obj[int(p)]
        else:
            return None, None
    return obj, parts[-1]


def apply() -> None:
    def patched(self, path):
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt['model_state_dict']
        cur = self.state_dict()
        param_names = {name for name, _ in self.named_parameters()}

        filtered, dropped_params, resized_buffers = {}, [], []
        for k, v in sd.items():
            if k not in cur:
                continue
            if cur[k].shape == v.shape:
                filtered[k] = v
                continue
            # Shape mismatch — decide by kind
            if k in param_names:
                dropped_params.append((k, list(v.shape), list(cur[k].shape)))
                continue
            # Buffer with shape mismatch — replace with ckpt buffer
            parent, leaf = _get_module_attr(self, k)
            if parent is None:
                dropped_params.append((k, list(v.shape), 'UNRESOLVED'))
                continue
            new_buf = v.detach().to(self.device).clone()
            try:
                delattr(parent, leaf)
            except AttributeError:
                pass
            parent.register_buffer(leaf, new_buf)
            resized_buffers.append((k, list(v.shape), list(cur[k].shape)))

        if dropped_params:
            print(f'[CKPT_LOAD] Dropped {len(dropped_params)} parameter(s) with shape mismatch (re-init):')
            for k, saved, current in dropped_params[:6]:
                print(f'  {k}: saved{saved} vs current{current}')
        if resized_buffers:
            print(f'[CKPT_LOAD] Resized {len(resized_buffers)} buffer(s) to ckpt shape:')
            for k, saved, current in resized_buffers[:6]:
                print(f'  {k}: current{current} → saved{saved}')

        self.load_state_dict(filtered, strict=False)
        print(f'Model loaded from {path}. Current Level: {self.curr_level}')

    cad.Cascade_Avatar.load_model = patched
    print('[CKPT_LOAD] Shape-mismatch filter installed (params drop, buffers resize)')

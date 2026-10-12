"""Inject SMPL-X pose / transl learning rates into `opt_conf` at every stage.

The legacy stage code calls `ECCVavatar_w_split_datasets_w_split_level(opt_conf=…)`
without specifying `full_pose` / `transl` lr. This patch sets those defaults at
the call boundary so we don't have to touch each stage's invocation.

Uses `setdefault` semantics — does not overwrite values already present
(preserves any stage that intentionally chose its own lr).
"""
from __future__ import annotations

from typing import Optional

from core.optim import avatar_optimizer as opt_mod


def apply(pose_lr: Optional[float] = None,
          transl_lr: Optional[float] = None) -> None:
    orig_fn = opt_mod.AvatarOptimizer.ECCVavatar_w_split_datasets_w_split_level

    def wrapped(self, *args, **kwargs):
        if 'opt_conf' in kwargs and isinstance(kwargs['opt_conf'], dict):
            oc = kwargs['opt_conf']
            if pose_lr is not None and 'full_pose' not in oc:
                oc['full_pose'] = float(pose_lr)
                print(f'[OPT_CONF] inject full_pose={pose_lr}')
            if transl_lr is not None and 'transl' not in oc:
                oc['transl'] = float(transl_lr)
                print(f'[OPT_CONF] inject transl={transl_lr}')
        return orig_fn(self, *args, **kwargs)

    opt_mod.AvatarOptimizer.ECCVavatar_w_split_datasets_w_split_level = wrapped

"""Freeze RGB MLP / Refine RGB MLP grads for the first N iters of GS#1.

This is purely a class-attribute assignment — the actual freeze is implemented
inside `avatar_optimizer.py:3073` which reads
`getattr(FineAvatar, 'gs_freeze_rgb_iters', 0.0)`.
"""
from __future__ import annotations

from core.avatar.cascade import Cascade_Avatar


def apply(value: int) -> None:
    Cascade_Avatar.gs_freeze_rgb_iters = float(value)
    print(f'[GS_FREEZE_RGB] first {int(value)} iters of GS#1 → '
          f'RGB_MLP/Refine_RGB_MLP grads frozen')

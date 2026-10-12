"""Per-frame scaling clamp for GS (bodypart_scale=False path only).

Original code returns `scaling = exp(scale_out - 6.0)` (init ≈ 2.5 mm).
This patch:
  1. Shifts: `scaling = exp(scale_out - init_offset)` via multiplication.
     - init_offset=4.0 → init ≈ 18 mm (the v5 default).
  2. Clamps `scaling` to [min_clamp_m, max_clamp_m].

Wraps `Cascade_Avatar._generate_gs_at_level`. Inactive when
`self.bodypart_scale=True`.
"""
from __future__ import annotations

import math

import torch

from core.avatar import cascade as cad


def apply(max_clamp_m: float = 0.01,
          min_clamp_m: float = 0.0,
          init_offset: float = 6.0) -> None:
    orig_gen = cad.Cascade_Avatar._generate_gs_at_level
    shift_factor = math.exp(6.0 - init_offset) if abs(init_offset - 6.0) > 1e-6 else 1.0

    def patched_gen(self, *args, **kwargs):
        gs_params = orig_gen(self, *args, **kwargs)
        if isinstance(gs_params, dict) and 'scaling' in gs_params:
            if not bool(getattr(self, 'bodypart_scale', True)):
                if shift_factor != 1.0:
                    gs_params['scaling'] = gs_params['scaling'] * shift_factor
                if min_clamp_m > 0:
                    gs_params['scaling'] = torch.clamp(gs_params['scaling'],
                                                       min=float(min_clamp_m),
                                                       max=float(max_clamp_m))
                else:
                    gs_params['scaling'] = torch.clamp(gs_params['scaling'],
                                                       max=float(max_clamp_m))
                if gs_params['scaling'].shape[-1] == 1:
                    gs_params['scaling'] = gs_params['scaling'].repeat(1, 1, 3)
        return gs_params

    cad.Cascade_Avatar._generate_gs_at_level = patched_gen
    init_mm = math.exp(-init_offset) * 1000
    print(f'[SCALE_PURE_CLAMP] init_offset={init_offset} (init ≈ {init_mm:.2f}mm), '
          f'clamp [{min_clamp_m*1000:.1f}, {max_clamp_m*1000:.1f}] mm')

"""v5_release feature patches.

Each module here installs ONE monkey-patch on the legacy `Cascade_Avatar` /
`AvatarOptimizer` classes — capturing a behavioral change that v5 relies on
without modifying any existing code. The `apply_all` entry point applies the
full v5 stack in the order required.

Order matters: `mesh2splat_init` wraps `_generate_gs_at_level` AFTER
`scale_clamp` does so the clamp lives on top of the mesh2splat scaling.
"""
from __future__ import annotations

from typing import Any, Dict

from . import (
    ckpt_load,
    scale_clamp,
    head_data_fix,
    region_aware_lap,
    gs_freeze_rgb,
    mesh2splat_init,
    opt_conf_inject,
)


def apply_all(model_params: Dict[str, Any],
              mesh2splat_cfg: Dict[str, Any] = None,
              scale_clamp_cfg: Dict[str, Any] = None,
              training_cfg=None) -> None:
    """Apply every v5 feature patch and adjust `model_params` to match.

    Args:
        model_params: parsed model config (mutated in place).
                      `Cascade_Avatar` block gets the v5 hard-coded fields
                      (subdivide_count=2, bodypart_scale=False, …).
        mesh2splat_cfg: yaml block `mesh2splat:` (sigma, iso, …). None = legacy defaults.
        scale_clamp_cfg: yaml block `scale_clamp:` (pure_max, pure_min, init_offset).
        training_cfg: `cfg.training` (inject_pose_lr, inject_transl_lr, gs_freeze_rgb_iters).
    """
    mp = model_params['Cascade_Avatar']
    mp['subdivide_count'] = 2
    mp['bodypart_scale'] = False
    mp['scale_zero_last_init'] = False
    mp['use_init_rgb'] = False
    mp['gs_decoder']['scale_offset']['output_dim'] = 3

    sc = scale_clamp_cfg or {}
    sc_max = float(sc.get('pure_max', 0.05))
    sc_min = float(sc.get('pure_min', 0.0))
    sc_init_offset = float(sc.get('init_offset', 4.0))

    ms = mesh2splat_cfg or {}
    ms_sigma = float(ms.get('sigma', 0.65))
    ms_iso = str(ms.get('iso', 'none'))
    ms_head_sigma_mult = float(ms.get('head_sigma_mult', 1.0))
    ms_hand_sigma_mult = float(ms.get('hand_sigma_mult', 1.0))
    ms_scale_z_gof = float(ms.get('scale_z_gof', 1.0e-7))  # only used when gs_rasterizer=gof

    freeze_iters = int(getattr(training_cfg, 'gs_freeze_rgb_iters', 50))
    pose_lr = float(getattr(training_cfg, 'inject_pose_lr', 1.0e-4))
    transl_lr = float(getattr(training_cfg, 'inject_transl_lr', 1.0e-4))

    ckpt_load.apply()
    scale_clamp.apply(max_clamp_m=sc_max, min_clamp_m=sc_min, init_offset=sc_init_offset)
    head_data_fix.apply()
    region_aware_lap.apply()
    gs_freeze_rgb.apply(freeze_iters)
    # mesh2splat_init takes over scaling/rotation; its clamp [0, pure_max] is the
    # actual cap on init scale (matches legacy `--scale_pure_max_clamp`).
    mesh2splat_init.apply(sigma=ms_sigma, iso_mode=ms_iso,
                          max_clamp_m=sc_max, min_clamp_m=sc_min,
                          head_sigma_mult=ms_head_sigma_mult,
                          hand_sigma_mult=ms_hand_sigma_mult,
                          scale_z_gof=ms_scale_z_gof)
    ms_balance_strength = float(ms.get('scale_balance_strength', 0.0))
    opt_conf_inject.apply(pose_lr=pose_lr, transl_lr=transl_lr)

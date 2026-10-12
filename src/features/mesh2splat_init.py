"""Surface-aware mesh2splat-style init for Gaussian scale + rotation.

Active when `bodypart_scale=False` and `use_gs_geo=True` (the GS stage).

Two wrappers are installed:

  (1) `forward_gs_decoder`
      Caches the raw scale_out (BEFORE the -6.0 shift inside the original
      `_generate_gs_at_level`) on `self._last_scale_out`. Needed because
      the original return path discards it.

  (2) `_generate_gs_at_level`
      After the original returns, overrides:
          scaling  := scale_init × exp(iso_residual)
          rotation := quat([tangent_u, tangent_v, normal])
          scale_offset := scale_residual.pow(2)

      where:
          scale_init = [|e0|·σ, |e1·t_v|·σ, 1e-7]   (anisotropic, paper-thin z)
            or
          scale_init = avg_edge_len · σ, broadcast to all 3 axes (`iso_mode='full'`)

  Init behavior (random MLP weights ⇒ scale_out ≈ 0):
      scaling ≈ scale_init   (mesh edge length × σ)
"""
from __future__ import annotations

from typing import Optional

import torch

from core.avatar import cascade as cad
from core.gaussian_utils import matrix_to_quaternion


def apply(sigma: float = 0.65,
          max_clamp_m: Optional[float] = None,
          min_clamp_m: float = 0.0,
          zero_mlp_residual: bool = False,
          iso_mode: str = 'none',
          head_sigma_mult: float = 1.0,
          hand_sigma_mult: float = 1.0,
          scale_z_gof: float = 1.0e-7) -> None:
    """Install mesh2splat scale/rotation overrides.

    Args extra:
        head_sigma_mult: per-face σ multiplier for head region (anisotropic mode
            only). 1.0 = no change. 0.5 = halve init scale on face faces so the
            optimizer starts with sharper face gaussians.
        hand_sigma_mult: per-face σ multiplier for both hand regions (lhand +
            rhand) — same semantics as `head_sigma_mult`. Useful for sharper
            finger gaussians at initialization.

    Region indices come from `self.avatar_indices[curr_level].{head, lhand, rhand}`.
    """
    # (1) capture raw scale_out
    orig_decoder = cad.Cascade_Avatar.forward_gs_decoder

    def wrapped_decoder(self, feat):
        mean_offset, scale_out = orig_decoder(self, feat)
        if zero_mlp_residual:
            self._last_scale_out = torch.zeros_like(scale_out)
        else:
            self._last_scale_out = scale_out
        return mean_offset, scale_out

    cad.Cascade_Avatar.forward_gs_decoder = wrapped_decoder

    # (2) override scaling + rotation in _generate_gs_at_level
    orig_gen = cad.Cascade_Avatar._generate_gs_at_level

    def wrapped_gen(self, posed_vts, canon_vts, faces, lbs_motion, features,
                    pose_emb, w2c_rot, not_pooling, curr_level=-1):
        gs_params = orig_gen(self, posed_vts, canon_vts, faces, lbs_motion,
                              features, pose_emb, w2c_rot, not_pooling, curr_level)
        if (not getattr(self, 'use_gs_geo', False)) or getattr(self, 'bodypart_scale', True):
            return gs_params
        if not isinstance(gs_params, dict) or 'scaling' not in gs_params:
            return gs_params

        pvts = posed_vts if posed_vts.ndim == 3 else posed_vts.unsqueeze(0)
        v0 = pvts[:, faces[:, 0]]
        v1 = pvts[:, faces[:, 1]]
        v2 = pvts[:, faces[:, 2]]
        e0 = v1 - v0
        e1 = v2 - v0
        normal = torch.nn.functional.normalize(torch.cross(e0, e1, dim=-1), dim=-1)
        t_u = torch.nn.functional.normalize(e0, dim=-1)
        t_v = torch.nn.functional.normalize(torch.cross(normal, t_u, dim=-1), dim=-1)

        if iso_mode == 'full':
            len_e0 = torch.norm(e0, dim=-1)
            len_e1 = torch.norm(e1, dim=-1)
            len_e2 = torch.norm(v0 - v2, dim=-1)
            avg_len = (len_e0 + len_e1 + len_e2) / 3.0
            iso_scale = (avg_len * sigma).clamp_min(1e-6)
            scale_init = iso_scale.unsqueeze(-1).expand(-1, -1, 3)
        else:
            len_e0 = torch.norm(e0, dim=-1)
            e1_proj_v = torch.abs((e1 * t_v).sum(dim=-1))
            # Per-face σ — defaults to scalar `sigma`, but head faces get a
            # different multiplier if `head_sigma_mult` != 1.0.
            sigma_per_face = torch.full(
                (faces.shape[0],), float(sigma),
                device=faces.device, dtype=len_e0.dtype,
            )
            if curr_level >= 0 and hasattr(self, 'avatar_indices'):
                idx_store = self.avatar_indices[curr_level]
                if head_sigma_mult != 1.0:
                    sigma_per_face[idx_store.head] *= float(head_sigma_mult)
                if hand_sigma_mult != 1.0:
                    sigma_per_face[idx_store.lhand] *= float(hand_sigma_mult)
                    sigma_per_face[idx_store.rhand] *= float(hand_sigma_mult)
            scale_u = (len_e0 * sigma_per_face).clamp_min(1e-6)
            scale_v = (e1_proj_v * sigma_per_face).clamp_min(1e-6)
            # GoF ray-Gaussian intersection degenerates with paper-thin
            # z=1e-7. Use a larger value when GoF is active so the integral has
            # non-trivial extent along the normal direction.
            if getattr(self, 'gs_rasterizer', 'vanilla') == 'gof':
                scale_z_val = float(scale_z_gof)
            else:
                scale_z_val = 1e-7
            scale_z = torch.full_like(scale_u, scale_z_val)
            scale_init = torch.stack([scale_u, scale_v, scale_z], dim=-1)

        scale_out_raw = getattr(self, '_last_scale_out', None)
        if scale_out_raw is None:
            scale_residual = torch.zeros_like(scale_init)
        else:
            C_raw = scale_out_raw.shape[-1]
            if iso_mode == 'full':
                iso_residual = scale_out_raw.mean(dim=-1, keepdim=True)
                scale_residual = iso_residual.expand_as(scale_init)
            else:
                if C_raw >= 2:
                    res_u = scale_out_raw[..., 0:1]
                    res_v = scale_out_raw[..., 1:2]
                else:
                    res_u = scale_out_raw[..., 0:1]
                    res_v = scale_out_raw[..., 0:1]
                res_z = torch.zeros_like(res_u)
                scale_residual = torch.cat([res_u, res_v, res_z], dim=-1)

        scaling = scale_init * torch.exp(scale_residual)

        if max_clamp_m is not None and max_clamp_m > 0:
            if min_clamp_m and min_clamp_m > 0:
                scaling = torch.clamp(scaling, min=float(min_clamp_m),
                                       max=float(max_clamp_m))
            else:
                scaling = torch.clamp(scaling, max=float(max_clamp_m))

        if iso_mode == 'full':
            B = scale_init.shape[0]; Fn = scale_init.shape[1]
            quat = torch.zeros((B, Fn, 4), device=scale_init.device, dtype=scale_init.dtype)
            quat[..., 0] = 1.0
            quat = quat.unsqueeze(-2)
        else:
            # mesh2splat philosophy: rotation is determined by the mesh, not learned.
            # Detach is required to avoid grad-shape mismatch from the rasterizer.
            R = torch.stack([t_u, t_v, normal], dim=-1)
            quat = matrix_to_quaternion(R).detach().unsqueeze(-2)

        if not hasattr(self, '_m2s_dbg_count'):
            self._m2s_dbg_count = 0
        self._m2s_dbg_count += 1
        if self._m2s_dbg_count <= 3 or self._m2s_dbg_count % 100 == 0:
            sz = scaling[..., 2]
            print(f'[M2S_DBG #{self._m2s_dbg_count}] iso={iso_mode} '
                  f'scale_z: min={sz.min().item():.2e} '
                  f'median={sz.median().item():.2e} '
                  f'max={sz.max().item():.2e}  '
                  '(expected 1e-7 for anisotropic)')

        gs_params['scaling'] = scaling
        gs_params['rotation'] = quat
        gs_params['scale_offset'] = scale_residual.pow(2)
        return gs_params

    cad.Cascade_Avatar._generate_gs_at_level = wrapped_gen

    if max_clamp_m is not None and max_clamp_m > 0:
        clamp_str = f'clamp [{(min_clamp_m or 0)*1000:.1f}, {max_clamp_m*1000:.1f}] mm'
    else:
        clamp_str = 'no clamp'
    if iso_mode == 'full':
        mode_str = 'FULL ISOTROPIC sphere (avg edge × σ, identity rotation)'
        res_str = 'MLP residual=mean(scale_out_raw)'
    else:
        mode_str = ('anisotropic ellipse (paper-thin z=1e-7 FIXED, '
                    'tangent rotation FIXED)')
        res_str = 'MLP residual: u=scale_out[0], v=scale_out[1], z=0 (fixed)'
    head_str = (f', head σ× {head_sigma_mult}' if head_sigma_mult != 1.0 else '')
    hand_str = (f', hand σ× {hand_sigma_mult}' if hand_sigma_mult != 1.0 else '')
    print(f'[MESH2SPLAT_INIT] σ={sigma}{head_str}{hand_str}, {mode_str}, {res_str}, {clamp_str}')

"""Training launcher with OPTIONAL hard-zero of GS mean_offset on head/hand.

Monkey-patches core.avatar.cascade.Cascade_Avatar.forward_gs_decoder so that,
when the env var GS_ZERO_HEAD_HAND=1 is set (or `--zero_head_hand` is passed),
the model's per-face mean_offset is force-zeroed for the head/lhand/rhand
semantic face indices at EVERY forward call.

This preserves core code (no cascade.py edit) — the patch is applied at import
time before any model instantiation. The env var makes it opt-in per run.

Usage:
    # Enabled:
    GS_ZERO_HEAD_HAND=1 python scripts/main_hard_zero_headhand.py --config ...
    # Or via flag:
    python scripts/main_hard_zero_headhand.py --zero_head_hand --config ...
    # Disabled (equivalent to plain main.py):
    python scripts/main_hard_zero_headhand.py --config ...
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch  # needed by outer wrap closures below

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))


def _apply_patch():
    """Install the head/hand mean_offset zero patch on Cascade_Avatar.

    Wraps `_generate_gs_at_level` and, when the env var is set, subtracts the
    added mean_offset from the head/lhand/rhand faces of the returned `xyz`
    (posed face centers) and zeros the returned `mean_offset` for those faces.
    This correctly handles pooling (result tensors are post-pooling shape) and
    per-level curr_level (passed as arg to the wrapped call)."""
    from core.avatar.cascade import Cascade_Avatar

    _orig = Cascade_Avatar._generate_gs_at_level
    _step_counters = {}  # per curr_level counter

    def _wrapped(self, posed_vts, canon_vts, faces, lbs_motion, features,
                 pose_emb, w2c_rot, not_pooling, curr_level=-1):
        gs_params = _orig(self, posed_vts, canon_vts, faces, lbs_motion,
                          features, pose_emb, w2c_rot, not_pooling, curr_level)
        _zero_hh = os.environ.get('GS_ZERO_HEAD_HAND', '0') == '1'
        _body_delay = int(os.environ.get('GS_BODY_DELAY_ITERS', '0'))
        if (not _zero_hh) and _body_delay <= 0:
            return gs_params
        if not getattr(self, 'use_gs_geo', False):
            return gs_params
        try:
            curr_idxs = self.avatar_indices[curr_level]
            mean_offset = gs_params.get('mean_offset')
            xyz = gs_params.get('xyz')
            if mean_offset is None or xyz is None:
                return gs_params
            xyz_new = xyz.clone()
            mean_offset_new = mean_offset.clone()

            # Hard-zero head/hand (existing behavior)
            if _zero_hh:
                for region in ('head', 'lhand', 'rhand'):
                    r_idx = getattr(curr_idxs, region, None)
                    if r_idx is None or (hasattr(r_idx, 'numel') and r_idx.numel() == 0):
                        continue
                    xyz_new[:, r_idx] = xyz_new[:, r_idx] - mean_offset_new[:, r_idx]
                    mean_offset_new[:, r_idx] = 0.0

            # Body delay: zero BODY (all except head/hand) for first N training iters
            if _body_delay > 0:
                if getattr(self, 'training', False):
                    step = _step_counters.get(curr_level, 0) + 1
                    _step_counters[curr_level] = step
                else:
                    step = _step_counters.get(curr_level, 0)
                if step <= _body_delay:
                    F = mean_offset_new.shape[1]
                    body_mask = torch.ones(F, dtype=torch.bool, device=mean_offset_new.device)
                    for region in ('head', 'lhand', 'rhand'):
                        r_idx = getattr(curr_idxs, region, None)
                        if r_idx is not None and (not hasattr(r_idx, 'numel') or r_idx.numel() > 0):
                            body_mask[r_idx] = False
                    xyz_new[:, body_mask] = xyz_new[:, body_mask] - mean_offset_new[:, body_mask]
                    mean_offset_new[:, body_mask] = 0.0

            gs_params['xyz'] = xyz_new
            gs_params['mean_offset'] = mean_offset_new
        except Exception as e:
            print(f'[hard_zero_headhand] wrap fail (curr_level={curr_level}): {e}')
        return gs_params

    Cascade_Avatar._generate_gs_at_level = _wrapped
    print(f'[hard_zero_headhand] patch installed on Cascade_Avatar._generate_gs_at_level '
          f'(active={os.environ.get("GS_ZERO_HEAD_HAND", "0")})')

    # ----------------------------------------------------------------------
    # Optional: per-region sigma multiplier for iso=full mesh2splat_init.
    # iso=full uses uniform scaling and ignores head/hand_sigma_mult by design.
    # If MESH2GS_HEAD_SIGMA_MULT or MESH2GS_HAND_SIGMA_MULT != 1.0, wrap the
    # already-installed _generate_gs_at_level (which will include mesh2splat_init's
    # wrap installed later) to POST-SCALE the head/hand gaussians.
    # ----------------------------------------------------------------------
    _hmult = float(os.environ.get('MESH2GS_HEAD_SIGMA_MULT', '1.0'))
    _amult = float(os.environ.get('MESH2GS_HAND_SIGMA_MULT', '1.0'))
    _bmult = float(os.environ.get('MESH2GS_BODY_SIGMA_MULT', '1.0'))  # body = default (all except head/hand)
    if _hmult != 1.0 or _amult != 1.0 or _bmult != 1.0:
        # Wrap mesh2splat_init.apply — called ONCE inside setup_mesh_stages.
        # After it installs its own wrap, we install our outer wrap on top so
        # scaling can be modified per-region even in iso=full mode.
        import src.features.mesh2splat_init as _ms_init
        _orig_apply = _ms_init.apply

        def _patched_apply(*args, **kwargs):
            _orig_apply(*args, **kwargs)
            from core.avatar.cascade import Cascade_Avatar as _CA
            _outer_orig = _CA._generate_gs_at_level

            def _outer_wrap(self, posed_vts, canon_vts, faces, lbs_motion,
                            features, pose_emb, w2c_rot, not_pooling, curr_level=-1):
                gs_params = _outer_orig(self, posed_vts, canon_vts, faces, lbs_motion,
                                        features, pose_emb, w2c_rot, not_pooling, curr_level)
                if not getattr(self, 'use_gs_geo', False):
                    return gs_params
                if not isinstance(gs_params, dict) or 'scaling' not in gs_params:
                    return gs_params
                try:
                    idxs = self.avatar_indices[curr_level]
                    scaling = gs_params['scaling']
                    scaling_new = scaling.clone()
                    # Compute per-face avg edge length (matches mesh2splat_init iso=full path).
                    # This becomes the base scale for head/hand — MLP-independent, prevents
                    # optimizer from compensating via scale_residual.
                    _sigma = float(os.environ.get('MESH2GS_BASE_SIGMA', '0.8'))
                    pvts = posed_vts if posed_vts.ndim == 3 else posed_vts.unsqueeze(0)
                    _v0 = pvts[:, faces[:, 0]]
                    _v1 = pvts[:, faces[:, 1]]
                    _v2 = pvts[:, faces[:, 2]]
                    _avg_len = (torch.norm(_v1 - _v0, dim=-1)
                                + torch.norm(_v2 - _v0, dim=-1)
                                + torch.norm(_v0 - _v2, dim=-1)) / 3.0
                    _iso_base = (_avg_len * _sigma).clamp_min(1e-6)  # (B, F)
                    # Body first: fix all faces to body_mult, then head/hand override.
                    if _bmult != 1.0:
                        _fixed = (_iso_base * float(_bmult)).detach()
                        _fixed3 = _fixed.unsqueeze(-1).expand(-1, -1, 3)
                        scaling_new = _fixed3.clone()  # overwrite entire scaling
                    for region, _mult in (('head', _hmult), ('lhand', _amult), ('rhand', _amult)):
                        if _mult == 1.0:
                            continue
                        r_idx = getattr(idxs, region, None)
                        if r_idx is None or (hasattr(r_idx, 'numel') and r_idx.numel() == 0):
                            continue
                        # FIXED per-face scale: mesh geometry × sigma × mult, DETACHED so
                        # gradient does not flow back through this replacement to MLP.
                        _fixed = (_iso_base[:, r_idx] * float(_mult)).detach()
                        _fixed3 = _fixed.unsqueeze(-1).expand(-1, -1, 3)
                        scaling_new[:, r_idx] = _fixed3
                    gs_params['scaling'] = scaling_new
                except Exception as e:
                    print(f'[sigmaHH_mult] wrap fail (level={curr_level}): {e}')
                return gs_params

            _CA._generate_gs_at_level = _outer_wrap
            print(f'[sigmaHH_mult] outer wrap installed after mesh2splat_init.apply '
                  f'(head_mult={_hmult}, hand_mult={_amult})')

        # Patch both the module attribute AND src.features.__init__ (which imports it)
        _ms_init.apply = _patched_apply
        try:
            import src.features as _sf
            if hasattr(_sf, 'mesh2splat_init'):
                _sf.mesh2splat_init.apply = _patched_apply
        except Exception:
            pass
        print(f'[sigmaHH_mult] wrapping mesh2splat_init.apply (head_mult={_hmult}, hand_mult={_amult})')


def main() -> int:
    # Argparse just to allow --zero_head_hand flag; remaining args passed to real main.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--zero_head_hand', action='store_true',
                    help='force enable hard-zero of head/hand GS mean_offset (equivalent to env GS_ZERO_HEAD_HAND=1)')
    args, remaining = ap.parse_known_args()
    if args.zero_head_hand:
        os.environ['GS_ZERO_HEAD_HAND'] = '1'
    _apply_patch()

    # Replace sys.argv so downstream main() sees only remaining args
    sys.argv = [sys.argv[0]] + remaining

    from main import main as real_main
    return real_main()


if __name__ == '__main__':
    sys.exit(main())

"""Render canonical-pose sequence: canonical mesh from a fixed frontal camera, per-frame
Jacobian residual applied to Da_vts. Mirrors the OLD `novel_posed_front_canon` behavior
(see avatar_optimizer.py:novel_posed_front_canon) with RAW pose (no zeroing).

Two sequences per call: canon_train_v2 and canon_novel_v2 (frame counts differ per scene).

VIZ CKPT RULE: run twice, keep Mesh+Normal from All ckpt, keep GS from GS#1 ckpt.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.rendering._common import (
    apply_monkey_patch_if_needed,
    apply_pkl_patch,
    extract_frame_params,
    setup_avatar_from_yaml,
)

apply_monkey_patch_if_needed()

import torch

from scripts.save_all_outputs_v2 import Save_all_outputs_v2


def render_canon_seq(config: str, ckpt_path: str, out_dir: str) -> int:
    """Renders canon_train_v2/ + canon_novel_v2/ using debug_da_camera[0] (front, 6m).

    Uses raw smpl params (no global_orient/transl zeroing) to preserve pose-driven
    Jacobian input — matches OLD `novel_posed_front_canon`.
    """
    state, avatar = setup_avatar_from_yaml(config, ckpt_path)
    apply_pkl_patch(state, ckpt_path)

    train_ds = state.train_dataset
    zju = state.extras['ZJU_Novel_pose_dataset']
    gpo = state.extras['gpo_optimizer']
    smpl_mesh = state.smpl_mesh

    front_cam = state.extras['debug_cams'][0]

    def _sfd_to_list(sfd):
        B = None
        for v in sfd.values():
            if hasattr(v, 'dim') and v.dim() > 0:
                B = v.shape[0]; break
        if B is None:
            return []
        out = []
        for i in range(B):
            p = extract_frame_params(sfd, i, smpl_mesh)
            if p is not None:
                out.append(p)
        return out

    def _render(sfd, subname):
        params = _sfd_to_list(sfd)
        if not params:
            print(f'[render_canon_seq] {subname}: empty'); return
        sub_out = Path(out_dir) / subname
        sub_out.mkdir(parents=True, exist_ok=True)
        print(f'[render_canon_seq] → {subname} ({len(params)} frames) → {sub_out}')
        with torch.no_grad():
            Save_all_outputs_v2(
                avatar, gpo.mesh_renderer, str(sub_out),
                cameras=[front_cam] * len(params), smpl_params=params,
                canon=True, bg=[1.0, 1.0, 1.0])

    _render(train_ds.smpl_full_dict, 'canon_train_v2')
    _render(zju.smpl_full_dict, 'canon_novel_v2')
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt_path', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    return render_canon_seq(args.config, args.ckpt_path, args.out_dir)


if __name__ == '__main__':
    sys.exit(_main())

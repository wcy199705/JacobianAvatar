"""Shared helpers for rendering scripts (render_orbit, render_canon_seq, render_per_frame_cropped).

Isolates the "load model + apply pkl patch + build per-frame smpl params list" boilerplate
so each render script becomes a thin composition of these functions.

Dataset: CUSTOM.
"""
from __future__ import annotations
import glob
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Final_Jacobian_code/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))


def apply_monkey_patch_if_needed() -> None:
    """Apply main_hard_zero_headhand._apply_patch() if env vars indicate its use.

    Mirrors the head/hand hard-zero + sigma modulation used at training time. Loading a
    ckpt trained with these patches without re-applying them at eval time changes model
    behavior — see project_monkeypatch_eval_inconsistency memory.
    """
    if (os.environ.get('GS_ZERO_HEAD_HAND') == '1'
        or float(os.environ.get('MESH2GS_HEAD_SIGMA_MULT', '1.0')) != 1.0
        or float(os.environ.get('MESH2GS_HAND_SIGMA_MULT', '1.0')) != 1.0
        or float(os.environ.get('MESH2GS_BODY_SIGMA_MULT', '1.0')) != 1.0
        or int(os.environ.get('GS_BODY_DELAY_ITERS', '0')) > 0):
        # scripts.main_hard_zero_headhand exposes `_apply_patch()` that monkey-patches
        # `core.avatar.cascade.Cascade_Avatar.forward_gs_decoder`.
        from scripts.main_hard_zero_headhand import _apply_patch  # noqa: WPS433
        _apply_patch()


def setup_avatar_from_yaml(config_path: str, ckpt_path: str):
    """Load config → build trainer state → run Pre stage → load ckpt into Avatar.

    Returns:
        (state, avatar) tuple. Also configures avatar for GS-stage inference:
            avatar._gs_stage_active = True
            avatar.set_use_gs_geo(True)
            avatar.curr_level.fill_(avatar.max_level - 1)
    """
    import torch  # local; keep top-level clean
    from src.config import load_config
    from src.optim.setup import setup_trainer
    from src.optim.mesh_setup import setup_mesh_stages
    from src.stages.pre import run_pre_stage
    from src.stages.pre_loose_smpl import run_pre_stage_loose_smpl

    cfg = load_config(config_path)
    state = setup_trainer(cfg)
    if bool(cfg.stages.get('pre', {}).get('loose_smpl', False)):
        run_pre_stage_loose_smpl(cfg, state)
    else:
        run_pre_stage(cfg, state)
    setup_mesh_stages(cfg, state)
    avatar = state.avatar
    avatar.load_model(ckpt_path)
    avatar._gs_stage_active = True
    avatar.set_use_gs_geo(True)
    avatar.curr_level.fill_(avatar.max_level - 1)
    return state, avatar


def apply_pkl_patch(state, ckpt_path: str) -> None:
    """Patch novel/train sfd with fitted SMPL params from `{ckpt}_smplx_param.pkl`.

    Chooses which sfd (train vs novel) based on pkl's full_pose batch size:
      - matches novel dataset frame count → patch ZJU_Novel_pose_dataset.smpl_full_dict
      - matches train dataset frame count → patch train_dataset.smpl_full_dict
      - else: no-op
    Silent no-op if pkl doesn't exist.
    """
    import torch
    pkl = ckpt_path + '_smplx_param.pkl'
    if not os.path.exists(pkl):
        return
    with open(pkl, 'rb') as f:
        fitted = pickle.load(f)
    fp = fitted.get('full_pose')
    if fp is None or not hasattr(fp, 'shape'):
        return
    train_ds = state.train_dataset
    zju = state.extras['ZJU_Novel_pose_dataset']
    n_test = zju.num_frames
    try:
        n_train = train_ds.smpl_full_dict['full_pose'].shape[0]
    except Exception:
        n_train = None
    target = None
    label = None
    if fp.shape[0] == n_test:
        target = zju.smpl_full_dict; label = 'novel'
    elif n_train is not None and fp.shape[0] == n_train:
        target = train_ds.smpl_full_dict; label = 'train'
    if target is None:
        return
    for k in list(target.keys()):
        if k not in fitted:
            continue
        v = fitted[k]
        v = v.detach().clone() if hasattr(v, 'detach') else torch.as_tensor(v)
        target[k] = v.to(target[k].device if hasattr(target[k], 'device') else 'cpu')
    print(f'[render/_common] pkl matched {label} ({fp.shape[0]}) → sfd patched')


def extract_frame_params(sfd: Dict, i: int, smpl_mesh) -> Optional[Dict]:
    """Extract i-th frame's smpl params dict from a smpl_full_dict.

    Returns dict with all per-frame tensors sliced to [i:i+1] and non-batched
    fields shallow-cloned. If `full_pose` is absent, derives it via
    `smpl_mesh.smplx2full_pose_dict(...)`.
    """
    B = None
    for v in sfd.values():
        if hasattr(v, 'dim') and v.dim() > 0:
            B = v.shape[0]; break
    if B is None or i >= B:
        return None
    p = {}
    for k, v in sfd.items():
        if hasattr(v, 'dim') and v.dim() > 0 and v.shape[0] == B:
            p[k] = v[i:i+1].detach().clone()
        elif hasattr(v, 'dim') and v.dim() > 0:
            p[k] = v.detach().clone()
        else:
            p[k] = v
    if 'full_pose' not in p and 'body_pose' in p:
        fpd = smpl_mesh.smplx2full_pose_dict(p)
        for k2 in ('full_pose', 'transl', 'betas', 'expression'):
            if k2 in fpd:
                p[k2] = fpd[k2]
    return p


def load_bboxes(bbox_dir: str) -> Dict[str, List[float]]:
    """Read DWPose bbox txt files → dict[stem]=[x, y, w, h]."""
    d = {}
    for f in glob.glob(os.path.join(bbox_dir, '*.txt')):
        with open(f) as fp:
            v = [float(x.strip()) for x in fp.read().split(',') if x.strip()]
        if len(v) == 4:
            d[os.path.basename(f).rsplit('.', 1)[0]] = v
    return d


def scene_max_wh(bbox_dirs: List[str]) -> Tuple[int, int]:
    """Return (max_W, max_H) across all bbox files in given dirs (from DWPose)."""
    import numpy as np
    bs = []
    for d in bbox_dirs:
        for f in glob.glob(os.path.join(d, '*.txt')):
            with open(f) as fp:
                v = [float(x.strip()) for x in fp.read().split(',') if x.strip()]
            if len(v) == 4 and v[2] > 0 and v[3] > 0:
                bs.append([v[2], v[3]])
    if not bs:
        return 0, 0
    a = np.array(bs)
    return int(math.ceil(a[:, 0].max())), int(math.ceil(a[:, 1].max()))

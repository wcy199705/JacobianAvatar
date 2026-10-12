"""Mesh-stage shared setup — runs after Pre, before L0/L1/All.

Public API: `setup_mesh_stages(cfg, state)` — mutates `state` in place.

Internal structure:
  _build_debug_cameras()          — front + back debug cameras from cfg.debug_cam
  _build_avatar()                 — Cascade_Avatar + SDF grad volume + gs_rasterizer config
  _build_avatar_optimizer()       — optimizer + scheduler from Avatar.get_all_param_groups
  _load_flow_features()           — WAFT flow features (all training frames)

Builds (via `state.*` and `state.extras[*]`):
  state.avatar                      Cascade_Avatar
  state.avatar_optimizer/_scheduler
  state.extras['SDF_grad_volume']
  state.extras['flow_features']
  state.extras['debug_cams']        [front_camera, back_camera]
  state.extras['mesh_loss_conf']
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple

import torch

from core.avatar.cascade import Cascade_Avatar
from core.geometry.sdf_volume import GridSDFGrad
from core.optim.opt_utils import set_adam_optimizer
from core.camera import UnifiedCamera

from src.config import TrainingConfig
from src.optim.trainer_state import TrainerState
from src.optim.loss_conf import build_loss_conf
from src.features import apply_all as apply_v5_features




def _build_debug_cameras(cfg: TrainingConfig, train_dataset, device) -> List:
    """Build [front_camera, back_camera] from cfg.debug_cam (optional overrides).

    Front: R_front = diag(1, -1, -1), T=[0, 0, -6] → matches OLD `novel_posed_front_canon`.
    Back:  R_back  = diag(1, -1,  1) (only Y flipped), T=[0, 0, -6].
    K defaults to `train_dataset.cam_int` with cx/cy → W/2, H/2. Cfg overrides:
      debug_cam: width_scale, height_scale, width, height, fx, fy, cx, cy
    """
    _dbg_cfg = getattr(cfg, 'debug_cam', {}) or {}
    if not isinstance(_dbg_cfg, dict):
        _dbg_cfg = {}
    _orig_W = int(train_dataset.original_width)
    _orig_H = int(train_dataset.original_height)
    _w_scale = float(_dbg_cfg.get('width_scale', 1.0))
    _h_scale = float(_dbg_cfg.get('height_scale', 1.0))
    _dbg_W = int(_dbg_cfg.get('width') or round(_orig_W * _w_scale))
    _dbg_H = int(_dbg_cfg.get('height') or round(_orig_H * _h_scale))
    _K = torch.Tensor(train_dataset.cam_int).clone()
    if 'fx' in _dbg_cfg: _K[0, 0] = float(_dbg_cfg['fx'])
    if 'fy' in _dbg_cfg: _K[1, 1] = float(_dbg_cfg['fy'])
    _K[0, 2] = float(_dbg_cfg.get('cx', _dbg_W / 2.0))
    _K[1, 2] = float(_dbg_cfg.get('cy', _dbg_H / 2.0))
    print(f'[debug_cam] {_dbg_W}×{_dbg_H} fx={_K[0,0]:.1f} fy={_K[1,1]:.1f} '
          f'cx={_K[0,2]:.1f} cy={_K[1,2]:.1f}', flush=True)

    R_front = torch.eye(3); R_front[1:] = -R_front[1:]
    front_camera = UnifiedCamera.perspective_camera_from_K(
        K=_K, width=_dbg_W, height=_dbg_H, znear=0.01, zfar=1000.0,
        R=R_front, T=torch.Tensor([0, 0, -6]), device=device)
    R_back = torch.eye(3); R_back[1, :] = -R_back[1, :]
    back_camera = UnifiedCamera.perspective_camera_from_K(
        K=_K, width=_dbg_W, height=_dbg_H, znear=0.01, zfar=1000.0,
        R=R_back, T=torch.Tensor([0, 0, -6]), device=device)
    return [front_camera, back_camera]


def _build_avatar(cfg: TrainingConfig, state: TrainerState, device) -> Tuple[Cascade_Avatar, GridSDFGrad]:
    """Construct Cascade_Avatar + SDF_grad_volume; wire GoF rasterizer config if set."""
    smpl_mesh = state.smpl_mesh
    model_params = state.extras['model_params']

    smpl_mesh.rgb = smpl_mesh.rgb.squeeze()
    Avatar = Cascade_Avatar(
        config=model_params['Cascade_Avatar'],
        first_avatar=smpl_mesh, disp=smpl_mesh.disp, device=device)
    SDF_grad_volume = GridSDFGrad(
        verts=Avatar.jacobian_systems[-1].init_da_verts.squeeze(),
        faces=Avatar.jacobian_systems[-1].init_face,
        resolution=256, padding=0.2)

    # Optional GoF rasterizer (yaml `gs_rasterizer: gof`). cascade.render dispatches on
    # `self.gs_rasterizer`. `gof_legacy_wrapper` DEFAULTS to True. `gof_kernel_size` overrides mip-splatting AA
    # inflation (legacy default 0.0, current 0.1).
    Avatar.gs_rasterizer = cfg.model.get('gs_rasterizer', 'vanilla')
    Avatar.gof_legacy_wrapper = bool(cfg.model.get('gof_legacy_wrapper', True))
    _ms = cfg.mesh2splat
    _ks = _ms.get('gof_kernel_size', None)
    if _ks is not None:
        Avatar.gof_kernel_size = float(_ks)
    return Avatar, SDF_grad_volume


def _build_avatar_optimizer(Avatar: Cascade_Avatar) -> Tuple[Any, Any]:
    """Build optimizer + scheduler from Avatar param groups; also freezes grads + sets L0."""
    avatar_group_params = []
    for param_group in Avatar.get_all_param_groups():
        avatar_group_params.append({
            "params": param_group['params'],
            "lr": float(param_group['lr']),
        })
    avatar_optimizer, avatar_scheduler = set_adam_optimizer(
        avatar_group_params, lr=1e-3, decay=0.95)
    Avatar.set_all_grad_false()
    Avatar.set_curr_level(0)
    return avatar_optimizer, avatar_scheduler


def _load_flow_features(state: TrainerState, save_root: str):
    """Load / compute WAFT flow features for all training frames (cached on disk)."""
    with torch.no_grad():
        return state.flow_loader.get_features(
            save_root=save_root,
            images=state.train_dataset.images,
            filename='all_flow_feat.pt')


def setup_mesh_stages(cfg: TrainingConfig, state: TrainerState) -> None:
    """Build Avatar, optimizer, flow features, loss_conf, debug cameras.

    Thin composition of the 5 helper functions above.
    """
    device = state.extras['device']

    # 1. Install v5 feature patches on legacy Cascade_Avatar / AvatarOptimizer.
    #    Must run before iso_remesh + Cascade_Avatar construction (patches take effect on next
    #    instantiation).
    apply_v5_features(state.extras['model_params'],
                      mesh2splat_cfg=cfg.mesh2splat,
                      scale_clamp_cfg=cfg.scale_clamp,
                      training_cfg=cfg.training)

    # 2. Optional iso-remesh post-process on smpl_mesh (cached in preprocessing/).

    # 3. Debug cameras (front + back) — cfg-driven from `debug_cam` section.
    state.extras['debug_cams'] = _build_debug_cameras(cfg, state.train_dataset, device)

    # 4. Avatar + SDF volume.
    Avatar, SDF_grad_volume = _build_avatar(cfg, state, device)
    state.avatar = Avatar
    state.extras['SDF_grad_volume'] = SDF_grad_volume

    # 5. Optimizer + scheduler.
    state.avatar_optimizer, state.avatar_scheduler = _build_avatar_optimizer(Avatar)

    # 6. Flow features (WAFT, all frames, cached).
    state.extras['flow_features'] = _load_flow_features(state, state.save_root)

    # 7. Mesh-stage loss config.
    state.extras['mesh_loss_conf'] = build_loss_conf(cfg.losses.mesh)

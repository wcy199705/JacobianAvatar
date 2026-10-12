"""Trainer setup — builds TrainerState fields needed by every stage.

Public API: `setup_trainer(cfg)`.

Internal structure:
  _load_legacy_params()   — general/model/exp yamls + pose_noise/body_lr overrides
  _resolve_save_root()    — apply save_root_override
  _build_renderer_opt()   — UnifiedRenderer + GPOOptimizer
  _build_datasets()       — train (Custom) / ZJU / Exavatar novel_pose
  _build_avatar_mesh()    — AvatarMesh with MHR conversion + poisson remesh
  _prepare_save_folders() — mlp_log / pre_log / curr_exp folders
  _initial_sampling()     — full-dataset sampling indices + paired smpl batches

Supported dataset: Custom.
"""
from __future__ import annotations
import copy
import os
from typing import Any, Dict, Tuple

import torch
import yaml

from core.body_model.smplx import AvatarMesh
from core.avatar.cascade import Cascade_Avatar  # noqa: F401  (kept for legacy import chains)
from core.optim.avatar_optimizer import AvatarOptimizer as GPOOptimizer
from core.render.render_unified import UnifiedRenderer
from core.data.scene import SceneDataset
from core.data.novel_pose import Novelpose
from core.data.optical_flow import OpticalFlowLoader
from core.geometry.sampling_utils import get_sampled_indices_from_length
from core.data.utils import get_sampled_data
from core.geometry.avatar_utils import get_paired_smpl_batches
from core.geometry.mesh_utils import get_normalization_params
from core.io.smpl2mhr import convert2MHR

from src.config import TrainingConfig
from src.optim.trainer_state import TrainerState


# Per-dataset legacy yaml naming. Add new datasets here.
_GENERAL_YAML = {
    'Custom':      ('CUSTOM',      '{scene}.yaml'),  # same scene yaml as preprocessing
}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _apply_pose_noise_scale(exp_params: dict, nsf: float) -> None:
    """Multiply all top-level *_noise_scale / *_min_noise entries by nsf.

    Mirrors legacy `install_yaml_hook_full` behavior. In-place mutation.
    """
    if abs(nsf - 1.0) <= 1e-9:
        return
    for key in list(exp_params.keys()):
        kl = key.lower()
        if ('noise_scale' in kl) or ('min_noise' in kl):
            v = exp_params[key]
            if isinstance(v, list):
                exp_params[key] = [float(x) * nsf for x in v]
                print(f'[POSE_NOISE] {key}: {v} -> {exp_params[key]}')
            elif isinstance(v, (int, float)):
                exp_params[key] = float(v) * nsf
                print(f'[POSE_NOISE] {key}: {v} -> {exp_params[key]}')


def _apply_body_triplane_lr(model_params: dict, body_lr: float) -> None:
    """Override Cascade_Avatar.encoder.*.lr and gs_encoder.lr with body_lr."""
    ca = model_params.get('Cascade_Avatar', {})
    enc = ca.get('encoder', {})
    if isinstance(enc, dict):
        for lvl_key, lvl_cfg in enc.items():
            if isinstance(lvl_cfg, dict) and 'lr' in lvl_cfg:
                old = lvl_cfg['lr']
                lvl_cfg['lr'] = float(body_lr)
                print(f'[BODY_LR] encoder.{lvl_key}.lr {old} -> {body_lr}')
    gs_enc = ca.get('gs_encoder', {})
    if isinstance(gs_enc, dict) and 'lr' in gs_enc:
        old = gs_enc['lr']
        gs_enc['lr'] = float(body_lr)
        print(f'[BODY_LR] gs_encoder.lr {old} -> {body_lr}')


def _load_legacy_params(cfg: TrainingConfig) -> Tuple[dict, dict, dict, str]:
    """Load general/model/exp yamls from configs/legacy/ + apply cfg overrides.

    Returns (general_params, model_params, exp_params, curr_exp_name).
    """
    # legacy_model_name: model yaml under configs/legacy/models/
    # (`_bigplane` = full triplane resolutions, L0: 64, L1/GS: 128).
    legacy_model_name = cfg.model.get('legacy_name', 'world_normal_more_dump_bigplane')
    # legacy_exp_name: exp yaml under configs/legacy/exp/.
    legacy_exp_name = cfg.model.get('legacy_exp_name', 'single_of5')
    curr_exp_name = (
        'AFTER_ECCV_' + legacy_model_name + '_' + legacy_exp_name +
        '_w_mesh_strong_offset_reg_wo_head_lpips_modified_laps'
        '_w_more_color_laps_v7features'
    )

    legacy_cfg = os.path.join('configs', 'legacy')
    if cfg.dataset_name not in _GENERAL_YAML:
        raise ValueError(
            f"dataset_name={cfg.dataset_name!r} not registered. Known: {sorted(_GENERAL_YAML)}"
        )
    _subdir, _fname_tmpl = _GENERAL_YAML[cfg.dataset_name]
    _fname = _fname_tmpl.format(scene=cfg.scene_name)

    general_params = _load_yaml(
        os.path.join(legacy_cfg, 'SMPLX_dataloader', _subdir, _fname))
    model_params = _load_yaml(
        os.path.join(legacy_cfg, 'models', legacy_model_name + '.yaml'))
    exp_params = _load_yaml(
        os.path.join(legacy_cfg, 'exp', legacy_exp_name + '.yaml'))

    _apply_pose_noise_scale(exp_params, cfg.training.pose_noise_scale)
    _apply_body_triplane_lr(model_params, cfg.training.body_triplane_lr)
    return general_params, model_params, exp_params, curr_exp_name


def _resolve_save_root(cfg: TrainingConfig, general_params: dict) -> str:
    """Return save_root: cfg.save_root_override if set, else legacy Path.save_root."""
    if cfg.save_root_override:
        save_root = cfg.save_root_override
        os.makedirs(save_root, exist_ok=True)
        print(f'[setup] save_root_override → {save_root}')
        return save_root
    return general_params['Path']['save_root']


def _build_renderer_opt(device) -> Tuple[UnifiedRenderer, GPOOptimizer]:
    """UnifiedRenderer + GPOOptimizer (no gs_renderer at construction time)."""
    rndr = UnifiedRenderer(device=device)
    gpo_optimizer = GPOOptimizer(mesh_renderer=rndr.mesh_render, gs_renderer=None)
    return rndr, gpo_optimizer


def _build_datasets(cfg: TrainingConfig, general_params: dict, device) -> Tuple[Any, Any, Any]:
    """Construct (train_dataset, ZJU_Novel_pose_dataset, Exavatar_Novel_pose_dataset).

    Train set: core.data.custom.Custom (SceneDataset loader + valid_frames_file filter).
    """
    path_params = general_params['Path']
    smpl_config = general_params['SMPL']

    from core.data.custom import Custom
    _valid_frames_file = cfg.valid_frames_file or None
    if _valid_frames_file:
        print(f"[setup] valid_frames_file → {_valid_frames_file}", flush=True)
    train_dataset = Custom(
        root_folder=path_params['training_folder'],
        smpl_config=smpl_config,
        device=device, use_head_data=True,
        valid_frames_file=_valid_frames_file,
        anchor_frame_file=(cfg.anchor_frame_file or None),
    )
    novel_dataset = SceneDataset(
        root_folder=path_params['novel_pose_folder'],
        smpl_config=smpl_config,
        device=device,
    )
    # Optional driving sequence (ExAvatar format: cam_params/*.json + smplx_params_smoothed/*.json),
    # used only for the intermediate `extra_pose` renders. Empty → skipped.
    exavatar_dataset = None
    if path_params.get('exavatar_pose_folder'):
        exavatar_dataset = Novelpose(
            root_folder=path_params['exavatar_pose_folder'],
            smpl_config=smpl_config,
            device=device,
            sampling_rate=100,
        )
    return train_dataset, novel_dataset, exavatar_dataset


def _build_avatar_mesh(train_dataset, general_params: dict, device) -> AvatarMesh:
    """AvatarMesh init: init_smpl_mesh → da_pose conversion via MHR → poisson remesh."""
    smpl_mesh = AvatarMesh(smpl_config=general_params['SMPL'])
    target_smpl_param = train_dataset.smpl_list[0]
    target_vts = train_dataset.smpl_params['vertices'][0]
    _scale, _t = get_normalization_params(target_vts.detach().cpu().numpy())  # noqa: F841 (kept for parity)

    hunyuan_init = copy.deepcopy(target_smpl_param)
    smpl_mesh.init_smpl_mesh(hunyuan_init)
    da_pose = smpl_mesh.get_dapose_param(train_dataset.smpl_params['betas'])
    lod3_vts, lod3_faces = convert2MHR(general_params['SMPL'], da_pose, lod=3)
    smpl_mesh.set_da_canon(verts=lod3_vts * 0.01, faces=lod3_faces)
    smpl_mesh.poisson_remesh(8)
    smpl_mesh.rgb = torch.ones_like(smpl_mesh.v_canon.squeeze()).to(device) * 0.3
    return smpl_mesh


def _prepare_save_folders(save_root: str, curr_exp_name: str) -> Tuple[str, str, str]:
    """Create + return (mlp_log_folder, pre_log_folder, curr_exp_save_folder)."""
    mlp_log_folder = os.path.join(save_root, '3DGS_SMPLX_Version_poisson')
    pre_log_folder = os.path.join(mlp_log_folder, 'preprocessing')
    os.makedirs(pre_log_folder, exist_ok=True)
    curr_exp_save_folder = os.path.join(mlp_log_folder, curr_exp_name)
    os.makedirs(curr_exp_save_folder, exist_ok=True)
    return mlp_log_folder, pre_log_folder, curr_exp_save_folder


def _initial_sampling(train_dataset, gpo_optimizer, smpl_mesh) -> Dict[str, Any]:
    """Full-dataset sampling indices + paired smpl batches (sampling_interval=1)."""
    total_sampling_idxs = get_sampled_indices_from_length(train_dataset.num_frames, 1)
    sampling_imgs, sampling_masks, sampling_cams, sampling_normals, sampling_keypoints = \
        get_sampled_data(train_dataset, total_sampling_idxs)
    sampling_smplbatch, _ = get_paired_smpl_batches(
        train_smpl_params=train_dataset.smpl_params,
        sampling_idxs=total_sampling_idxs,
        except_frames=None,
        gpo_optimizer=gpo_optimizer,
        smpl_mesh=smpl_mesh,
    )
    return {
        'sampling_imgs': sampling_imgs,
        'sampling_masks': sampling_masks,
        'sampling_cams': sampling_cams,
        'sampling_normals': sampling_normals,
        'sampling_keypoints': sampling_keypoints,
        'sampling_smplbatch': sampling_smplbatch,
        'total_sampling_idxs': total_sampling_idxs,
    }


def setup_trainer(cfg: TrainingConfig) -> TrainerState:
    """Build initial TrainerState from a v5 config.

    Thin composition of the 7 helper functions above.
    """
    state = TrainerState()
    device = torch.device('cuda')

    # 1. Legacy params + curr_exp_name
    general_params, model_params, exp_params, curr_exp_name = _load_legacy_params(cfg)
    state.curr_exp_name = curr_exp_name
    state.extras['general_params'] = general_params
    state.extras['model_params'] = model_params
    state.extras['exp_params'] = exp_params
    state.save_root = _resolve_save_root(cfg, general_params)

    # 2. Renderer + Optimizer
    rndr, gpo_optimizer = _build_renderer_opt(device)
    state.extras['gpo_optimizer'] = gpo_optimizer
    state.extras['renderer'] = rndr

    # 3. Datasets
    train_dataset, ZJU_Novel_pose_dataset, Exavatar_Novel_pose_dataset = \
        _build_datasets(cfg, general_params, device)
    state.train_dataset = train_dataset
    # Populate the head_data_fix cache up-front. The SceneDataset.__init__ patch
    # installed by `apply_v5_features` (in mesh_setup) runs AFTER this point,
    # so the cache would otherwise stay empty and the optimizer wrapper would
    # silently no-op — leaving the legacy head-idx misalignment in place.
    from src.features.head_data_fix import register_train_dataset
    register_train_dataset(train_dataset)
    state.extras['NVS_dataset'] = None
    state.extras['ZJU_Novel_pose_dataset'] = ZJU_Novel_pose_dataset
    state.extras['Exavatar_Novel_pose_dataset'] = Exavatar_Novel_pose_dataset

    state.flow_loader = OpticalFlowLoader(waft_config=general_params['WAFT'], device=device)

    # 4. AvatarMesh
    state.smpl_mesh = _build_avatar_mesh(train_dataset, general_params, device)

    # 5. Save folders
    mlp_log, pre_log, curr_exp_folder = _prepare_save_folders(state.save_root, curr_exp_name)
    state.extras['mlp_log_folder'] = mlp_log
    state.extras['pre_log_folder'] = pre_log
    state.extras['curr_exp_save_folder'] = curr_exp_folder
    state.extras['device'] = device

    # 6. Initial sampling
    state.extras.update(_initial_sampling(train_dataset, gpo_optimizer, state.smpl_mesh))

    return state

"""GS#1 stage — Gaussian Splatting fine-tuning with both training_levels [0,1].


Differs from mesh stages:
  - Builds its own GS-specific loss_conf (separate common + level overrides)
  - Avatar.set_use_gs_geo(True), set_optimize_mesh(False) called once before loop
  - exp_name pattern: `GS_iter_*_sampling_*_w_both_training_levels_w_strong_color_lap`
  - samplings: exp_params['GS_sampling'] (singular, not GS_samplings)
  - batch_size=3, test_step=6, max_iters=300
  - No pose_noise_dict passed
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch


from core.geometry.sampling_utils import (                                      # noqa: E402
    get_safe_triple_indices_from_current, get_sampled_indices_from_length,
)
from core.geometry.avatar_utils import get_smpl_batch_from_indices               # noqa: E402
from core.data.utils import get_sampled_data                               # noqa: E402
from core.io.smplx import save_optimizer_outputs, load_optimizer_outputs         # noqa: E402

from src.config import TrainingConfig
from src.optim.trainer_state import TrainerState
from src.optim.loss_conf import build_loss_conf


def run_gs_stage(cfg: TrainingConfig, state: TrainerState) -> None:
    """Run all GS#1 sampling iterations defined in exp_params['GS_sampling']."""
    Avatar = state.avatar
    # enable stage-aware GoF dispatch for the duration of GS#1.
    Avatar._gs_stage_active = True
    gpo_optimizer = state.extras['gpo_optimizer']
    train_dataset = state.train_dataset
    smpl_mesh = state.smpl_mesh
    flow_loader = state.flow_loader
    save_root = state.save_root
    curr_exp_save_folder = state.extras['curr_exp_save_folder']
    SDF_grad_volume = state.extras['SDF_grad_volume']
    flow_features = state.extras['flow_features']
    NVS_dataset = state.extras['NVS_dataset']
    ZJU_Novel_pose_dataset = state.extras['ZJU_Novel_pose_dataset']
    Exavatar_Novel_pose_dataset = state.extras['Exavatar_Novel_pose_dataset']
    debug_cams = state.extras['debug_cams']
    exp_params = state.extras['exp_params']

    avatar_optimizer = state.avatar_optimizer
    avatar_scheduler = state.avatar_scheduler

    loss_conf = build_loss_conf(cfg.losses.gs)
    state.extras['gs_loss_conf'] = loss_conf

    GS_samplings = exp_params['GS_sampling']

    Avatar.set_use_gs_geo(True)
    Avatar.set_optimize_mesh(False)

    for iter_idx, sampling_interval in enumerate(GS_samplings):
        exp_name = (
            f'GS_iter_{str(iter_idx)}_sampling_{str(sampling_interval)}'
            f'_w_both_training_levels_w_strong_color_lap'
        )
        update_rgb = True
        total_len = len(train_dataset.images)

        if sampling_interval < 3:
            sampling_idxs = list(range(0, train_dataset.num_frames))
            imgs, masks, cams, normals, kpts = get_sampled_data(
                train_dataset, sampling_idxs,
            )
            seq_len = len(imgs)
            batch_idxs = (
                np.arange(0, seq_len - 2, sampling_interval),
                np.arange(1, seq_len - 1, sampling_interval),
                np.arange(2, seq_len, sampling_interval),
            )
            flow = flow_loader.get_flow_low_interval(
                save_root, train_dataset.images,
                idx_p=batch_idxs[0], idx_c=batch_idxs[1], idx_n=batch_idxs[2],
                level_name='L0', interval=sampling_interval,
            )
            curr_smpl_batch = train_dataset.smpl_params
            smpl_param_idxs = sampling_idxs
        else:
            sampling_idxs = get_sampled_indices_from_length(
                train_dataset.num_frames, sampling_interval,
            )
            prev, curr, next_ = get_safe_triple_indices_from_current(
                total_len, sampling_idxs, None, None, None,
            )
            triplet_idxs = np.stack([prev, curr, next_], axis=1).flatten()
            imgs, masks, cams, normals, kpts = get_sampled_data(
                train_dataset, triplet_idxs,
            )
            curr_smpl_batch = get_smpl_batch_from_indices(
                train_dataset.smpl_params, gpo_optimizer, smpl_mesh, triplet_idxs,
            )
            flow = flow_loader.get_flow(
                save_root, train_dataset.images, triplet_idxs, 'L0', sampling_interval,
            )
            seq_len = len(triplet_idxs)
            batch_idxs = (
                np.arange(0, seq_len, 3),
                np.arange(1, seq_len, 3),
                np.arange(2, seq_len, 3),
            )
            smpl_param_idxs = triplet_idxs

        save_path = os.path.join(curr_exp_save_folder, exp_name + '_Avatar.pth')
        log_folder = os.path.join(curr_exp_save_folder, exp_name)

        if not os.path.exists(save_path):
            Avatar.set_all_grad_false()
            _gscfg = cfg.stages.get('gs', {})
            _oc = {'DoubleAvatar': 1.0, 'training_levels': [0, 1]}
            if bool(_gscfg.get('update_pose', True)):
                _oc['full_pose'] = float(_gscfg.get('full_pose_lr', 1e-4))
                _oc['transl']    = float(_gscfg.get('transl_lr',    1e-5))
            curr_smpl_batch, Avatar, avatar_optimizer, avatar_scheduler = \
                gpo_optimizer.ECCVavatar_w_split_datasets_w_split_level(
                    Avatar, curr_smpl_batch,
                    optimizer=avatar_optimizer, scheduler=avatar_scheduler,
                    sdf_volume=SDF_grad_volume,
                    novel_view_dataset=NVS_dataset,
                    novel_pose_dataset=ZJU_Novel_pose_dataset,
                    extra_dataset=Exavatar_Novel_pose_dataset,
                    opt_flag={'update_pose': bool(_gscfg.get('update_pose', True)), 'use_rgb': update_rgb},
                    pose_conf={'global_orient': 0.1, 'body': 1.0},
                    use_6d_rot=True,
                    all_loss_conf=loss_conf,
                    opt_conf=_oc,
                    invalid_sample=None,
                    images=imgs, masks=masks, cameras=cams, normals=normals,
                    OF_model=flow_loader.model,
                    waft_of=flow, waft_features=flow_features,
                    batch_size=int(cfg.stages.get('gs', {}).get('batch_size', 3)),
                    test_step=int(cfg.stages.get('gs', {}).get('test_step', 6)),
                    head_imgs=train_dataset.head_images,
                    head_masks=train_dataset.head_masks,
                    head_bboxs=train_dataset.head_aabb,
                    paired_prev_idxs=batch_idxs[0],
                    paired_curr_idxs=batch_idxs[1],
                    paired_next_idxs=batch_idxs[2],
                    debug_da_camera=debug_cams,
                    debug_folder=log_folder,
                    max_iters=int(cfg.stages.get('gs', {}).get('max_iters', 300)),
                    anchor_head_only=bool(cfg.anchor_head_only),
                    anchor_idx=getattr(state.train_dataset, 'anchor_frame_idx', None),
                )
            for key in curr_smpl_batch.keys():
                if key != 'betas':
                    train_dataset.smpl_params[key][smpl_param_idxs] = \
                        curr_smpl_batch[key].detach().clone()
            train_dataset.smpl_params['betas'] = curr_smpl_batch['betas'].detach().clone()
            checkpoint = {
                'optimizer_state_dict': avatar_optimizer.state_dict(),
                'scheduler_state_dict': avatar_scheduler.state_dict(),
            }
            save_optimizer_outputs(save_path, train_dataset.smpl_params, Avatar, checkpoint)
        else:
            load_smpl_batch, Avatar, checkpoint = load_optimizer_outputs(Avatar, save_path)
            train_dataset.smpl_params = load_smpl_batch
            avatar_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            avatar_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # clear stage flag so any subsequent mesh re-eval reverts to vanilla.
    Avatar._gs_stage_active = False
    state.avatar = Avatar
    state.avatar_optimizer = avatar_optimizer
    state.avatar_scheduler = avatar_scheduler

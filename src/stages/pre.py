"""Pre stages — Pre0 (pose update) + Pre_Step1/2/3 (mesh & pose optim).


Each Pre step is gated by ckpt presence: if the target file already exists, we
load it and skip recomputation. This matches legacy behavior so reruns are
incremental.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import torch


from core.io.smplx import preprocessing_save, preprocessing_load          # noqa: E402
from core.io.keypoints import tmp_kpt                                      # noqa: E402
from core.geometry.sdf_volume import GridSDFGrad              # noqa: E402

from src.config import TrainingConfig
from src.optim.trainer_state import TrainerState


def _compute_head_vts_idxs(smpl_mesh):
    """Vertex indices whose dominant LBS joint is in the head/face joint set
    (12 neck, 15 head, 22 jaw, 23 l_eye, 24 r_eye). Mirrors
    `cascade.set_head_idxs` logic but standalone for Pre stage."""
    lbs = smpl_mesh._lbs_weights.squeeze()  # (V, J)
    head_joints = (12, 15, 22, 23, 24)
    head_mask = torch.zeros_like(lbs)[..., 0]
    for hj in head_joints:
        head_mask = head_mask + (torch.argmax(lbs, dim=-1) == hj).to(head_mask.dtype)
    return torch.where(head_mask > 0)[0]


def run_pre_stage(cfg: TrainingConfig, state: TrainerState) -> None:
    """Run Pre0 -> Pre_Step1 -> Pre_Step2 -> Pre_Step3 sequentially.

    Mutates `state.smpl_mesh` (its `disp`/`v_canon` fields) and writes ckpts
    under `state.extras['pre_log_folder']`. Subsequent stages read the final
    SMPL params from `state.train_dataset.smpl_params` (set at the end).
    """
    gpo_optimizer = state.extras['gpo_optimizer']
    smpl_mesh = state.smpl_mesh
    pre_log_folder = state.extras['pre_log_folder']

    sampling_imgs = state.extras['sampling_imgs']
    sampling_masks = state.extras['sampling_masks']
    sampling_cams = state.extras['sampling_cams']
    sampling_normals = state.extras['sampling_normals']
    sampling_keypoints = state.extras['sampling_keypoints']
    sampling_smplbatch = state.extras['sampling_smplbatch']

    # ----------------------------------------------------------------- Pre0
    exp_name = 'Pre0_smpl_pose_update'
    smpl_param_path = os.path.join(pre_log_folder, exp_name + '_smplx_param.pkl')
    exp_log_folder = os.path.join(pre_log_folder, exp_name)

    if not os.path.exists(smpl_param_path):
        pose_conf = {'global_orient': 0.1, 'body': 1.0, 'hand': 0.1}
        loss_conf = {
            "masks": 1.0,
            "normal_all_cosine": 0.5,
            'init_full_pose': 0.1,
            'keypoint': 10.0,
        }
        new_smpl_params_batch = gpo_optimizer.pose_update_only(
            opt_conf={'full_pose': 1e-1}, loss_conf=loss_conf,
            smpl_mesh=smpl_mesh, da_verts=smpl_mesh._v_canon.detach(),
            fullpose_dict=sampling_smplbatch, pose_conf=pose_conf,
            taget_keypoints=sampling_keypoints,
            images=sampling_imgs, cameras=sampling_cams,
            normals=sampling_normals, masks=sampling_masks,
            batch_size=int(cfg.stages.get('pre', {}).get('batch_size', 48)), downsampling=1, max_iters=int(cfg.stages.get('pre', {}).get('max_iters', 500)),
            kpt_idx=tmp_kpt['idx'],
            use_weight=False, use_6d_rot=True,
            debug_folder=exp_log_folder,
        )
        smpl_mesh.v_canon_update_disp_base()
        preprocessing_save(None, smpl_param_path, None, new_smpl_params_batch)
        for key in new_smpl_params_batch.keys():
            new_smpl_params_batch[key].requires_grad = False
    else:
        new_smpl_params_batch = preprocessing_load(None, smpl_param_path, None)

    # ------------------------------------------------------------ Pre_Step1
    exp_name = 'Pre_Step1_w_lap_w_color3000'
    smpl_mesh_path = os.path.join(pre_log_folder, exp_name + '_smpl_mesh.pth')
    smpl_param_path = os.path.join(pre_log_folder, exp_name + '_smplx_param.pkl')
    exp_log_folder = os.path.join(pre_log_folder, exp_name)

    if not os.path.exists(smpl_mesh_path):
        print("Running Preprocessing: Pre_Step1 (mesh + pose)...")
        SDF_volume = GridSDFGrad(verts=smpl_mesh._v_canon.squeeze(),
                                 faces=smpl_mesh.face,
                                 resolution=256, padding=0.2)
        pose_conf = {'global_orient': 0.1, 'body': 1.0}
        # SDF weights yaml-overridable for loose-cloth datasets.
        _pre_cfg = cfg.stages.get('pre', {})
        loss_conf = {
            "masks": 1.0, "rgb_l1": 1.0,
            "normal_all_cosine": float(_pre_cfg.get('normal_all_cosine', 0.1)),
            "laplacian": 10000.0, 'rgb_laplacian': 1.0, "reg_l1": 0.0,
            "iso_sdf_vts_norm":  float(_pre_cfg.get('iso_sdf_vts_norm',  1.0)),
            "iso_sdf_face_norm": float(_pre_cfg.get('iso_sdf_face_norm', 0.1)),
            'init_full_pose': 0.1,
        }
        # Optional head-region SDF reduction (default disabled).
        _hscale = float(_pre_cfg.get('head_scale_sdf', 0.0))
        _hidxs = _compute_head_vts_idxs(smpl_mesh) if _hscale > 0 else None
        if _hidxs is not None:
            print(f"[pre] head_scale_sdf={_hscale} on {len(_hidxs)} head vts", flush=True)
        new_da_vts, _, new_smpl_params_batch, _ = gpo_optimizer.largesteps_w_rgb(
            opt_conf={'Mesh': 1.0, 'full_pose': 1e-1, 'mesh_rgb': 1e-2},
            loss_conf=loss_conf,
            smpl_mesh=smpl_mesh, da_verts=smpl_mesh._v_canon.detach(),
            fullpose_dict=new_smpl_params_batch,
            use_6d_rot=True, pose_conf=pose_conf,
            images=sampling_imgs, cameras=sampling_cams,
            normals=sampling_normals, masks=sampling_masks,
            batch_size=int(cfg.stages.get('pre', {}).get('batch_size', 48)), step_size=1e-2, lambda_=10.0,
            downsampling=1, max_iters=int(cfg.stages.get('pre', {}).get('max_iters', 500)),
            SDF_volume=SDF_volume, grad_clip_hand=True, use_weight=True,
            debug_folder=exp_log_folder,
            head_idxs=_hidxs, head_scale=_hscale,
        )
        smpl_mesh.disp = new_da_vts.squeeze() - smpl_mesh.v_canon.squeeze()
        smpl_mesh.v_canon_update_disp_base()
        preprocessing_save(smpl_mesh_path, smpl_param_path, smpl_mesh,
                           new_smpl_params_batch)
        for key in new_smpl_params_batch.keys():
            new_smpl_params_batch[key].requires_grad = False
    else:
        smpl_mesh, new_smpl_params_batch = preprocessing_load(
            smpl_mesh_path, smpl_param_path, smpl_mesh)

    # ------------------------------------------------------------ Pre_Step2
    exp_name = 'Pre_Step2_more'
    smpl_param_path = os.path.join(pre_log_folder, exp_name + '_smplx_param.pkl')
    exp_log_folder = os.path.join(pre_log_folder, exp_name)

    if not os.path.exists(smpl_param_path):
        pose_conf = {'global_orient': 0.1, 'body': 1.0}
        loss_conf = {"masks": 1.0, "normal_all_cosine": 0.5, 'init_full_pose': 0.1}
        new_smpl_params_batch = gpo_optimizer.pose_update_only(
            opt_conf={'full_pose': 1e-1}, loss_conf=loss_conf,
            smpl_mesh=smpl_mesh, da_verts=smpl_mesh._v_canon.detach(),
            fullpose_dict=new_smpl_params_batch, pose_conf=pose_conf,
            taget_keypoints=sampling_keypoints,
            images=sampling_imgs, cameras=sampling_cams,
            normals=sampling_normals, masks=sampling_masks,
            batch_size=int(cfg.stages.get('pre', {}).get('batch_size', 48)), downsampling=1, max_iters=int(cfg.stages.get('pre', {}).get('max_iters', 500)),
            kpt_idx=tmp_kpt['idx'],
            use_weight=False, use_6d_rot=True,
            debug_folder=exp_log_folder,
        )
        smpl_mesh.v_canon_update_disp_base()
        preprocessing_save(None, smpl_param_path, None, new_smpl_params_batch)
        for key in new_smpl_params_batch.keys():
            new_smpl_params_batch[key].requires_grad = False
    else:
        new_smpl_params_batch = preprocessing_load(None, smpl_param_path, None)

    # smpl_full_dict carries Pre_Step2 result for downstream sampling
    state.train_dataset.smpl_full_dict = new_smpl_params_batch

    # ------------------------------------------------------------ Pre_Step3
    exp_name = 'Pre_Step3_w_pose_w_lap_w_rgb'
    smpl_mesh_path = os.path.join(pre_log_folder, exp_name + '_smpl_mesh.pth')
    smpl_param_path = os.path.join(pre_log_folder, exp_name + '_smplx_param.pkl')
    exp_log_folder = os.path.join(pre_log_folder, exp_name)

    if not os.path.exists(smpl_mesh_path):
        print("Running Preprocessing: Pre_Step3 (mesh + pose, final)...")
        SDF_volume = GridSDFGrad(verts=smpl_mesh._v_canon.squeeze(),
                                 faces=smpl_mesh.face,
                                 resolution=256, padding=0.2)
        pose_conf = {'global_orient': 0.1, 'body': 1.0}
        # SDF weights yaml-overridable for loose-cloth datasets.
        _pre_cfg = cfg.stages.get('pre', {})
        loss_conf = {
            "masks": 1.0, "rgb_l1": 1.0,
            "normal_all_cosine": float(_pre_cfg.get('normal_all_cosine', 0.1)),
            "laplacian": 10000.0, 'rgb_laplacian': 1.0, "reg_l1": 0.0,
            "iso_sdf_vts_norm":  float(_pre_cfg.get('iso_sdf_vts_norm',  1.0)),
            "iso_sdf_face_norm": float(_pre_cfg.get('iso_sdf_face_norm', 0.1)),
            'init_full_pose': 0.1,
        }
        # Optional head-region SDF reduction (default disabled).
        _hscale = float(_pre_cfg.get('head_scale_sdf', 0.0))
        _hidxs = _compute_head_vts_idxs(smpl_mesh) if _hscale > 0 else None
        if _hidxs is not None:
            print(f"[pre] head_scale_sdf={_hscale} on {len(_hidxs)} head vts", flush=True)
        new_da_vts, _, new_smpl_params_batch, _ = gpo_optimizer.largesteps_w_rgb(
            opt_conf={'Mesh': 1.0, 'full_pose': 1e-1, 'mesh_rgb': 1e-2},
            loss_conf=loss_conf,
            smpl_mesh=smpl_mesh, da_verts=smpl_mesh._v_canon.detach(),
            fullpose_dict=new_smpl_params_batch,
            use_6d_rot=True, pose_conf=pose_conf,
            images=sampling_imgs, cameras=sampling_cams,
            normals=sampling_normals, masks=sampling_masks,
            batch_size=int(cfg.stages.get('pre', {}).get('batch_size', 48)), step_size=1e-2, lambda_=10.0,
            downsampling=1, max_iters=int(cfg.stages.get('pre', {}).get('max_iters', 500)),
            SDF_volume=SDF_volume, grad_clip_hand=True, use_weight=True,
            debug_folder=exp_log_folder,
            head_idxs=_hidxs, head_scale=_hscale,
        )
        smpl_mesh.disp = new_da_vts.squeeze() - smpl_mesh.v_canon.squeeze()
        smpl_mesh.v_canon_update_disp_base()
        preprocessing_save(smpl_mesh_path, smpl_param_path, smpl_mesh,
                           new_smpl_params_batch)
        for key in new_smpl_params_batch.keys():
            new_smpl_params_batch[key].requires_grad = False
    else:
        smpl_mesh, new_smpl_params_batch = preprocessing_load(
            smpl_mesh_path, smpl_param_path, smpl_mesh)

    # Pre done — push final smpl params into train_dataset for downstream stages
    for key in new_smpl_params_batch.keys():
        new_smpl_params_batch[key] = new_smpl_params_batch[key].to(torch.float32)
    state.train_dataset.smpl_params = new_smpl_params_batch
    state.smpl_mesh = smpl_mesh

    # --- Validation dump: per-frame silhouette vs SAMURAI mask.
    # Custom dataset → auto-enabled (so joint_valid_frames.txt is published to
    # training_folder during the first run, and subsequent runs auto-filter).
    # Other datasets → opt-in only.
    _pre_cfg_check = cfg.stages.get('pre', {})
    _cfg_dump = _pre_cfg_check.get('dump_frame_validity', None)
    if _cfg_dump is None:
        _cfg_dump = (cfg.dataset_name.lower() == 'custom')
    print(f"[validity_gate] dump={_cfg_dump!r} "
          f"viz={_pre_cfg_check.get('frame_validity_viz', False)!r}  "
          f"(auto-enabled for Custom)" if _cfg_dump and _pre_cfg_check.get('dump_frame_validity') is None else
          f"[validity_gate] dump={_cfg_dump!r} viz={_pre_cfg_check.get('frame_validity_viz', False)!r}",
          flush=True)
    _valid_list = os.path.join(state.train_dataset.root_folder, 'joint_valid_frames.txt')
    if _cfg_dump and os.path.exists(_valid_list):
        # Frame set is fixed once the list exists: the dataset was already filtered at
        # load time, and re-validating would rewrite the list mid-project.
        print(f"[validity] {_valid_list} exists → frame set fixed, skipping re-validation", flush=True)
        _cfg_dump = False
    if _cfg_dump:
        from src.stages._validate_pre import dump_frame_validity
        _thr = float(_pre_cfg_check.get('frame_validity_threshold', 0.3))
        _out = os.path.join(pre_log_folder, 'frame_validity_post_pre.tsv')
        _viz = (os.path.join(pre_log_folder, 'frame_validity_viz')
                if _pre_cfg_check.get('frame_validity_viz', False) else None)
        try:
            dump_frame_validity(state, smpl_mesh, new_smpl_params_batch,
                                out_tsv_path=_out, threshold=_thr, viz_dir=_viz)
        except Exception as _e:
            import traceback
            print(f"[validity] ERROR: {_e}", flush=True)
            traceback.print_exc()
        if os.path.exists(_valid_list):
            with open(_valid_list) as _f:
                _n_valid = sum(1 for _ln in _f if _ln.strip())
            _n_all = len(state.train_dataset.file_names)
            if _n_valid < _n_all:
                print(f"[validity] WARNING: {_n_all - _n_valid}/{_n_all} frames were marked invalid, but this run "
                      f"keeps training on all {_n_all} frames. To train on the filtered set, start a NEW save_root "
                      f"(scripts/usage/01_train_custom.sh does this automatically).", flush=True)

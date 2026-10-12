"""Pre stages — loose-clothing / long-hair variant.

Companion to `pre.py` for scenes where the mesh cannot be
constrained to the SMPL body shell (long hair, loose clothing, etc.).
Reproduces the v7 `--preproc_loose_smpl` bundle:

  - masks         : 1.0    → 5.0    (silhouette stronger, x5)
  - laplacian     : 10000  → 50000  (smoothness stronger, x5)
  - iso_sdf_*     : 1.0/0.1 → 0.1/0.01 (SDF relaxed, x0.1)
  - normal_all_cosine at Pre_Step1/3 : 0.1 → 1.0 (added normal supervision, x10)

Selection: `stages.pre.loose_smpl: true` in the dataset yaml routes the trainer
to this file instead of `src.stages.pre`. Yaml-overridable fields
(`normal_all_cosine`, `iso_sdf_vts_norm`, `iso_sdf_face_norm`, `head_scale_sdf`,
`batch_size`, `max_iters`, `dump_frame_validity`, ...) still work identically to
`pre.py` — only the loose-cloth-specific weights are hardcoded here.
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


# Hardcoded loose_smpl scale multipliers (v7 --preproc_loose_smpl bundle).
_MASKS_SCALE = 5.0
_LAP_SCALE   = 5.0
# Iso-SDF override defaults for loose bodies. Individual yaml override still
# wins (via `_pre_cfg.get(...)`).
_LOOSE_SDF_VTS_NORM_DEFAULT  = 0.1
_LOOSE_SDF_FACE_NORM_DEFAULT = 0.01
# Normal cosine bump for Pre_Step1/3 (default in pre.py is 0.1; loose = 1.0).
_LOOSE_NORMAL_COSINE_DEFAULT = 1.0


def _compute_head_vts_idxs(smpl_mesh):
    """Vertex indices whose dominant LBS joint is in the head/face joint set."""
    lbs = smpl_mesh._lbs_weights.squeeze()
    head_joints = (12, 15, 22, 23, 24)
    head_mask = torch.zeros_like(lbs)[..., 0]
    for hj in head_joints:
        head_mask = head_mask + (torch.argmax(lbs, dim=-1) == hj).to(head_mask.dtype)
    return torch.where(head_mask > 0)[0]


def run_pre_stage_loose_smpl(cfg: TrainingConfig, state: TrainerState) -> None:
    """Run Pre0 → Pre_Step1 → Pre_Step2 → Pre_Step3 with the loose_smpl preset.

    Mirrors `src.stages.pre.run_pre_stage` but with the loose_smpl hardcoded
    weight adjustments applied throughout.
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

    print(f"[pre_loose_smpl] LOOSE preset active: masks×{_MASKS_SCALE}, "
          f"laplacian×{_LAP_SCALE}, iso_sdf default {_LOOSE_SDF_VTS_NORM_DEFAULT}/"
          f"{_LOOSE_SDF_FACE_NORM_DEFAULT}, normal_cosine default "
          f"{_LOOSE_NORMAL_COSINE_DEFAULT} (yaml `stages.pre.*` still overrides)",
          flush=True)

    # ----------------------------------------------------------------- Pre0
    exp_name = 'Pre0_smpl_pose_update'
    smpl_param_path = os.path.join(pre_log_folder, exp_name + '_smplx_param.pkl')
    exp_log_folder = os.path.join(pre_log_folder, exp_name)

    if not os.path.exists(smpl_param_path):
        pose_conf = {'global_orient': 0.1, 'body': 1.0, 'hand': 0.1}
        loss_conf = {
            "masks": 1.0 * _MASKS_SCALE,
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
        print("Running Preprocessing: Pre_Step1 (mesh + pose) [loose_smpl]...")
        SDF_volume = GridSDFGrad(verts=smpl_mesh._v_canon.squeeze(),
                                 faces=smpl_mesh.face,
                                 resolution=256, padding=0.2)
        pose_conf = {'global_orient': 0.1, 'body': 1.0}
        _pre_cfg = cfg.stages.get('pre', {})
        loss_conf = {
            "masks": 1.0 * _MASKS_SCALE, "rgb_l1": 1.0,
            "normal_all_cosine": float(_pre_cfg.get('normal_all_cosine', _LOOSE_NORMAL_COSINE_DEFAULT)),
            "laplacian": 10000.0 * _LAP_SCALE, 'rgb_laplacian': 1.0, "reg_l1": 0.0,
            "iso_sdf_vts_norm":  float(_pre_cfg.get('iso_sdf_vts_norm',  _LOOSE_SDF_VTS_NORM_DEFAULT)),
            "iso_sdf_face_norm": float(_pre_cfg.get('iso_sdf_face_norm', _LOOSE_SDF_FACE_NORM_DEFAULT)),
            'init_full_pose': 0.1,
        }
        _hscale = float(_pre_cfg.get('head_scale_sdf', 0.0))
        _hidxs = _compute_head_vts_idxs(smpl_mesh) if _hscale > 0 else None
        if _hidxs is not None:
            print(f"[pre_loose_smpl] head_scale_sdf={_hscale} on {len(_hidxs)} head vts", flush=True)
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
        loss_conf = {"masks": 1.0 * _MASKS_SCALE, "normal_all_cosine": 0.5, 'init_full_pose': 0.1}
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

    state.train_dataset.smpl_full_dict = new_smpl_params_batch

    # ------------------------------------------------------------ Pre_Step3
    exp_name = 'Pre_Step3_w_pose_w_lap_w_rgb'
    smpl_mesh_path = os.path.join(pre_log_folder, exp_name + '_smpl_mesh.pth')
    smpl_param_path = os.path.join(pre_log_folder, exp_name + '_smplx_param.pkl')
    exp_log_folder = os.path.join(pre_log_folder, exp_name)

    if not os.path.exists(smpl_mesh_path):
        print("Running Preprocessing: Pre_Step3 (mesh + pose, final) [loose_smpl]...")
        SDF_volume = GridSDFGrad(verts=smpl_mesh._v_canon.squeeze(),
                                 faces=smpl_mesh.face,
                                 resolution=256, padding=0.2)
        pose_conf = {'global_orient': 0.1, 'body': 1.0}
        _pre_cfg = cfg.stages.get('pre', {})
        loss_conf = {
            "masks": 1.0 * _MASKS_SCALE, "rgb_l1": 1.0,
            "normal_all_cosine": float(_pre_cfg.get('normal_all_cosine', _LOOSE_NORMAL_COSINE_DEFAULT)),
            "laplacian": 10000.0 * _LAP_SCALE, 'rgb_laplacian': 1.0, "reg_l1": 0.0,
            "iso_sdf_vts_norm":  float(_pre_cfg.get('iso_sdf_vts_norm',  _LOOSE_SDF_VTS_NORM_DEFAULT)),
            "iso_sdf_face_norm": float(_pre_cfg.get('iso_sdf_face_norm', _LOOSE_SDF_FACE_NORM_DEFAULT)),
            'init_full_pose': 0.1,
        }
        _hscale = float(_pre_cfg.get('head_scale_sdf', 0.0))
        _hidxs = _compute_head_vts_idxs(smpl_mesh) if _hscale > 0 else None
        if _hidxs is not None:
            print(f"[pre_loose_smpl] head_scale_sdf={_hscale} on {len(_hidxs)} head vts", flush=True)
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

    for key in new_smpl_params_batch.keys():
        new_smpl_params_batch[key] = new_smpl_params_batch[key].to(torch.float32)
    state.train_dataset.smpl_params = new_smpl_params_batch
    state.smpl_mesh = smpl_mesh

    # Validation dump (identical to pre.py behavior).
    _pre_cfg_check = cfg.stages.get('pre', {})
    _cfg_dump = _pre_cfg_check.get('dump_frame_validity', None)
    if _cfg_dump is None:
        _cfg_dump = (cfg.dataset_name.lower() == 'custom')
    print(f"[validity_gate] dump={_cfg_dump!r} viz={_pre_cfg_check.get('frame_validity_viz', False)!r}",
          flush=True)
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

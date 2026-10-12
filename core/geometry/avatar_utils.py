import os
import json
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.cuda.amp import custom_fwd
from skimage.metrics import structural_similarity

from core.mesh import Mesh
from core.geometry.mesh_utils import face_to_vertex_color_batch
from core.geometry.camera_utils import projection
from core.lpips.lpips import LPIPS_Evaluation


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class Evaluator_ExAvatar(nn.Module):
    """Adapted from ExAvatar (https://github.com/mks0601/ExAvatar_RELEASE)."""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb*2-1.0, rgb_gt*2-1.0),
        }


class TorchEvaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):  # (N,C,H,W)
        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


class Evaluator:
    """ #From Gomavatar
    copied from https://github.com/zju3dv/neuralbody/blob/6bf1905822f71d1e568ef831110728fd1d06c94d/lib/evaluators/neural_volume.py
    adapted from https://github.com/escapefreeg/humannerf-eval/blob/master/eval.py
    """

    def __init__(self):
        self.lpips_model = LPIPS_Evaluation(net='vgg').cuda()
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt):
        ssim = structural_similarity(img_pred, img_gt, multichannel=True, channel_axis=-1, data_range=2)
        return ssim

    def lpips_metric(self, img_pred, img_gt):
        # convert range from 0-1 to -1-1
        processed_pred = torch.from_numpy(img_pred).float().unsqueeze(0).cuda() * 2. - 1.
        processed_gt = torch.from_numpy(img_gt).float().unsqueeze(0).cuda() * 2. - 1.

        lpips_loss = self.lpips_model(processed_pred.permute(0, 3, 1, 2), processed_gt.permute(0, 3, 1, 2))
        return torch.mean(lpips_loss).cpu().detach().item() * 1000

    def evaluate(self, rgb_pred, rgb_gt):
        mse = np.mean((rgb_pred - rgb_gt) ** 2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt)
        self.ssim.append(ssim)

        lpips = self.lpips_metric(rgb_pred, rgb_gt)
        self.lpips.append(lpips)

    def summarize(self, path):
        result_path = path
        print(os.path.basename(path))

        mse_mean = np.mean(self.mse)
        psnr_mean = np.mean(self.psnr)
        ssim_mean = np.mean(self.ssim)
        lpips_mean = np.mean(self.lpips)

        with open(result_path, 'w') as f:
            f.write(f'mse: {mse_mean}\n')
            f.write(f'psnr: {psnr_mean}\n')
            f.write(f'ssim: {ssim_mean}\n')
            f.write(f'lpips: {lpips_mean}\n')

        print(f'Results saved to {result_path}')
        print(f'mse: {mse_mean}')
        print(f'psnr: {psnr_mean}')
        print(f'ssim: {ssim_mean}')
        print(f'lpips: {lpips_mean}')

        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []


# ---------------------------------------------------------------------------
# SMPL batch helpers
# ---------------------------------------------------------------------------

def get_paired_smpl_batches(train_smpl_params, gpo_optimizer, smpl_mesh, sampling_idxs=None, except_frames=None):
    """
    Return (current, previous) frame SMPL parameter batches.

    Args:
        sampling_idxs (list or None): indices to extract. None uses all frames.
        except_frames (list or set): frames whose previous entry is replaced with Default Pose.
    """
    train_params_list = gpo_optimizer.batch2list(train_smpl_params)

    if sampling_idxs is None:
        sampling_idxs = list(range(len(train_params_list)))
    else:
        sampling_idxs = list(sampling_idxs)

    current_params_list = [train_params_list[idx] for idx in sampling_idxs]

    prev_idxs = [idx - 1 for idx in sampling_idxs]
    prev_params_list = [train_params_list[idx] for idx in prev_idxs]

    except_set = set(except_frames) if except_frames is not None else set()

    def _process_batch(params_list):
        if not params_list:
            return {}
        batch = gpo_optimizer.list2batch(params_list)
        batch['betas'] = params_list[0]['betas']
        return smpl_mesh.smplx2full_pose_dict(batch)

    sampling_smplbatch = _process_batch(current_params_list)
    prev_sampling_smplbatch = _process_batch(prev_params_list)
    prev_sampling_smplbatch_list = gpo_optimizer.batch2list(prev_sampling_smplbatch)
    for list_idx, original_idx in enumerate(sampling_idxs):
        if original_idx in except_set:
            prev_sampling_smplbatch_list[list_idx]['full_pose'] = smpl_mesh.get_da_fullpose()
    prev_sampling_smplbatch = gpo_optimizer.list2batch(prev_sampling_smplbatch_list)

    return sampling_smplbatch, prev_sampling_smplbatch


def get_smpl_batch_from_indices(train_smpl_params, gpo_optimizer, smpl_mesh, sampling_idxs):
    """
    Extract an SMPL parameter batch from the given index list.
    Indices of -1 are filled with Default Pose.
    """
    train_params_list = gpo_optimizer.batch2list(train_smpl_params)

    default_full_pose = smpl_mesh.get_da_fullpose().squeeze()

    batch_params_list = []
    invalid_mask = []

    for i, idx in enumerate(sampling_idxs):
        if idx < 0:
            batch_params_list.append(train_params_list[0])
            invalid_mask.append(i)
        else:
            batch_params_list.append(train_params_list[idx])

    if not batch_params_list:
        return {}

    batch = gpo_optimizer.list2batch(batch_params_list)
    batch['betas'] = train_params_list[0]['betas']

    if 'full_pose' not in train_smpl_params.keys():
        full_pose_batch = smpl_mesh.smplx2full_pose_dict(batch)
    else:
        full_pose_batch = batch

    if invalid_mask:
        full_poses = full_pose_batch['full_pose'].clone()
        default_pose_tensor = default_full_pose.to(full_poses.device)
        for i in invalid_mask:
            full_poses[i] = default_pose_tensor
        full_pose_batch['full_pose'] = full_poses

    return full_pose_batch


# ---------------------------------------------------------------------------
# Forward / inference helpers
# ---------------------------------------------------------------------------

def forward_cascadeavatar_split_use_gs(single_avatar,
                                       batch_params,
                                       cameras,
                                       bg_color,
                                       render_out_levels=[],
                                       noise_size=None,
                                       canon=False,
                                       ):
    model_outputs = single_avatar(target_pose=batch_params,
                                  cameras=cameras,
                                  training_levels=render_out_levels,
                                  noise_size=noise_size,
                                  canon=canon,
                                  )
    gs_outs = []
    if single_avatar.use_gs_geo:
        for level in range(single_avatar.curr_level + 1):
            curr_gs = model_outputs['levels'][level]['gs']
            gs_out = single_avatar.render_mesh2gs_batch(curr_gs,
                                                        cameras=cameras,
                                                        bg_color=bg_color)
            gs_outs.append(gs_out)
    return model_outputs, gs_outs


def forward_cascadeavatar_split_use_gs_w_OF(single_avatar,
                                            batch_params,
                                            cameras,
                                            bg_color,
                                            of_cur_idxs,
                                            of_next_idxs,
                                            tmp_camera_K,
                                            tmp_camera_R,
                                            tmp_camera_t,
                                            render_out_levels=[],
                                            noise_size=None,
                                            canon=False,
                                            ):
    model_outputs = single_avatar(target_pose=batch_params,
                                  cameras=cameras,
                                  training_levels=render_out_levels,
                                  noise_size=noise_size,
                                  canon=canon,
                                  )
    gs_outs = []
    if single_avatar.use_gs_geo:
        for level in range(single_avatar.curr_level + 1):
            curr_gs = model_outputs['levels'][level]['gs']
            xyz = curr_gs['xyz']

            current_vts = torch.concatenate([xyz[of_cur_idxs]], dim=0)
            next_vts = torch.concatenate([xyz[of_next_idxs]], dim=0)
            curr_uv = projection(current_vts, tmp_camera_K[of_cur_idxs],
                                 tmp_camera_R[of_cur_idxs],
                                 -tmp_camera_t[of_cur_idxs].unsqueeze(-1))

            next_uv = projection(next_vts, tmp_camera_K[of_next_idxs],
                                 tmp_camera_R[of_next_idxs],
                                 -tmp_camera_t[of_next_idxs].unsqueeze(-1))
            OF = next_uv - curr_uv

            curr_gs['OF'] = OF
            gs_out = single_avatar.render_mesh2gs_batch_w_OF(curr_gs,
                                                             cameras=cameras,
                                                             curr_idxs=of_cur_idxs,
                                                             bg_color=bg_color)
            gs_outs.append(gs_out)
    return model_outputs, gs_outs


# ---------------------------------------------------------------------------
# Save / benchmark utilities
# ---------------------------------------------------------------------------

@torch.no_grad
def save_render_cascade_general_version_split_gs(single_avatar,
                                                 mesh_renderer,
                                                 save_root,
                                                 render_out_levels=[],
                                                 is_benchmark=False,
                                                 benchmark_data=None,
                                                 full_dict_list=None,
                                                 canon=False,
                                                 bg=[1.0, 1.0, 1.0]):

    device = single_avatar.device
    if full_dict_list is None:
        full_dict_list = benchmark_data['smpl_params']
    if not isinstance(full_dict_list, list):
        full_dict_list = single_avatar.mesh_avatars[0].batch2list(benchmark_data['smpl_params'])

    test_images = None
    test_masks = None
    normal_gt = None

    if benchmark_data is not None:
        test_cameras = benchmark_data['cameras']
        test_images = copy.deepcopy(benchmark_data['test_images'])
        test_masks = benchmark_data['test_masks']

        if 'test_normals' in benchmark_data.keys():
            normal_gt = benchmark_data['test_normals']

    os.makedirs(save_root, exist_ok=True)

    if not canon:
        P_normal_overlay_folder = os.path.join(save_root, 'P_normal_overlay')
        os.makedirs(P_normal_overlay_folder, exist_ok=True)
        if normal_gt is not None:
            gt_normal_folder = os.path.join(save_root, 'gt_normal')
            os.makedirs(gt_normal_folder, exist_ok=True)

    P_normal_folder = os.path.join(save_root, 'P_normal')
    os.makedirs(P_normal_folder, exist_ok=True)
    GS_surf_normal_folder = os.path.join(save_root, 'GS_surf_normal')
    os.makedirs(GS_surf_normal_folder, exist_ok=True)
    GS_rndr_normal_folder = os.path.join(save_root, 'GS_rndr_normal')
    os.makedirs(GS_rndr_normal_folder, exist_ok=True)
    GS_rgb_folder = os.path.join(save_root, 'GS_RGB')
    os.makedirs(GS_rgb_folder, exist_ok=True)
    GS_albedo_folder = os.path.join(save_root, 'GS_albedo')
    os.makedirs(GS_albedo_folder, exist_ok=True)
    Mesh_rgb_folder = os.path.join(save_root, 'Mesh_RGB')
    os.makedirs(Mesh_rgb_folder, exist_ok=True)
    Mesh_albedo_folder = os.path.join(save_root, 'Mesh_albedo')
    os.makedirs(Mesh_albedo_folder, exist_ok=True)
    # MLP displacement mesh viz folders (created lazily if MLP branch present)
    MLP_Mesh_rgb_folder = os.path.join(save_root, 'MLP_Mesh_RGB')
    MLP_Mesh_albedo_folder = os.path.join(save_root, 'MLP_Mesh_albedo')
    MLP_P_normal_folder = os.path.join(save_root, 'MLP_P_normal')
    MLP_P_normal_overlay_folder = os.path.join(save_root, 'MLP_P_normal_overlay') if not canon else None

    B = len(full_dict_list)
    if is_benchmark:
        mesh_evaluator = Evaluator()
        gs_evaluator = Evaluator()

    for i in range(B):
        test_camera = [test_cameras[i]]

        curr_img_np = None
        curr_mask_tensor = None

        if not canon and benchmark_data is not None and test_images is not None:
            curr_img_data = test_images[i]
            curr_mask_data = test_masks[i]

            if curr_img_data.max() <= 1.0:
                curr_img_np = (curr_img_data * 255.0).astype(np.uint8)
            else:
                curr_img_np = curr_img_data.astype(np.uint8)

            curr_img_tensor = torch.from_numpy(curr_img_data).float().to(device)
            if curr_img_tensor.max() > 1.0:
                curr_img_tensor /= 255.0
            if curr_img_tensor.dim() == 3:
                curr_img_tensor = curr_img_tensor.permute(2, 0, 1).unsqueeze(0)

            curr_mask_tensor = torch.from_numpy(curr_mask_data).float().to(device)
            if curr_mask_tensor.max() > 1.0:
                curr_mask_tensor /= 255.0
            if curr_mask_tensor.dim() == 2:
                curr_mask_tensor = curr_mask_tensor.unsqueeze(-1)
            if curr_mask_tensor.dim() == 3:
                curr_mask_tensor = curr_mask_tensor.permute(2, 0, 1).unsqueeze(0)

        bg_tensor = torch.tensor(bg, device=device).view(1, 3, 1, 1)

        curr_batch_params = full_dict_list[i]
        F_outputs_all, gs_out_all = forward_cascadeavatar_split_use_gs(single_avatar,
                                                                        cameras=test_camera,
                                                                        bg_color=bg_tensor,
                                                                        render_out_levels=render_out_levels,
                                                                        batch_params=curr_batch_params,
                                                                        noise_size=None,
                                                                        canon=canon,
                                                                        )
        Render_GS = len(gs_out_all) != 0
        if Render_GS:
            GS2D_output = gs_out_all[-1]

        P_da_vts = F_outputs_all['Da_vts']
        P_posed_vts = F_outputs_all['posed_vts']
        render_vts = P_da_vts if canon else P_posed_vts
        F_face = single_avatar.mesh_avatars[single_avatar.curr_level].face
        refine_vts_rgb = face_to_vertex_color_batch(F_outputs_all['gs_finest']['rgb_refine'], F_face)
        corase_vts_rgb = face_to_vertex_color_batch(F_outputs_all['gs_finest']['rgb'], F_face)

        P_curr_mesh = Mesh(render_vts.squeeze(), F_outputs_all['faces'],
                           colors=torch.ones_like(render_vts.squeeze()), device=device)

        P_mesh_rendered_out = mesh_renderer.render_SH_batch_w_albedo_refine_rgb(P_curr_mesh,
                                                                                 rgb=corase_vts_rgb,
                                                                                 refine_rgb=refine_vts_rgb,
                                                                                 cameras=test_camera, bg=bg,
                                                                                 disp_dict=None)
        P_normal = P_mesh_rendered_out['normal']

        p_normal_np = P_normal[0].detach().cpu().numpy()
        p_normal_np = normal_save_converter(p_normal_np, minus=False)
        p_normal_vis = (p_normal_np * 255).astype(np.uint8)

        if not canon and curr_img_np is not None:
            overlay = (curr_img_np / 2 + p_normal_vis / 2).astype(np.uint8)
            cv2.imwrite(os.path.join(P_normal_overlay_folder, str(i).zfill(5) + '.png'), overlay)

            if normal_gt is not None:
                gt_n_data = normal_gt[i]
                if isinstance(gt_n_data, torch.Tensor):
                    gt_n_data = gt_n_data.detach().cpu().numpy()
                if gt_n_data.shape[0] == 3:
                    gt_n_data = gt_n_data
                gt_n_vis = ((gt_n_data + 1.0) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(gt_normal_folder, str(i).zfill(5) + '.png'), gt_n_vis)

        cv2.imwrite(os.path.join(P_normal_folder, str(i).zfill(5) + '.png'), p_normal_vis)

        mesh_rgb = P_mesh_rendered_out['rgba'][0, ..., :3].detach().cpu().numpy()
        cv2.imwrite(os.path.join(Mesh_rgb_folder, str(i).zfill(5) + '.png'),
                    (np.clip(mesh_rgb, 0, 1) * 255.0).astype(np.uint8))

        mesh_coarse_rgb = P_mesh_rendered_out['rgba_coarse'][0, ..., :3].detach().cpu().numpy()
        cv2.imwrite(os.path.join(Mesh_albedo_folder, str(i).zfill(5) + '.png'),
                    (np.clip(mesh_coarse_rgb, 0, 1) * 255.0).astype(np.uint8))

        # ============================================================
        # MLP displacement mesh viz — render & save if MLP branch present
        # ============================================================
        _levels = F_outputs_all.get('levels', [])
        _curr_level_out = _levels[single_avatar.curr_level] if len(_levels) > single_avatar.curr_level else {}
        if 'mesh_mlp_canon_vts' in _curr_level_out and 'mesh_mlp_posed_vts' in _curr_level_out:
            os.makedirs(MLP_Mesh_rgb_folder, exist_ok=True)
            os.makedirs(MLP_Mesh_albedo_folder, exist_ok=True)
            os.makedirs(MLP_P_normal_folder, exist_ok=True)
            if MLP_P_normal_overlay_folder is not None:
                os.makedirs(MLP_P_normal_overlay_folder, exist_ok=True)
            _mlp_canon = _curr_level_out['mesh_mlp_canon_vts']
            _mlp_posed = _curr_level_out['mesh_mlp_posed_vts']
            _mlp_render_vts = _mlp_canon if canon else _mlp_posed
            _mlp_render_vts_sq = _mlp_render_vts.squeeze() if _mlp_render_vts.dim() == 3 else _mlp_render_vts
            _mlp_mesh_obj = Mesh(_mlp_render_vts_sq, F_outputs_all['faces'],
                                 colors=torch.ones_like(_mlp_render_vts_sq), device=device)
            _mlp_rendered = mesh_renderer.render_SH_batch_w_albedo_refine_rgb(
                _mlp_mesh_obj, rgb=corase_vts_rgb, refine_rgb=refine_vts_rgb,
                cameras=test_camera, bg=bg, disp_dict=None)
            _mlp_p_normal = _mlp_rendered['normal'][0].detach().cpu().numpy()
            _mlp_p_normal = normal_save_converter(_mlp_p_normal, minus=False)
            _mlp_p_normal_vis = (_mlp_p_normal * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(MLP_P_normal_folder, str(i).zfill(5) + '.png'), _mlp_p_normal_vis)
            if MLP_P_normal_overlay_folder is not None and curr_img_np is not None:
                _overlay = (curr_img_np / 2 + _mlp_p_normal_vis / 2).astype(np.uint8)
                cv2.imwrite(os.path.join(MLP_P_normal_overlay_folder, str(i).zfill(5) + '.png'), _overlay)
            _mlp_rgb = _mlp_rendered['rgba'][0, ..., :3].detach().cpu().numpy()
            cv2.imwrite(os.path.join(MLP_Mesh_rgb_folder, str(i).zfill(5) + '.png'),
                        (np.clip(_mlp_rgb, 0, 1) * 255.0).astype(np.uint8))
            _mlp_albedo = _mlp_rendered['rgba_coarse'][0, ..., :3].detach().cpu().numpy()
            cv2.imwrite(os.path.join(MLP_Mesh_albedo_folder, str(i).zfill(5) + '.png'),
                        (np.clip(_mlp_albedo, 0, 1) * 255.0).astype(np.uint8))
            del _mlp_rendered
        # ============================================================

        if Render_GS:
            if 'rend_normal' in GS2D_output.keys():
                gs_rend_n = GS2D_output['rend_normal'][0].detach().cpu().numpy()
                cv2.imwrite(os.path.join(GS_rndr_normal_folder, str(i).zfill(5) + '.png'),
                            ((gs_rend_n + 1.0) * 127.5).astype(np.uint8))

            gs_surf_n = GS2D_output['surf_normal'][0].detach().cpu().numpy()
            gs_surf_n = normal_save_converter(gs_surf_n)
            cv2.imwrite(os.path.join(GS_surf_normal_folder, str(i).zfill(5) + '.png'),
                        ((gs_surf_n + 1.0) * 127.5).astype(np.uint8))

            gs_rgb = GS2D_output['render'][0].detach().cpu().numpy()
            cv2.imwrite(os.path.join(GS_rgb_folder, str(i).zfill(5) + '.png'),
                        (np.clip(gs_rgb, 0, 1) * 255.0).astype(np.uint8))

            gs_albedo = GS2D_output['render_albedo'][0].detach().cpu().numpy()
            cv2.imwrite(os.path.join(GS_albedo_folder, str(i).zfill(5) + '.png'),
                        (np.clip(gs_albedo, 0, 1) * 255.0).astype(np.uint8))

        if is_benchmark:
            gt_img = curr_img_np.astype(np.float32) / 255.0
            mesh_evaluator.evaluate(mesh_rgb, gt_img)
            if Render_GS:
                gs_evaluator.evaluate(gs_rgb, gt_img)

        if Render_GS:
            del GS2D_output
        del P_mesh_rendered_out

    if is_benchmark:
        gt_img = curr_img_np.astype(np.float32) / 255.0
        mesh_evaluator.summarize(os.path.join(save_root, 'mesh_metric.txt'))
        if Render_GS:
            gs_evaluator.summarize(os.path.join(save_root, 'gs_metric.txt'))


@torch.no_grad
def benchmarking_EaxAvatar_version(single_avatar,
                                   mesh_renderer,
                                   save_root,
                                   render_out_levels=[],
                                   is_benchmark=False,
                                   benchmark_data=None,
                                   full_dict_list=None,
                                   canon=False,
                                   bg=[1.0, 1.0, 1.0],
                                   eval_subsample_stride=0):
    """
    eval_subsample_stride: if > 0 and test set has more than that many frames,
        render every Nth frame (`np.arange(0, B, stride)`). For training-time
        intermediate validation. Final evaluation MUST pass 0 (default) so all
        frames count toward the reported PSNR/SSIM/LPIPS.
    """
    bg = [1.0, 1.0, 1.0]
    device = single_avatar.device
    if full_dict_list is None:
        full_dict_list = benchmark_data['smpl_params']
    if not isinstance(full_dict_list, list):
        full_dict_list = single_avatar.mesh_avatars[0].batch2list(benchmark_data['smpl_params'])

    test_images = None
    test_masks = None
    normal_gt = None

    if benchmark_data is not None:
        test_cameras = benchmark_data['cameras']
        test_images = copy.deepcopy(benchmark_data['test_images'])
        test_masks = benchmark_data['test_masks']

        if 'test_normals' in benchmark_data.keys() and benchmark_data['test_normals'] is not None:
            normal_gt = benchmark_data['test_normals']

    # optional stride sub-sampling for training-time intermediate
    # validation (disk + GPU savings on 100+ frame test sets). 30 → 1 per 30
    # frames. Final eval passes 0 → full.
    B_full = len(full_dict_list)
    if eval_subsample_stride and eval_subsample_stride > 0 \
            and B_full > eval_subsample_stride:
        _sample_idxs = list(range(0, B_full, int(eval_subsample_stride)))
        full_dict_list = [full_dict_list[i] for i in _sample_idxs]
        if benchmark_data is not None:
            test_cameras = [test_cameras[i] for i in _sample_idxs]
            if test_images is not None:
                test_images = [test_images[i] for i in _sample_idxs]
            if test_masks is not None:
                test_masks = [test_masks[i] for i in _sample_idxs]
            if normal_gt is not None:
                normal_gt = [normal_gt[i] for i in _sample_idxs]

    os.makedirs(save_root, exist_ok=True)

    if not canon:
        P_normal_overlay_folder = os.path.join(save_root, 'P_normal_overlay')
        os.makedirs(P_normal_overlay_folder, exist_ok=True)
        if normal_gt is not None:
            gt_normal_folder = os.path.join(save_root, 'gt_normal')
            os.makedirs(gt_normal_folder, exist_ok=True)

    P_normal_folder = os.path.join(save_root, 'P_normal')
    os.makedirs(P_normal_folder, exist_ok=True)
    GS_surf_normal_folder = os.path.join(save_root, 'GS_surf_normal')
    os.makedirs(GS_surf_normal_folder, exist_ok=True)
    GS_rndr_normal_folder = os.path.join(save_root, 'GS_rndr_normal')
    os.makedirs(GS_rndr_normal_folder, exist_ok=True)
    GS_rgb_folder = os.path.join(save_root, 'GS_RGB')
    os.makedirs(GS_rgb_folder, exist_ok=True)
    GS_albedo_folder = os.path.join(save_root, 'GS_albedo')
    os.makedirs(GS_albedo_folder, exist_ok=True)
    Mesh_rgb_folder = os.path.join(save_root, 'Mesh_RGB')
    os.makedirs(Mesh_rgb_folder, exist_ok=True)
    Mesh_albedo_folder = os.path.join(save_root, 'Mesh_albedo')
    os.makedirs(Mesh_albedo_folder, exist_ok=True)
    # MLP displacement mesh viz folders (created lazily if MLP branch present)
    MLP_Mesh_rgb_folder = os.path.join(save_root, 'MLP_Mesh_RGB')
    MLP_Mesh_albedo_folder = os.path.join(save_root, 'MLP_Mesh_albedo')
    MLP_P_normal_folder = os.path.join(save_root, 'MLP_P_normal')
    MLP_P_normal_overlay_folder = os.path.join(save_root, 'MLP_P_normal_overlay') if not canon else None

    B = len(full_dict_list)
    if is_benchmark:
        mesh_evaluator = Evaluator_ExAvatar()
        mesh_evaluator = mesh_evaluator.cuda()
        gs_evaluator = Evaluator_ExAvatar()
        gs_evaluator = gs_evaluator.cuda()
        mesh_evaluator.eval()
        gs_evaluator.eval()
        mesh_results = []
        gs_results = []
    all_times = []
    for i in range(B):
        test_camera = [test_cameras[i]]

        curr_img_np = None
        curr_mask_tensor = None

        if not canon and benchmark_data is not None and test_images is not None:
            curr_img_data = test_images[i]
            curr_mask_data = test_masks[i]

            if curr_img_data.max() <= 1.0:
                curr_img_np = (curr_img_data * 255.0).astype(np.uint8)
            else:
                curr_img_np = curr_img_data.astype(np.uint8)

            curr_img_tensor = torch.from_numpy(curr_img_data).float().to(device)
            if curr_img_tensor.max() > 1.0:
                curr_img_tensor /= 255.0
            if curr_img_tensor.dim() == 3:
                curr_img_tensor = curr_img_tensor.permute(2, 0, 1).unsqueeze(0)

            curr_mask_tensor = torch.from_numpy(curr_mask_data).float().to(device)
            if curr_mask_tensor.max() > 1.0:
                curr_mask_tensor /= 255.0
            if curr_mask_tensor.dim() == 2:
                curr_mask_tensor = curr_mask_tensor.unsqueeze(-1)
            if curr_mask_tensor.dim() == 3:
                curr_mask_tensor = curr_mask_tensor.permute(2, 0, 1).unsqueeze(0)

        bg_tensor = torch.tensor(bg, device=device).view(1, 3, 1, 1)

        curr_batch_params = full_dict_list[i]
        import time
        start = time.time()
        with torch.inference_mode():
            F_outputs_all, gs_out_all = forward_cascadeavatar_split_use_gs(single_avatar,
                                                                            cameras=test_camera,
                                                                            bg_color=bg_tensor,
                                                                            render_out_levels=render_out_levels,
                                                                            batch_params=curr_batch_params,
                                                                            noise_size=None,
                                                                            canon=canon,
                                                                            )
        end_time = time.time()
        all_times.append(end_time - start)
        Render_GS = len(gs_out_all) != 0
        if Render_GS:
            GS2D_output = gs_out_all[-1]

        P_da_vts = F_outputs_all['Da_vts']
        P_posed_vts = F_outputs_all['posed_vts']
        render_vts = P_da_vts if canon else P_posed_vts
        F_face = single_avatar.mesh_avatars[single_avatar.curr_level].face
        refine_vts_rgb = face_to_vertex_color_batch(F_outputs_all['gs_finest']['rgb_refine'], F_face)
        corase_vts_rgb = face_to_vertex_color_batch(F_outputs_all['gs_finest']['rgb'], F_face)

        P_curr_mesh = Mesh(render_vts.squeeze(), F_outputs_all['faces'],
                           colors=torch.ones_like(render_vts.squeeze()), device=device)

        P_mesh_rendered_out = mesh_renderer.render_SH_batch_w_albedo_refine_rgb(P_curr_mesh,
                                                                                 rgb=corase_vts_rgb,
                                                                                 refine_rgb=refine_vts_rgb,
                                                                                 cameras=test_camera, bg=bg,
                                                                                 disp_dict=None)
        P_normal = P_mesh_rendered_out['normal']

        p_normal_np = P_normal[0].detach().cpu().numpy()
        p_normal_np = normal_save_converter(p_normal_np, minus=False)
        p_normal_vis = (p_normal_np * 255).astype(np.uint8)

        if not canon and curr_img_np is not None:
            overlay = (curr_img_np / 2 + p_normal_vis / 2).astype(np.uint8)
            cv2.imwrite(os.path.join(P_normal_overlay_folder, str(i).zfill(5) + '.png'), overlay)

            if normal_gt is not None:
                gt_n_data = normal_gt[i]
                if isinstance(gt_n_data, torch.Tensor):
                    gt_n_data = gt_n_data.detach().cpu().numpy()
                if gt_n_data.shape[0] == 3:
                    gt_n_data = gt_n_data
                gt_n_vis = ((gt_n_data + 1.0) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(gt_normal_folder, str(i).zfill(5) + '.png'), gt_n_vis)

        cv2.imwrite(os.path.join(P_normal_folder, str(i).zfill(5) + '.png'), p_normal_vis)

        mesh_rgb = P_mesh_rendered_out['rgba'][0, ..., :3].detach().cpu().numpy()
        cv2.imwrite(os.path.join(Mesh_rgb_folder, str(i).zfill(5) + '.png'),
                    (np.clip(mesh_rgb, 0, 1) * 255.0).astype(np.uint8))

        mesh_coarse_rgb = P_mesh_rendered_out['rgba_coarse'][0, ..., :3].detach().cpu().numpy()
        cv2.imwrite(os.path.join(Mesh_albedo_folder, str(i).zfill(5) + '.png'),
                    (np.clip(mesh_coarse_rgb, 0, 1) * 255.0).astype(np.uint8))

        # ============================================================
        # MLP displacement mesh viz — render & save if MLP branch present
        # ============================================================
        _levels = F_outputs_all.get('levels', [])
        _curr_level_out = _levels[single_avatar.curr_level] if len(_levels) > single_avatar.curr_level else {}
        if 'mesh_mlp_canon_vts' in _curr_level_out and 'mesh_mlp_posed_vts' in _curr_level_out:
            os.makedirs(MLP_Mesh_rgb_folder, exist_ok=True)
            os.makedirs(MLP_Mesh_albedo_folder, exist_ok=True)
            os.makedirs(MLP_P_normal_folder, exist_ok=True)
            if MLP_P_normal_overlay_folder is not None:
                os.makedirs(MLP_P_normal_overlay_folder, exist_ok=True)
            _mlp_canon = _curr_level_out['mesh_mlp_canon_vts']
            _mlp_posed = _curr_level_out['mesh_mlp_posed_vts']
            _mlp_render_vts = _mlp_canon if canon else _mlp_posed
            _mlp_render_vts_sq = _mlp_render_vts.squeeze() if _mlp_render_vts.dim() == 3 else _mlp_render_vts
            _mlp_mesh_obj = Mesh(_mlp_render_vts_sq, F_outputs_all['faces'],
                                 colors=torch.ones_like(_mlp_render_vts_sq), device=device)
            _mlp_rendered = mesh_renderer.render_SH_batch_w_albedo_refine_rgb(
                _mlp_mesh_obj, rgb=corase_vts_rgb, refine_rgb=refine_vts_rgb,
                cameras=test_camera, bg=bg, disp_dict=None)
            _mlp_p_normal = _mlp_rendered['normal'][0].detach().cpu().numpy()
            _mlp_p_normal = normal_save_converter(_mlp_p_normal, minus=False)
            _mlp_p_normal_vis = (_mlp_p_normal * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(MLP_P_normal_folder, str(i).zfill(5) + '.png'), _mlp_p_normal_vis)
            if MLP_P_normal_overlay_folder is not None and curr_img_np is not None:
                _overlay = (curr_img_np / 2 + _mlp_p_normal_vis / 2).astype(np.uint8)
                cv2.imwrite(os.path.join(MLP_P_normal_overlay_folder, str(i).zfill(5) + '.png'), _overlay)
            _mlp_rgb = _mlp_rendered['rgba'][0, ..., :3].detach().cpu().numpy()
            cv2.imwrite(os.path.join(MLP_Mesh_rgb_folder, str(i).zfill(5) + '.png'),
                        (np.clip(_mlp_rgb, 0, 1) * 255.0).astype(np.uint8))
            _mlp_albedo = _mlp_rendered['rgba_coarse'][0, ..., :3].detach().cpu().numpy()
            cv2.imwrite(os.path.join(MLP_Mesh_albedo_folder, str(i).zfill(5) + '.png'),
                        (np.clip(_mlp_albedo, 0, 1) * 255.0).astype(np.uint8))
            del _mlp_rendered
        # ============================================================

        if Render_GS:
            if 'rend_normal' in GS2D_output.keys():
                gs_rend_n = GS2D_output['rend_normal'][0].detach().cpu().numpy()
                cv2.imwrite(os.path.join(GS_rndr_normal_folder, str(i).zfill(5) + '.png'),
                            ((gs_rend_n + 1.0) * 127.5).astype(np.uint8))

            gs_surf_n = GS2D_output['surf_normal'][0].detach().cpu().numpy()
            gs_surf_n = normal_save_converter(gs_surf_n)
            cv2.imwrite(os.path.join(GS_surf_normal_folder, str(i).zfill(5) + '.png'),
                        ((gs_surf_n + 1.0) * 127.5).astype(np.uint8))

            gs_rgb = GS2D_output['render'][0].detach().cpu().numpy()
            cv2.imwrite(os.path.join(GS_rgb_folder, str(i).zfill(5) + '.png'),
                        (np.clip(gs_rgb, 0, 1) * 255.0).astype(np.uint8))

            gs_albedo = GS2D_output['render_albedo'][0].detach().cpu().numpy()
            cv2.imwrite(os.path.join(GS_albedo_folder, str(i).zfill(5) + '.png'),
                        (np.clip(gs_albedo, 0, 1) * 255.0).astype(np.uint8))

        if is_benchmark:
            if len(curr_mask_data.shape) == 3:
                curr_mask_data = torch.tensor(curr_mask_data)[None].cuda().permute(0, 3, 1, 2)
            else:
                curr_mask_data = torch.tensor(curr_mask_data)[None, :, :, None].cuda().permute(0, 3, 1, 2)
            if torch.max(curr_mask_data) > 2.0:
                curr_mask_data = curr_mask_data / 255.0
            benchmark_mask = curr_mask_data
            gt_img = curr_img_np.astype(np.float32) / 255.0

            gt_img = torch.tensor(gt_img).cuda().unsqueeze(0).permute(0, 3, 1, 2)
            gt_img = gt_img * benchmark_mask + 1 * (1 - benchmark_mask) * torch.tensor(bg).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            mesh_rgb_input = torch.tensor(mesh_rgb).cuda().unsqueeze(0).permute(0, 3, 1, 2)
            mesh_rgb_input = mesh_rgb_input * benchmark_mask + 1 * (1 - benchmark_mask) * torch.tensor(bg).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mesh_results.append(mesh_evaluator(mesh_rgb_input, gt_img))
            if Render_GS:
                gs_rgb = torch.tensor(gs_rgb).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                gs_rgb = gs_rgb * benchmark_mask + 1 * (1 - benchmark_mask) * torch.tensor(bg).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                gs_results.append(gs_evaluator(gs_rgb, gt_img))

        if Render_GS:
            del GS2D_output
        del P_mesh_rendered_out

    if is_benchmark:
        gt_img = curr_img_np.astype(np.float32) / 255.0

        with open(os.path.join(save_root, 'mesh_metric.txt'), "w") as f:
            psnr = torch.stack([r['psnr'] for r in mesh_results]).mean().item()
            print(f"PSNR: {psnr:.2f}")
            f.write(f"PSNR: {psnr:.2f}\n")
            ssim = torch.stack([r['ssim'] for r in mesh_results]).mean().item()
            print(f"SSIM: {ssim:.4f}")
            f.write(f"SSIM: {ssim:.4f}\n")
            lpips = torch.stack([r['lpips'] for r in mesh_results]).mean().item()
            print(f"LPIPS: {lpips:.4f}")
            f.write(f"LPIPS: {lpips:.4f}\n")

        if Render_GS:
            with open(os.path.join(save_root, 'gs_metric.txt'), "w") as f:
                psnr = torch.stack([r['psnr'] for r in gs_results]).mean().item()
                print(f"PSNR: {psnr:.2f}")
                f.write(f"PSNR: {psnr:.2f}\n")
                ssim = torch.stack([r['ssim'] for r in gs_results]).mean().item()
                print(f"SSIM: {ssim:.4f}")
                f.write(f"SSIM: {ssim:.4f}\n")
                lpips = torch.stack([r['lpips'] for r in gs_results]).mean().item()
                print(f"LPIPS: {lpips:.4f}")
                f.write(f"LPIPS: {lpips:.4f}\n")
        print("avg_time:", sum(all_times) / len(all_times))


def normal_save_converter(normal, minus=False):
    if not minus:  # -1 to 1
        normal = 2 * normal - 1.0
    normal[..., 1:] = -normal[..., 1:]
    normal = normal[..., ::-1]
    if not minus:
        normal = (normal + 1.0) / 2.0
    return normal


# ---------------------------------------------------------------------------
# SMPL mesh save helper
# ---------------------------------------------------------------------------

def save_render_smplmesh_params(smpl_mesh, mesh_renderer, camera, save_root, smpl_param_batch, images=None):
    os.makedirs(save_root, exist_ok=True)
    if 'left_eye_pose' in smpl_param_batch:
        smpl_param_batch['leye_pose'] = smpl_param_batch['left_eye_pose']
    if 'right_eye_pose' in smpl_param_batch:
        smpl_param_batch['reye_pose'] = smpl_param_batch['right_eye_pose']
    out = smpl_mesh.smpl_model(**smpl_param_batch)
    posed_verts = out.vertices
    rgb_save_folder = os.path.join(save_root, 'rgb')
    normal_save_folder = os.path.join(save_root, 'normal')

    os.makedirs(rgb_save_folder, exist_ok=True)
    os.makedirs(normal_save_folder, exist_ok=True)
    B = smpl_param_batch['body_pose'].shape[0]
    for i in range(B):
        mesh = Mesh(vertices=posed_verts[i], indices=smpl_mesh.smpl_model.faces,
                    colors=torch.ones_like(posed_verts[i]), device='cuda')

        if isinstance(camera, list):
            rendered = mesh_renderer.render_batch(mesh, [camera[i]], bg=[0., 0., 0.])
        else:
            rendered = mesh_renderer.render_batch(mesh, [camera], bg=[0., 0., 0.])

        normal = rendered['normal'][0].detach().cpu().numpy() * 255
        if images is not None:
            if isinstance(images, torch.Tensor):
                image = images[i].detach().cpu().numpy() * 255.0
            else:
                image = images[i]
            cv2.imwrite(os.path.join(normal_save_folder, str(i).zfill(5) + '_normal.png'), image / 2 + normal / 2)
        else:
            cv2.imwrite(os.path.join(normal_save_folder, str(i).zfill(5) + '_normal.png'), normal / 2)


def save_ply(filename, xyz, rgb):
    """
    xyz: (N, 3) float array
    rgb: (N, 3) uint8 array (0~255)
    """
    header = f"""ply
format ascii 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(xyz)):
            line = f"{xyz[i][0]} {xyz[i][1]} {xyz[i][2]} {int(rgb[i][0])} {int(rgb[i][1])} {int(rgb[i][2])}\n"
            f.write(line)


def save_dict_as_json(data_dict, filename):
    """Save a dict containing Tensors and numpy arrays as JSON."""
    processed_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            processed_dict[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            processed_dict[key] = value.tolist()
        else:
            processed_dict[key] = value
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_dict, f, indent=4)

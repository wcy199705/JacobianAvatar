"""Save_all_outputs_v2 — like core.geometry.avatar_utils.Save_all_outputs but with
GS-native alpha for GS_* RGBA outputs (mesh alpha only for Mesh_*/P_normal).

The base function in core/geometry/avatar_utils.py applies the mesh renderer's
fg_mask alpha to ALL RGBA outputs (including GS_RGB_rgba, GS_albedo_rgba, etc.).
This version uses GS2D_output['render_alpha'] for GS-related RGBA, so gaussian
splat transparency is faithfully represented.

Alpha assignment:
    Mesh_RGB_rgba, Mesh_albedo_rgba, P_normal_rgba → mesh alpha (nvdiffrast fg_mask)
    GS_RGB_rgba, GS_albedo_rgba, GS_surf_normal_rgba, GS_rndr_normal_rgba → GS render_alpha

Usage: drop-in replacement for Save_all_outputs in scripts/eval_only_best.py.
"""
from __future__ import annotations
import os
import time

import cv2
import numpy as np
import torch
from tqdm import trange

from core.mesh import Mesh
from core.geometry.mesh_utils import face_to_vertex_color_batch
from core.geometry.avatar_utils import (
    forward_cascadeavatar_split_use_gs, normal_save_converter,
    save_ply, save_dict_as_json,
)


def _mask_to_uint8(mask_tensor):
    """Accept [H,W], [1,H,W], [H,W,1] etc. Return uint8 [H,W]."""
    m = mask_tensor
    if m.dim() == 3 and m.shape[0] == 1:
        m = m.squeeze(0)
    if m.dim() == 3 and m.shape[-1] == 1:
        m = m.squeeze(-1)
    np_m = m.detach().cpu().numpy()
    if np_m.max() > 1.0:
        np_m = np_m / 255.0
    return (np.clip(np_m, 0, 1) * 255).astype(np.uint8)


def Save_all_outputs_v2(
    single_avatar, mesh_renderer, save_root, cameras, smpl_params,
    canon=False, bg=(1.0, 1.0, 1.0),
):
    device = single_avatar.device
    full_dict_list = smpl_params
    os.makedirs(save_root, exist_ok=True)

    folders_3ch = [
        'P_normal', 'GS_surf_normal', 'GS_rndr_normal',
        'GS_RGB', 'GS_albedo', 'Mesh_RGB', 'Mesh_albedo',
    ]
    folders_4ch = [f + '_rgba' for f in folders_3ch]
    save_folders = {}
    for f in folders_3ch + folders_4ch + ['Mesh', 'GS', 'smpl_param']:
        p = os.path.join(save_root, f)
        os.makedirs(p, exist_ok=True)
        save_folders[f] = p

    bg_tensor = torch.tensor(list(bg), device=device).view(1, 3, 1, 1)
    all_times = []

    for i in trange(len(full_dict_list)):
        test_camera = [cameras[i]]
        curr_batch_params = full_dict_list[i]
        start = time.time()
        with torch.inference_mode():
            F_outputs_all, gs_out_all = forward_cascadeavatar_split_use_gs(
                single_avatar, cameras=test_camera, bg_color=bg_tensor,
                render_out_levels=[0, 1], batch_params=curr_batch_params,
                noise_size=None, canon=canon,
            )
        all_times.append(time.time() - start)

        Render_GS = len(gs_out_all) != 0
        GS2D = gs_out_all[-1] if Render_GS else None

        P_da_vts = F_outputs_all['Da_vts']
        P_posed_vts = F_outputs_all['posed_vts']
        render_vts = P_da_vts if canon else P_posed_vts

        F_face = single_avatar.mesh_avatars[single_avatar.curr_level].face
        refine_vts_rgb = face_to_vertex_color_batch(F_outputs_all['gs_finest']['rgb_refine'], F_face)
        corase_vts_rgb = face_to_vertex_color_batch(F_outputs_all['gs_finest']['rgb'], F_face)

        P_curr_mesh = Mesh(render_vts.squeeze(), F_outputs_all['faces'],
                           colors=torch.ones_like(render_vts.squeeze()), device=device)
        P_mesh_rendered_out = mesh_renderer.render_SH_batch_w_albedo_refine_rgb(
            P_curr_mesh, rgb=corase_vts_rgb, refine_rgb=refine_vts_rgb,
            cameras=test_camera, bg=list(bg), disp_dict=None,
        )
        P_normal = P_mesh_rendered_out['normal']

        # ---- Save mesh outputs (ply + json) ----
        P_curr_mesh.to_trimesh().export(os.path.join(save_folders['Mesh'], f"{i:04d}.ply"))
        save_ply(os.path.join(save_folders['GS'], f"{i:04d}.ply"),
                 render_vts.squeeze().detach().cpu().numpy(),
                 refine_vts_rgb.squeeze().detach().cpu().numpy())
        smpl_param_version = single_avatar.mesh_avatars[-1].full_pose_dict2smplx(curr_batch_params)
        save_dict_as_json(smpl_param_version, os.path.join(save_folders['smpl_param'], f"{i:04d}.json"))

        filename = f"{i:05d}.png"

        # ---- Mesh alpha (nvdiffrast fg_mask) → for Mesh_*/P_normal RGBA ----
        if 'mask' in P_mesh_rendered_out:
            mesh_alpha_u8 = _mask_to_uint8(P_mesh_rendered_out['mask'][0])
        else:
            h, w = P_normal.shape[2], P_normal.shape[3]
            mesh_alpha_u8 = np.full((h, w), 255, dtype=np.uint8)

        # ---- GS alpha (render_alpha from gaussian rasterizer) → for GS_* RGBA ----
        gs_alpha_u8 = None
        if Render_GS:
            if 'render_alpha' in GS2D:
                gs_alpha_u8 = _mask_to_uint8(GS2D['render_alpha'][0])
            elif 'render' in GS2D and GS2D['render'].shape[-1] == 4:
                gs_alpha_u8 = _mask_to_uint8(GS2D['render'][0][..., 3])

        # ---- 3-channel saves ----
        p_normal_np = normal_save_converter(P_normal[0].detach().cpu().numpy())
        p_normal_vis = (np.clip(p_normal_np, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_folders['P_normal'], filename), p_normal_vis)

        mesh_rgb = P_mesh_rendered_out['rgba'][0, ..., :3].detach().cpu().numpy()
        mesh_rgb_u8 = (np.clip(mesh_rgb, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_folders['Mesh_RGB'], filename), mesh_rgb_u8)

        mesh_coarse = P_mesh_rendered_out['rgba_coarse'][0, ..., :3].detach().cpu().numpy()
        mesh_coarse_u8 = (np.clip(mesh_coarse, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_folders['Mesh_albedo'], filename), mesh_coarse_u8)

        gs_rgb_u8 = gs_albedo_u8 = gs_surf_u8 = gs_rend_u8 = None
        if Render_GS:
            if 'rend_normal' in GS2D:
                gs_rend = GS2D['rend_normal'][0].detach().cpu().numpy()
                gs_rend_u8 = ((gs_rend + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_folders['GS_rndr_normal'], filename), gs_rend_u8)
            gs_surf = GS2D['surf_normal'][0].detach().cpu().numpy()
            gs_surf_u8 = ((gs_surf + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_folders['GS_surf_normal'], filename), gs_surf_u8)

            gs_rgb = GS2D['render'][0].detach().cpu().numpy()
            gs_rgb_3ch = gs_rgb[..., :3] if gs_rgb.shape[-1] == 4 else gs_rgb
            gs_rgb_u8 = (np.clip(gs_rgb_3ch, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_folders['GS_RGB'], filename), gs_rgb_u8)

            gs_albedo = GS2D['render_albedo'][0].detach().cpu().numpy()
            gs_albedo_3ch = gs_albedo[..., :3] if gs_albedo.shape[-1] == 4 else gs_albedo
            gs_albedo_u8 = (np.clip(gs_albedo_3ch, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_folders['GS_albedo'], filename), gs_albedo_u8)

        # ---- RGBA saves — mesh alpha for Mesh_*/P_normal, GS alpha for GS_* ----
        cv2.imwrite(os.path.join(save_folders['P_normal_rgba'], filename),
                    np.dstack((p_normal_vis, mesh_alpha_u8)))
        cv2.imwrite(os.path.join(save_folders['Mesh_RGB_rgba'], filename),
                    np.dstack((mesh_rgb_u8, mesh_alpha_u8)))
        cv2.imwrite(os.path.join(save_folders['Mesh_albedo_rgba'], filename),
                    np.dstack((mesh_coarse_u8, mesh_alpha_u8)))

        if Render_GS:
            _gs_alpha = gs_alpha_u8 if gs_alpha_u8 is not None else mesh_alpha_u8
            if gs_rend_u8 is not None:
                cv2.imwrite(os.path.join(save_folders['GS_rndr_normal_rgba'], filename),
                            np.dstack((gs_rend_u8, _gs_alpha)))
            cv2.imwrite(os.path.join(save_folders['GS_surf_normal_rgba'], filename),
                        np.dstack((gs_surf_u8, _gs_alpha)))
            cv2.imwrite(os.path.join(save_folders['GS_RGB_rgba'], filename),
                        np.dstack((gs_rgb_u8, _gs_alpha)))
            cv2.imwrite(os.path.join(save_folders['GS_albedo_rgba'], filename),
                        np.dstack((gs_albedo_u8, _gs_alpha)))

        if Render_GS:
            del GS2D
        del P_mesh_rendered_out

    if all_times:
        print(f"avg_time: {sum(all_times) / len(all_times):.4f} sec/frame")

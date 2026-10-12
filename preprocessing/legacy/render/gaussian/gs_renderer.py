#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
import numpy as np
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except:
    pass
from utils.graphics.sh_utils import eval_sh
# from scene.object.gaussian import GaussianModel
# import torch.nn as nn


class GSRenderer:

    def render(self, 
            gaussain,
            camera, 
            compute_cov3D_python=False,
            override_color=None,
            separate_sh=None,
            bg_color=[1.0, 1.0, 1.0],
            scaling_modifier=1.0,
            debug=False,
            convert_SHs_python=False #default option is white.
            ):
        screenspace_points = torch.zeros_like(gaussain.xyz, dtype=gaussain.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        fov_x = np.deg2rad(camera.FoVx)
        fov_y = np.deg2rad(camera.FoVy)
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(fov_y * 0.5)
        # camera.camera_center = camera.camera_center * -1.0
        # check device ( all inputs should be in cuda.)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform.cuda(), #w2c world to camera matrix
            projmatrix=camera.full_proj_transform.cuda(),
            sh_degree=gaussain.active_sh_degree,
            campos=camera.camera_center.cuda(),
            prefiltered=False,
            debug=debug,
            antialiasing=True # - Gaussian Avatar's rasterizer does not operate mip-splatting(anti-aliasing)
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gaussain.xyz
        means2D = screenspace_points
        opacity = gaussain.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None

        if compute_cov3D_python:
            cov3D_precomp = gaussain.get_covariance(scaling_modifier)
        else:
            scales = gaussain.scaling
            rotations = gaussain.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = gaussain.features.transpose(1, 2).view(-1, 3, (gaussain.max_sh_degree+1)**2)
                dir_pp = (gaussain.xyz - camera.camera_center.repeat(gaussain.features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussain.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if separate_sh:
                    dc, shs = gaussain.features_dc, gaussain.features_rest
                else:
                    shs = gaussain.features
        else:
            colors_precomp = self.override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        if separate_sh:
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                dc = dc,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        else:
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        #print("mean:", scales.mean())
        # Apply exposure to rendered image (training only)
        #if use_trained_exp:
        #    exposure = gaussain.get_exposure_from_name(camera.image_name)
        #    rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        #rendered_image = rendered_image.clamp(0, 1)
        out = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            #"depth" : depth_image
            }
        
        return out




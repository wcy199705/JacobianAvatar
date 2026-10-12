# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import torch
import numpy as np
import nvdiffrast.torch as dr
from typing import Tuple, Literal
from core.math_ops import vector_ops as V
from core.mesh import Mesh, TexturedMesh


class NVDiffRenderer:
    def __init__(
            self,
            use_opengl: bool = False, 
            lighting_type: Literal['constant', 'front'] = 'front',
            lighting_space: Literal['camera', 'world'] = 'camera',
            device=torch.device("cuda:0")
        ):
        super().__init__()
        self.device = device
        self.use_opengl = use_opengl
        self.lighting_type = lighting_type
        self.lighting_space = lighting_space
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    def mvp_from_camera_param(self, RT, K, image_size):
        # projection matrix
        proj = self.projection_from_intrinsics(K, image_size)

        # Modelview and modelview + projection matrices.
        if RT.shape[-2] == 3:
            mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
            mv[..., 3, 3] = 1
        elif RT.shape[-2] == 4:
            mv = RT
        mvp = torch.bmm(proj, mv)
        return mvp
    
    def projection_from_intrinsics(self, K: torch.Tensor, image_size: Tuple[int], near: float=0.1, far:float=10):
        """
        Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
        Args:
            K: Intrinsic matrix, (N, 3, 3)
                K = [[
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1],
                    ]
                ]
            image_size: (height, width)
        Output:
            proj = [[
                    [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                    [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                    [0.0,    0.0,     -(far+near) / (far-near), -2*far*near / (far-near)],
                    [0.0,    0.0,     -1.0,                     0.0                     ]
                ]
            ]
        """

        B = K.shape[0]
        h, w = image_size

        if K.shape[-2:] == (3, 3):
            fx = K[..., 0, 0]
            fy = K[..., 1, 1]
            cx = K[..., 0, 2]
            cy = K[..., 1, 2]
        elif K.shape[-1] == 4:
            fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        else:
            raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

        proj = torch.zeros([B, 4, 4], device=K.device)
        proj[:, 0, 0]  = fx * 2 / w 
        proj[:, 1, 1]  = fy * 2 / h
        proj[:, 0, 2]  = (w - 2 * cx) / w
        proj[:, 1, 2]  = (h - 2 * cy) / h
        proj[:, 2, 2]  = -(far+near) / (far-near)
        proj[:, 2, 3]  = -2*far*near / (far-near)
        proj[:, 3, 2]  = -1
        return proj
    
    
    def world_to_clip(self, vtx, RT, K, image_size, mvp=None):
        """Transform vertex positions from the world space to the clip space"""
        if mvp == None:
            mvp = self.mvp_from_camera_param(RT, K, image_size)

        mvp = torch.from_numpy(mvp).cuda() if isinstance(mvp, np.ndarray) else mvp
        # (x,y,z) -> (x',y',z',w)
        posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        return torch.bmm(posw, mvp.transpose(-1, -2))
    
    def shade(self, normal, lighting_coeff=None):
        if self.lighting_type == 'constant':
            diffuse = torch.ones_like(normal[..., :3])
        elif self.lighting_type == 'front':
            diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')), 0.0, 1.0)
        else:
            raise NotImplementedError(f"Unknown lighting type: {self.lighting_type}")
        return diffuse

    def get_vert_visibility(self, mesh, camera):
        verts = mesh.xyz if len(mesh.xyz.shape) == 3 else mesh.xyz[None, ...]
        
        faces = mesh.face.int()
        num_view, num_verts = verts.shape[0], verts.shape[1]
        vis_mask = torch.zeros(size=(num_view, num_verts), device=self.device).bool()  # visibility mask
        mvp_batch = torch.stack([c.mvp.squeeze(0) for c in camera], dim=0).contiguous()  # [n_views, 4, 4]

        with torch.no_grad():
            verts_clip = self.world_to_clip(verts, None, None, camera[0].image_size, mvp=mvp_batch)
            rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, camera[0].image_size)

            face_ids = rast_out[..., -1].long()
            for v in range(num_view):
                masked_face_idxs = face_ids[v, face_ids[v] != 0] - 1  # num_masked_face Tensor
                masked_vert_idxs = faces[masked_face_idxs].long()
                vis_mask[v, masked_vert_idxs] = 1
        return vis_mask


    #memory leakage
    


    def render_batch(self, mesh, cameras, bg=[1., 1., 1.]):
        """
        Renders meshes into RGBA images
        Need to be updated.
        """
        verts = mesh.xyz
        faces = mesh.face.int()
        rgb = mesh.rgb
        norm_xyz = mesh.norm_xyz

        # Expand to batch dimension
        n_views = len(cameras)
        if verts.shape[0] != n_views:
            # verts_batch = verts[None, ...].expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            verts_batch = verts.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            verts_batch = verts.contiguous()

        if rgb is not None:
            if rgb.shape[0] != n_views:
                rgb_batch = rgb.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                rgb_batch = rgb.contiguous()

        if norm_xyz.shape[0] != n_views:
            norm_xyz_batch = norm_xyz.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            norm_xyz_batch = norm_xyz.contiguous()

        rot_batch = torch.stack([c.RT.squeeze(0)[:3,:3] for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]
        view_space_normals = torch.matmul(norm_xyz_batch, rot_batch.transpose(1, 2))
        mvp_batch = torch.stack([c.mvp.squeeze(0) for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]

        # if isinstance(mesh, TexturedMesh):
        #     uv = mesh.uv
        #     tex = mesh.tex
        # elif isinstance(mesh, Mesh):
        #     rgb = mesh.rgb
        # else:
        #     assert "only support Mesh or TexturedMesh classes"
        image_size=cameras[0].image_size

        # Transform to clip space
        
        verts_clip = self.world_to_clip(verts_batch, None, None, image_size, mvp=mvp_batch)  # [n_views, n_verts, 4]
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, image_size)
        """

            x_img = (verts_clip[...,0] + 1) * 0.5 * cameras[0].image_width
            y_img = (verts_clip[...,1] + 1) * 0.5 * cameras[0].image_height

        """
        # Foreground mask
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1)  # [n_views, h, w, 1]
        fg_mask = dr.antialias(fg_mask, rast_out, verts_clip, faces)
        fg_bool = fg_mask.clone().bool() #leage on cloning

        # Render normal
        #normal, _ = dr.interpolate(norm_xyz_batch, rast_out, faces)  # [n_views, h, w, 3]
        view_space_normal_map, _ = dr.interpolate(view_space_normals, rast_out, faces)
        #opengl->opencv
        view_space_normal_map[...,1:] = - view_space_normal_map[...,1:]

        # Render position (depth)

        position, _ = dr.interpolate(verts_clip, rast_out, faces)
        position = dr.antialias(position, rast_out, verts_clip, faces)

        

        # Render albedo
        albedo, _ = dr.interpolate(rgb_batch, rast_out, faces)  # [n_views, h, w, 3]
        # if isinstance(mesh, TexturedMesh):
        #     max_mip_level = 9
        #     texc, texd = dr.interpolate(uv_batch, rast_out, faces_batch, rast_db=rast_out_db, diff_attrs='all')
        #     albedo = dr.texture(tex_batch, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        # else:
        #     albedo, _ = dr.interpolate(rgb_batch, rast_out, faces_batch)  # [n_views, h, w, 3]

        # Apply background
        bg_tensor = torch.tensor(bg, device=self.device).expand_as(albedo) if isinstance(bg, list) else bg
        albedo = torch.where(rast_out[..., 3:] > 0, albedo, bg_tensor)

        # Shading
        diffuse = self.shade(view_space_normal_map)  # [n_views, h, w, 1]

        # Combine RGB and alpha
        rgb_ = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb_, alpha], dim=-1)  # [n_views, h, w, 4]

        # Handle background
        if isinstance(bg, list):
            rgba_bg = torch.tensor(bg + [0], device=self.device).expand_as(rgba)  # [n_views, h, w, 4]
        elif isinstance(bg, torch.Tensor):
            rgba_bg = torch.cat([bg, torch.zeros_like(bg[..., :1])], dim=-1)  # [n_views, h, w, 4]
        else:
            raise ValueError(f"Unknown background type: {type(bg)}")
        rgba_bg = rgba_bg.flip(1)  # flip y-axis

        # Apply mask
        view_space_normal_map = torch.where(fg_bool, view_space_normal_map, rgba_bg[..., :3])
        diffuse = torch.where(fg_bool, diffuse, rgba_bg[..., :3])
        rgba = torch.where(fg_bool, rgba, rgba_bg)

        # Anti-aliasing

        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces)
        #normal = dr.antialias(normal, rast_out, verts_clip, faces)

        view_space_normal_map = dr.antialias(view_space_normal_map, rast_out, verts_clip, faces)
        view_space_normal_map = view_space_normal_map/(torch.norm(view_space_normal_map,dim=-1)[...,None]+1e-8)

        view_space_normal_map = (view_space_normal_map + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        #n#ormal = (normal + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        #normal = normal/(torch.norm(normal,dim=-1)[...,None]+1e-8)

        torch.cuda.empty_cache()

        return {
            'albedo': albedo.flip(1),
            'normal': view_space_normal_map.flip(1),
            'diffuse': diffuse.flip(1),
            'rgba': rgba_aa.flip(1),
            'depth': position.flip(1),
            'mask': fg_mask.flip(1)
        }

    def render_SH_batch(self, mesh, cameras, bg=[1., 1., 1.], return_depth=False, use_rgb=False, vts_attributes=None, disp_dict=None, w_albedo=False):
        """
        Renders meshes into RGBA images
        Need to be updated.
        """
        #import time
        #start = time.time()

        from core.graphics.sh_utils import eval_sh
        import math
        verts = mesh.xyz
        faces = mesh.face.int()
        sh = mesh.rgb
        #sh: [..., C, (deg + 1) ** 2]
        sh_dim = sh.shape[-1]
        sh_degree = int(math.sqrt(sh_dim/3)) -1 
        #assert sh.shape[-1] != 3
        norm_xyz = mesh.norm_xyz
        H,W = cameras[0].image_size
        #view_dir_map = get_rays_cam_space_pytorch(H=H,W=W,K=cameras[0].K)
        # Expand to batch dimension
        n_views = len(cameras)
        if verts.shape[0] != n_views:
            # verts_batch = verts[None, ...].expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            verts_batch = verts.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            verts_batch = verts.contiguous()

        if sh is not None:
            if sh.shape[0] != n_views:
                sh_batch = sh.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                sh_batch = sh.contiguous()

        if norm_xyz.shape[0] != n_views:
            norm_xyz_batch = norm_xyz.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            norm_xyz_batch = norm_xyz.contiguous()

        if vts_attributes is not None:
            if vts_attributes.shape[0] != n_views:
                 vts_attributes= vts_attributes.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                vts_attributes = vts_attributes.contiguous()

        #get norm_xyz for viewing direction.

        if disp_dict is not None : #or len(disp_dict.keys()!=0):
            disp_dict['disp'] = disp_dict['disp'].contiguous()

        
        rot_batch = torch.stack([c.RT.squeeze(0)[:3,:3] for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]
        view_space_normals = torch.matmul(norm_xyz_batch, rot_batch.transpose(1, 2))
        mvp_batch = torch.stack([c.mvp.squeeze(0) for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]

        # if isinstance(mesh, TexturedMesh):
        #     uv = mesh.uv
        #     tex = mesh.tex
        # elif isinstance(mesh, Mesh):
        #     rgb = mesh.rgb
        # else:
        #     assert "only support Mesh or TexturedMesh classes"
        image_size=cameras[0].image_size

        # Transform to clip space
        
        verts_clip = self.world_to_clip(verts_batch, None, None, image_size, mvp=mvp_batch)  # [n_views, n_verts, 4]
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, image_size)
        """

            x_img = (verts_clip[...,0] + 1) * 0.5 * cameras[0].image_width
            y_img = (verts_clip[...,1] + 1) * 0.5 * cameras[0].image_height

        """
        # Foreground mask
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1)  # [n_views, h, w, 1]
        fg_mask = dr.antialias(fg_mask, rast_out, verts_clip, faces)
        fg_bool = fg_mask.clone().bool() #leage on cloning

        # Render normal
        #normal, _ = dr.interpolate(norm_xyz_batch, rast_out, faces)  # [n_views, h, w, 3]
        view_space_normal_map, _ = dr.interpolate(view_space_normals, rast_out, faces)
        #opengl->opencv
        view_space_normal_map[...,1:] = - view_space_normal_map[...,1:]
        
        # Render position (depth)
        if return_depth:
            position, _ = dr.interpolate(verts_clip, rast_out, faces)
            position = dr.antialias(position, rast_out, verts_clip, faces)
            position = position.flip(1)
        else: 
            position = None
    
        # Render vts_attributes
        if vts_attributes is not None:
            render_attribute,_ =dr.interpolate(vts_attributes, rast_out, faces) 
        if disp_dict is not None:
            displacement,_ = dr.interpolate(disp_dict['disp'], rast_out[disp_dict['curr_idxs']], faces) 

            #render_attribute = torch.clamp(render_attribute,min=0.0,max=1.0)

        # Render albedo
        
        sh, _ = dr.interpolate(sh_batch, rast_out, faces)  # [n_views, h, w, (sh_dim+1)**2 *3 ]
        
        # if isinstance(mesh, TexturedMesh):
        #     max_mip_level = 9
        #     texc, texd = dr.interpolate(uv_batch, rast_out, faces_batch, rast_db=rast_out_db, diff_attrs='all')
        #     albedo = dr.texture(tex_batch, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        # else:
        #     albedo, _ = dr.interpolate(rgb_batch, rast_out, faces_batch)  # [n_views, h, w, 3]

        # Apply background
        bg_tensor = torch.zeros_like(sh)
        bg_tensor[...,:3] = torch.tensor(bg, device=self.device)
        #bg_tensor = torch.tensor(bg, device=self.device).expand_as(sh[...,:3]) if isinstance(bg, list) else bg
        sh = torch.where(rast_out[..., 3:] > 0, sh, bg_tensor)

        # Shading
        #diffuse = self.shade(normal)  # [n_views, h, w, 1]

        # Combine RGB and alpha
        #check values (ray directions..)
        #import cv2
        #cv2.imwrite('view_normal.png',(view_space_normal_map+1.0).squeeze().detach().cpu().numpy()*127.5)
        #v2.imwrite('global_normal.png',(normal+1.0).squeeze().detach().cpu().numpy()*127.5)            
        
        if use_rgb:
            rgb_ = sh.view(view_space_normal_map.shape[0],H,W,3)
            rgb_ = torch.clamp(rgb_, 0.0, 1.0)
        else:
            rgb_ = eval_sh(deg=sh_degree,sh=sh.view(view_space_normal_map.shape[0],H,W,sh_dim//3,3).transpose(-1,-2),dirs=view_space_normal_map) 
        alpha = fg_mask.float()
        rgba = torch.cat([rgb_, alpha], dim=-1)  # [n_views, h, w, 4]

        # Handle background
        if isinstance(bg, list):
            rgba_bg = torch.tensor(bg + [0], device=self.device).expand_as(rgba)  # [n_views, h, w, 4]
        elif isinstance(bg, torch.Tensor):
            rgba_bg = torch.cat([bg, torch.zeros_like(bg[..., :1])], dim=-1)  # [n_views, h, w, 4]
        else:
            raise ValueError(f"Unknown background type: {type(bg)}")
        rgba_bg = rgba_bg.flip(1)  # flip y-axis

        # Apply mask
        #normal = torch.where(fg_bool, normal, rgba_bg[..., :3])
        #diffuse = torch.where(fg_bool, diffuse, rgba_bg[..., :3])
        rgba = torch.where(fg_bool, rgba, rgba_bg)

        # Anti-aliasing

        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces)
        if w_albedo:
            albedo =eval_sh(deg=0,sh=sh.view(view_space_normal_map.shape[0],H,W,sh_dim//3,3).transpose(-1,-2)[...,0:1],dirs=None) 
            albedo = torch.where(fg_bool, albedo, rgba_bg[...,:3])
            albedo = dr.antialias(albedo, rast_out, verts_clip, faces)

        else:
            albedo = rgba_aa[...,:3]

        view_space_normal_map = dr.antialias(view_space_normal_map, rast_out, verts_clip, faces)
        view_space_normal_map = view_space_normal_map/(torch.norm(view_space_normal_map,dim=-1)[...,None]+1e-8)

        view_space_normal_map = (view_space_normal_map + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        #print("renderingtime:", time.time()- start)
        torch.cuda.empty_cache()
        output = {
                'albedo': albedo.flip(1),
                #'view_normal_cv2': view_space_normal_map.flip(1),
                'normal': view_space_normal_map.flip(1),
                #'diffuse': diffuse.flip(1),
                'rgba': rgba_aa.flip(1),
                'depth': position,
                'mask': fg_mask.flip(1),
            }
        if return_depth:
            output['depth_cy'] = position[...,2:3]
        if vts_attributes is not None:
            output['attribute'] = render_attribute.flip(1)
        if disp_dict is not None:
            output['displacement'] = displacement.flip(1)

        return output


    def render_SH_batch_w_albedo_refine_rgb(self, mesh, rgb, refine_rgb, cameras, bg=[1., 1., 1.], return_depth=False,  vts_attributes=None, disp_dict=None):
        """
        Renders meshes into RGBA images
        Need to be updated.
        """
        #import time
        #start = time.time()

        verts = mesh.xyz
        faces = mesh.face.int()
        #sh: [..., C, (deg + 1) ** 2]
        sh_dim = 3  #sh.shape[-1]
        sh_degree = 0 # int(math.sqrt(sh_dim/3)) -1 
        #assert sh.shape[-1] != 3
        norm_xyz = mesh.norm_xyz
        H,W = cameras[0].image_size
        #view_dir_map = get_rays_cam_space_pytorch(H=H,W=W,K=cameras[0].K)
        # Expand to batch dimension
        n_views = len(cameras)
        if verts.shape[0] != n_views:
            # verts_batch = verts[None, ...].expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            verts_batch = verts.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            verts_batch = verts.contiguous()

        if rgb is not None:
            if rgb.shape[0] != n_views:
                rgb_batch = rgb.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                rgb_batch = rgb.contiguous()

        if refine_rgb is not None:
            if refine_rgb.shape[0] != n_views:
                refine_rgb_batch = refine_rgb.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                refine_rgb_batch = refine_rgb.contiguous()


        if norm_xyz.shape[0] != n_views:
            norm_xyz_batch = norm_xyz.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            norm_xyz_batch = norm_xyz.contiguous()

        if vts_attributes is not None:
            if vts_attributes.shape[0] != n_views:
                 vts_attributes= vts_attributes.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                vts_attributes = vts_attributes.contiguous()

        #get norm_xyz for viewing direction.

        if disp_dict is not None : #or len(disp_dict.keys()!=0):
            disp_dict['disp'] = disp_dict['disp'].contiguous()

        
        rot_batch = torch.stack([c.RT.squeeze(0)[:3,:3] for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]
        view_space_normals = torch.matmul(norm_xyz_batch, rot_batch.transpose(1, 2))
        mvp_batch = torch.stack([c.mvp.squeeze(0) for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]

        # if isinstance(mesh, TexturedMesh):
        #     uv = mesh.uv
        #     tex = mesh.tex
        # elif isinstance(mesh, Mesh):
        #     rgb = mesh.rgb
        # else:
        #     assert "only support Mesh or TexturedMesh classes"
        image_size=cameras[0].image_size

        # Transform to clip space
        
        verts_clip = self.world_to_clip(verts_batch, None, None, image_size, mvp=mvp_batch)  # [n_views, n_verts, 4]
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, image_size)
        """

            x_img = (verts_clip[...,0] + 1) * 0.5 * cameras[0].image_width
            y_img = (verts_clip[...,1] + 1) * 0.5 * cameras[0].image_height

        """
        # Foreground mask
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1)  # [n_views, h, w, 1]
        fg_mask = dr.antialias(fg_mask, rast_out, verts_clip, faces)
        fg_bool = fg_mask.clone().bool() #leage on cloning

        # Render normal
        #normal, _ = dr.interpolate(norm_xyz_batch, rast_out, faces)  # [n_views, h, w, 3]
        view_space_normal_map, _ = dr.interpolate(view_space_normals, rast_out, faces)
        #opengl->opencv
        view_space_normal_map[...,1:] = - view_space_normal_map[...,1:]
        
        # Render position (depth)
        if return_depth:
            position, _ = dr.interpolate(verts_clip, rast_out, faces)
            position = dr.antialias(position, rast_out, verts_clip, faces)
            position = position.flip(1)
        else: 
            position = None
    
        # Render vts_attributes
        if vts_attributes is not None:
            render_attribute,_ =dr.interpolate(vts_attributes, rast_out, faces) 
        if disp_dict is not None:
            displacement,_ = dr.interpolate(disp_dict['disp'], rast_out[disp_dict['curr_idxs']], faces) 

            #render_attribute = torch.clamp(render_attribute,min=0.0,max=1.0)

        # Render albedo
        
        render_rgb_refine, _ = dr.interpolate(refine_rgb_batch, rast_out, faces)  # [n_views, h, w, (sh_dim+1)**2 *3 ]
        render_rgb, _ = dr.interpolate(rgb_batch, rast_out, faces)  # [n_views, h, w, (sh_dim+1)**2 *3 ]

        # if isinstance(mesh, TexturedMesh):
        #     max_mip_level = 9
        #     texc, texd = dr.interpolate(uv_batch, rast_out, faces_batch, rast_db=rast_out_db, diff_attrs='all')
        #     albedo = dr.texture(tex_batch, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        # else:
        #     albedo, _ = dr.interpolate(rgb_batch, rast_out, faces_batch)  # [n_views, h, w, 3]

        # Apply background
        bg_tensor = torch.zeros_like(render_rgb_refine)
        bg_tensor[...,:3] = torch.tensor(bg, device=self.device)
        #bg_tensor = torch.tensor(bg, device=self.device).expand_as(sh[...,:3]) if isinstance(bg, list) else bg
        render_rgb_refine = torch.where(rast_out[..., 3:] > 0, render_rgb_refine, bg_tensor)
        render_rgb = torch.where(rast_out[..., 3:] > 0, render_rgb, bg_tensor)

        # Shading
        #diffuse = self.shade(normal)  # [n_views, h, w, 1]

        # Combine RGB and alpha
        #check values (ray directions..)
        #import cv2
        #cv2.imwrite('view_normal.png',(view_space_normal_map+1.0).squeeze().detach().cpu().numpy()*127.5)
        #v2.imwrite('global_normal.png',(normal+1.0).squeeze().detach().cpu().numpy()*127.5)            
        
        render_rgb_refine = render_rgb_refine.view(view_space_normal_map.shape[0],H,W,3)
        render_rgb_refine = torch.clamp(render_rgb_refine, 0.0, 1.0)
        render_rgb = render_rgb.view(view_space_normal_map.shape[0],H,W,3)
        render_rgb = torch.clamp(render_rgb, 0.0, 1.0)

        alpha = fg_mask.float()
        rgba_coarse = torch.cat([render_rgb, alpha], dim=-1)  # [n_views, h, w, 4]

        rgba_refine = torch.cat([render_rgb_refine, alpha], dim=-1)  # [n_views, h, w, 4]

        # Handle background
        if isinstance(bg, list):
            rgba_bg = torch.tensor(bg + [0], device=self.device).expand_as(rgba_coarse)  # [n_views, h, w, 4]
        elif isinstance(bg, torch.Tensor):
            rgba_bg = torch.cat([bg, torch.zeros_like(bg[..., :1])], dim=-1)  # [n_views, h, w, 4]
        else:
            raise ValueError(f"Unknown background type: {type(bg)}")
        rgba_bg = rgba_bg.flip(1)  # flip y-axis

        # Apply mask
        #normal = torch.where(fg_bool, normal, rgba_bg[..., :3])
        #diffuse = torch.where(fg_bool, diffuse, rgba_bg[..., :3])
        rgba_coarse = torch.where(fg_bool, rgba_coarse, rgba_bg)
        rgba_refine = torch.where(fg_bool, rgba_refine, rgba_bg)

        # Anti-aliasing

        rgba_coarse_aa = dr.antialias(rgba_coarse, rast_out, verts_clip, faces)
        rgba_refine_aa = dr.antialias(rgba_refine, rast_out, verts_clip, faces)

        view_space_normal_map = dr.antialias(view_space_normal_map, rast_out, verts_clip, faces)
        view_space_normal_map = view_space_normal_map/(torch.norm(view_space_normal_map,dim=-1)[...,None]+1e-8)

        view_space_normal_map = (view_space_normal_map + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        #print("renderingtime:", time.time()- start)
        torch.cuda.empty_cache()
        output = {
                'rgba_coarse': rgba_coarse_aa.flip(1),
                'rgba': rgba_refine_aa.flip(1),

                #'view_normal_cv2': view_space_normal_map.flip(1),
                'normal': view_space_normal_map.flip(1),
                #'diffuse': diffuse.flip(1),
                'depth': position,
                'mask': fg_mask.flip(1),
            }
        if return_depth:
            output['depth_cy'] = position[...,2:3]
        if vts_attributes is not None:
            output['attribute'] = render_attribute.flip(1)
        if disp_dict is not None:
            output['displacement'] = displacement.flip(1)

        return output


    def render_SH_batch_w_depth(self, mesh, cameras, bg=[1., 1., 1.], w_albedo=False):
        """
        Renders meshes into RGBA images
        Need to be updated.
        """
        #import time
        #start = time.time()

        from core.graphics.sh_utils import eval_sh
        import math
        verts = mesh.xyz
        faces = mesh.face.int()
        sh = mesh.rgb
        #sh: [..., C, (deg + 1) ** 2]
        sh_dim = sh.shape[-1]
        sh_degree = int(math.sqrt(sh_dim/3)) -1 
        #assert sh.shape[-1] != 3
        norm_xyz = mesh.norm_xyz
        H,W = cameras[0].image_size
        #view_dir_map = get_rays_cam_space_pytorch(H=H,W=W,K=cameras[0].K)
        # Expand to batch dimension
        n_views = len(cameras)
        if verts.shape[0] != n_views:
            # verts_batch = verts[None, ...].expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            verts_batch = verts.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            verts_batch = verts.contiguous()

        if sh is not None:
            if sh.shape[0] != n_views:
                sh_batch = sh.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
            else:
                sh_batch = sh.contiguous()

        if norm_xyz.shape[0] != n_views:
            norm_xyz_batch = norm_xyz.expand(n_views, -1, -1).contiguous()  # [n_views, n_verts, 3]
        else:
            norm_xyz_batch = norm_xyz.contiguous()
        #get norm_xyz for viewing direction.
        rot_batch = torch.stack([c.RT.squeeze(0)[:3,:3] for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]
        view_space_normals = torch.matmul(norm_xyz_batch, rot_batch.transpose(1, 2))
        mvp_batch = torch.stack([c.mvp.squeeze(0) for c in cameras], dim=0).contiguous()  # [n_views, 4, 4]

        # if isinstance(mesh, TexturedMesh):
        #     uv = mesh.uv
        #     tex = mesh.tex
        # elif isinstance(mesh, Mesh):
        #     rgb = mesh.rgb
        # else:
        #     assert "only support Mesh or TexturedMesh classes"
        image_size=cameras[0].image_size

        # Transform to clip space
        
        verts_clip = self.world_to_clip(verts_batch, None, None, image_size, mvp=mvp_batch)  # [n_views, n_verts, 4]
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, image_size)
        """

            x_img = (verts_clip[...,0] + 1) * 0.5 * cameras[0].image_width
            y_img = (verts_clip[...,1] + 1) * 0.5 * cameras[0].image_height

        """
        # Foreground mask
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1)  # [n_views, h, w, 1]
        fg_mask = dr.antialias(fg_mask, rast_out, verts_clip, faces)
        fg_bool = fg_mask.clone().bool() #leage on cloning

        # Render normal
        #normal, _ = dr.interpolate(norm_xyz_batch, rast_out, faces)  # [n_views, h, w, 3]
        view_space_normal_map, _ = dr.interpolate(view_space_normals, rast_out, faces)
        #opengl->opencv
        view_space_normal_map[...,1:] = - view_space_normal_map[...,1:]
        
        # Render position (depth)

        position, _ = dr.interpolate(verts_clip, rast_out, faces)
        position = dr.antialias(position, rast_out, verts_clip, faces)

        
        #depth_.squeeze()
        depth_ = verts_batch[..., 2:3].squeeze()
        if len(depth_.shape) == 2:
            depth_ = depth_[...,None]
        else:
            depth_ = depth_[None,:,None]
        #[None,:,None]
        #depth = torch.Tensor([[[(z_val/1)] for z_val in depth_.squeeze()]], dtype=torch.float32)
        depth_ = depth_.contiguous()
        depth, _ = dr.interpolate(depth_, rast_out, faces)
        #depth = depth.flip(1)
        #cv2.imwrite('depthmap.png',depth.cpu().squeeze().numpy()*30)
        # Render albedo 

        # Render albedo
        sh, _ = dr.interpolate(sh_batch, rast_out, faces)  # [n_views, h, w, (sh_dim+1)**2 *3 ]
        
        # if isinstance(mesh, TexturedMesh):
        #     max_mip_level = 9
        #     texc, texd = dr.interpolate(uv_batch, rast_out, faces_batch, rast_db=rast_out_db, diff_attrs='all')
        #     albedo = dr.texture(tex_batch, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        # else:
        #     albedo, _ = dr.interpolate(rgb_batch, rast_out, faces_batch)  # [n_views, h, w, 3]

        # Apply background
        bg_tensor = torch.zeros_like(sh)
        bg_tensor[...,:3] = torch.tensor(bg, device=self.device)
        #bg_tensor = torch.tensor(bg, device=self.device).expand_as(sh[...,:3]) if isinstance(bg, list) else bg
        sh = torch.where(rast_out[..., 3:] > 0, sh, bg_tensor)

        # Shading
        #diffuse = self.shade(normal)  # [n_views, h, w, 1]

        # Combine RGB and alpha
        #check values (ray directions..)
        #import cv2
        #cv2.imwrite('view_normal.png',(view_space_normal_map+1.0).squeeze().detach().cpu().numpy()*127.5)
        #v2.imwrite('global_normal.png',(normal+1.0).squeeze().detach().cpu().numpy()*127.5)            
        rgb_ = eval_sh(deg=sh_degree,sh=sh.view(view_space_normal_map.shape[0],H,W,sh_dim//3,3).transpose(-1,-2),dirs=view_space_normal_map) 
        alpha = fg_mask.float()
        rgba = torch.cat([rgb_, alpha], dim=-1)  # [n_views, h, w, 4]

        # Handle background
        if isinstance(bg, list):
            rgba_bg = torch.tensor(bg + [0], device=self.device).expand_as(rgba)  # [n_views, h, w, 4]
        elif isinstance(bg, torch.Tensor):
            rgba_bg = torch.cat([bg, torch.zeros_like(bg[..., :1])], dim=-1)  # [n_views, h, w, 4]
        else:
            raise ValueError(f"Unknown background type: {type(bg)}")
        rgba_bg = rgba_bg.flip(1)  # flip y-axis

        # Apply mask
        #normal = torch.where(fg_bool, normal, rgba_bg[..., :3])
        #diffuse = torch.where(fg_bool, diffuse, rgba_bg[..., :3])
        rgba = torch.where(fg_bool, rgba, rgba_bg)

        # Anti-aliasing

        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces)
        if w_albedo:
            albedo =eval_sh(deg=0,sh=sh.view(view_space_normal_map.shape[0],H,W,sh_dim//3,3).transpose(-1,-2)[...,0:1],dirs=None) 
            albedo = torch.where(fg_bool, albedo, rgba_bg[...,:3])
            albedo = dr.antialias(albedo, rast_out, verts_clip, faces)

        else:
            albedo = rgba_aa[...,:3]

        view_space_normal_map = dr.antialias(view_space_normal_map, rast_out, verts_clip, faces)
        view_space_normal_map = view_space_normal_map/(torch.norm(view_space_normal_map,dim=-1)[...,None]+1e-8)

        view_space_normal_map = (view_space_normal_map + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        #print("renderingtime:", time.time()- start)
        torch.cuda.empty_cache()

        return {
            'albedo': albedo.flip(1),
            #'view_normal_cv2': view_space_normal_map.flip(1),
            'normal': view_space_normal_map.flip(1),
            #'diffuse': diffuse.flip(1),
            'rgba': rgba_aa.flip(1),
            'depth': depth.flip(1),
            'mask': fg_mask.flip(1)
        }


    def render_mask(self, mesh, camera, verts=None):
        """
        Renders meshes into RGBA images
        Need to be updated.
        """
        if verts is None:
            verts = mesh._xyz[None, ...]

        faces = mesh.face.int()
        verts_clip = self.world_to_clip(mesh._xyz[None, ...],None, None, camera.image_size, mvp=camera.mvp)
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, camera.image_size)
        # fg_mask = torch.clamp(rast_out[..., -1:], 0, 1)

        albedo, _ = dr.interpolate(mesh._rgb[None, ...], rast_out, faces)
        albedo = torch.sum(albedo, dim=3, keepdim=True)
        return albedo.flip(1)

    def render(self, mesh, camera, bg=[1., 1., 1.]):
        """
        Renders meshes into RGBA images
        Need to be updated.
        """
        verts = mesh.xyz
        if len(verts.shape) == 3:
            verts = verts.squeeze(0)

        faces = mesh.face.int()
        rgb = mesh.rgb

        if isinstance(mesh, TexturedMesh):
            uv = mesh.uv
            tex = mesh.tex
        elif isinstance(mesh, Mesh):
            rgb = mesh.rgb
        else:
            assert "only support Mesh or TexturedMesh classes"            
        verts_clip = self.world_to_clip(verts[None, ...], None, None, camera.image_size, mvp=camera.mvp)
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, faces, camera.image_size)

        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        fg_mask = dr.antialias(fg_mask, rast_out, verts_clip, faces) # if with_antialiasing else mask[0]
        fg_bool = fg_mask.clone().bool()
        normal, _ = dr.interpolate(mesh.norm_xyz[None, ...], rast_out, faces)

        # return normal
        position, _ = dr.interpolate(verts_clip, rast_out, faces)
        position = dr.antialias(position, rast_out, verts_clip, faces)
        # gbuffer["depth"] = camera.project(position, depth_as_distance=True)[..., 2:3]

        if isinstance(mesh, TexturedMesh):
            max_mip_level = 9
            texc, texd = dr.interpolate(uv[None, ...], rast_out, faces, rast_db=rast_out_db, diff_attrs='all')
            albedo = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear',
                                max_mip_level=max_mip_level)

        elif isinstance(mesh, Mesh):
            # per-vertex coloring
            albedo, _ = dr.interpolate(rgb[None, ...], rast_out, faces)
        else:
            albedo = torch.ones_like(normal)

        albedo = torch.where(rast_out[..., 3:] > 0, albedo, torch.tensor(bg).to(self.device))
        # ---- shading ----
        diffuse = self.shade(normal)

        rgb = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(bg, list):
            """Background as a constant color"""
            rgba_bg = torch.tensor(bg + [0]).to(rgba).expand_as(rgba)  # RGBA
        elif isinstance(bg, torch.Tensor):
            """Background as a image"""
            rgba_bg = bg
            rgba_bg = torch.cat([rgba_bg, torch.zeros_like(rgba_bg[..., :1])], dim=-1)  # RGBA
        else:
            raise ValueError(f"Unknown background type: {type(bg)}")
        rgba_bg = rgba_bg.flip(1)  # opengl camera has y-axis up, needs flipping

        normal = torch.where(fg_bool, normal, rgba_bg[..., :3])
        diffuse = torch.where(fg_bool, diffuse, rgba_bg[..., :3])
        rgba = torch.where(fg_bool, rgba, rgba_bg)

        # ---- AA on both RGB and alpha channels ----
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())
        normal = (normal + 1.0) / 2.0  # [-1, 1] -> [0, 1] to save the image
        normal = dr.antialias(normal, rast_out, verts_clip, faces)

        return {
            'albedo': albedo.flip(1),
            'normal': normal.flip(1),
            'diffuse': diffuse.flip(1),
            'rgba': rgba_aa.flip(1),
            'depth': position.flip(1), # do we need depth?
            'mask': fg_mask.flip(1)
        }

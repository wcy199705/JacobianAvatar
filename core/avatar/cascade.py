import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import copy
import math
from core.gaussian_utils import *

# Import your custom modules
from core.body_model.smplx import AvatarMesh
from core.neural.mlp_base import UniversalMLP 
from core.neural.triplane import TriplaneEncoder
from core.neural.triplane_dino import TriplaneEncoderDino
from core.neural.hashgrid_dino import HashGridEncoderDino
from core.neural.pose_embedder import Pose_embedder
from core.neural.jacobian_system import jacobian_system
from core.mesh import Mesh
from core.neural.jacobian_net import jacobian_Net
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion
from core.geometry.camera_utils import compute_face_laplacian
from core.neural.pe_embedder import get_embedder
# Diff-Surfel-Rasterization Imports
#from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gauss import GaussianRasterizationSettings as gauss_raster_setting #https://github.com/slothfulxtx/diff-gaussian-rasterization/issues/23
from diff_gauss import GaussianRasterizer as gauss_raster


from core.geometry.mesh_utils import vertex_to_face_color_batch
import numpy as np

def save_ply_for_meshlab(filename, vertices, colors):
    """
    Saves a PLY so that colors display correctly in MeshLab
    vertices: (N, 3) float array
    colors: (N, 3) float array (range 0.0 ~ 1.0)
    """
    # 1. Check number of points
    num_points = len(vertices)
    
    # 2. If colors are in 0~1, convert to 0~255 (key for MeshLab compatibility)
    # If already in 0~255, this part should be skipped or modified.
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # 3. Write header (this is what makes MeshLab recognize the colors)
    # Must declare 'uchar red', 'uchar green', 'uchar blue'.
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # 4. Write file
    with open(filename, 'w') as f:
        f.write(header)
        # Fast save using numpy savetxt
        # Write coordinates (3 floats) + colors (3 ints) on a single line.
        data = np.hstack([vertices, colors_uint8])
        np.savetxt(f, data, fmt='%.6f %.6f %.6f %d %d %d')

    print(f"✅ Saved MeshLab file: {filename}")
class AvatarIndexStore(nn.Module):
    def __init__(self, head, lhand, rhand, head_vts, lhand_vts, rhand_vts):
        super().__init__()
        # register_buffer is needed so it is saved in state_dict and moves with device automatically
        self.register_buffer('head', head)
        self.register_buffer('lhand', lhand)
        self.register_buffer('rhand', rhand)
        self.register_buffer('head_vts', head_vts)
        self.register_buffer('lhand_vts', lhand_vts)
        self.register_buffer('rhand_vts', rhand_vts)

class Cascade_Avatar(nn.Module):
    def __init__(self, 
                 config: Dict, 
                 first_avatar: AvatarMesh, 
                 disp: List[torch.Tensor] = None, 
                 device: torch.device = torch.device('cuda:0')):
        super().__init__()
        self.device = device
        self.config = config
        self.double_shading = config['double_shading']
        self.use_albedo_act = config['use_albedo_act']
        self.max_level = config['max_level']
        self.pooling_return = 'pooling_return' in config and config['pooling_return']
        # ------------------------------------------------------------------
        # 1. Mesh Hierarchy Construction (Coarse -> Fine)
        # ------------------------------------------------------------------
        self.use_shading = config['use_shading']
        self.use_residual_feat = config['use_residual_feat']
        self.share_color_geo_feat = config['share_color_geo_feat']
        # Variant options
        self.bodypart_scale = config.get('bodypart_scale', True)
        self.canon_use_posed_vts = config.get('canon_use_posed_vts', True)
        self.mesh_avatars = []
        self.face_c2f_indexs = [] 
        self.vts_c2f_mats = [] 
        self.register_buffer('use_gs_geo', torch.tensor(False))
        # Level 0 (Coarse)
        self.mesh_avatars.append(copy.deepcopy(first_avatar))
        self.attributes = {}
        if disp is not None:
            self._init_explicit_attribute('disp', disp)
            self._update_v_canon_init()
        #dubug_mesh = copy.deepcopy(self.mesh_avatars[-1])

        prev_mesh = copy.deepcopy(self.mesh_avatars[-1])
        self.opt_mesh=True

        subdivide_count = config.get('subdivide_count', 2)
        if subdivide_count == 1:
            c2f_face_index_tensor, c2f_vts_sparse_mat = self.once_subdivide(prev_mesh)
        else:
            c2f_face_index_tensor, c2f_vts_sparse_mat = self.twice_subdivide(prev_mesh)
        self.mesh_avatars.append(prev_mesh)
        self.face_c2f_indexs.append(c2f_face_index_tensor)
        self.vts_c2f_mats.append(c2f_vts_sparse_mat)
        self.use_init_rgb = self.config['use_init_rgb']

        if self.max_level==2:
            self.register_buffer(f'c2f_face_idx_0', c2f_face_index_tensor)        # [Control Buffer] Current Optimization Level
            self.register_buffer(f'c2f_vts_mat_0', c2f_vts_sparse_mat)        # [Control Buffer] Current Optimization Level

        if self.use_init_rgb:
            vts_color = first_avatar.rgb.squeeze()[None]
            face_color_init = vertex_to_face_color_batch(vts_color,first_avatar.face)
            face_color_param = self.rgb_act_inverse(face_color_init)
            self.init_face_rgbs = [] #face_color_param #nn.Parameter(face_color_param, requires_grad=False) 
            for level in range(self.max_level):
                curr_face_color_param = face_color_param.clone()
                if level !=0:
                    c2f_face_idx = getattr(self, f'c2f_face_idx_{level-1}')
                    curr_face_color_param = face_color_param[:, c2f_face_idx,:]

                self.init_face_rgbs.append(curr_face_color_param)

        self.avatar_indices = nn.ModuleList()        
        for  mesh_avatar in self.mesh_avatars:
            head_idxs,head_vst_idxs = self.set_head_idxs(mesh_avatar)
            head_idxs =head_idxs.to(self.device)
            head_vst_idxs =head_vst_idxs.to(self.device)

            #self.head_idxs.append(head_idxs)
            #self.register_buffer(f'head_idx_{str(idx)}', head_idxs)        # [Control Buffer] Current Optimization Level

            lhand_idxs, rhand_idxs, lhand_vst_idxs, rhand_vst_idxs = self.set_hand_idxs(mesh_avatar)
            #self.rhand_idxs.append(rhand_idxs.to(self.device))
            #self.lhand_idxs.append(lhand_idxs.to(self.device))
            store = AvatarIndexStore(head_idxs, lhand_idxs, rhand_idxs, head_vst_idxs, lhand_vst_idxs, rhand_vst_idxs)

            self.avatar_indices.append(store)
            #self.register_buffer(f'lhand_idx_{str(idx)}', lhand_idxs)        # [Control Buffer] Current Optimization Level
        self.zero_rotation = matrix_to_quaternion(torch.eye(3).float().to(device)[None,:,:])
        self.register_buffer('curr_level', torch.tensor(0, device=device))

        # ------------------------------------------------------------------
        # 2. Config & Attributes
        # ------------------------------------------------------------------
        self.deformation_pose_model_type = config['deform_pose']['deform_type']


        self._compute_all_face_centers()
        self.get_scene_bounds(padding=0.1)

        # ------------------------------------------------------------------
        # 3. Build Layers (Dual Path Encoders)
        # ------------------------------------------------------------------
        # Path A: Jacobian Encoders (Independent per level)
        self._init_jacobian_encoders()
        
        # Path B: GS Encoder (Shared Finest Source)
        # Decoders & System
        self._init_pose_base_decoder() # Jacobian Decoders
        self._init_gs_encoder()
        self._init_gs_decoder()

        self._init_jacob_system()      # Finest Level System
        
        # GS Modules (MLPs)
        self._init_gs_modules()
        
        self.set_face_laplacian()

    def get_c2f_index2tensor(self,subdivide_info):
        #subdivide_info: output of trimesh subdivide
        flat_fine_indices = np.concatenate(list(subdivide_info.values()))
        
        # 2. Repeat the coarse indices (keys) by the number of children
        # e.g. keys=[0, 1], counts=[4, 4] -> [0, 0, 0, 0, 1, 1, 1, 1]
        keys = np.array(list(subdivide_info.keys()))
        counts = np.array([len(v) for v in subdivide_info.values()])
        flat_coarse_indices = np.repeat(keys, counts)
        
        # 3. Convert to GPU tensor
        # Create an empty tensor sized by the max fine index
        num_fine_faces = flat_fine_indices.max() + 1
        c2f_index_tensor = torch.zeros(num_fine_faces, dtype=torch.long, device=self.device)
        
        # 4. Assign all at once (vectorized assignment)
        # Handled in one shot via indexing, without a loop.
        tensor_fine = torch.from_numpy(flat_fine_indices).to(self.device).long()
        tensor_coarse = torch.from_numpy(flat_coarse_indices).to(self.device).long()
        
        c2f_index_tensor[tensor_fine] = tensor_coarse
        return c2f_index_tensor


    def once_subdivide(self, mesh_avatar):
        """Single subdivision: L0 → L1 (4x faces)."""
        curr_mesh_avatar = copy.deepcopy(mesh_avatar)
        subdivide_info = mesh_avatar.subdivide_w_index_cy()
        face_map = self.get_c2f_index2tensor(subdivide_info)
        vts_mat = self.get_topology_based_matrix(curr_mesh_avatar, mesh_avatar)
        return face_map, vts_mat

    def twice_subdivide(self,mesh_avatar):
        #subdivide_info: output of trimesh subdivide
        curr_mesh_avatar =copy.deepcopy(mesh_avatar)
        num_vts = max(mesh_avatar.v_canon.shape)

        subdivide_info_1 = mesh_avatar.subdivide_w_index_cy()
        face_map_1_to_0 = self.get_c2f_index2tensor(subdivide_info_1)
        vts_mat_1_to_0 = self.get_topology_based_matrix(curr_mesh_avatar, mesh_avatar)

        num_vts = max(mesh_avatar.v_canon.shape)

        curr_mesh_avatar =copy.deepcopy(mesh_avatar)

        subdivide_info_2 = mesh_avatar.subdivide_w_index_cy()
        face_map_2_to_1 = self.get_c2f_index2tensor(subdivide_info_2)
        vts_mat_2_to_1 = self.get_topology_based_matrix(curr_mesh_avatar, mesh_avatar)


        face_map_2_to_0 = face_map_1_to_0[face_map_2_to_1]
        vts_mat_2_to_0 = torch.sparse.mm(vts_mat_2_to_1, vts_mat_1_to_0)
        return face_map_2_to_0, vts_mat_2_to_0

    # ------------------------------------------------------------------
    # Initialization Helpers
    # ------------------------------------------------------------------
    def _init_jacobian_encoders(self):
        """ Initialize Independent Encoders for each level (Geometry Path)
            L0, L1: TriplaneEncoderDino (sum mode + DINOv2 projection)
            L2+: original TriplaneEncoder
        """
        self.jacobian_encoders = nn.ModuleList()
        self.dino_levels = []  # levels that use DINOv2 triplane

        for level in range(self.max_level):
            enc_cfg = self.config['encoder'][f'level_{str(level+1)}']

            if enc_cfg['type'] == 'mlp':
                enc = UniversalMLP(enc_cfg, device=self.device)
            elif enc_cfg['type'] == 'triplane':
                if level <= 1:  # L0, L1: DINOv2 triplane
                    enc = TriplaneEncoderDino(enc_cfg, bounds=self.bounds, device=self.device)
                    self.dino_levels.append(level)
                else:
                    enc = TriplaneEncoder(enc_cfg, bounds=self.bounds, device=self.device)
            elif enc_cfg['type'] == 'hashgrid':
                enc = HashGridEncoderDino(enc_cfg, bounds=self.bounds, device=self.device)
                self.dino_levels.append(level)
            elif enc_cfg['type'] == 'identity':
                enc = nn.Identity()
            elif enc_cfg['type'] == 'PE':
                enc,_ = get_embedder(enc_cfg['multires'])
            else:
                raise NotImplementedError
            self.jacobian_encoders.append(enc)


    def _init_gs_encoder(self):
        """ Initialize Single Encoder for Finest Level (Appearance Path) """
        gs_enc_cfg = self.config['gs_encoder'] #, self.config['encoder'])
        if not self.share_color_geo_feat:
        
            if gs_enc_cfg['type'] == 'mlp':
                self.gs_feature_encoder = UniversalMLP(gs_enc_cfg, device=self.device)
            elif gs_enc_cfg['type'] == 'triplane':
                self.gs_feature_encoder = TriplaneEncoder(gs_enc_cfg, bounds=self.bounds, device=self.device)
            elif gs_enc_cfg['type'] == 'identity':
                self.gs_feature_encoder = nn.Identity()
            elif gs_enc_cfg['type'] == 'PE':
                self.gs_feature_encoder,_ = get_embedder(gs_enc_cfg['multires'],self.bounds)
            self.gs_enc_dim = self.gs_feature_encoder.output_dim
        else:
            assert self.jacobian_encoders[0].output_dim == self.jacobian_encoders[-1].output_dim
            self.gs_enc_dim = self.jacobian_encoders[0].output_dim

    def _init_gs_decoder(self):
        self.gs_dec_cfg = self.config['gs_decoder']

        self.geo_feat_cfg = self.config['gs_decoder']['geo_feat']
        refine_input_dim = self.gs_enc_dim
        if self.geo_feat_cfg['use_pose']:#, False):
            refine_input_dim += self.pose_base_encoder.out_dim
        self.geo_feat_cfg['input_dim'] = refine_input_dim


        self.gs_geo_encoder = UniversalMLP(self.geo_feat_cfg, device=self.device)

        self.gs_geo_feat_enc_dim = self.geo_feat_cfg['output_dim']
        self.mean_offset_cfg = self.config['gs_decoder']['mean_offset']
        self.mean_offset_cfg['input_dim'] = self.gs_geo_feat_enc_dim
        self.scale_offset_cfg = self.config['gs_decoder']['scale_offset']
        self.scale_offset_cfg['input_dim'] = self.gs_geo_feat_enc_dim

        self.mean_offset_mlp = UniversalMLP(self.mean_offset_cfg, device=self.device)
        self.scale_offset_mlp = UniversalMLP(self.scale_offset_cfg, device=self.device)

        # (option d): pose-conditioned global mean offset.
        # Replaces per-face mean_offset with single (B,1,3) global offset broadcast to all faces.
        # Default: False (per-face offset preserved). Enable via config['gs_decoder']['use_global_mean_offset']=True.
        self.use_global_mean_offset = bool(self.gs_dec_cfg.get('use_global_mean_offset', False))
        if self.use_global_mean_offset:
            _gmo_cfg = {
                'input_dim': self.pose_base_encoder.out_dim,
                'output_dim': 3,
                'hidden_dim': 64,
                'num_layers': 2,
                'type': 'standard',
                'use_gn': False,
                'zero_last_init': True,           # init output ≈ 0 → no shift at start
                'hidden_activation': 'relu',
            }
            self.global_mean_offset_mlp = UniversalMLP(_gmo_cfg, device=self.device)
            print('[Cascade_Avatar] use_global_mean_offset=True: pose-cond (B,1,3) global offset replaces per-face mean_offset')

        # pose-conditioned global scale offset (Option d for scale).
        # Replaces per-face scale_out residual with single (B,1,1) global scale offset (isotropic).
        # Bodypart-aware base (log(sqrt_area*1.3) for body, ×0.1 for hands) is preserved.
        # Default: False (per-face scale_out preserved).
        self.use_global_scale_offset = bool(self.gs_dec_cfg.get('use_global_scale_offset', False))
        if self.use_global_scale_offset:
            _gso_cfg = {
                'input_dim': self.pose_base_encoder.out_dim,
                'output_dim': 1,                  # isotropic scalar offset
                'hidden_dim': 64,
                'num_layers': 2,
                'type': 'standard',
                'use_gn': False,
                'zero_last_init': True,           # init output ≈ 0 → no scale shift at start
                'hidden_activation': 'relu',
            }
            self.global_scale_offset_mlp = UniversalMLP(_gso_cfg, device=self.device)
            print('[Cascade_Avatar] use_global_scale_offset=True: pose-cond (B,1,1) global scale offset replaces per-face scale_out')


    def forward_gs_decoder(self, feat):
        geo_feats = self.gs_geo_encoder(feat)
        mean_offset = self.mean_offset_mlp(geo_feats)
        scale_offset=  self.scale_offset_mlp(geo_feats)
        return mean_offset, scale_offset


    def _init_pose_base_decoder(self):
        # Global Pose Embedder
        self.pose_encoding_type = self.config['deform_pose']['Pose_embedder']['embedder']
        self.pose_base_encoder = Pose_embedder(self.config['deform_pose']['Pose_embedder'], 
                                               self.mesh_avatars[0], device=self.device)
        
        self.pose_base_decoders = nn.ModuleList()
        deform_type = self.deformation_pose_model_type

        for level in range(self.max_level):
            pose_config = copy.deepcopy(self.config['deform_pose'])
            
            # Decoder Input Dim = Jacobian Feature Dim + Pose Emb Dim
            # Assuming Jacobian Encoder has same dim as config
            curr_enc = self.jacobian_encoders[level]
            enc_out_dim = curr_enc.output_dim  #if hasattr(curr_enc, 'output_dim') else self.config['encoder'].get('dim', 32)
            
            if self.encode_inputs[level]  == 'face_vts': enc_out_dim *= 3
            pose_config['input_dim'] = enc_out_dim + self.pose_base_encoder.out_dim
            
            if deform_type == 'jacob':
                decoder = UniversalMLP(config=pose_config, device=self.device)
            elif deform_type == 'multiple_jacob':
                decoder = jacobian_Net(config=pose_config, device=self.device)
            else:
                decoder = nn.Identity()
            
            self.pose_base_decoders.append(decoder)
        self.pose_config = pose_config 
        self.pose_dec = deform_type

    def _init_gs_modules(self):
        self.gs_config = self.config['GS2D']
        self.RGB_MLP_cfg = self.gs_config['RGB_MLP']
        self.Refine_MLP_cfg = self.gs_config['Refine_RGB_MLP']
        
        self.active_sh_degree = 0
        self.scaling_factor = self.gs_config.get('scaling_factor', 1.0)
        self.learn_xyz = self.gs_config.get('learn_xyz', False)

        # 1. RGB MLP (Coarse Color)
        # Input: Pooled GS Feature
        self.RGB_MLP_cfg['input_dim'] = self.gs_enc_dim
        self.RGB_MLP = UniversalMLP(self.RGB_MLP_cfg, device=self.device)

        # 2. Refine MLP (Residual Color)
        # Input: Pooled GS Feature + Pose + (Optional) Normal       
        refine_input_dim= 0
        if self.Refine_MLP_cfg['use_feat']:#, False):
            refine_input_dim += self.gs_enc_dim
        if self.Refine_MLP_cfg['use_pose']:#, False):
            refine_input_dim += self.pose_base_encoder.out_dim
        if self.Refine_MLP_cfg['use_norm']:#, False):
            if self.Refine_MLP_cfg['use_norm_PE']:
                self.Shadow_PE,_ = get_embedder(6)
                refine_input_dim += self.Shadow_PE.output_dim
                #from gomavaatar multires is 6
            else:
                refine_input_dim += 3
        
        self.Refine_MLP_cfg['input_dim'] = refine_input_dim
        self.Refine_RGB_MLP = UniversalMLP(self.Refine_MLP_cfg, device=self.device)


        # Global Scale Parameter
        self.global_scale = nn.Parameter(torch.tensor([0.0], device=self.device).requires_grad_(True))

    def _init_jacob_system(self):
        self.jacobian_systems = nn.ModuleList()

        for level, mesh in enumerate(self.mesh_avatars):
            v_canon = mesh.v_canon.squeeze()
            init_da_mesh = Mesh(v_canon, mesh.face, mesh.rgb, device=self.device)
            self.jacobian_systems.append(jacobian_system(self.config['jacobian_system'][f'level_{str(level+1)}'], init_da_mesh=init_da_mesh, device=self.device))
        
        

    def _compute_all_face_centers(self):
        self.encode_inputs = []
        for level, mesh in enumerate(self.mesh_avatars):
            self.encode_inputs.append(self.config['encoder'][f'level_{str(level+1)}']['input_type'])

            da_vts = mesh.v_canon
            if da_vts.dim() == 3: da_vts = da_vts.squeeze(0)
            centers = da_vts[mesh.face].mean(dim=1).to(self.device)
            self.register_buffer(f'face_center_{level}', centers)
            
            # Input Data Buffers for Encoders
            if self.encode_inputs[level] == 'face_vts':
                face_vts = da_vts[mesh.face].to(self.device)
                self.register_buffer(f'face_vts_{level}', face_vts)
                
            if level == self.max_level - 1: self.face_center = centers 

    def _init_explicit_attribute(self, attribute_name: str, attribute: torch.Tensor):
        self.attributes[attribute_name] = attribute

    def _update_v_canon_init(self):
        if 'disp' not in self.attributes: return
        target_mesh = self.mesh_avatars[-1]
        if len(target_mesh.v_canon.shape) == 3 and len(self.attributes['disp'][0])!=3:
            self.attributes['disp'] = self.attributes['disp'][None,:]
        if torch.sum(self.attributes['disp'])>0:
            target_mesh.v_canon = target_mesh._v_canon + self.attributes['disp']
        self.attributes['disp'] = torch.zeros_like(target_mesh._v_canon)
        target_mesh.disp = self.attributes['disp']

    def get_scene_bounds(self, padding: float = 0.1):
        centers = getattr(self, 'face_center_0')
        min_bound = torch.min(centers, dim=0)[0]
        max_bound = torch.max(centers, dim=0)[0]
        scale = max_bound - min_bound 
        if padding > 0:
            min_bound -= padding * scale
            max_bound += padding * scale
        self.bounds = {'min': min_bound, 'max': max_bound}
        
    def set_face_laplacian(self):
        """
        Computes the face Laplacian for the mesh at every level and stores it in a list.
        self.face_laps[0] -> Coarse Level Laplacian
        self.face_laps[-1] -> Finest Level Laplacian
        """
        self.face_laps = []
        
        for level, mesh in enumerate(self.mesh_avatars):
            # Compute Laplacian using each level's face information
            # compute_face_laplacian is assumed to be defined in utils.geometry.geo_utils
            lap = compute_face_laplacian(mesh.face)
            
            # Append to list (device follows mesh.face)
            self.face_laps.append(lap)

    
    def rgb_act(self, rgb):
        return (torch.tanh(rgb) + 1) / 2
    def rgb_act_inverse(self, rgb_normalized):
        epsilon = 1e-6
        rgb_safe = torch.clamp(rgb_normalized, min=epsilon, max=1.0-epsilon)
        return torch.arctanh((rgb_safe * 2) - 1)
    # ------------------------------------------------------------------
    # Differentiable Pooling & Helpers
    # ------------------------------------------------------------------
    def _generate_gs_at_level(self, posed_vts, canon_vts, faces, lbs_motion,  features, pose_emb, w2c_rot, not_pooling, curr_level=-1):
        use_pooling =  curr_level != (self.max_level-1) and not not_pooling
        if canon_vts.ndim ==2:
            canon_vts = canon_vts[None,:]
        if posed_vts.ndim==2:
            posed_vts = posed_vts[None,:]

        B, V, _ = canon_vts.shape
        F = faces.shape[0]
        #with torch.no_grad():

        if self.use_gs_geo:

            if lbs_motion.ndim==2:
                lbs_motion = lbs_motion[None,:,:]
            face_motions = lbs_motion[:, faces, :]
            face_motions = face_motions.mean(dim=2)
            """ Generate GS parameters for a specific level using Mesh + Features """
            # ... (Same logic as provided in previous explanation) ...
            #finest features
            


            if self.geo_feat_cfg['use_pose']: #, False):
                geo_feat = torch.cat([features, pose_emb], dim=-1)
            else:
                geo_feat = features.clone()

            mean_offset, scale_out = self.forward_gs_decoder(geo_feat)
            if use_pooling:
                mean_offset = self.feature_pooling(mean_offset,F)
                scale_out = self.feature_pooling(scale_out,F)

            # (option d): override per-face mean_offset with pose-cond global (B,1,3) broadcast.
            # Default path above is preserved when self.use_global_mean_offset is False.
            if getattr(self, 'use_global_mean_offset', False):
                # pose_emb shape: (B, num_faces_or_1, pose_dim) — take batch-level pose (face axis is broadcasted)
                _pose_global = pose_emb[:, :1, :]                       # (B, 1, pose_dim)
                _g_offset = self.global_mean_offset_mlp(_pose_global)   # (B, 1, 3)
                mean_offset = _g_offset.expand(-1, mean_offset.shape[1], -1).contiguous()  # (B, F, 3)

            # override per-face scale_out with pose-cond global (B,1,1) broadcast.
            # Bodypart-aware base (sqrt_area*1.3 etc.) preserved by downstream code; only the LEARNED offset is global.
            if getattr(self, 'use_global_scale_offset', False):
                _pose_global_s = pose_emb[:, :1, :]                       # (B, 1, pose_dim)
                _g_scale = self.global_scale_offset_mlp(_pose_global_s)   # (B, 1, 1)
                scale_out = _g_scale.expand(-1, scale_out.shape[1], scale_out.shape[2]).contiguous()  # (B, F, C)

            v0 = canon_vts[:, faces[:, 0]]
            v1 = canon_vts[:, faces[:, 1]]
            v2 = canon_vts[:, faces[:, 2]]

            canon_face_centers = (v0 + v1 + v2) / 3.0
            edge0, edge1, edge2 = v1 - v0, v2 - v1, v0 - v2

            normal_raw = torch.cross(edge0, edge1, dim=-1)
            triangle_area_x2 = torch.norm(normal_raw, dim=-1)
            normal_world = torch.nn.functional.normalize(normal_raw, dim=-1)

            if self.bodypart_scale:
                # Body-part-aware scale: area-based prior with hand attenuation
                sqrt_area = torch.sqrt(triangle_area_x2/2*math.pi)[...,None].detach()
                if scale_out.shape[-1]==3:
                    scale_out = scale_out.mean(dim=-1)[...,None]
                scale_out= scale_out.repeat(1,1,3)
                scale = torch.zeros_like(scale_out)
                num_points = scale.shape[1]
                mask = torch.ones(num_points, dtype=torch.bool, device=scale.device)
                curr_idxs = self.avatar_indices[curr_level]
                mask[curr_idxs.lhand] = False
                mask[curr_idxs.rhand] = False
                scale[:, mask] = torch.log(sqrt_area * 1.3)[:, mask, :]
                scale[:,curr_idxs.lhand] =  torch.log(sqrt_area*0.1)[:,curr_idxs.lhand,:]
                scale[:,curr_idxs.rhand] =  torch.log(sqrt_area*0.1)[:,curr_idxs.rhand,:]
                scale = scale + scale_out
                # scale clamp to prevent visible-Gaussian explosion (rasterizer fillrate OOM).
                # Default: cap at 5x sqrt_area (≈ exp(log(5*sqrt_area)) → splat radius ≤ 5x face area sqrt).
                # Set self.gs_scale_clamp_max < 0 to disable.
                _clamp_max_factor = float(getattr(self, 'gs_scale_clamp_max', 5.0))
                if _clamp_max_factor > 0:
                    _max_log = torch.log(sqrt_area * _clamp_max_factor)  # (B, F, 1)
                    scale = torch.minimum(scale, _max_log.expand_as(scale))
            else:
                # Flat scale: simple constant bias
                scale = scale_out - 6.0
                sqrt_area = None
                mask = None
                if mean_offset.shape[-1] == 1:
                    mean_offset = mean_offset * normal_world

            canon_face_centers = canon_face_centers + mean_offset
            posed_faces = canon_face_centers + face_motions

        
        v0 = posed_vts[:, faces[:, 0]]
        v1 = posed_vts[:, faces[:, 1]]
        v2 = posed_vts[:, faces[:, 2]]

        edge0, edge1, edge2 = v1 - v0, v2 - v1, v0 - v2
        
        normal_raw = torch.cross(edge0, edge1, dim=-1)
        #triangle_area_x2 = torch.norm(normal_raw, dim=-1)
        normal_world = torch.nn.functional.normalize(normal_raw, dim=-1)
        
        #if w2c_rot is not None:
            #    normal_cam = torch.bmm(normal_world, w2c_rot.transpose(1, 2))
            #    normal_cam = torch.nn.functional.normalize(normal_cam, dim=-1)
            #else:
            #    normal_cam = normal_world


        

        # Appearance
        coarse_rgb = self.RGB_MLP(features)
                    

        if use_pooling:
            coarse_rgb = self.feature_pooling(coarse_rgb,F)
        if self.use_init_rgb:
            coarse_rgb = coarse_rgb + self.init_face_rgbs[curr_level]

        #   if not self.pooling_return:
        #        init_face_rgb= self.init_face_rgbs[curr_level]
                #for level in range(self.max_level-1):
                #    init_face_rgb = init_face_rgb[:,self.c2f_indexs[level],:]


        #check mlp for gs..
        #Offset,  scale, rotation add. but . i think currently we need to use this setting. (initial and will be improved. )
        
        refine_inputs = []
        
        if self.Refine_MLP_cfg['use_feat']: #.get('use_feat', False):
            refine_inputs.append(features)
            
        if self.Refine_MLP_cfg['use_pose']:#.get('use_pose', False):
            refine_inputs.append(pose_emb.detach())
            
        if self.Refine_MLP_cfg['use_norm']:  #.get('use_norm', False):
            if use_pooling:
                # Indexing
                normal_world = normal_world[:, self.c2f_face_idx_0, :]

            if self.Refine_MLP_cfg['use_norm_PE']:
                normal_world = self.Shadow_PE(normal_world)
            refine_inputs.append(normal_world.detach())

        if refine_inputs:
            refine_in = torch.cat(refine_inputs, dim=-1)
        else:
            refine_in = None # or raise an error
        if self.use_albedo_act:
            albedo = self.rgb_act(coarse_rgb)
        else:
            albedo = coarse_rgb

        rgb_residual = self.Refine_RGB_MLP(refine_in)
        if use_pooling:
            rgb_residual = self.feature_pooling(rgb_residual, F)
        # Removed commas and tidied parentheses
        refined_rgb = self.rgb_act(coarse_rgb + rgb_residual)  #          refined_rgb = self.rgb_act(coarse_rgb + rgb_residual)
        #if use_pooling:
        #    rgb_residual = self.feature_pooling(rgb_residual,F)

        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(face_centers.squeeze().cpu().numpy())
        colors = normal_world.squeeze().cpu().numpy()
        colors_rgb = (colors[:, [2, 1, 0]] +1.0)/2.0
        pcd.colors = o3d.utility.Vector3dVector((255*colors_rgb).astype(np.uint8))
        o3d.io.write_point_cloud('test.ply', pcd,write_ascii=True)
        save_ply_for_meshlab("meshlab_output.ply", face_centers.squeeze().cpu().numpy(), colors_rgb)
        """
        
        if self.use_gs_geo:
            gs_params = {
                'xyz': posed_faces,
                'rotation': self.zero_rotation[None,:].repeat(B,F,1,1).clone(),
                'scaling': torch.exp(scale),
                'opacity': 0.9999 * torch.ones_like(scale)[..., 0],
                'rgb': albedo.clamp_max(1.0),
                'rgb_refine': refined_rgb.clamp_max(1.0),
                'rgb_residual': rgb_residual,    # expose raw residual for regularizer
                'mean_offset':mean_offset,
                'scale_offset':scale,
                "normal_world":normal_world,
            }
            if self.bodypart_scale:
                gs_params['sqrt_area'] = sqrt_area
                gs_params['hand_except_mask'] = mask
        else:
            gs_params= {
                'rgb': albedo,
                'rgb_refine': refined_rgb,
                'rgb_residual': rgb_residual,    # expose raw residual for regularizer
                "normal_world":normal_world,
                }
        return gs_params


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, 
                target_pose: Optional[Dict] = None, 
                cameras: Optional[List] = None, 
                training_levels =[],
                init_offset = None,
                noise_size=None,
                canon=False,
                ) -> Dict:
        
        outputs = {'levels': []}
        batch_size = target_pose['full_pose'].shape[0] if target_pose else 1
        
        # Camera Info
        w2c_rot = None
        if cameras is not None:
            if isinstance(cameras, list):
                w2c_rot = torch.stack([cam.world_view_transform[:3, :3].T for cam in cameras], dim=0)
            else:
                w2c_rot = cameras.world_view_transform[:3, :3].T.unsqueeze(0).expand(batch_size, -1, -1)

        # ------------------------------------------------
        # 1. GS Feature Extraction & Pooling (Top-Down)
        # ------------------------------------------------
        finest_lvl = self.max_level - 1
        num_finest = self.mesh_avatars[finest_lvl].face.shape[0]
        
        # Input for Finest GS Encoder
        if not self.share_color_geo_feat:

            if self.encode_inputs[-1] == 'face_centers':
                gs_inp = getattr(self, f'face_center_{finest_lvl}')
                gs_finest_feat = self.gs_feature_encoder(gs_inp)
            elif self.encode_inputs[-1] == 'face_vts':
                gs_inp = getattr(self, f'face_vts_{finest_lvl}')
                gs_finest_feat = self.gs_feature_encoder(gs_inp.view(3*num_finest, 3)).view(1, num_finest, -1)
            # Batch Expand
            if gs_finest_feat.dim() == 2: 
                gs_finest_feat = gs_finest_feat.unsqueeze(0).expand(batch_size, -1, -1)
            elif gs_finest_feat.shape[0] != batch_size:
                gs_finest_feat = gs_finest_feat.repeat(batch_size, 1, 1)

        # ------------------------------------------------
        pose_embed = None
        if target_pose is not None:
            pose_input = {k: v.detach() for k, v in target_pose.items() if v is not None}
            pose_input = copy.deepcopy(pose_input)
            pose_emb_wo_noise = self.pose_base_encoder(pose_input, face_number=1)

            if noise_size:
                # Exp J v13: optional body-only noise. Flag set by subclass; default False → original.
                if getattr(self, '_body_only_pose_noise', False):
                    _body_noise = torch.zeros_like(pose_input['full_pose'])
                    _body_noise[..., 3:66] = torch.randn_like(pose_input['full_pose'][..., 3:66]) * noise_size
                    pose_input['full_pose'] = pose_input['full_pose'] + _body_noise
                else:
                    pose_input['full_pose'] += torch.randn_like(pose_input['full_pose']) * noise_size
                pose_emb_w_noise = self.pose_base_encoder(pose_input, face_number=1)
                train_pose_embed= pose_emb_w_noise.detach()
            else:
                train_pose_embed = pose_emb_wo_noise.detach()

        current_jacob_matrix = None
        curr_offset = None
        curr_offset_up =None
        for level in range(self.curr_level+1):
            level_out = {}
            if level in training_levels:
                pose_embed = train_pose_embed
            else:
                pose_embed = pose_emb_wo_noise
            num_faces = self.mesh_avatars[level].face.shape[0]
            pose_emb_curr = pose_embed.view(batch_size, 1, -1).expand(-1, num_faces, -1)

            curr_mesh = self.mesh_avatars[level]
            num_faces = curr_mesh.face.shape[0]

            # [A] Jacobian Feature (Independent)
            if self.encode_inputs[level] == 'face_centers':
                jac_inp = getattr(self, f'face_center_{level}')
                jac_feat = self.jacobian_encoders[level](jac_inp)
                if self.use_residual_feat:
                    if level==0:
                        prev_jac_feat = jac_feat
                    else:
                        face_c2f_idx = getattr(self, f'c2f_face_idx_{level-1}')
                        #import pdb;pdb.set_trace()
                        jac_feat = jac_feat + self.c2f_face_mapping(face_c2f_idx,prev_jac_feat)
                        prev_jac_feat = jac_feat
            #elif self.encode_inputs[level]  == 'face_vts':
            #    jac_inp = getattr(self, f'face_vts_{level}')
            #    jac_feat = self.jacobian_encoders[level](jac_inp.view(3*num_faces, 3)).view(1, num_faces, -1)
            
            if jac_feat.dim() == 2: 
                jac_feat = jac_feat.unsqueeze(0).expand(batch_size, -1, -1)
            elif jac_feat.shape[0] != batch_size:
                jac_feat = jac_feat.repeat(batch_size, 1, 1)

            if level > 0:
                vts_c2f_idx = getattr(self, f'c2f_vts_mat_{level-1}')
                if self.pose_dec == 'multiple_jacob':
                    curr_offset_up = self.c2f_vts_mapping(vts_c2f_idx, curr_offset )

            # Active Level: Decode Residual
            pose_emb_curr = pose_embed.view(batch_size, 1, -1).expand(-1, num_faces, -1)
            dec_in = torch.cat([jac_feat, pose_emb_curr], dim=-1)
            
            if self.pose_dec == 'multiple_jacob':
                curr_rot, curr_scale, curr_shear, _ = self.pose_base_decoders[level](dec_in)
                level_out['pose_rot'] = curr_rot
                level_out['pose_scale'] = curr_scale
                level_out['pose_shear'] = curr_shear
                R = rotation_6d_to_matrix(curr_rot.view(-1, 6)).view(batch_size, num_faces, 3, 3)
                scale_vec = curr_scale
                shear_vec = curr_shear
                U = torch.zeros_like(R)
                U[..., 0, 0], U[..., 1, 1], U[..., 2, 2] = scale_vec[..., 0], scale_vec[..., 1], scale_vec[..., 2]
                U[..., 0, 1], U[..., 0, 2], U[..., 1, 2] = shear_vec[..., 0], shear_vec[..., 1], shear_vec[..., 2]
                current_jacob_matrix = torch.matmul(R, U).view(batch_size, num_faces, 9)


            else:
                current_jacob_matrix = self.pose_base_decoders[level](dec_in)


            curr_jacob_sys = self.jacobian_systems[level]
            if curr_offset_up is not None:

                pose_base_canon = curr_jacob_sys.vertices_from_jacobians(
                            current_jacob_matrix.view(batch_size, -1, 3, 3)
                        ) + curr_jacob_sys.init_da_mean  + curr_offset_up
            else:
                pose_base_canon = curr_jacob_sys.vertices_from_jacobians(
                            current_jacob_matrix.view(batch_size, -1, 3, 3)
                        ) + curr_jacob_sys.init_da_mean 

            init_vts = curr_jacob_sys.init_vts
                
            curr_offset = pose_base_canon - init_vts 
            if curr_offset_up is not None:
                curr_offset = curr_offset  + curr_offset_up
            level_out['jac_feat'] = jac_feat
            #level_out['gs_feat'] = gs_feat
            level_out['jacobian'] = current_jacob_matrix
            level_out['Da_vts'] = pose_base_canon
            if curr_offset_up is not None:
                level_out['offset'] = curr_offset - curr_offset_up
            else:
                level_out['offset'] = curr_offset
            if canon and not self.canon_use_posed_vts:
                # Base variant: skip FS_fullposedict when canon
                input_vts = pose_base_canon
            else:
                pose_base_posed_vts = self.mesh_avatars[level].FS_fullposedict(full_pose_dict=target_pose, v_canon=pose_base_canon)
                input_vts = pose_base_canon if canon else pose_base_posed_vts

            if self.share_color_geo_feat:
                gs_finest_feat = jac_feat

            lbs_motion = input_vts - pose_base_canon #lbs skinning
            level_out['posed_vts'] = input_vts
            if self.share_color_geo_feat:
                # canon_use_posed_vts: in canon mode, feed posed_vts for normal computation
                if canon and self.canon_use_posed_vts:
                    gs_posed_vts = pose_base_posed_vts
                else:
                    gs_posed_vts = input_vts
                gs = self._generate_gs_at_level(
                    posed_vts = gs_posed_vts,
                    canon_vts=pose_base_canon,
                    faces=self.mesh_avatars[level].face,
                    lbs_motion=lbs_motion,
                    features=gs_finest_feat,
                    pose_emb=pose_embed.view(batch_size, 1, -1).expand(-1, num_faces,-1),
                    w2c_rot=w2c_rot,
                    curr_level= level,
                    not_pooling=True,
                )

            else:
                gs = self._generate_gs_at_level(
                    posed_vts = input_vts, 
                    canon_vts=pose_base_canon,
                    faces=self.mesh_avatars[level].face,
                    lbs_motion=lbs_motion,
                    features=gs_finest_feat,#gs_feat_pyramid[level],
                    pose_emb=pose_embed.view(batch_size, 1, -1).expand(-1, self.face_center.shape[0], -1),
                    w2c_rot=w2c_rot,
                    curr_level= level,
                    not_pooling=False,

                )
            level_out['gs'] = gs


            outputs['levels'].append(level_out)

                #pooling to current level?


        outputs['Da_vts'] = outputs['levels'][self.curr_level]['Da_vts']
        outputs['posed_vts'] = outputs['levels'][self.curr_level]['posed_vts']
        outputs['faces'] = self.mesh_avatars[self.curr_level].face
        
        # Pass through Finest GS for rendering outside if needed
        outputs['gs_finest'] = outputs['levels'][self.curr_level]['gs']

        return outputs

    # ------------------------------------------------------------------
    # Rendering & Optimization
    # ------------------------------------------------------------------
    def render_mesh2gs_batch(self, level_gs_params, cameras, bg_color=[1.0, 1.0, 1.0]):
        """ Render pre-calculated GS parameters """
        # Flatten batch for iteration
        batch_size = level_gs_params['xyz'].shape[0]
        batch_results = {}

        for i in range(batch_size):
            single_gs = {k: v[i] for k, v in level_gs_params.items()}
            current_cam = cameras[i] if isinstance(cameras, list) else cameras
            out_dict = self.render(single_gs, current_cam, bg_color=bg_color)
            for k, v in out_dict.items():
                if k not in batch_results: batch_results[k] = []
                batch_results[k].append(v)
                
        final_output = {}
        for k, v_list in batch_results.items():
            if k != 'visibility_filter':
                if len(v_list) > 0 and isinstance(v_list[0], torch.Tensor):
                    final_output[k] = torch.stack(v_list, dim=0)
                else: final_output[k] = v_list
        return final_output

    def render_mesh2gs_batch_w_OF(self, level_gs_params, cameras, curr_idxs, bg_color=[1.0, 1.0, 1.0]):
        """ Render pre-calculated GS parameters """
        # Flatten batch for iteration
        batch_size = level_gs_params['xyz'].shape[0]
        batch_results = {}

        for i in range(batch_size):
            if i in curr_idxs:
                single_gs = {k: v[i] for k, v in level_gs_params.items()}

            else:
                single_gs = {k: v[i] for k, v in level_gs_params.items() if k!='OF'}
            current_cam = cameras[i] if isinstance(cameras, list) else cameras
            out_dict = self.render(single_gs, current_cam, bg_color=bg_color)
            for k, v in out_dict.items():
                if k not in batch_results: batch_results[k] = []
                batch_results[k].append(v)
                
        final_output = {}
        for k, v_list in batch_results.items():
            if k == 'visibility_filter': #and k!='render_flow':
                pass
            elif k=='render_flow':
                if len(v_list) > 0 and isinstance(v_list[0], torch.Tensor):
                    total_v_list = [v for v in v_list if v is not None]
                    final_output[k] = torch.stack(total_v_list, dim=0)
            else:
                if len(v_list) > 0 and isinstance(v_list[0], torch.Tensor):
                    final_output[k] = torch.stack(v_list, dim=0)
                else: final_output[k] = v_list
        
        return final_output


    def render(self, gaussian, camera, bg_color=[1.0, 1.0, 1.0], scaling_modifier=1.0, debug=False):
        # GoF rasterizer dispatch is STAGE-AWARE.

        # 1. Set up screen-space points and camera parameters
        screenspace_points = torch.zeros_like(gaussian['xyz'], dtype=gaussian['xyz'].dtype, requires_grad=True, device="cuda")
        try: screenspace_points.retain_grad()
        except: pass

        fov_x = np.deg2rad(camera.FoVx)
        fov_y = np.deg2rad(camera.FoVy)
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(fov_y * 0.5)

        # Rasterizer settings
        raster_settings = gauss_raster_setting(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform.cuda(), # w2c (RT)
            projmatrix=camera.full_proj_transform.cuda(),
            sh_degree=self.active_sh_degree if hasattr(self, 'active_sh_degree') else 0,
            campos=camera.camera_center.cuda(),
            prefiltered=False,
            debug=debug 
        )
        rasterizer = gauss_raster(raster_settings=raster_settings)

        # ----------------------------------------------------------------
        # 2. Merge extra render attributes (Albedo + Optical Flow)
        # ----------------------------------------------------------------
        # Merge gaussian['rgb'] (Albedo) and gaussian['optical_flow'].
        extra_list = []
        
        # 2-1. Albedo (base color, 3 channels)
        # Use gaussian['rgb'] or the DC base color from SH
        albedo = gaussian['rgb']
        extra_list.append(albedo)
        
        # 2-2. Optical Flow (if present, usually 2 channels)
        has_flow =  'OF' in gaussian.keys()
        if has_flow:
            OF = gaussian['OF']

            flow_dim = OF.shape[-1]
            extra_list.append(OF)
            
        # Concatenate into [N, 3(Albedo) + C(Flow)] to pass to the rasterizer.
        extra_features = torch.cat(extra_list, dim=-1)

        # ---------------------------------------------------
        # 3. Set up rasterizer and call render
        # ---------------------------------------------------
        xyz = gaussian['xyz']
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try: screenspace_points.retain_grad()
        except: pass
        
        fov_x = camera.FoVx
        fov_y = camera.FoVy
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(fov_y * 0.5)

        raster_settings = gauss_raster_setting(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            # camera paramter, i.e. the w2c RT matrix
            viewmatrix=camera.world_view_transform.cuda(), 
            projmatrix=camera.full_proj_transform.cuda(),
            sh_degree=0,
            campos=camera.camera_center.cuda(),
            prefiltered=False,
            debug=debug 
        )
        rasterizer = gauss_raster(raster_settings=raster_settings)

        # Render all data in a single call
        rendered_image, rendered_depth, rend_normal, render_alpha, radii, extra = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=gaussian['rgb_refine'], # main RGB
            opacities=gaussian['opacity'],
            scales=gaussian['scaling'],
            rotations=gaussian['rotation'],
            cov3Ds_precomp=None,
            extra_attrs=extra_features     # merged Albedo & Flow
        )

        # ---------------------------------------------------
        # 4. Slice and split outputs
        # ---------------------------------------------------
        # extra has shape [Channels, H, W]. Split back in the order it was concatenated.
        render_albedo = extra[0:3, ...].permute(1, 2, 0)
        render_depth_expected = torch.nan_to_num(rendered_depth, 0, 0)
        render_depth_expected = (render_depth_expected / (render_alpha + 1e-6))

        # 6.4 Pseudo Surface Normal
        surf_depth = render_depth_expected  #* (1 - depth_ratio) + (depth_ratio) * render_depth_median

        surf_normal = depth_to_normal_camera(camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()
        surf_normal  = surf_normal.permute(1,2,0)

        #cv2.imwrite('surf_normal.png',(surf_normal+1.0).detach().cpu().numpy()*127.5)
        #cv2.imwrite('rndr_norml.png',rend_normal.permute(1, 2, 0).detach().cpu().numpy()*255)
        render_flow = None
        if has_flow:
            render_flow = extra[3:3+flow_dim, ...].permute(1, 2, 0)

        out = {
            "render": rendered_image.permute(1, 2, 0).clamp(0, 1), # main RGB
            "render_albedo": render_albedo.clamp(0, 1),                            # Albedo
            "render_flow": render_flow,                                # Optical Flow
            #"rend_normal": rend_normal.permute(1, 2, 0),           # Depth
            "surf_normal": surf_normal,         # Normal
            "render_alpha": render_alpha.permute(1, 2, 0),           # Alpha mask
            #"viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii
        }
        
        return out
        
        
    def set_all_grad_false(self):
        """
        Freezes all trainable parameters in the model.
        Sets requires_grad=False and clears remaining gradients (None).
        """
        
        # [Helper] freeze helper function
        def freeze(module_or_tensor):
            if isinstance(module_or_tensor, nn.Module):
                params = module_or_tensor.parameters()
            elif isinstance(module_or_tensor, torch.Tensor):
                params = [module_or_tensor]
            else:
                return # skip if None or not trainable

            for p in params:
                p.requires_grad = False
                p.grad = None  # [important] clear stale gradients

        # -----------------------------------------------------------

        # 1. Jacobian Encoders (All Levels)
        for enc in self.jacobian_encoders:
            freeze(enc)

        # 2. Jacobian Decoders (All Levels)
        for dec in self.pose_base_decoders:
            if not isinstance(dec, nn.Identity):
                freeze(dec)
        #if hasattr(self, 'init_face_rgb'):
        #     self.init_face_rgb.requires_grad = False
        #     self.init_face_rgb.grad = None
        # 3. GS Modules & MLPs
        if not self.share_color_geo_feat:
            freeze(self.gs_feature_encoder)

        freeze(self.gs_geo_encoder)    
         
        freeze(self.scale_offset_mlp)
        freeze(self.mean_offset_mlp)
        if getattr(self, 'use_global_mean_offset', False):
            freeze(self.global_mean_offset_mlp)
        if getattr(self, 'use_global_scale_offset', False):
            freeze(self.global_scale_offset_mlp)
        freeze(self.RGB_MLP)
        freeze(self.Refine_RGB_MLP)
        
        # Global Scale
        freeze(self.global_scale)

        # 4. Global Pose Embedder
        freeze(self.pose_base_encoder)

    def set_grad_params_groups(self, training_levels, freeze_decoders=False):
        """
        Without returning optimizer groups,
        sets requires_grad=False for modules not in the current training_levels
        and resets their existing gradients to None.
        """
        
        # [Helper] decides whether gradients are enabled for a module or tensor
        def set_trainable(module_or_tensor, is_active):
            # 1. If the target is an nn.Module, iterate via parameters()
            if isinstance(module_or_tensor, nn.Module):
                params = module_or_tensor.parameters()
            # 2. If the target is a Tensor (Parameter), wrap it in a list
            elif isinstance(module_or_tensor, torch.Tensor):
                params = [module_or_tensor]
            else:
                return # skip if None or another type

            for p in params:
                p.requires_grad = is_active
                # [Key] when disabling, clear stale previously computed gradients
                if not is_active:
                    p.grad = None

        #if hasattr(self, 'init_face_rgb'):
        #    is_active = 0 in training_levels
        #    self.init_face_rgb.requires_grad = is_active
        #    if not is_active:
        #        self.init_face_rgb.grad = None

        # 1. Jacobian Encoders (Active Levels Only)
        if self.opt_mesh:
            for i, enc in enumerate(self.jacobian_encoders):
                is_active = (i in training_levels)
                set_trainable(enc, is_active)

            # 2. Jacobian Decoders (Active Levels Only, unless freeze_decoders=True)
            for i, dec in enumerate(self.pose_base_decoders):
                if not isinstance(dec, nn.Identity):
                    is_active = (i in training_levels) and (not freeze_decoders)
                    set_trainable(dec, is_active)
        else:
            # Defensive freeze when opt_mesh=False (GS stage).
            # Without this, encoders/decoders retain whatever requires_grad they had —
            # which is False after set_all_grad_false() but is fragile. Make explicit.
            for enc in self.jacobian_encoders:
                set_trainable(enc, False)
            for dec in self.pose_base_decoders:
                if not isinstance(dec, nn.Identity):
                    set_trainable(dec, False)

        # 3. GS Modules (Always Active / Shared Source)
        # These modules are assumed to always be trained to preserve color, etc.
        if not self.share_color_geo_feat:
            set_trainable(self.gs_feature_encoder, True)
        if self.use_gs_geo:
            set_trainable(self.gs_geo_encoder, True)
            set_trainable(self.scale_offset_mlp, True)
            set_trainable(self.mean_offset_mlp, True)
            if getattr(self, 'use_global_mean_offset', False):
                set_trainable(self.global_mean_offset_mlp, True)
            if getattr(self, 'use_global_scale_offset', False):
                set_trainable(self.global_scale_offset_mlp, True)

        set_trainable(self.RGB_MLP, True)
        set_trainable(self.Refine_RGB_MLP, True)
        
        # Global Scale (Tensor)
        set_trainable(self.global_scale, True)

        # 4. Global Pose Embedder (Always Active)
        set_trainable(self.pose_base_encoder, True)


    def get_all_param_groups(self):
        param_groups = []
        #current_active_level = self.curr_level.item()
        #color_lr = self.config.get('rgb_init', {}).get('lr', 1e-2) 
        
        #if hasattr(self, 'init_face_rgb'):
        #    param_groups.append({
        #        'params': [self.init_face_rgb], 
        #        'lr': float(color_lr), 
         #       'name': 'base_face_color'
         #  })        # -----------------------------------------------------------

        # 1. Jacobian Encoders (Active Levels Only)
        for i, enc in enumerate(self.jacobian_encoders):
            enc_lr = self.config['encoder'][f'level_{str(i+1)}'].get('lr', 1e-3)

            param_groups.append({'params': enc.parameters(), 'lr': float(enc_lr), 'name': f'jac_enc_L{i}'})

        # 2. Jacobian Decoders (Active Levels Only)
        dec_lr = self.config['deform_pose'].get('lr', 1e-3)
        for i, dec in enumerate(self.pose_base_decoders):
            if not isinstance(dec, nn.Identity): #i <= current_active_level:
                param_groups.append({'params': dec.parameters(), 'lr': float(dec_lr), 'name': f'jac_dec_L{i}'})

        # 3. GS Modules (Shared Source -> Always Optimized or Controlled)
        # Since GS relies on Finest Encoder via Pooling, we typically optimize it always 
        # to ensure color consistency, or only when Finest level is active?
        # Recommendation: Optimize GS modules always so low-level training also updates color via pooling.
        if not self.share_color_geo_feat:
            gs_lr = self.config.get('gs_encoder', {}).get('lr', enc_lr)
            param_groups.append({'params': self.gs_feature_encoder.parameters(), 'lr': float(gs_lr), 'name': 'gs_enc'})
        
        mlp_lr = self.RGB_MLP_cfg.get('lr', 1e-3)
        param_groups.append({'params': self.RGB_MLP.parameters(), 'lr': float(mlp_lr), 'name': 'rgb_mlp'})
        mlp_lr = self.Refine_MLP_cfg.get('lr', 1e-3)
        param_groups.append({'params': self.Refine_RGB_MLP.parameters(), 'lr': float(mlp_lr), 'name': 'refine_mlp'})

        mlp_lr = self.geo_feat_cfg.get('lr', 1e-3)
        param_groups.append({'params': self.gs_geo_encoder.parameters(), 'lr': float(mlp_lr), 'name': 'gs_geo_encoder'})
        mlp_lr = self.scale_offset_cfg.get('lr', 1e-3)
        param_groups.append({'params': self.scale_offset_mlp.parameters(), 'lr': float(mlp_lr), 'name': 'scale_offset_mlp'})
        mlp_lr = self.mean_offset_cfg.get('lr', 1e-3)
        param_groups.append({'params': self.mean_offset_mlp.parameters(), 'lr': float(mlp_lr), 'name': 'mean_offset_mlp'})
        if getattr(self, 'use_global_mean_offset', False):
            param_groups.append({'params': self.global_mean_offset_mlp.parameters(), 'lr': float(mlp_lr), 'name': 'global_mean_offset_mlp'})
        if getattr(self, 'use_global_scale_offset', False):
            _scale_lr = self.scale_offset_cfg.get('lr', 1e-3)
            param_groups.append({'params': self.global_scale_offset_mlp.parameters(), 'lr': float(_scale_lr), 'name': 'global_scale_offset_mlp'})


        
        global_scale_lr = float(self.gs_config['lr'].get('global_scale', 1e-3))
        param_groups.append({'params': [self.global_scale], 'lr': float(global_scale_lr), 'name': 'global_scale'})

        # Global Pose Embedder
        emb_lr = self.config['deform_pose']['Pose_embedder'].get('lr', 1e-3)
        param_groups.append({'params': self.pose_base_encoder.parameters(), 'lr': float(emb_lr), 'name': 'pose_emb'})


        return param_groups


    def set_curr_level(self, level: int):
        if level < 0 or level >= self.max_level: return
        self.curr_level.fill_(level)
        print(f"Active Level Set to: {level}")
    def set_use_gs_geo(self, boolean):
        self.use_gs_geo.fill_(boolean)
    def set_optimize_mesh(self, boolean):
        self.opt_mesh =boolean

    def save_model(self, path: str, optimizer=None, epoch=None):
        avatar_states = [mesh.get_mesh_state_dict() for mesh in self.mesh_avatars]
        # Custom dict construction...
        save_dict = {
            'model_state_dict': self.state_dict(),
            'avatar_states_list': avatar_states,
            # ... (Other attributes as previously discussed)
        }
        torch.save(save_dict, path)

    def load_model(self, path: str, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        
        # 1. Restore attributes (disp, curr_level, etc.)
        if 'attributes' in checkpoint:
            self.attributes = checkpoint['attributes']

        # 2. Restore avatar mesh state first (geometry recovery)
        # The mesh structure must be in place to initialize the system correctly.
        if 'avatar_states_list' in checkpoint:
            saved_states = checkpoint['avatar_states_list']
            for i, state in enumerate(saved_states):
                if i < len(self.mesh_avatars): 
                    self.mesh_avatars[i].load_mesh_state_dict(state)
        
        # 3. Update v_canon (after the mesh is restored)
        if 'disp' in self.attributes:
            self._update_v_canon_init()

        # 3.5. Recompute face_center_{level} / face_vts_{level} buffers (based on the restored mesh topology)
        # iso_remesh non-determinism / regenerated preprocessing can make the initial face_center face count mismatch the ckpt
        # load_state_dict(strict=False) silently skips shape mismatches, so re-register here to match shapes
        self._compute_all_face_centers()

        # 4. Re-initialize Jacobian system (based on restored mesh info)
        # Note: if this function includes trainable parameters (weights) it must run before load_state_dict;
        # if it only computes buffers it can run here. Placed before state_dict loading to be safe.
        self._init_jacob_system()

        # 5. Restore model weights (including ModuleList and buffers)
        # Weights must be overwritten after the Jacobian system is re-created.
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 6. Restore optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model and Optimizer loaded from {path}")
            return optimizer
            
        print(f"Model loaded from {path}. Current Level: {self.curr_level.item()}")
        return None


    def feature_pooling(self, fine_feats, num_coarse):
        """ Vectorized Differentiable Pooling (Fine -> Coarse) """
        B, N_fine, D = fine_feats.shape
        
        # Determine c2f_idx based on fine_feats size (Reverse mapping logic required)
        # Here we assume the caller knows which level's pooling is happening.
        # But for robustness, we can pass c2f_idx.
        # However, to match your requested signature, we must infer or store the correct idx.
        # NOTE: Caller should pass c2f_idx or we need to find it. 
        # Let's Modify signature slightly to be safer, or look up based on N_fine.
        
        # Finding c2f_idx by N_fine match (Heuristic)
        found_idx = None
        for i in range(self.max_level - 1):
            idx = getattr(self, f'c2f_face_idx_{i}')
            if idx.shape[0] == N_fine:
                found_idx = idx
                break
        
        if found_idx is None:
            raise ValueError(f"Could not find c2f_face_idx for N_fine={N_fine}")

        coarse_feats = torch.zeros((B, num_coarse, D), device=fine_feats.device, dtype=fine_feats.dtype)
        
        # Counts
        ones = torch.ones(N_fine, device=fine_feats.device, dtype=fine_feats.dtype)
        counts = torch.zeros(num_coarse, device=fine_feats.device, dtype=fine_feats.dtype)
        counts.index_add_(0, found_idx, ones)
        counts = torch.clamp(counts, min=1.0).view(1, num_coarse, 1)

        # Sum Pooling
        coarse_feats.index_add_(1, found_idx, fine_feats)
        
        # Avg Pooling
        coarse_feats = coarse_feats / counts
        return coarse_feats

    def c2f_vts_mapping(self, c2f_mat, vts_attributes):
        N_fine = c2f_mat.shape[0] # number of rows of the sparse matrix
        if vts_attributes.ndim==2:
            vts_attributes = vts_attributes.unsqueeze(0)
        B, N_coarse, C = vts_attributes.shape
        feat_permuted = vts_attributes.permute(1, 0, 2)
        feat_flattened = feat_permuted.reshape(N_coarse, -1)
        fine_feat_flattened = torch.sparse.mm(c2f_mat, feat_flattened)
        fine_attributes = fine_feat_flattened.view(N_fine, B, C).permute(1, 0, 2)   
        return fine_attributes
    

    def c2f_face_mapping(self, c2f_idx, face_attributes):
        if face_attributes.ndim==2:
            face_attributes = face_attributes.unsqueeze(0)
        face_attributes = face_attributes[:, c2f_idx, :]
        return face_attributes
        
    def extract_midpoint_recipe(self, new_faces, num_org_verts, num_new_verts_count):
        """
        Traces back the fine mesh face information to find the parents (old vertices) of each midpoint.
        """
        # 1. Extract all edges
        edges = torch.cat([
            new_faces[:, [0, 1]],
            new_faces[:, [1, 2]],
            new_faces[:, [2, 0]]
        ], dim=0)
        
        # 2. Sort and deduplicate
        # (sorted as (smaller index, larger index))
        edges, _ = torch.sort(edges, dim=1) 
        edges = torch.unique(edges, dim=0) 
        
        # 3. Keep only [Old Vertex, New Vertex] edges
        # Condition: column 0 < N (Old), column 1 >= N (New)
        mask_mixed = (edges[:, 0] < num_org_verts) & (edges[:, 1] >= num_org_verts)
        mixed_edges = edges[mask_mixed]
        
        # 4. Sort by New Vertex (column 1)
        # Result: [[Old_A, New_100], [Old_B, New_100], [Old_C, New_101], ...]
        sort_indices = torch.argsort(mixed_edges[:, 1])
        sorted_mixed = mixed_edges[sort_indices]
        
        # 5. Extract parent indices
        # (2*M, 2) -> take only column 0 (Old) and reshape to (M, 2)
        # Each New Vertex must have exactly 2 parents
        parents = sorted_mixed[:, 0].reshape(num_new_verts_count, 2)
        
        return parents

    def get_topology_based_matrix(self, coarse_mesh, fine_mesh):
        """
        Builds the Coarse -> Fine transform matrix by tracing back the topology
        """
        device = self.device
        
        # 1. Convert face info to tensor
        if isinstance(fine_mesh.face, torch.Tensor):
            fine_faces = fine_mesh.face
        else:
            fine_faces = torch.from_numpy(fine_mesh.face).to(device).long()
            
        N = max(coarse_mesh.v_canon.shape)          # number of coarse vertices
        Total = max(fine_mesh.v_canon.shape)        # number of fine vertices
        M = Total - N                          # number of newly created vertices
        
        # 2. Find parents (using the recipe)
        # parents shape: (M, 2) -> [[p1, p2], [p3, p4], ...]
        parents = self.extract_midpoint_recipe(fine_faces, N, M)
        
        # ---------------------------------------------------------
        # 3. Build sparse matrix
        # ---------------------------------------------------------
        
        # (A) Identity part: existing points map 1:1 (weight 1.0)
        # Row: 0 ~ N-1, Col: 0 ~ N-1
        idx_id = torch.arange(N, device=device)
        vals_id = torch.ones(N, device=device)
        
        # (B) Midpoint part: new points take 0.5 from each of their 2 parents
        # Rows: N ~ Total-1 (indices of new points)
        # Each point has 2 parents, so repeat each index twice
        rows_new = torch.arange(N, Total, device=device).repeat_interleave(2)
        
        # Cols: indices of the found parents
        cols_new = parents.flatten() # (M*2, )
        
        # Values: 0.5
        vals_new = torch.full((M * 2,), 0.5, device=device)
        
        # (C) Merge
        rows = torch.cat([idx_id, rows_new])
        cols = torch.cat([idx_id, cols_new])
        vals = torch.cat([vals_id, vals_new])
        
        indices = torch.stack([rows, cols], dim=0)
        
        matrix = torch.sparse_coo_tensor(
            indices, vals, size=(Total, N), device=device
        )
        
        return matrix


    def set_head_idxs(self,mesh_avatar):
        with torch.no_grad():
            lbs_squeeze = copy.deepcopy(mesh_avatar._lbs_weights).squeeze()
            head_idxs = [12,15 ,22,23,24]
            head_mask = torch.zeros_like(lbs_squeeze)[...,0]

            for head_idx in head_idxs:
                head_mask += torch.argmax(lbs_squeeze,dim=-1) == head_idx
            head_vst_idxs = torch.where( head_mask)[0]

            head_idxs = torch.where(torch.isin(mesh_avatar.face, head_vst_idxs).any(dim=1))[0]
        return head_idxs, head_vst_idxs
    
    def set_hand_idxs(self, mesh_avatar):
        with torch.no_grad():
            lbs_squeeze = copy.deepcopy(mesh_avatar._lbs_weights).squeeze()
            lhand_idxs = [i for i in range(25,40)]
            lhand_mask = torch.zeros_like(lbs_squeeze)[...,0]

            for lhand_idx in lhand_idxs:
                lhand_mask += torch.argmax(lbs_squeeze,dim=-1) == lhand_idx
            lhand_vst_idxs = torch.where( lhand_mask)[0]

            lhand_idxs = torch.where(torch.isin(mesh_avatar.face, lhand_vst_idxs).any(dim=1))[0]
            
            lbs_squeeze = copy.deepcopy(mesh_avatar._lbs_weights).squeeze()
            rhand_idxs = [i for i in range(40,55)]
            rhand_mask = torch.zeros_like(lbs_squeeze)[...,0]

            for rhand_idx in rhand_idxs:
                rhand_mask += torch.argmax(lbs_squeeze,dim=-1) == rhand_idx
            rhand_vst_idxs = torch.where( rhand_mask)[0]

            rhand_idxs = torch.where(torch.isin(mesh_avatar.face, rhand_vst_idxs).any(dim=1))[0]
        return lhand_idxs, rhand_idxs, lhand_vst_idxs, rhand_vst_idxs

import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque
from pytorch3d.transforms import (
    matrix_to_rotation_6d, 
    axis_angle_to_matrix
)

class Pose_embedder(nn.Module): # [change 1] inherit from nn.Module
    def __init__(self, cfg, smpl_mesh, device='cuda'):
        super().__init__() # [change 2] initialize parent class
        self.device = device
        self.embedding_type = cfg['embedder']
        
        # smpl_mesh is an nn.Module injected from outside.
        # Storing it as self.smpl_mesh lets PyTorch register it as a submodule automatically.
        # (If LoDAvatar already manages it, this is just a reference)
        self.smpl_mesh = smpl_mesh

        # --- 1. Base (Canonical) Pose Initialization ---
        pose_aa = torch.zeros((1, 63), device=self.device)
        
        # Init legs
        pose_aa[:, 5] = math.radians(smpl_mesh.canon_leg_angle)
        pose_aa[:, 8] = -math.radians(smpl_mesh.canon_leg_angle)
        # Init shoulders
        pose_aa[:, 53] = math.radians(smpl_mesh.canon_sholder_angle)
        pose_aa[:, 50] = -math.radians(smpl_mesh.canon_sholder_angle)

        # [change 3] use register_buffer
        # This way da_pose_aa is moved to the GPU together on model.cuda().
        self.register_buffer('da_pose_aa', pose_aa.view(1, 21, 3))
        
        # The mask is initialized later; it must be declared as None or registered conditionally via register_buffer.
        # For convenience it is set to None here and registered in the init function.
        self.adj_joint_mask = None 

        # --- 2. Select Embedder Function ---
        if self.embedding_type == 'naive':
            self.embed_mode = 'naive' # store a mode string rather than the function itself
            self.out_dim = 126 
        elif self.embedding_type == 'sample':
            self.embed_mode = 'sample' # store a mode string rather than the function itself
            self.out_dim =  78 #126 -48 
            body_idxs = np.arange(1, 22)
            except_idxs = [7,8,10,11,12,15,20,21]
            valid_mask = ~np.isin(body_idxs, except_idxs)
            target_idxs = body_idxs[valid_mask] 
            self.target_idxs = target_idxs
            pose_aa = pose_aa.view(1,-1,3)
            pose_aa = pose_aa[:,target_idxs,:]
            self.register_buffer('da_pose_aa', pose_aa)

        elif self.embedding_type == 'ring':
            self.embed_mode = 'ring'
            self.out_dim = 126
        elif self.embedding_type == 'diffpose':
            self.embed_mode = 'diffpose'
            self.out_dim = 126 + 3 + 6 
        
        if self.embedding_type != 'naive':
            self.init_ring_base_embed_fn(smpl_mesh)

    def forward(self, *args, **kwargs):
        # [change 4] branch on mode in forward
        if self.embed_mode == 'naive':
            return self.naive_embed_fn(*args, **kwargs)
        elif self.embedding_type == 'sample':
            return self.sampled_embed_fn(*args, **kwargs)
        elif self.embed_mode == 'ring':
            return self.ring_base_embed_fn(*args, **kwargs)
        elif self.embed_mode == 'diffpose':
            return self.diff_pose_base_embed_fn(*args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown embedder type: {self.embed_mode}")

    # =========================================================================
    #   Core Helper Function
    # =========================================================================
    def _compute_relative_6d(self, curr_aa, base_aa):
        R_curr = axis_angle_to_matrix(curr_aa) 
        R_base = axis_angle_to_matrix(base_aa) 
        R_diff = torch.matmul(R_base.transpose(-1, -2), R_curr)
        return matrix_to_rotation_6d(R_diff)

    # =========================================================================
    #   Embedder Implementations
    # =========================================================================

    def naive_embed_fn(self, fullpose_dict, face_number):
        fullpose = fullpose_dict['full_pose']
        if fullpose.dim() == 3: 
            body_pose = fullpose[:, 1:22, :]
        else: 
            body_pose = fullpose[:, 3:66].view(-1, 21, 3)

        # self.da_pose_aa is a buffer, so it is directly accessible
        pose_6d = self._compute_relative_6d(body_pose, self.da_pose_aa) 
        pose_flat = pose_6d.view(-1, 1, 21*6)
        return pose_flat.repeat(1, face_number, 1)

    def sampled_embed_fn(self, fullpose_dict, face_number):
        fullpose = fullpose_dict['full_pose']
        #body_idxs = np.arange(1, 22)
        #except_idxs = [7,8,10,11,12,15,20,21]
        #valid_mask = ~np.isin(body_idxs, except_idxs)
        if fullpose.dim() == 3: 
            #target_idxs = body_idxs[valid_mask] 
            body_pose = fullpose[:, self.target_idxs, :]
        else: 
            temp_body_pose = fullpose[:, 3:66].view(-1, 21, 3)
            body_pose = temp_body_pose[:, self.target_idxs, :]
        #ignore foot, head, twist of hand.
        
        # self.da_pose_aa is a buffer, so it is directly accessible
        pose_6d = self._compute_relative_6d(body_pose, self.da_pose_aa) 
        pose_flat = pose_6d.view(-1, 1, 13*6)
        return pose_flat.repeat(1, face_number, 1)


    def ring_base_embed_fn(self, smpl_param, face_number):
        body_pose = smpl_param['full_pose'][..., 3:66].view(-1, 21, 3)
        pose_6d = self._compute_relative_6d(body_pose, self.da_pose_aa)
        B = pose_6d.shape[0]
        pose_expanded = pose_6d[:, None, :, :].repeat(1, face_number, 1, 1)
        
        # use self.adj_joint_mask
        mask = self.adj_joint_mask[None, :, :, None]
        pose_masked = pose_expanded * mask
        return pose_masked.view(B, face_number, -1)



    def diff_pose_base_embed_fn(self, smpl_param_1, smpl_param_2, face_number):
        trans_diff = smpl_param_2['transl'].view(-1, 3) - smpl_param_1['transl'].view(-1, 3)
        pose1 = smpl_param_1['full_pose'] #prev
        pose2 = smpl_param_2['full_pose'] #curr
        
        global_1 = pose1[..., :3].view(-1, 1, 3)
        global_2 = pose2[..., :3].view(-1, 1, 3)
        body_1 = pose1[..., 3:66].view(-1, 21, 3)
        body_2 = pose2[..., 3:66].view(-1, 21, 3)

        global_diff_6d = self._compute_relative_6d(global_2, global_1) 
        body_diff_6d = self._compute_relative_6d(body_2, body_1)       

        B = body_diff_6d.shape[0]
        global_feat = torch.cat([trans_diff, global_diff_6d.view(B, 6)], dim=-1)
        global_feat_expanded = global_feat[:, None, :].repeat(1, face_number, 1) 

        body_feat_expanded = body_diff_6d[:, None, :, :].repeat(1, face_number, 1, 1)
        mask = self.adj_joint_mask[None, :, :, None]
        body_feat_masked = body_feat_expanded * mask
        body_feat_final = body_feat_masked.view(B, face_number, -1) 

        return torch.cat([global_feat_expanded, body_feat_final], dim=-1)

    # =========================================================================
    #   Initialization Logic (Kept mostly same, just cleaned up)
    # =========================================================================
    def init_ring_base_embed_fn(self, smpl_mesh, ring_size=4):
        # (original logic kept)
        parents = smpl_mesh.smpl_model.parents.tolist()
        graph = {i: [] for i in range(len(parents))}
        for child, parent in enumerate(parents):
            if parent != -1:
                graph[child].append(parent)
                graph[parent].append(child)
        
        N = len(parents)
        distance_matrix = np.full((N, N), np.inf)
        for start in range(N):
            queue = deque([(start, 0)])
            visited = set([start])
            while queue:
                current, dist = queue.popleft()
                distance_matrix[start, current] = dist
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        adj_joint_mask = torch.FloatTensor(distance_matrix <= ring_size)
        
        # Only body mask
        body_mask = torch.zeros((N, N))
        body_mask[1:22, 1:22] = 1.0
        adj_joint_mask = adj_joint_mask * body_mask
        
        self.adj_joint_mask = adj_joint_mask.to(self.device)
        
        # Map to Face
        face_lbs_weights = self.smpl_mesh._lbs_weights[0][self.smpl_mesh.face].mean(dim=1)
        joint_idxs = torch.argmax(face_lbs_weights, -1).squeeze()
        self.adj_joint_mask = self.adj_joint_mask[joint_idxs]
        self.adj_joint_mask = self.adj_joint_mask[:, 1:22] # Keep only body columns

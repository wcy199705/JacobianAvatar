
import os
import json
import smplx
import torch
# import igl
# import itertools
import trimesh
import numpy as np
from sklearn.neighbors import KDTree

from scene.object.body_model.smpl_base import SMPLBase
from scene.object.mesh import Mesh
from scene.object.body_model.smpl_utils import batch_rigid_transform, transform_homogeneous
import math
from pytorch3d.transforms import matrix_to_rotation_6d,  axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix
from natsort import natsorted
import glob
from scene.camera.cameraunified import UnifiedCamera
# class SMPLMeshUV(SMPLMesh):
#     pass

# deformable smpl mesh with texture map.
# what if we use face or vertex color instead of texture coordinate?
# -> vertex colors should be updated
class AvatarMesh(SMPLBase, Mesh):
    """
    Assume V_canon is star-posed(Da-posed.) but j canon is v-posed.
    Convert SMPL(-X) template mesh to clothed human mesh
    """
    def __init__(self, smpl_config, device=torch.device('cuda:0')):
        SMPLBase.__init__(self, smpl_config, device=device)

        # newly added variable is displacement vector, not in both Mesh and SMPLBase
        # disp, lbs_weights, v_canon, j_canon supports batch operations (B x V/J x 3) otherwise,
        self._disp = None
        self._t_pose = None  # T-posed smpl vertices, used for calculating chamfer distance
        self._idx2smpl = None
        self._T_da_inv = None
        self.canon_leg_angle = smpl_config['canon_leg_angle']
        self.canon_sholder_angle = smpl_config['canon_sholder_angle']
        self._optimize_lbs = False
        self.num_pca_comp = smpl_config['num_pca_comp']
        self.is_da = True
    @property
    def disp(self):
        return self._disp

    @disp.setter
    def disp(self, disp):
        self._disp = self.to_torch(disp, device=self.device)

    def inverse_softmax(self, x, eps=1e-6):
            """Convert probability (0~1) to logits (-inf~inf)"""
            return torch.log(torch.clamp(x, min=eps))

        
    @property
    def lbs_weights(self):
        # In optimization mode, apply softmax to the logits and return
        if hasattr(self, '_optimize_lbs') and self._optimize_lbs:
            return torch.nn.functional.softmax(self._lbs_logits, dim=-1)
        return self._lbs_weights
    @lbs_weights.setter

    def lbs_weights(self, lbs_weights):
        # If optimizing, decide whether to reset the logic or update the value
        # when the setter is called. Here only the base value is updated.
        self._lbs_weights = self.to_torch(lbs_weights, device=self.device)
        # If a value is forcibly set from outside during optimization, also update the optimization variable
        if hasattr(self, '_optimize_lbs') and self._optimize_lbs:
             self._lbs_logits.data = self.inverse_softmax(self._lbs_weights)

    @property
    def t_pose(self):
        return self._t_pose

    @property
    def v_canon(self):
        """
        Getter function for self.v_shaped
        Returns:
        _v_shaped(Tensor): t-posed smpl(-h/x) vertices after applying shape deformation
        """
        return self._v_canon

    @property
    def j_canon(self):
        """
        Getter function for self.v_shaped
        Returns:
        _v_shaped(Tensor): t-posed smpl(-h/x) vertices after applying shape deformation
        """
        return self._j_canon
    @v_canon.setter
    def v_canon(self, betas):
        if max(betas.shape)<=100:
            _v_canon = self.smpl_model.v_template + smplx.lbs.blend_shapes(betas, self.smpl_model.shapedirs)
            _j_canon = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, _v_canon)
            self._j_canon = _j_canon #self.v_pose2da_pose(_v_canon,betas)
            self._v_canon = self.v_pose2da_pose(_v_canon) # self.v_pose2da_pose(betas)

            #convert2da_posed verts

        else:
            if len(betas.shape) != 3:
                betas = betas.squeeze()[None,:,:]
            self._v_canon = betas  #self.smpl_model.v_template + smplx.lbs.blend_shapes(betas, self.smpl_model.shapedirs)
            #self._j_canon = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, self._v_canon)

    @property
    def T_da_inv(self):
        return self._T_da_inv 
    @T_da_inv.setter
    def T_da_inv(self, da_rot):
        #da_pose = self.get_da_fullpose()
        if da_rot is None:
            da_pose = self.get_da_fullpose()
            da_rot = self.rel_transform(da_pose, inverse=False)
        #da_rot = self.rel_transform(da_pose, inverse=False)
        T_da = torch.matmul(self.lbs_weights.expand(1, -1, -1), da_rot).view(1, -1, 4, 4) 
        T_da_inv = torch.linalg.inv(T_da)
        self._T_da_inv = T_da_inv.detach()



    def init_smpl_mesh(self, smpl_params):
        smpl_output = self.predict(smpl_params)
        if smpl_output.betas.shape[0] != 1:
            smpl_output.betas = torch.mean(smpl_output.betas,dim=0,keepdims=True)
        self.v_canon = smpl_output.betas
        self.rgb = torch.ones_like(self.v_canon).squeeze()*0.3
        #change to da pose.
        
        self.T_da_inv = None

        #self._t_pose = self.v_canon.detach().clone().requires_grad_(False)  # for future purpose
        self._disp = torch.zeros_like(self.v_canon)
        if self.fill_mouth:
            smpl_faces = self.fill_mouth_and_eyes()
        else:
            smpl_faces = self.smpl_model.faces
        Mesh.__init__(self, self.v_canon.squeeze(0), smpl_faces,
                      colors=torch.ones_like(self.v_canon.squeeze(0)) * 0.3, device=self.device)
        return smpl_output

    def detach(self):
        self._xyz = self._xyz.detach()
        self._face = self._face.detach()
        if self.rgb is not None:
            self.rgb = self.rgb.detach()#.cpu().numpy()
        if self.lbs_weights is not None:
            self.lbs_weights = self.rgb.detach()#.cpu().numpy()
        if self.disp is not None:
            self.disp = self.disp.detach()#.cpu().numpy()
        if self.v_canon is not None:
            self.v_canon = self.v_canon.detach()#.cpu().numpy()

        # self.disp.detach().squeeze(0).cpu().numpy(),
        # self.v_canon.detach().squeeze(0).cpu().numpy(),
        # self.lbs_weights.detach().squeeze(0).cpu().numpy())),
    def v_template(self, betas):
        return self.smpl_model.v_template + smplx.lbs.blend_shapes(betas, self.smpl_model.shapedirs)
    # [added] Return parameters to pass to the optimizer
        
    """  
    def subdivide(self):
        vertices, faces = trimesh.remesh.subdivide(
            vertices=np.hstack((self.xyz.detach().cpu().numpy(),
                                self.rgb.detach().cpu().numpy(),
                                self.disp.detach().squeeze(0).cpu().numpy(),
                                self.v_canon.detach().squeeze(0).cpu().numpy(),
                                self.lbs_weights.detach().squeeze(0).cpu().numpy())),
            faces=self.face.detach().cpu().numpy())

        self.xyz = vertices[:, 0:3]
        self.rgb = vertices[:, 3:6]
        self.disp = vertices[:, 6:9]
        # v_canon's setter function takes betas, so we need direct initialization.
        self._v_canon = self.to_torch(vertices[:, 9:12], device=self.device)
        self.lbs_weights = vertices[:, 12:]

        # these are batch variables
        self._disp = self._disp.unsqueeze(0)
        self._v_canon = self._v_canon.unsqueeze(0)
        self._lbs_weights = self._lbs_weights.unsqueeze(0)
        self.face = faces
        self.set_connectivity()
    """
    def subdivide_w_index_cy(self):
        """
        subdivide the mesh with assigned attributes
        lbs_weights, canonical_vertices, displacements, colors, and positions
        relying on trimesh.remesh function, which means the
        gradient will not flow anymore.
        """
        vertices, faces, index_dict = trimesh.remesh.subdivide(
            vertices=np.hstack((self.xyz.squeeze().detach().cpu().numpy(),
                                self.rgb.detach().cpu().numpy(),
                                self.disp.detach().squeeze().cpu().numpy(),
                                self.v_canon.detach().squeeze().cpu().numpy(),
                                self.lbs_weights.detach().squeeze().cpu().numpy())),
            faces=self.face.detach().cpu().numpy(),
            return_index=True)

        self.xyz = vertices[:, 0:3]
        self.rgb = vertices[:, 3:6]
        self.disp =vertices[:,6:9]
        self.v_canon = self.to_torch(vertices[:, 9:12],device=self.device)
        # v_canon's setter function takes betas, so we need direct initialization.
        self.lbs_weights = vertices[:, 12:]

        # these are batch variables
        self._disp = self._disp.squeeze().unsqueeze(0)
        self._v_canon = self._v_canon.squeeze().unsqueeze(0)
        self._lbs_weights = self._lbs_weights.squeeze().unsqueeze(0)
        self.face = faces
        self.set_connectivity()
        self.init_T_da_inv()
        return index_dict
    def poisson_remesh(self, depth=9):
        from utils.geometry.mesh_utils import poisson_trimesh, postprocess_mesh
        #assume self._v_canon is da posed verts
        #self._v_canon = self.deform_canon2star(inverse=False)
        mesh_before = self.to_trimesh(return_canon=True)
        mesh_after = poisson_trimesh(mesh_before, depth=depth)
        mesh_after = postprocess_mesh(mesh_after)
        self._lbs_weights = self.lbs_transfer_nearest(mesh_before.vertices, mesh_after.vertices, self.lbs_weights,
                                                 device=self.device)
        self._v_canon = self.to_torch(mesh_after.vertices, device=self.device)
        self.face = mesh_after.faces
        self.xyz = mesh_after.vertices[None, ...]
        self.rgb = torch.ones_like(self._xyz).to(self.device) * 0.5
        self.disp = torch.zeros_like(self._v_canon).to(self.device)
        self.set_connectivity()
        da_pose = self.get_da_fullpose()
        da_rot = self.rel_transform(da_pose, inverse=False)
        #T_da = torch.matmul(self.lbs_weights.expand(1, -1, -1), da_rot).view(1, -1, 4, 4) 
        #T_da_inv = torch.linalg.inv(T_da)
        self.T_da_inv =da_rot #T_da_inv
        self.disp = torch.zeros_like(self._v_canon)



        #self._v_canon = self.deform_canon2star(inverse=True)

    def deform_canon2star(self, v_canon=None, lbs_weights=None, inverse=False):
        import pdb;pdb.set_trace()
        """
        #full_pose = torch.zeros(size=(1, self.smpl_model.NUM_POSES)).to(self.device) # 87 -> self.smpl_model.NUM_POSES
        full_pose = self.smpl_model(**self.get_empty_params(pca_dim=self.num_pca_comp,device=self.device),return_full_pose=True)['full_pose']
        
        # legs (5, 8), shoulders (50, 53)
        full_pose[:, 5] = math.radians(self.canon_leg_angle)
        full_pose[:, 8] = -math.radians(self.canon_leg_angle)

        # shoulders
        full_pose[:, 53] = math.radians(self.canon_sholder_angle)
        full_pose[:, 50] = -math.radians(self.canon_sholder_angle)

        if v_canon is None:
            v_canon = self.v_canon

        if len(v_canon.shape) == 2:
            v_canon = v_canon[None, ...]

        B = v_canon.shape[0]
        if B > 1:
            v_canon = v_canon.expand(B, -1, -1)

        rel_rot = self.rel_transform(full_pose, inverse=inverse)
        if lbs_weights is None:
            T_v = torch.matmul(self._lbs_weights.expand(B, -1, -1), rel_rot).view(B, -1, 4, 4)
        else:
            T_v = torch.matmul(lbs_weights.expand(B, -1, -1), rel_rot).view(B, -1, 4, 4)

        return transform_homogeneous(T_v, v_canon)
        """


    def get_dapose_param(self,betas, scale=1.0):
        zero_poses = self.get_empty_params(pca_dim=self.num_pca_comp,device=self.device)
        zero_poses['body_pose'][:,2] =  math.radians(self.canon_leg_angle)
        zero_poses['body_pose'][:,5] =  -math.radians(self.canon_leg_angle)
        zero_poses['body_pose'][:,47] =  -math.radians(self.canon_sholder_angle)
        zero_poses['body_pose'][:,50] =  math.radians(self.canon_sholder_angle)
        zero_poses['betas'] = betas
        if isinstance(scale, float):
            zero_poses['scale'][:,0] = scale
        else:
            zero_poses['scale']= scale
        zero_poses = self.to_param(zero_poses,device=self.device)
        return zero_poses

    def set_da_canon(self, verts, faces):
        #transfer verts
        self._lbs_weights = self.lbs_transfer_nearest(self._v_canon, verts, self.lbs_weights ) #check lbs transfer for hands. - might be not aligned ...
        #Mesh(vertices=self._v_canon.squeeze(),indices=self.face,colors=torch.ones_like(self._v_canon.squeeze())).to_trimesh().export('smpl.ply')
        #Mesh(vertices=verts.squeeze(),indices=faces,colors=torch.ones_like(verts.squeeze())).to_trimesh().export('hmr.ply')

        self._v_canon = verts
        self.xyz = self._v_canon.clone().squeeze()
        self.face = faces
        #fullbody!!!!
        da_pose = self.get_da_fullpose()
        da_rot = self.rel_transform(da_pose, inverse=False)
        #T_da = torch.matmul(self.lbs_weights.expand(1, -1, -1), da_rot).view(1, -1, 4, 4) 
        #T_da_inv = torch.linalg.inv(T_da)
        self.T_da_inv =da_rot #T_da_inv
        self.disp = torch.zeros_like(verts)


    def init_T_da_inv(self):
        #transfer verts
        #self._lbs_weights = self.lbs_transfer_nearest(self._v_canon, verts, self.lbs_weights ) #check lbs transfer for hands. - might be not aligned ...
        #Mesh(vertices=self._v_canon.squeeze(),indices=self.face,colors=torch.ones_like(self._v_canon.squeeze())).to_trimesh().export('smpl.ply')
        #Mesh(vertices=verts.squeeze(),indices=faces,colors=torch.ones_like(verts.squeeze())).to_trimesh().export('hmr.ply')

        #self._v_canon = verts
        #self.xyz = self._v_canon.clone().squeeze()
        #self.face = faces
        #fullbody!!!!
        da_pose = self.get_da_fullpose()
        da_rot = self.rel_transform(da_pose, inverse=False)
        #T_da = torch.matmul(self.lbs_weights.expand(1, -1, -1), da_rot).view(1, -1, 4, 4) 
        #T_da_inv = torch.linalg.inv(T_da)
        self.T_da_inv = da_rot #T_da_inv
        #elf.disp = torch.zeros_like(verts)


    
    def FS_fullposedict(self, full_pose_dict, j_canon=None, rel_rot=None, v_canon=None, T_da_inv=None, return_J=False):
        B = full_pose_dict['full_pose'].shape[0]

        if T_da_inv is None:
            T_da_inv = self.T_da_inv.clone()

            #if da_rot is None:
            #    da_rot = self.rel_transform(self.get_dapose_param(betas=smpl_params['betas']), inverse=False)
            #T_da = torch.matmul(self.lbs_weights.expand(1, -1, -1), da_rot).view(1, -1, 4, 4)                
            #T_da_inv = torch.linalg.inv(T_da)
        if T_da_inv.shape[0] != B:
            T_da_inv = T_da_inv.expand(B, -1, -1, -1)
        if v_canon is None:
            v_canon = self.v_canon.expand(B,-1,-1)
        else:
            v_canon = v_canon.expand(B,-1,-1)


        if rel_rot is None:  # can be pre-computed
            rel_rot = self.rel_transform(full_pose_dict['full_pose'].view(B,-1,3), inverse=False)
        T_v = torch.matmul(self._lbs_weights.expand(B, -1, -1), rel_rot).view(B, -1, 4, 4)
        T_da2pose = T_v@ T_da_inv

        v_deformed = transform_homogeneous(T_da2pose, v_canon)
        
        if full_pose_dict is not None:
            v_deformed = (v_deformed + full_pose_dict['transl'].view(B,-1,3)) #* full_pose_dict['scale']
            if 'scale' in full_pose_dict.keys():
                v_deformed = v_deformed*full_pose_dict['scale']
        if return_J:
            rel_rot_mat = rel_rot.view(B, -1, 4, 4)
            j_deforemd = rel_rot_mat[..., :3, 3]
            #if j_canon is None:
                
            #    j_canon = self.j_canon
            j_deforemd =  (j_deforemd + full_pose_dict['transl'].view(B,-1,3)) #* smpl_params['scale']
            if 'scale' in full_pose_dict.keys():
                j_deforemd = j_deforemd*full_pose_dict['scale']

            return v_deformed.squeeze(0),  j_deforemd.squeeze(0)

        return v_deformed.squeeze(0)

    def rel_transform(self, full_pose, inverse=False):
        """
        Function for calculating relative transform matrix A
        Args:
            full_pose: full_pose parameters (smpl_ouput)
            inverse: inverse transformation matrix if this is set to true otherwise forward transformation
        Returns:
            rel_rot: relative transformation matrix
        """
        B = full_pose.shape[0]

        #if len(full_pose.shape)==3:
        full_pose = full_pose.view(-1,3)
        rot_mats = smplx.lbs.batch_rodrigues(full_pose).view([B, -1, 3, 3])
        _, rel_rot = batch_rigid_transform(rot_mats,
                                           self._j_canon[:, :55, :].repeat(B,1,1),
                                           self.smpl_model.parents,
                                           inverse=inverse,
                                           dtype=torch.float32)
        rel_rot = rel_rot.reshape(B, self.num_joints, 16)
        return rel_rot  # relative transform matrices (A matrix)

    def forward_skinning(self, smpl_output, j_canon=None, rel_rot=None, v_canon=None, smpl_params=None, T_da_inv=None, return_J=False):
        B = smpl_output.poses.shape[0]

        if T_da_inv is None:
            T_da_inv = self.T_da_inv.clone()

            #if da_rot is None:
            #    da_rot = self.rel_transform(self.get_dapose_param(betas=smpl_params['betas']), inverse=False)
            #T_da = torch.matmul(self.lbs_weights.expand(1, -1, -1), da_rot).view(1, -1, 4, 4)                
            #T_da_inv = torch.linalg.inv(T_da)
        if T_da_inv.shape[0] != B:
            T_da_inv = T_da_inv.expand(B, -1, -1, -1)
        if v_canon is None:
            v_canon = self.v_canon.expand(B,-1,-1)
        else:
            v_canon = v_canon.expand(B,-1,-1)

        #if v_canon is None:
        #    if self.v_canon is None:
        #        # initialize canonical SMPL(-H/X) vertices from betas
        #        self.v_canon = smpl_output.betas
        #    v_canon = self.v_canon

        if rel_rot is None:  # can be pre-computed
            rel_rot = self.rel_transform(smpl_output.full_pose, inverse=False)
        T_v = torch.matmul(self._lbs_weights.expand(B, -1, -1), rel_rot).view(B, -1, 4, 4)
        T_da2pose = T_v@ T_da_inv

        v_deformed = transform_homogeneous(T_da2pose, v_canon)

        if smpl_params is not None:
            v_deformed = (v_deformed + smpl_params['transl']) * smpl_params['scale']
        if return_J:
            if j_canon is None:
                j_canon = self.j_canon
                j_deforemd = transform_homogeneous(T_v, j_canon)
                j_deforemd =  (j_deforemd + smpl_params['transl']) * smpl_params['scale']
            return v_deformed.squeeze(0),  j_deforemd.squeeze(0)

        return v_deformed.squeeze(0)


    def inverse_skinning(self, smpl_output, v_posed=None, rel_rot=None, smpl_params=None):
        if v_posed is None:
            v_posed = self.xyz
        else:
            v_posed = self.to_torch(v_posed, device=self.device)

        if len(v_posed.shape) == 2:
            v_posed = v_posed.unsqueeze(0)

        if smpl_params is not None:
            v_posed = v_posed / smpl_params['scale'] - smpl_params['transl']

        if rel_rot is None:  # can be pre-computed
            rel_rot = self.rel_transform(smpl_output.full_pose, inverse=True)

        T = torch.matmul(self.lbs_weights, rel_rot).view(1, -1, 4, 4)
        v_canon = transform_homogeneous(T, v_posed)

        #v_canon2_da_pose
        self.forward_skinning()

        return v_canon

    def save_smpl_mesh(self, filename):
        mesh_state=self.get_mesh_state_dict()
        torch.save(mesh_state, filename)
        print("Mesh state saved successfully.")
    """
    def save_smpl_mesh(self, filename):
        # pickle? npy?
        output = {"xyz": self._xyz,
                  "face": self._face,
                  "lbs_weights": self._lbs_weights,
                  "v_canon": self.v_canon,
                  "index": self._idx2smpl,
                  "v_normal": self.norm_xyz,
                  "rgb": self._rgb}

        with open(filename, "wb") as f:  # "wb": binary write mode
            pickle.dump(output, f)
    """

    def decimate(self, num_faces=5000, canonical=True):
        from igl import decimate
        if canonical:
            V1 = self.v_canon.squeeze(0).detach().cpu().numpy().astype(np.double)
        #else:
        #    V1 = self.xyz.detach().cpu().numpy().astype(np.double)
        F1 = self.face.detach().cpu().numpy().astype(np.int32)
        # V2, F2, _, J = decimate(V1, F1, num_faces=num_faces)  # from gpytoolbox import decimate
        _, V2, F2, _, J = decimate(V1, F1, max_m=num_faces)  # from igl import decimate

        if canonical:
            self._v_canon = self.to_torch(V2, device=self.device).unsqueeze(0)
        else:
            self.xyz = V2

        self.face = F2
        J = torch.IntTensor(J).to(self.device)
        self.lbs_weights = self.lbs_weights[:, J, :]
        if self.rgb is not None:
            self.rgb = self.rgb[J, :]
        if self.disp is not None:
            self.disp = self.disp[:, J, :]
        else:
            self.disp = torch.zeros_like(self.v_canon).to(self.device)
        self.set_connectivity()

    def lbs_transfer_nearest(self, V1, V2, W, device=torch.device('cuda')):
        if isinstance(V1,torch.Tensor):
            V1 = V1.detach().cpu().numpy()
        if isinstance(V2,torch.Tensor):
            V2 = V2.detach().cpu().numpy()
        if isinstance(W,torch.Tensor):
            W = W.detach().cpu().numpy()

        W = W.squeeze()
        W = W[None,:,:]
        V1 = V1.squeeze()
        V2 = V2.squeeze()
        kdtree = KDTree(V1, leaf_size=30)
        kd_idx = kdtree.query(V2, k=1, return_distance=False)
        W_new = W[:, kd_idx.squeeze(), :]
        return self.to_torch(W_new, device=device)
    


    def lbs_transfer_robust(self, V1, F1, N1, V2, F2, N2, W):
        """
        Robust Skin Weights Transfer via Weight Inpainting
        in SIGGRAPH Asia 2023 Technical Communications
        by Epic Games
        Args:
            V1: source vertices (V1 x 3)
            F1: source faces (F1 x 3)
            N1: source vertex normals (V1 x 3)
            V2: target vertices (V2 x 3)
            F2: target faces (F2 x 3)
            N2: target vertex normals (V2 x 3)
            W: linear blend skinning weights (V1 x J) J: number of joints
        Returns:
            W2: transferred lbs weights for V2 (target mesh vertices)
        """
        from utils.geometry.robust_lbs_transfer import find_matches_closest_surface, inpaint, smooth
        dDISTANCE_THRESHOLD = 0.05  # threshold distance D
        dDISTANCE_THRESHOLD_SQRD = dDISTANCE_THRESHOLD * dDISTANCE_THRESHOLD
        dANGLE_THRESHOLD_DEGREES = 30  # threshold angle theta in degrees

        # for every vertex on the target mesh find the closest point on the source mesh and copy weights over
        matched, lbs_interpolated = find_matches_closest_surface(V1.astype(np.float64), F1.astype(np.int64), N1.astype(np.float64),
                                                                 V2.astype(np.float64), F2.astype(np.int64), N2.astype(np.float64),
                                                                 W.astype(np.float64),
                                                                 dDISTANCE_THRESHOLD_SQRD,
                                                                 dANGLE_THRESHOLD_DEGREES)
        # lbs_weights = lbs_interpolated
        lbs_weights = inpaint(V2, F2, lbs_interpolated, matched)
        lbs_weights, _ = smooth(V2, F2, lbs_weights, matched, dDISTANCE_THRESHOLD,
                                num_smooth_iter_steps=10, smooth_alpha=0.2)
        # np.sum(matched == False) / (np.sum(matched == True) + np.sum(matched == False)) # % of false matches
        return self.to_torch(lbs_weights, device=self.device).unsqueeze(0)

    def to_trimesh(self, posed_vts=None, return_canon=True):

        if self.is_da:
            if return_canon:
                vts = self.v_canon
            else:
                vts = posed_vts
        else:
            if return_canon:
                vts = self._v_canon + self._disp
        vertex_color = self.rgb.detach().cpu()
        if vertex_color.max() <= 1.0:
            vertex_color = vertex_color.numpy() * 255
        else:
            vertex_color = vertex_color.numpy()

        mesh = trimesh.Trimesh(vertices=vts.squeeze(0).detach().cpu().numpy(),
                               faces=self.face.detach().cpu().numpy(),
                               vertex_colors=vertex_color.astype(np.int32),
                               process=False)
        return mesh

    def set_semantic_labels(self, vertices):
        def get_index(key, threshold=1e-3):
            dist, _ = self.semantic[key].query(vertices, k=1, return_distance=True)
            tmp_list = (dist.squeeze(1) < threshold).tolist()
            index = []
            for i, v in enumerate(tmp_list):
                if v is True:
                    index.append(i)
            return index

        self.idx2body = {}
        for key in self.target_parts.keys():
            self.idx2body[key] = get_index(key, threshold=self.target_parts[key])

        if 'leftHand' in self.idx2body.keys() and 'rightHand' in self.idx2body.keys():
            no_hands_index = []
            for k in range(vertices.shape[0]):
                if k not in self.idx2body['leftHand'] and k not in self.idx2body['rightHand']:
                    no_hands_index.append(k)
            self.idx2body['noHands'] = no_hands_index

    #copy from smplx_ezmocap.py.
    """
    def transfer_semantic_labels(self, betas):
        v_template = self.v_template(betas)

        smpl_vts = v_template.squeeze(0).detach().cpu().numpy()
        custom_vts = self._v_canon.squeeze(0).detach().cpu().numpy()

        kdtree = KDTree(smpl_vts, leaf_size=30)
        kd_idx = kdtree.query(custom_vts, k=1, return_distance=False)

        semantic = np.asarray([0 for k in range(10475)])
        semantic[self.v_label['left_arm']] = 1
        semantic[self.v_label['right_arm']] = 2
        semantic[self.v_label['left_leg']] = 3
        semantic[self.v_label['right_leg']] = 4

        new_idx = semantic[kd_idx]
        new_num = len(kd_idx)
        self.v_label['custom_body'] = [k for k in range(new_num) if new_idx[k] == 0]
        self.v_label['custom_left_arm'] = [k for k in range(new_num) if new_idx[k] == 1]
        self.v_label['custom_right_arm'] = [k for k in range(new_num) if new_idx[k] == 2]
        self.v_label['custom_left_leg'] = [k for k in range(new_num) if new_idx[k] == 3]
        self.v_label['custom_right_leg'] = [k for k in range(new_num) if new_idx[k] == 4]
    """


    def v_canon_update_disp_base(self):
        self.v_canon = self.v_canon + self.disp
        self.disp = torch.zeros_like(self.disp)
        self.xyz = self.v_canon.clone()

    @staticmethod
    def list2batch(params,avg_beta=True):
        # works for batch size of 1 (parametric models)
        batch_params = {}
        for key in params[0].keys():
            batch_params[key] = [e[key].detach().clone() for e in params]
            batch_params[key] = torch.concatenate(batch_params[key], dim=0)
        
        batch_params['betas'] = torch.mean(batch_params['betas'],dim=0,keepdim=True)
        return batch_params

    @staticmethod
    def batch2list(batch_params):
        params_new = []
        if 'poses' in batch_params:
            batch = batch_params['poses'].shape[0]
        elif 'body_pose' in batch_params:
            batch = batch_params['body_pose'].shape[0]
        elif 'full_pose' in batch_params:
            batch = batch_params['full_pose'].shape[0]

        else:
            assert "invalid template model parameters."

        for k in range(batch):
            params = {}
            for key in batch_params.keys():
                if batch_params[key].shape[0] == batch:
                    params[key] = batch_params[key][k:k+1].detach().clone()
                elif batch_params[key].shape[0] == 1:  # merged shapes
                    params[key] = batch_params[key].detach().clone()
            params_new.append(params)
        return params_new
    
    def to_param(self,params,device=torch.device('cuda')):
        if isinstance(params,dict):
            for key in params.keys():
                params[key] = params[key].to(device)
        return params

    def v_pose2da_pose(self, v_verts):

        da_pose_fullpose= self.get_da_fullpose()
        rel_rot = self.rel_transform(da_pose_fullpose, inverse=False)

        T = torch.matmul(self.lbs_weights, rel_rot).view(1, -1, 4, 4)
        da_verts = transform_homogeneous(T, v_verts)
        return da_verts#.squeeze(0)
    

        #if smpl_params is not None:
        #    v_deformed = (v_deformed + smpl_params['transl']) * smpl_params['scale']
    def get_da_fullpose(self):
        if self.smpl_model.use_pca:
            da_pose_fullpose = self.smpl_model(**self.get_empty_params(pca_dim=self.num_pca_comp,device=self.device),return_full_pose=True)['full_pose']
        else:
            da_pose_fullpose = self.smpl_model(**self.get_empty_params(pca_dim=45,device=self.device),return_full_pose=True)['full_pose']

        # legs (5, 8), shoulders (50, 53)
        da_pose_fullpose[:, 5] = math.radians(self.canon_leg_angle)
        da_pose_fullpose[:, 8] = -math.radians(self.canon_leg_angle)

        # shoulders
        da_pose_fullpose[:, 53] = math.radians(self.canon_sholder_angle)
        da_pose_fullpose[:, 50] = -math.radians(self.canon_sholder_angle)
        return da_pose_fullpose
    

    #def smplx2full_pose_dict(self,smplx_batch_params):
    #    full_posedict = {}
    ##    smpl_output = self.smpl_model(**smplx_batch_params, return_full_pose=True)
     #   full_posedict['full_pose'] = smpl_output['full_pose']
     #   full_posedict['transl'] =  smplx_batch_params['transl']
     #   full_posedict['betas'] = torch.mean(smpl_output['betas'],dim=0,keepdim=True)
    #    full_posedict['expression'] = smpl_output['expression']

    #    return full_posedict        
    
    def full_pose_dict2smplx(self, full_posedict):
        full_pose = full_posedict['full_pose']  # shape: (B, 165)
        if len(full_pose.shape) !=2:
            full_pose = full_pose.view(full_pose.shape[0],-1)
        # Define SMPL-X parameter dimensions (axis-angle)
        # Global Orient: 3
        # Body Pose: 21 * 3 = 63
        # Jaw Pose: 3
        # Left Eye: 3
        # Right Eye: 3
        # Left Hand: 15 * 3 = 45
        # Right Hand: 15 * 3 =full_pose_dict2smplx 45
        # Total: 165
        
        # Slice in order
        global_orient = full_pose[:, 0:3]
        body_pose = full_pose[:, 3:66]
        jaw_pose = full_pose[:, 66:69]
        leye_pose = full_pose[:, 69:72]
        reye_pose = full_pose[:, 72:75]
        left_hand_pose = full_pose[:, 75:120]
        right_hand_pose = full_pose[:, 120:165]
        if self.smpl_model.use_pca:
            left_comps = torch.tensor(self.smpl_model.np_left_hand_components).to(self.device)
            right_comps = torch.tensor(self.smpl_model.np_right_hand_components).to(self.device)

            # 2. Prepare the mean pose (very important!)
            left_mean = self.smpl_model.left_hand_mean #torch.tensor(self.smpl_model.left_hand_mean).to(self.device)
            right_mean = self.smpl_model.right_hand_mean  #torch.tensor(self.smpl_model.right_hand_mean).to(self.device)

            left_hand_pose = self.full_pose2pca(left_hand_pose, left_comps, left_mean)
            right_hand_pose = self.full_pose2pca(right_hand_pose, right_comps, right_mean)
        smplx_batch_params = {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'jaw_pose': jaw_pose,
            'left_eye_pose': leye_pose,
            'right_eye_pose': reye_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'transl': full_posedict.get('transl'),      # take it if present, otherwise None
            'betas': full_posedict.get('betas'),
            'expression': full_posedict.get(
            'expression', 
            torch.zeros((full_pose.shape[0], self.smpl_model.num_expression_coeffs), device=self.device))
            } 
        return smplx_batch_params
    
    def pca2full_pose(self, pca_pose, components, mean_pose):
        """
        PCA params -> Axis-Angle (45 dims)
        Formula: Full = (PCA @ Components) + Mean
        """
        # pca_pose: (B, num_pca_comps)
        # components: (num_pca_comps, 45)
        # mean_pose: (45,)
        
        # 1. Dot product (reconstruction)
        recon_pose = torch.matmul(pca_pose, components)
        
        # 2. Add the mean
        recon_pose = recon_pose + mean_pose
        
        return recon_pose

    def smplx2full_pose_dict(self, smplx_batch_params):
        # 1. Get reference tensor info (batch size, device, dtype)
        # global_orient is assumed to be required.
 #smplx_batch_params['global_orient']
        body_pose = smplx_batch_params['body_pose']             # (B, 63)

        batch_size = body_pose.shape[0]
        device = body_pose.device
        dtype = body_pose.dtype

        # 2. Get each parameter (create zeros if missing)
        global_orient = smplx_batch_params.get('global_orient', 
                torch.zeros((batch_size, 3), device=device, dtype=dtype))

        # Jaw & Eyes: (B, 3) zero vector if missing
        jaw_pose = smplx_batch_params.get('jaw_pose', 
                                          torch.zeros((batch_size, 3), device=device, dtype=dtype))
        if 'leye_pose' in smplx_batch_params.keys():
            leye_pose = smplx_batch_params.get('leye_pose', 
                                    torch.zeros((batch_size, 3), device=device, dtype=dtype))
        else:
            leye_pose = smplx_batch_params.get('left_eye_pose', 
                                               torch.zeros((batch_size, 3), device=device, dtype=dtype))
        if 'reye_pose' in smplx_batch_params.keys():
            reye_pose = smplx_batch_params.get('reye_pose', 
                                    torch.zeros((batch_size, 3), device=device, dtype=dtype))
        else:
            reye_pose = smplx_batch_params.get('right_eye_pose', 
                                               torch.zeros((batch_size, 3), device=device, dtype=dtype))

        # 3. Determine hand parameter dims (depends on whether PCA is used)
        if self.smpl_model.use_pca:
            # With PCA: dims equal to the number of components (usually 6, 12, or 45)
            # Infer dims from the shape of components in smpl_model
            n_comps = self.smpl_model.np_left_hand_components.shape[0]
            hand_dim = n_comps
        else:
            # Without PCA: full axis-angle (15 joints * 3 = 45)
            hand_dim = 45

        left_hand_pose = smplx_batch_params.get('left_hand_pose', 
                                                torch.zeros((batch_size, hand_dim), device=device, dtype=dtype))
        right_hand_pose = smplx_batch_params.get('right_hand_pose', 
                                                 torch.zeros((batch_size, hand_dim), device=device, dtype=dtype))

        # Shape fix (flatten in case input comes as (B, J, 3))
        if len(body_pose.shape) > 2: body_pose = body_pose.view(body_pose.shape[0], -1)
        
        # Hand PCA handling (if in PCA form -> restore to full axis-angle)
        if self.smpl_model.use_pca:
            left_comps = torch.tensor(self.smpl_model.np_left_hand_components).to(self.device)
            right_comps = torch.tensor(self.smpl_model.np_right_hand_components).to(self.device)
            left_mean = self.smpl_model.left_hand_mean
            right_mean = self.smpl_model.right_hand_mean

            # If zeros are given -> (0 @ Comp) + Mean = mean pose (naturally gives the mean hand shape)
            left_hand_pose = self.pca2full_pose(left_hand_pose, left_comps, left_mean)
            right_hand_pose = self.pca2full_pose(right_hand_pose, right_comps, right_mean)

        # Concatenate in order
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        full_pose = torch.cat([
            global_orient,
            body_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            left_hand_pose,
            right_hand_pose
        ], dim=1) # (B, 165)
        if 'transl' in smplx_batch_params.keys():
            transl = smplx_batch_params.get('transl')
        elif 'trans' in smplx_batch_params.keys():
            transl = smplx_batch_params.get('trans')
        else:
            transl = torch.zeros_like(reye_pose)

        full_pose_dict = {
            'full_pose': full_pose,
            'transl': transl,
            'betas': smplx_batch_params.get('betas'),
            'expression': smplx_batch_params.get('expression',torch.zeros((full_pose.shape[0], self.smpl_model.num_expression_coeffs),device=self.device))
        }
        return full_pose_dict

    def full_pose2pca(self, full_hand_pose, hand_components, hand_mean):
        """
        full_hand_pose: (B, 45) - Axis-angle rotations for 15 joints (flattened)
        hand_components: (N_pca, 45) - PCA Basis vectors
        hand_mean: (45,) or (B, 45) - Mean hand pose
        """
        
        # 1. Subtract the mean pose (centering)
        # PCA represents deviation from the mean, so the mean must be subtracted first.
        centered_pose = full_hand_pose - hand_mean
        
        # 2. Prepare the inverse (or transpose) of the component matrix
        # SMPL PCA basis is usually orthonormal so .T (transpose) suffices,
        # but pinv can be used for generality, or .T for speed.
        # shape: (N_pca, 45) -> (45, N_pca)
        comp_inv = torch.linalg.pinv(hand_components)
        
        # 3. Recover PCA coefficients
        # (B, 45) x (45, N_pca) -> (B, N_pca)
        pca_coeffs = torch.matmul(centered_pose, comp_inv)
        
        return pca_coeffs
    def full_pose_to_sixrot(self,full_pose):
        original_shape_dim= len(full_pose.shape)

        if original_shape_dim==2:
            full_pose = full_pose.view(full_pose.shape[0],-1,3)
        full_pose = axis_angle_to_matrix(full_pose)
        sixrot = matrix_to_rotation_6d(full_pose)
        if original_shape_dim==2:
            sixrot = sixrot.view(sixrot.shape[0],-1)
        return sixrot
    def sixrot_to_full_pose(self, sixrot):
        original_shape_dim= len(sixrot.shape)
        if original_shape_dim==2:
            sixrot = sixrot.view(sixrot.shape[0],-1,6)

        rotmat = rotation_6d_to_matrix(sixrot)
        fullpose = matrix_to_axis_angle(rotmat)#, rotation_6d_to_matrix
        if original_shape_dim==2:
            fullpose = fullpose.view(fullpose.shape[0],-1)
        return fullpose

    def get_mesh_state_dict(self):
            """
            Returns the core state of the current mesh (topology, geometry, attributes).
            Used to save the state modified by subdivision, etc.
            """
            state = {
                'v_canon': self._v_canon,       # most important: deformed canonical vertices
                'lbs_weights': self._lbs_weights, # changes when the vertex count changes
                'disp': self._disp,
                'face': self.face,              # topology (connectivity)
                'rgb': self.rgb,
                'T_da_inv': self._T_da_inv,     # cached computed inverse matrix
                'j_canon': self._j_canon,
                # add if needed:
                # 'idx2smpl': self._idx2smpl, 
            }
            return state

    def load_ckpt(self, file_name):
        state = torch.load(file_name)
        self.load_mesh_state_dict(state)

    def load_mesh_state_dict(self, state):
        """
        Restores mesh information from a saved state dictionary.
        """
        # Restore tensors while moving them to the current device
        if 'v_canon' in state: 
            self._v_canon = state['v_canon'].to(self.device)
            self.xyz = self._v_canon.clone().squeeze()
        if 'lbs_weights' in state:
            self._lbs_weights = state['lbs_weights'].to(self.device)
        if 'disp' in state:
            self._disp = state['disp'].to(self.device)
        if 'face' in state:
            self.face = state['face'].to(self.device)
            # May need to call the Mesh parent-class method to reset connectivity (neighbor info, etc.)
            self.set_connectivity() 
        if 'rgb' in state:
            self.rgb = state['rgb'].to(self.device)
        if 'T_da_inv' in state:
            self._T_da_inv = state['T_da_inv'].to(self.device) if state['T_da_inv'] is not None else None


    def load_animation_poses_exavatar(self, scene_folder,sampling_rate=1):

        cam_param_folder = os.path.join(scene_folder,'cam_params')
        smplx_param_folder = os.path.join(scene_folder,'smplx_params_smoothed')

        smplx_param_files = natsorted(glob.glob(os.path.join(smplx_param_folder,'*.json')))[::sampling_rate]
        cam_files = natsorted(glob.glob(os.path.join(cam_param_folder,'*.json')))[::sampling_rate]


        cameras =[]
        smplx_param_list = []
        for idx, smplx_param_file in enumerate(smplx_param_files):
            cam_file = cam_files[idx]
            with open(smplx_param_file, 'r', encoding='utf-8') as f:
                smplx_param = json.load(f)

            with open(cam_file, 'r', encoding='utf-8') as f:
                cam_param = json.load(f)
            smplx_param['global_orient'] = torch.tensor(smplx_param['root_pose']).to(self.device).view(1,-1)
            for key in smplx_param.keys():
                smplx_param[key] = torch.tensor(smplx_param[key]).to(self.device)[None,:].view(1,-1)
            full_pose = self.smplx2full_pose_dict(smplx_param)
            #smplx_params = self.full_pose_dict2smplx(full_pose)

            intrinsic = np.eye(4)
            intrinsic[0,0] = cam_param['focal'][0]
            intrinsic[1,1] = cam_param['focal'][0]
            intrinsic[0,2] = cam_param['princpt'][0]
            intrinsic[1,2] = cam_param['princpt'][1]
            extrinsic = np.eye(4)
            extrinsic[:3,:3] = cam_param['R']
            extrinsic[:3,3] = cam_param['t']
            intrinsic = torch.Tensor(intrinsic)[:3,:3]
            extrinsic = torch.Tensor(extrinsic)

            extrinsic[1:,3]*= -1
            camera = UnifiedCamera.perspective_camera_from_K(K=intrinsic,width=int(cam_param['princpt'][0]*2),height=int(cam_param['princpt'][1]*2),znear=0.01, zfar=1000.0, R=extrinsic[:3,:3],T=extrinsic[:3,3],device=torch.device('cuda:0'))
            cameras.append(camera)
            smplx_param_list.append(full_pose)

        return cameras, smplx_param_list

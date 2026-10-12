
import smplx
import torch
import pickle
# import igl
# import itertools
import trimesh
import numpy as np
from sklearn.neighbors import KDTree

from scene.object.body_model.smpl_base import SMPLBase
from scene.object.mesh import Mesh
from scene.object.body_model.smpl_utils import batch_rigid_transform, transform_homogeneous

# class SMPLMeshUV(SMPLMesh):
#     pass

# deformable smpl mesh with texture map.
# what if we use face or vertex color instead of texture coordinate?
# -> vertex colors should be updated
class SMPLMesh(SMPLBase, Mesh):
    """
    Convert SMPL(-X) template mesh to clothed human mesh
    """
    def __init__(self, smpl_config, device=torch.device('cuda:0')):
        SMPLBase.__init__(self, smpl_config, device=device)

        # newly added variable is displacement vector, not in both Mesh and SMPLBase
        # disp, lbs_weights, v_canon, j_canon supports batch operations (B x V/J x 3) otherwise,
        self._disp = None
        self._t_pose = None  # T-posed smpl vertices, used for calculating chamfer distance
        self._idx2smpl = None

    @property
    def disp(self):
        return self._disp

    @disp.setter
    def disp(self, disp):
        self._disp = self.to_torch(disp, device=self.device)

    @property
    def lbs_weights(self):
        return self._lbs_weights

    @property
    def t_pose(self):
        return self._t_pose

    @lbs_weights.setter
    def lbs_weights(self, lbs_weights):
        self._lbs_weights = self.to_torch(lbs_weights, device=self.device)

    def init_smpl_mesh(self, smpl_params):
        smpl_output = self.predict(smpl_params)
        if smpl_output.betas.shape[0] != 1:
            smpl_output.betas = torch.mean(smpl_output.betas,dim=0,keepdims=True)
        self.v_canon = smpl_output.betas
        self._t_pose = self.v_canon.detach().clone().requires_grad_(False)  # for future purpose
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
        # self.rgb.detach().cpu().numpy(),
        # self.disp.detach().squeeze(0).cpu().numpy(),
        # self.v_canon.detach().squeeze(0).cpu().numpy(),
        # self.lbs_weights.detach().squeeze(0).cpu().numpy())),

    def subdivide(self):
        """
        subdivide the mesh with assigned attributes
        lbs_weights, canonical_vertices, displacements, colors, and positions
        relying on trimesh.remesh function, which means the
        gradient will not flow anymore.
        """
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


    def forward_skinning(self, smpl_output, rel_rot=None, v_canon=None, smpl_params=None):
        if v_canon is None:
            if self.v_canon is None:
                # initialize canonical SMPL(-H/X) vertices from betas
                self.v_canon = smpl_output.betas
            v_canon = self.v_canon

        if rel_rot is None:  # can be pre-computed
            rel_rot = self.rel_transform(smpl_output.full_pose, inverse=False)

        T = torch.matmul(self.lbs_weights, rel_rot).view(1, -1, 4, 4)
        v_deformed = transform_homogeneous(T, v_canon)

        if smpl_params is not None:
            v_deformed = (v_deformed + smpl_params['transl']) * smpl_params['scale']
        return v_deformed.squeeze(0)


    def forward_v_canon(self, smpl_params, return_mesh=False, j_canon=None, v_canon=None):
        smpl_output = self.predict(smpl_params)
        full_pose = smpl_output.full_pose
        B = full_pose.shape[0]

        rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([B, -1, 3, 3])
        if j_canon!=None:
            joint = j_canon
        else:
            joint = self.j_canon.expand(B, -1, -1).contiguous()


        _, rel_rot = batch_rigid_transform(rot_mats, joint,
                                           self.smpl_model.parents,
                                           inverse=False)
        rel_rot = rel_rot.reshape(B, self.num_joints, 16)
        T = torch.matmul(self._lbs_weights, rel_rot).view(B, -1, 4, 4)
        if v_canon==None:
            v_deformed = transform_homogeneous(T, self._v_canon)
        else:
            v_deformed = transform_homogeneous(T, v_canon)
        if smpl_params is not None:
            v_deformed = (v_deformed + smpl_params['transl'][:,None,:]) * smpl_params['scale'][:,None,:]

        if return_mesh:
            posed_mesh = Mesh(vertices=v_deformed,indices=self.face,colors=self.rgb,device=self.device)
            return v_deformed, posed_mesh
        return v_deformed

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

    def decimate(self, num_faces=5000, canonical=True):
        from igl import decimate
        if canonical:
            V1 = self.v_canon.squeeze(0).detach().cpu().numpy().astype(np.double)
        else:
            V1 = self.xyz.detach().cpu().numpy().astype(np.double)
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

    def lbs_transfer_nearest(self, V1, V2, W):
        kdtree = KDTree(V1, leaf_size=30)
        kd_idx = kdtree.query(V2, k=1, return_distance=False)
        W_new = W[:, kd_idx.squeeze(), :]
        return self.to_torch(W_new, device=self.device)


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

    def to_trimesh(self, return_canon=True):
        if return_canon:
            vts = self._v_canon + self._disp
        else:
            vts = self._xyz.unsqueeze(0)

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

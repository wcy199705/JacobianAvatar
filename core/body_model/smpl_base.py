import cv2
import os
import json
import smplx
import torch
# import itertools
import trimesh
import numpy as np

from core.body_model.smpl_utils import batch_rigid_transform, transform_homogeneous

class MANO:
    def __init__(self):
        pass

    def create(self):
        pass

    def predict(self):
        pass

class Flame:
    def __init__(self, params):
        pass

    def create(self):
        pass

    def predict(self):
        pass

class SMPLBase:
    """
    Simple SMPL handler to predict smpl_outputs
    smpl_config: configuration parameters (dictionary) must be given
    supports semantic labels and texture map for smpl, smplx models
    """
    def __init__(self, smpl_config, device=torch.device('cpu')):
        # need to define smpl_config file to use this class
        self.device = device

        self.smpl_config = smpl_config
        self.smpl_model = self.create()
        #import pdb;pdb.set_trace()
        # set semantic labels to each vertex if necessary
        self.v_label = self.set_v_label() if self.smpl_config['use_semantic'] else None

        # initially pre-defined weights are used.
        self._lbs_weights = self.smpl_model.lbs_weights.detach().expand([1, -1, -1]).to(self.device)
        self.num_joints = self._lbs_weights.shape[-1]
        self._v_canon, self._j_canon = None, None

        # additional variables (fill mouth, fill tooth, etc.)
        if smpl_config['fill_mouth'] is True and smpl_config['model_type'] == 'smplx':
            self.fill_mouth = True
            self.eyeball_fid = np.load(os.path.join(smpl_config['smpl_root'], 'eyeball_fid.npy'))
            self.fill_mouth_fid = np.load(os.path.join(smpl_config['smpl_root'], 'fill_mouth_fid.npy'))
        else:
            self.fill_mouth = False

    def fill_mouth_and_eyes(self):
        """
        Fill holes in the SMPL-X (only) mesh, eyes and mouth
        Returns:
            New faces with filled eyes and mouth (20908 -> 18762 faces for the same vertices)
        """
        smpl_faces_not_watertight = self.smpl_model.faces
        smpl_faces_no_eybeballs = smpl_faces_not_watertight[~self.eyeball_fid]
        faces = np.concatenate([smpl_faces_no_eybeballs, self.fill_mouth_fid], axis=0)
        return faces

    @property
    def lbs_weights(self):
        return self._lbs_weights

    @lbs_weights.setter
    def lbs_weights(self, weights):
        self._lbs_weights = self.smpl_model.lbs_weights.detach().expand([1, -1, -1]).to(self.device)

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
        self._v_canon = self.smpl_model.v_template + smplx.lbs.blend_shapes(betas, self.smpl_model.shapedirs)
        self._j_canon = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, self._v_canon)

    @staticmethod
    def get_empty_params(pca_dim=6, device=torch.device('cpu')):
        """
        Defined for SMPL-X model (need to be updated)
        Returns:
            Dictionary of SMPL-X parameters
        """
        return {'global_orient': torch.zeros(1, 3).to(device),
                'expression': torch.zeros(1, 10).to(device),
                'betas': torch.zeros(1, 10).to(device),
                'jaw_pose': torch.zeros(1, 3).to(device),
                'body_pose': torch.zeros(1, 63).to(device),
                'left_hand_pose': torch.zeros(1, pca_dim).to(device),
                'right_hand_pose': torch.zeros(1, pca_dim).to(device),
                'transl': torch.zeros(1, 3).to(device),
                'scale': torch.FloatTensor([[1.0]]).to(device)
                }

    def rel_transform(self, full_pose, inverse=False):
        """
        Function for calculating relative transform matrix A
        Args:
            full_pose: full_pose parameters (smpl_ouput)
            inverse: inverse transformation matrix if this is set to true otherwise forward transformation
        Returns:
            rel_rot: relative transformation matrix
        """
        rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
        _, rel_rot = batch_rigid_transform(rot_mats,
                                           self._j_canon[:, :55, :],
                                           self.smpl_model.parents,
                                           inverse=inverse,
                                           dtype=torch.float32)
        rel_rot = rel_rot.reshape(1, self.num_joints, 16)
        return rel_rot  # relative transform matrices (A matrix)


    def set_v_label(self):
        # target_parts = {"eyeballs": 2e-4, "leftHand": 5e-2, "rightHand": 5e-2,
        #                      "leftForeArm": 5e-2, "rightForeArm": 5e-2}  # "leftArm": 5e-2, "rightArm": 5e-2,
        path2semantic = os.path.join(self.smpl_config['smpl_root'], self.smpl_config['segmentation'])
        with open(path2semantic) as f:
            v_label = json.load(f)  # for original smplx

        hand_idx, non_hand_idx = [], []
        v_label['leftWrist'], v_label['rightWrist'] = [], []
        v_label['hand_idx'], v_label['non_hand_idx'] = [], []
        v_label['left_wrist_idx'], v_label['right_wrist_idx'] = [], []

        for k in v_label['leftHand']:
            if k in v_label['leftForeArm']:
                v_label['left_wrist_idx'].append(k)
        for k in v_label['rightHand']:
            if k in v_label['rightForeArm']:
                v_label['right_wrist_idx'].append(k)
        for key in v_label.keys():
            if 'leftHand' in key or 'rightHand' in key:
                v_label['hand_idx'].append(v_label[key])
            else:
                v_label['non_hand_idx'].append(v_label[key])

        nonbody_idx = v_label['head'] + v_label['eyeballs'] + \
                      v_label['leftToeBase'] + v_label['rightToeBase'] + \
                      v_label['leftEye'] + v_label['rightEye'] + \
                      hand_idx
        nonbody_idx = np.asarray(list(set(nonbody_idx)))  # unique idx
        body_idx = np.asarray([i for i in range(0, 10475) if i not in nonbody_idx])
        v_label['body_idx'] = body_idx
        v_label['non_body_idx'] = nonbody_idx
        # v_label['leftArm'], v_label['rightArm'] are excluded
        v_label['left_arm'] = np.unique(v_label['leftForeArm'] + v_label['leftHand'] +
                                        v_label['leftHandIndex1'] + v_label['leftArm'])
        v_label['right_arm'] = np.unique(v_label['rightForeArm'] + v_label['rightHand'] +
                                         v_label['rightArm'] + v_label['rightHandIndex1'])
        # + v_label['leftUpLeg'], + v_label['rightUpLeg']
        v_label['left_leg'] = np.unique(v_label['leftLeg'] + v_label['leftFoot'] + v_label['leftToeBase'] +
                                        v_label['leftUpLeg'])
        v_label['right_leg'] = np.unique(v_label['rightLeg'] + v_label['rightFoot'] + v_label['rightToeBase'] +
                                         v_label['rightUpLeg'])
        arm_leg_index = v_label['leftArm'] + v_label['leftForeArm'] + v_label['leftHand'] + v_label['leftHandIndex1'] +\
                        v_label['rightArm'] + v_label['rightForeArm'] + v_label['rightHand'] + v_label['rightHandIndex1'] + \
                        v_label['leftLeg'] + v_label['leftFoot'] + v_label['leftToeBase'] + v_label['leftUpLeg'] + \
                        v_label['rightLeg'] + v_label['rightFoot'] + v_label['rightToeBase'] + v_label['rightUpLeg']
        body_idx = np.asarray([i for i in range(0, 10475) if i not in arm_leg_index])
        v_label['non_arm_leg'] = body_idx
        return v_label

    def create(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        if self.smpl_config['model_type'] == 'smpl':
            return smplx.create(self.smpl_config['smpl_root'],
                                model_type=self.smpl_config['model_type'],
                                gender=self.smpl_config['gender'],
                                ext=self.smpl_config['ext']
                                ).to(self.device)
        elif self.smpl_config['model_type'] == 'smplh':
            return smplx.create(self.smpl_config['smpl_root'],
                                model_type=self.smpl_config['model_type'],
                                gender=self.smpl_config['gender'],
                                num_betas=self.smpl_config['num_betas'],
                                ext=self.smpl_config['ext'],
                                flat_hand_mean=self.smpl_config['use_flat_hand'],
                                use_pca=self.smpl_config['use_pca'],
                                num_pca_comps=self.smpl_config['num_pca_comp']
                                ).to(self.device)
        elif self.smpl_config['model_type'] == 'smplx':
            if 'num_betas' not in self.smpl_config.keys():
                num_shapes = 10
            else:
                num_shapes = self.smpl_config['num_betas']
            if 'num_expr' not in self.smpl_config.keys():
                num_expr = 10
            else:
                num_expr = self.smpl_config['num_expr']

            if num_shapes != 10:
                smpl_path = self.smpl_config['smpl_root']
            else:
                smpl_path=self.smpl_config['smpl_root']
            #import pdb;pdb.set_trace()
            return smplx.create(smpl_path,
                                model_type=self.smpl_config['model_type'],
                                gender=self.smpl_config['gender'],
                                num_betas=num_shapes,
                                num_expression_coeffs=num_expr,
                                ext=self.smpl_config['ext'],
                                use_face_contour=self.smpl_config['use_face_contour'],
                                flat_hand_mean=self.smpl_config['use_flat_hand'],
                                use_pca=self.smpl_config['use_pca'],
                                num_pca_comps=self.smpl_config['num_pca_comp']
                                ).to(self.device)
        else:
            assert f"{self.smpl_config['model_type']} is not supported model type."

    def predict(self, smpl_params, return_mesh=False):
        if self.smpl_config['model_type'] == 'smpl':
            if smpl_params['body_pose'].shape[1] == 63:
                smpl_params['body_pose'] = (
                    torch.concatenate((smpl_params['body_pose'],
                                       torch.zeros_like(smpl_params['body_pose'][:, :6])), dim=1))
            smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                          body_pose=smpl_params['body_pose'],
                                          betas=smpl_params['betas'],
                                          global_orient=smpl_params['global_orient'],
                                          return_full_pose=True,
                                          return_verts=True)
        elif self.smpl_config['model_type'] == 'smplx':
            smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                expression=smpl_params['expression'],
                                body_pose=smpl_params['body_pose'],
                                betas=smpl_params['betas'],
                                global_orient=smpl_params['global_orient'],
                                jaw_pose=smpl_params['jaw_pose'],
                                left_hand_pose=smpl_params['left_hand_pose'],
                                right_hand_pose=smpl_params['right_hand_pose'],
                                reye_pose = smpl_params['right_eye_pose'],
                                leye_pose = smpl_params['left_eye_pose'],
                                return_full_pose=True,
                                return_verts=True)
            """
            smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                          expression=smpl_params['expression'],
                                          body_pose=smpl_params['body_pose'],
                                          betas=smpl_params['betas'],
                                          global_orient=smpl_params['global_orient'],
                                          jaw_pose=smpl_params['jaw_pose'],
                                          left_hand_pose=smpl_params['left_hand_pose'],
                                          right_hand_pose=smpl_params['right_hand_pose'],
                                          return_full_pose=True,
                                          return_verts=True)
            """
        else:
            assert f"{self.smpl_config['model_type']} is not supported model type."

        if 'scale' in smpl_params and smpl_params['scale'] is not None:
            scale = smpl_params['scale'].unsqueeze(-1)       # (B,1,1)
            transl = smpl_params['transl'].unsqueeze(1)      # (B,1,3)
            # body * scale + transl: scale body only, keep translation intact
            # smplx model already added transl internally, so compensate:
            # (body+transl)*scale + transl*(1-scale) = body*scale + transl
            smpl_output.joints = smpl_output.joints * scale + transl * (1 - scale)
            smpl_output.vertices = smpl_output.vertices * scale + transl * (1 - scale)

        if return_mesh:
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy(),
                                        faces=self.smpl_model.faces, process=False)
            return smpl_output, smpl_mesh
        else:
            return smpl_output

# uv smpl uv (uv position file)
class SMPLBaseUV(SMPLBase):
    def __init__(self, smpl_config, device=torch.device('cpu')):
        super(SMPLBase, self).__init__(smpl_config, device=device)

        # set texture map and uv coordinates if necessary
        self.tex = None
        self.tex_eye = None
        self.tex_eye_mask = None
        self.tex_face_mask = None
        self.tex_hand_mask = None
        self.set_texture_maps()

    # @property
    # def xyz(self, x):
    #     return self.MLP(x)

    @staticmethod
    def load_texture_img(path2tex):
        tex = cv2.imread(path2tex, cv2.IMREAD_ANYCOLOR)
        return np.flip(tex, axis=0)

    def set_texture_maps(self):
        # initialize texture map and uv coordinates
        path2uv_texture = os.path.join(self.smpl_config["smpl_root"],
                                       self.smpl_config["model_type"] + '_uv',
                                       self.smpl_config["model_type"] + '_uv_with_eyes5.png')
        tex_eye = self.load_texture_img(path2uv_texture)
        self.tex = torch.FloatTensor(tex_eye.copy()).to(self.device) / 255.0 * 0.9
        self.tex_eye = self.tex.clone()
        self.tex_eye_mask = torch.FloatTensor(np.sum(tex_eye, axis=2, keepdims=True) > 0).to(self.device)
        path2uv_face = os.path.join(self.smpl_config["smpl_root"],
                                    self.smpl_config["model_type"] + '_uv',
                                    self.smpl_config["model_type"] + '_face_mask.png')
        tex_face = self.load_texture_img(path2uv_face)
        tex_face = cv2.cvtColor(tex_face, cv2.COLOR_RGB2GRAY)
        tex_np = np.zeros_like(tex_face)
        tex_np[tex_face > 0.5] = 1.0
        self.tex_face_mask = torch.FloatTensor(tex_np).to(self.device)
        path2uv_hand = path2uv_face.replace('_face_', '_hand_')
        tex_hand = self.load_texture_img(path2uv_hand)
        tex_hand = cv2.cvtColor(tex_hand, cv2.COLOR_RGB2GRAY)
        tex_np = np.zeros_like(tex_hand)
        tex_np[tex_hand > 0.5] = 1.0
        self.tex_hand_mask = torch.FloatTensor(tex_np).to(self.device)

class SMPLBaseUVFlame(SMPLBaseUV):
    def __init__(self):
        pass



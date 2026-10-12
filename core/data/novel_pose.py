import os
import glob
import torch
import numpy as np
# WAFT-related imports (use sys.path if path setup is needed)
from core.camera import UnifiedCamera
from core.body_model.smplx import AvatarMesh
import json

    
from natsort import natsorted
class Novelpose:
    def __init__(self, root_folder, smpl_config, intrinsic=None, sampling_rate=1, save_root=None, device='cuda'):
        """
        Args:
            root_folder (str): folder containing the raw data (images, masks, etc.)
            smpl_config (dict): SMPL config
            save_root (str, optional): root folder where experiment results are saved.
                                     If given, normal maps are loaded from its 'normals' subfolder.
            device (str): torch device
            is_train (bool): train/test mode flag
        """
        self.root_folder = root_folder
        self.save_root = save_root
        self.smpl_config = smpl_config
        self.device = device
        self.cam_param_folder = os.path.join(root_folder,'cam_params')
        self.smplx_param_folder = os.path.join(root_folder,'smplx_params_smoothed')

        self.sampling_rate = sampling_rate
        self.intrinsic = intrinsic
        # Data containers
        self.images = []
        self.masks = []
        self.cameras = []
        self.smpl_params = {} 
        self.smpl_list = []   
        self.normals = None 
        self.smpl_mesh = AvatarMesh(smpl_config=self.smpl_config)

        # 2. Load base data (images, masks, cameras, SMPL)
        self.load_animation_poses_exavatar() # load view keys first
        
        
        # 3. Load normal data (if save_root exists)


    def load_animation_poses_exavatar(self,):
        sampling_rate = self.sampling_rate

        smplx_param_files = natsorted(glob.glob(os.path.join(self.smplx_param_folder,'*.json')))[::sampling_rate]
        cam_files = natsorted(glob.glob(os.path.join(self.cam_param_folder,'*.json')))[::sampling_rate]


        self.cameras =[]
        self.smplx_param_list = []
        for idx, smplx_param_file in enumerate(smplx_param_files):
            cam_file = cam_files[idx]
            with open(smplx_param_file, 'r', encoding='utf-8') as f:
                smplx_param = json.load(f)

            with open(cam_file, 'r', encoding='utf-8') as f:
                cam_param = json.load(f)
            # fix: Exavatar json uses short keys `lhand_pose` /
            # `rhand_pose`, but `smplx2full_pose_dict` only reads
            # `left_hand_pose` / `right_hand_pose`. Without this rename,
            # every novel-pose render placed the hands in the SMPL-X mean
            # pose regardless of the source data. Also mirror `leye_pose` /
            # `reye_pose` to the long form for the same reason.
            for _short, _long in (('lhand_pose', 'left_hand_pose'),
                                   ('rhand_pose', 'right_hand_pose'),
                                   ('leye_pose',  'left_eye_pose'),
                                   ('reye_pose',  'right_eye_pose')):
                if _short in smplx_param and _long not in smplx_param:
                    smplx_param[_long] = smplx_param[_short]

            smplx_param['global_orient'] = torch.tensor(smplx_param['root_pose']).to(self.device).view(1,-1)
            for key in smplx_param.keys():
                if smplx_param[key] is not None:
                    smplx_param[key] = torch.tensor(smplx_param[key]).to(self.device)[None,:].view(1,-1)
            if 'betas' not in smplx_param.keys():
                smplx_param['betas'] = torch.zeros((1,self.smpl_mesh.smpl_model.num_betas)).to(self.smpl_mesh.device)

            full_pose = self.smpl_mesh.smplx2full_pose_dict(smplx_param)
            #smplx_params = self.full_pose_dict2smplx(full_pose)
            if self.intrinsic is None:
                intrinsic = np.eye(4)
                intrinsic[0,0] = cam_param['focal'][0]
                intrinsic[1,1] = cam_param['focal'][0]
                intrinsic[0,2] = cam_param['princpt'][0]
                intrinsic[1,2] = cam_param['princpt'][1]
            else:
                if isinstance(self.intrinsic,torch.Tensor):
                    self.intrinsic = self.intrinsic.cpu().numpy()
                intrinsic = self.intrinsic
                
            extrinsic = np.eye(4)
            extrinsic[:3,:3] = cam_param['R']
            extrinsic[:3,3] = cam_param['t']
            intrinsic = torch.Tensor(intrinsic)[:3,:3]
            extrinsic = torch.Tensor(extrinsic)

            extrinsic[1:,3]*= -1
            camera = UnifiedCamera.perspective_camera_from_K(K=intrinsic,width=int(intrinsic[0,2] *2),height=int(intrinsic[1,2] *2),znear=0.01, zfar=1000.0, R=extrinsic[:3,:3],T=extrinsic[:3,3],device=torch.device('cuda:0'))
            self.cameras.append(camera)
            self.smplx_param_list.append(full_pose)


    def get_all_data(self):
        return {
            'test_images': None,
            'test_masks': None,
            'cameras': self.cameras,
            'smpl_params': self.smplx_param_list,
            'smpl_batch': self.smpl_mesh.list2batch(self.smplx_param_list)

        }



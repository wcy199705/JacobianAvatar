import os
import torch
import json
import trimesh
import numpy as np
from torch.utils.data import Dataset
# from mesh_utils import *
from utils.io.mesh import *
from plyfile import PlyData


class DataLoader(Dataset):
    def __init__(self,
                 data_root="",
                 datalist=["TH2.1"],
                 model_type='SMPLX'
                 ):
        self.path2gt = []
        self.path2smpl = []
        model_type = model_type.upper()
        for data in datalist:
            path2gt = sorted(glob.glob(os.path.join(data_root, data, "MESH/*")))
            self.path2gt.extend(path2gt)

            path2smpl = sorted(glob.glob(os.path.join(data_root, data, f"{model_type}_UV/*")))
            self.path2smpl.extend(path2smpl)

        self.num_data = len(self.path2gt)

    @property
    def exists(self):
        return True if self.num_data > 0 else False

    def skip_n_data(self, n):
        self.path2gt = self.path2gt[205:]
        self.path2smpl = self.path2smpl[205:]

    def fetch_next_data(self, height=1.8, device=torch.device('cuda:0')):
        path2gt = self.path2gt.pop(0)
        path2smpl = self.path2smpl.pop(0)
        self.num_data -= 1

        filename = path2gt.split('/')[-1]
        dataset = path2gt.split('/')[-3]
        print(f"CURRENT DATASET: {dataset}, NAME: {filename}")

        path2gt_ = os.path.join(path2gt, filename + ".obj")
        if not os.path.exists(path2gt_):
            path2gt_ = os.path.join(path2gt, "model.obj")
        path2smpl_ = os.path.join(path2smpl, filename + ".obj")

        vts, faces, uv_vts, tex = DataLoader.load_gt_mesh(path2gt_, height=height)
        smpl_params = self.load_smpl_params(path2smpl_.replace(".obj", ".json"), device=device)
        output = {"file": filename, "dataset": dataset, "vert": vts, "faces": faces, "uv": uv_vts, "tex": tex, "smpl_params": smpl_params}
        return output

    def load_images(self, path2dir, return_mask=True):
        img_list = [f for f in sorted(glob.glob(f"{path2dir}/*")) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]
        images = []
        for img in img_list:
            cur_img = cv2.imread(img)
            #cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            images.append(cur_img)

        if return_mask:
            masks = []
            for img in images:
                cur_mask = 1 - np.asarray(np.sum(img, axis=2) == 765).astype(np.float32)
                masks.append(cur_mask)
            return images, masks, img_list
        else:
            return images, img_list
    def load_masks(self, path2dir, return_mask=True):
        img_list = [f for f in sorted(glob.glob(f"{path2dir}/*")) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]
        images = []
        for img in img_list:
            cur_img = cv2.imread(img)
            images.append(cur_img)

        if return_mask:
            masks = []
            for img in images:
                cur_mask =  np.asarray(np.sum(img, axis=2) == 765).astype(np.float32)
                masks.append(cur_mask)
            return images, masks, img_list
        else:
            return images, img_list


    def load_keypoints(self, keypoints_list):
        keypoints = []
        for keypoint_fname in keypoints_list:
            if not os.path.exists(keypoint_fname):
                cur_keypoint = None
            else:
                with open(keypoint_fname, 'r') as f:
                    cur_keypoint = json.load(f)
                cur_keypoint = torch.FloatTensor(cur_keypoint)
            keypoints.append(cur_keypoint)
        return keypoints

    def detect_keypoints(self, images, pose_detector, keypoints_list):
        keypoints = []
        for i, keypoint_name in enumerate(keypoints_list):
            cur_keypoint = pose_detector.openpose_ezmocap(images[i])
            with open(keypoint_name, 'w') as f:
                json.dump(cur_keypoint.detach().cpu().numpy().tolist(), f)
            keypoints.append(cur_keypoint)
        return keypoints

    def load_normals(self):
        pass

    #def load_masks(self):
    #    pass

    def load_depth(self):
        pass

    @staticmethod
    def load_gt_mesh(path2obj, height=180.0):
        vertices, faces, textures, texture_image = load_gt_data(path2obj, height=height)
        return vertices, faces, textures, texture_image

    @staticmethod
    def load_smpl_mesh(path2obj):
        m = trimesh.load(path2obj, process=False)
        return m
        # return m.vertices, m.faces, m.visual.uv

    @staticmethod
    def load_smpl_params(path2json, device='cpu'):
        if '.pkl' in path2json or '.npy' in path2json:
            smpl_params = np.load(path2json, allow_pickle=True)
        elif '.json' in path2json:
            with open(path2json, "r") as f:
                smpl_params = json.load(f)
        else:
            assert "unknown smpl parameter format, allows .pkl, .npy, and .json only"

        for key in smpl_params.keys():
            if isinstance(smpl_params[key], list):
                smpl_params[key] = torch.FloatTensor(smpl_params[key]).reshape(1, -1).to(device)
            elif isinstance(smpl_params[key], float):
                smpl_params[key] = torch.FloatTensor([smpl_params[key]]).reshape(1, -1).to(device)
            elif isinstance(smpl_params[key], np.ndarray):
                smpl_params[key] = torch.FloatTensor([smpl_params[key]]).reshape(1, -1).to(device)
            elif isinstance(smpl_params[key], torch.Tensor):
                smpl_params[key] = smpl_params[key].reshape(1, -1).to(device)

        return smpl_params

def load_gaussian(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return positions, colors, normals

def load_smpl_info(path2param, path2mesh=None):
    with open(path2param, 'r') as f:
        smpl_params = json.load(f)
    for key in smpl_params.keys():
        smpl_params[key] = torch.FloatTensor(smpl_params[key]).reshape(1, -1)

    if path2mesh is not None:
        smpl_mesh = trimesh.load(path2mesh, process=False)
        return smpl_params, smpl_mesh
    else:
        return smpl_params

def init_semantic_labels(path2label):
    """
    Set semantic labels for SMPL(-X) vertices
    :param path2label: path to the semantic information (json file)
    :results are saved in instance variables
    """
    # semantic labels for smplx vertices.
    if os.path.isfile(path2label):
        left_wrist_idx, right_wrist_idx = [], []
        hand_idx, non_hand_idx = [], []
        with open(path2label, "r") as json_file:
            v_label = json.load(json_file)
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
        return v_label
import os
import glob
import cv2
import torch
import numpy as np

from scene.camera.cameraunified import UnifiedCamera
from utils.dataloader.base_dataset import BaseDataset


def est_calib(height, width):
    """ Roughly estimate intrinsics by image dimensions """
    focal = np.max([height, width])
    cx, cy = width / 2., height / 2.
    return [focal, focal, cx, cy]


class CUSTOM(BaseDataset):
    def __init__(self, root_folder, smpl_config, split, original_size=False, image_size=None, use_head_data=False, img_scale_factor=1.0, device='cuda', normal_source='sapiens2'):
        self.root_folder = root_folder
        self.smpl_config = smpl_config
        self.device = device
        if image_size is not None:
            self.image_size = image_size

        root_folder_name = os.path.basename(root_folder)
        if split == 'train' and original_size == False:
            self.image_folder = os.path.join(root_folder, "image_crop")
        else:
            self.image_folder = os.path.join(root_folder, "images")

        if 'novel_view' in root_folder_name or 'novel_pose' in root_folder_name or original_size:
            self.mask_folder = os.path.join(root_folder, "masks")
        else:
            self.mask_folder = os.path.join(root_folder, "mask_crop")

        self.smplx_folder = os.path.join(root_folder, 'global_smpl_smoothed')
        self.keypoint_folder = os.path.join(root_folder, "mmpose_result_mmpose", "keypoints_whole_body")
        self.img_scale_factor = img_scale_factor

        self.bgcolor = np.array([0, 0, 0])
        self.bgcolor_list = [0, 0, 0]
        self.images = []
        self.masks = []
        self.smpl_params = {}
        self.smpl_list = []
        self.normals = None

        self._load_smplx_params()
        self.file_names = sorted(glob.glob(os.path.join(self.image_folder, '*.png')))
        original_image = cv2.imread(glob.glob(os.path.join(root_folder, "images", '*.png'))[0])
        self.original_height, self.original_width, _ = original_image.shape
        self._load_images_and_masks()
        self.split = split
        if split == 'train' and original_size == False:
            self._load_crop_cameras()
        else:
            self._load_cameras_test()
        # Normal source selection.
        # Default 'sapiens2' uses crop_normals_sapiens2/ if present, else falls back to sapiens 1b.
        self.normal_source = str(normal_source).lower()
        if self.normal_source == 'sapiens2':
            sp2_folder = os.path.join(root_folder, 'crop_normals_sapiens2')
            if os.path.exists(sp2_folder):
                self.normal_folder = sp2_folder
                print(f"[CUSTOM] normal_source=sapiens2 → {self.normal_folder}")
            else:
                self.normal_folder = os.path.join(root_folder, 'crop_normals')
                print(f"[CUSTOM] sapiens2 folder missing, fallback to sapiens 1b: {self.normal_folder}")
        else:
            self.normal_folder = os.path.join(root_folder, 'crop_normals')
        if os.path.exists(self.normal_folder) and not original_size:
            self._load_normals()
        if os.path.exists(self.keypoint_folder):
            self._load_keypoints_smplx()
        else:
            self.target_kpts = None
        if use_head_data:
            self._load_head_data()

    @torch.no_grad
    def _load_cameras_test(self):
        self.cameras = []
        cam_int = est_calib(self.original_height, self.original_width)
        cam_int = UnifiedCamera.intrinsic_from_param_each(fx=cam_int[0], fy=cam_int[1], cx=cam_int[2], cy=cam_int[3])
        self.cam_int = cam_int
        for frame_id in range(self.num_frames):
            unified_cam = UnifiedCamera.perspective_camera_from_K(K=torch.Tensor(cam_int), width=self.original_width, height=self.original_height,
                                                                   znear=0.01, zfar=1000.0, device=torch.device('cuda:0'))
            self.cameras.append(unified_cam)

    @torch.no_grad
    def _load_crop_cameras(self):
        self.crop_bbox_folder = os.path.join(self.root_folder, 'mmpose_result_mmpose', 'cropped_bboxes')
        crop_bboxes_files = sorted(glob.glob(os.path.join(self.crop_bbox_folder, '*.txt')))
        self.cameras = []
        cam_int = est_calib(self.original_height, self.original_width)
        cam_int = UnifiedCamera.intrinsic_from_param_each(fx=cam_int[0], fy=cam_int[1], cx=cam_int[2], cy=cam_int[3])
        self.cam_int = cam_int
        for frame_id in range(self.num_frames):
            unified_cam = UnifiedCamera.perspective_camera_from_K(K=torch.Tensor(cam_int), width=self.original_width, height=self.original_height,
                                                                   znear=0.01, zfar=1000.0, device=torch.device('cuda:0'))
            with open(crop_bboxes_files[frame_id], 'r') as file:
                content = file.read()
                values = [int(item.strip()) for item in content.split(',') if item.strip()]
            if self.img_scale_factor != 1.0:
                values[0] = int(values[0] * self.img_scale_factor)
                values[1] = int(values[1] * self.img_scale_factor)
                values[2] = int(values[2] * self.img_scale_factor)
                values[3] = int(values[3] * self.img_scale_factor)
            unified_cam.K_update_bbox(values)
            self.cameras.append(unified_cam)

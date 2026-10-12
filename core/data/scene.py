import os
import glob
import cv2
import torch
import numpy as np

from core.camera import UnifiedCamera
from core.data.base import BaseDataset


class SceneDataset(BaseDataset):
    def __init__(self, root_folder, smpl_config, no_crop=False, image_size=None, use_head_data=False, img_scale_factor=1.0, device='cuda'):
        self.root_folder = root_folder
        self.smpl_config = smpl_config
        self.device = device
        if image_size is not None:
            self.image_size = image_size

        root_folder_name = os.path.basename(root_folder)
        if 'training' in root_folder_name and not no_crop:
            self.image_folder = os.path.join(root_folder, "image_crop")
        else:
            self.image_folder = os.path.join(root_folder, "images")

        original_image = cv2.imread(glob.glob(os.path.join(root_folder, "images", '*.png'))[0])
        self.original_height, self.original_width, _ = original_image.shape

        # 'test' is the folder name used for a novel_pose split
        # (replaces legacy `<scene>_novel_pose`). Without this, the loader falls
        # through to training-mode mask_crop which can be empty for test data.
        if ('novel_view' in root_folder_name
                or 'novel_pose' in root_folder_name
                or root_folder_name == 'test'
                or no_crop):
            self.mask_folder = os.path.join(root_folder, "benchmark_mask")
        else:
            self.mask_folder = os.path.join(root_folder, "mask_crop")

        self.camera_file = os.path.join(root_folder, 'K.npy')
        self.keypoint_folder = os.path.join(root_folder, "mmpose_result_mmpose", "keypoints_whole_body")
        self.img_scale_factor = img_scale_factor
        self.smplx_folder = os.path.join(root_folder, 'global_smpl_optimized')

        self.bgcolor = np.array([0, 0, 0])
        self.bgcolor_list = [0, 0, 0]
        self.images = []
        self.masks = []
        self.smpl_params = {}
        self.smpl_list = []
        self.normals = None

        self._load_smplx_params()
        self.file_names = sorted(glob.glob(os.path.join(self.image_folder, '*.png')))
        self._load_images_and_masks()
        if 'training' in root_folder_name and not no_crop:
            self._load_crop_cameras()
        else:
            self._load_cameras_test()
        self.normal_folder = os.path.join(root_folder, 'normals')
        if os.path.exists(self.normal_folder) and 'training' in root_folder_name and not no_crop:
            self._load_normals()
        if os.path.exists(self.keypoint_folder) and 'training' in root_folder_name:
            self._load_keypoints_smplx()
        else:
            self.target_kpts = None
        if use_head_data:
            self._load_head_data()

    @torch.no_grad
    def _load_cameras_test(self):
        self.cameras = []
        cam_int = torch.Tensor(np.load(self.camera_file))
        if self.img_scale_factor != 1.0:
            cam_int[0, :] = cam_int[0, :] * self.img_scale_factor
            cam_int[1, :] = cam_int[1, :] * self.img_scale_factor
        self.cam_int = cam_int
        for frame_id in range(self.num_frames):
            camera = UnifiedCamera.perspective_camera_from_K(K=torch.Tensor(cam_int), width=self.w, height=self.h, znear=0.01, zfar=1000.0, device=torch.device('cuda:0'))
            self.cameras.append(camera)

    @torch.no_grad
    def _load_crop_cameras(self):
        self.crop_bbox_folder = os.path.join(self.root_folder, 'mmpose_result_mmpose', 'cropped_bboxes')
        crop_bboxes_files = sorted(glob.glob(os.path.join(self.crop_bbox_folder, '*.txt')))
        self.cameras = []
        cam_int = torch.Tensor(np.load(self.camera_file))
        if self.img_scale_factor != 1.0:
            cam_int[0, :] = cam_int[0, :] * self.img_scale_factor
            cam_int[1, :] = cam_int[1, :] * self.img_scale_factor
        self.cam_int = cam_int
        for frame_id in range(self.num_frames):
            unified_cam = UnifiedCamera.perspective_camera_from_K(K=torch.Tensor(cam_int), width=self.w, height=self.h, znear=0.01, zfar=1000.0, device=torch.device('cuda:0'))
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

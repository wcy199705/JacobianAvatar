import os
import glob
import cv2
import torch
import numpy as np
import tqdm
import copy
import json

from core.body_model.smplx import AvatarMesh
from core.data.smplx_param_loader import load_smplx_params
from core.data.lazy_loader import LazyImageList, LazyMaskList, LazyNormalArray


class _LazyPairedImageList:
    """
    Lazy loader for image-mask pairs. Since image preprocessing depends on mask,
    both are computed together. Uses a shared cache with refcount so that
    the pair is loaded once and freed after both image and mask are consumed.
    """
    _shared_cache = {}   # (cache_id, idx) -> (img, mask)
    _shared_refcount = {}  # (cache_id, idx) -> access count

    def __init__(self, path_pairs, preprocess_fn, data_type='image'):
        """
        Args:
            path_pairs: list of (img_path, mask_path)
            preprocess_fn: fn(img_path, mask_path) -> (img, mask)
            data_type: 'image' or 'mask'
        """
        self.path_pairs = path_pairs
        self.preprocess_fn = preprocess_fn
        self.data_type = data_type
        self._cache_id = id(path_pairs)  # shared between image/mask instances

    def _load_pair(self, idx):
        cache_key = (self._cache_id, idx)
        if cache_key not in _LazyPairedImageList._shared_cache:
            img, mask = self.preprocess_fn(*self.path_pairs[idx])
            _LazyPairedImageList._shared_cache[cache_key] = (img, mask)
            _LazyPairedImageList._shared_refcount[cache_key] = 0
        return _LazyPairedImageList._shared_cache[cache_key]

    def _get_single(self, idx):
        pair = self._load_pair(idx)
        result = pair[0] if self.data_type == 'image' else pair[1]
        # Increment refcount; free after both image and mask accessed
        cache_key = (self._cache_id, idx)
        _LazyPairedImageList._shared_refcount[cache_key] += 1
        if _LazyPairedImageList._shared_refcount[cache_key] >= 2:
            del _LazyPairedImageList._shared_cache[cache_key]
            del _LazyPairedImageList._shared_refcount[cache_key]
        return result

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._get_single(idx)
        elif isinstance(idx, slice):
            return [self._get_single(i) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, (list, np.ndarray)):
            return [self._get_single(int(i)) for i in idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")


    def __iter__(self):
        for i in range(len(self)):
            yield self._get_single(i)


class _LazyNormalList:
    """
    Lazy loader for normals. Supports numpy-style indexing.
    preprocess_fn(file_path, idx) -> np.ndarray
    """
    def __init__(self, file_paths, preprocess_fn):
        self.file_paths = file_paths
        self.preprocess_fn = preprocess_fn

    def _load(self, idx):
        return self.preprocess_fn(self.file_paths[idx], idx)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._load(idx)
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return np.stack([self._load(i) for i in indices])
        elif isinstance(idx, (list, np.ndarray)):
            return np.stack([self._load(int(i)) for i in idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")


    def __iter__(self):
        for i in range(len(self)):
            yield self._load(i)


class BaseDataset:

    @torch.no_grad
    def _load_head_data(self):
        head_img_paths = sorted(glob.glob(os.path.join(self.root_folder, 'Head_img', '*.png')))
        head_mask_paths = sorted(glob.glob(os.path.join(self.root_folder, 'Head_mask', '*.png')))
        lazy_load = getattr(self, 'lazy_load', False)

        if lazy_load:
            bgcolor = (np.ones(3) * self.bgcolor).astype(np.float32)

            def _preprocess_head(img_path, mask_path):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask.max() < 127:
                    mask = mask * 255
                mask = (mask > 127).astype(np.float32)
                img = (mask[..., None] * img + (1.0 - mask[..., None]) * bgcolor[None, None, :]).astype(np.uint8)
                mask = (255 * mask).astype(np.uint8)
                return img, mask

            path_pairs = list(zip(head_img_paths, head_mask_paths))
            self.head_images = _LazyPairedImageList(path_pairs, _preprocess_head, data_type='image')
            self.head_masks = _LazyPairedImageList(path_pairs, _preprocess_head, data_type='mask')
        else:
            self.head_images = []
            self.head_masks = []
            for idx, head_img_path in enumerate(head_img_paths):
                img = cv2.imread(head_img_path)
                mask = cv2.imread(head_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                if mask.max() < 127:
                    mask = mask * 255
                mask = (mask > 127).astype(np.float32)
                bgcolor = (np.ones(3) * self.bgcolor).astype(np.float32)
                img = (mask[..., None] * img + (1.0 - mask[..., None]) * bgcolor[None, None, :]).astype(np.uint8)
                mask = (255 * mask).astype(np.uint8)
                self.head_masks.append(mask)
                self.head_images.append(img)

        self.head_aabb = np.load(os.path.join(self.root_folder, 'Head_aabb.npy'))
        self.head_aabb = self.head_aabb * self.img_scale_factor

    @torch.no_grad
    def _load_images_and_masks(self):
        mask_files = sorted(glob.glob(os.path.join(self.mask_folder, '*.png')))
        lazy_load = getattr(self, 'lazy_load', False)

        if os.path.exists(self.image_folder):
            need_resize = (self.img_scale_factor != 1.0)
            bgcolor = (np.ones(3) * self.bgcolor).astype(np.float32)
            scale = self.img_scale_factor

            if lazy_load:
                # --- Lazy Loading: store file paths only, read from disk on access ---
                def _preprocess_img_mask(img_path, mask_path):
                    """Returns (processed_img, processed_mask) tuple."""
                    img = cv2.imread(img_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if need_resize:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    if mask.max() < 127:
                        mask = mask * 255
                    mask_f = (mask > 127).astype(np.float32)
                    img = (mask_f[..., None] * img + (1.0 - mask_f[..., None]) * bgcolor[None, None, :]).astype(np.uint8)
                    mask = (255 * mask_f).astype(np.uint8)
                    return img, mask

                # Store paths + preprocessing info for lazy access
                self._img_mask_paths = list(zip(self.file_names, mask_files))
                self._img_preprocess_fn = _preprocess_img_mask
                self._bgcolor = bgcolor
                self._need_resize = need_resize

                self.images = _LazyPairedImageList(self._img_mask_paths, _preprocess_img_mask, data_type='image')
                self.masks = _LazyPairedImageList(self._img_mask_paths, _preprocess_img_mask, data_type='mask')

                self.num_frames = len(self.file_names)
                # Get dimensions from first frame
                sample_img = self.images[0]
                self.h, self.w = sample_img.shape[:2]
                if hasattr(self.images, '_cache'):
                    self.images._cache.clear()
                    self.masks._cache.clear()
            else:
                # --- Eager Loading: original behavior ---
                for img_path, mask_path in zip(self.file_names, mask_files):
                    img = cv2.imread(img_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if need_resize:
                        img = cv2.resize(img, None,
                                         fx=self.img_scale_factor,
                                         fy=self.img_scale_factor,
                                         interpolation=cv2.INTER_LANCZOS4)
                        mask = cv2.resize(mask, None,
                                          fx=self.img_scale_factor,
                                          fy=self.img_scale_factor,
                                          interpolation=cv2.INTER_NEAREST)
                    if mask.max() < 127:
                        mask = mask * 255
                    mask = (mask > 127).astype(np.float32)
                    img = (mask[..., None] * img + (1.0 - mask[..., None]) * bgcolor[None, None, :]).astype(np.uint8)
                    mask = (255 * mask).astype(np.uint8)
                    self.masks.append(mask)
                    self.images.append(img)
                self.num_frames = len(self.images)
                if self.images:
                    self.h, self.w, _ = self.images[0].shape
        else:
            self.images = None
            self.masks = None
            self.h, self.w = self.image_size

    @torch.no_grad
    def _load_normals(self):
        normal_files = sorted(glob.glob(os.path.join(self.normal_folder, '*.npy')))
        if not normal_files:
            print("[Warning] No .npy files found in normal folder.")
            return

        lazy_load = getattr(self, 'lazy_load', False)

        first_normal = np.load(normal_files[0])
        h, w, c = first_normal.shape
        new_h = int(h * self.img_scale_factor)
        new_w = int(w * self.img_scale_factor)

        if lazy_load:
            # --- Lazy Loading: read from disk on access ---
            scale = self.img_scale_factor
            masks_ref = self.masks  # reference for masking

            def _preprocess_normal(normal_path, idx):
                normal = np.load(normal_path)
                if scale != 1.0:
                    normal = cv2.resize(normal, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                normal[..., 1:] *= -1
                norm = np.linalg.norm(normal, axis=-1, keepdims=True)
                normal = normal / (norm + 1e-7)
                if masks_ref is not None and idx < len(masks_ref):
                    mask = masks_ref[idx]
                    normal = normal * (mask[..., None] / 255.0)
                return normal.astype(np.float16)

            self.normals = _LazyNormalList(normal_files, _preprocess_normal)
        else:
            # --- Eager Loading: original behavior ---
            num_files = len(normal_files)
            self.normals = np.empty((num_files, new_h, new_w, c), dtype=np.float16)
            for i, normal_file in enumerate(tqdm.tqdm(normal_files, desc="Loading Normals")):
                normal = np.load(normal_file)
                if self.img_scale_factor != 1.0:
                    normal = cv2.resize(normal, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                normal[..., 1:] *= -1
                norm = np.linalg.norm(normal, axis=-1, keepdims=True)
                normal = normal / (norm + 1e-7)
                if i < len(self.masks):
                    mask = self.masks[i]
                    if mask.shape[:2] != normal.shape[:2]:
                        pass
                    normal = normal * (mask[..., None] / 255.0)
                self.normals[i] = normal.astype(np.float16)

    @torch.no_grad
    def _load_keypoints_smplx(self):
        from core.io.keypoints import change_kpt_name, dw_pose_kpt, tmp_kpt

        kpt_path_list = sorted(glob.glob(os.path.join(self.keypoint_folder, '*.json')))
        kpts = {}
        dw_kpts = np.zeros((len(kpt_path_list), 133, 3))
        for frame_idx, kpt_path in enumerate(kpt_path_list):
            with open(kpt_path) as f:
                kpt = np.array(json.load(f), dtype=np.float32)
            dw_kpts[frame_idx] = kpt
            kpt = change_kpt_name(kpt, dw_pose_kpt['name'], tmp_kpt['name'])
            kpts[frame_idx] = torch.Tensor(kpt)
        B = len(kpts)
        target_kpts = torch.zeros((B, 135, 3))
        for idx in range(B):
            target_kpts[idx][..., :2] = kpts[idx][..., :2] * self.img_scale_factor
            target_kpts[idx][..., 2] = kpts[idx][..., 2]
        self.target_kpts = torch.Tensor(target_kpts).to(self.device)

    @torch.no_grad
    def _load_smplx_params(self):
        if not hasattr(torch.storage.UntypedStorage, 'dtype'):
            torch.storage.UntypedStorage.dtype = torch.uint8
        smpl_mesh = AvatarMesh(smpl_config=self.smpl_config)
        smpl_params = load_smplx_params(self.smplx_folder, fmt='auto', device=self.device)
        smpl_param_batch = smpl_mesh.list2batch(smpl_params)
        smpl_out = smpl_mesh.predict(smpl_param_batch)
        self.smpl_params = smpl_param_batch
        self.smpl_params['vertices'] = smpl_out.vertices
        self.smpl_params['joints'] = smpl_out.joints
        self.smpl_list = smpl_mesh.batch2list(smpl_param_batch)
        self.smpl_full_dict = smpl_mesh.smplx2full_pose_dict(copy.deepcopy(smpl_param_batch))
        self.smpl_batch = smpl_param_batch

    @torch.no_grad
    def get_all_data(self):
        data = {}
        data['cameras'] = self.cameras
        if self.images is not None:
            data['test_images'] = self.images
        data['test_masks'] = self.masks
        if self.normals is not None:
            data['test_normals'] = self.normals
        data['smpl_params'] = self.smpl_full_dict
        return data

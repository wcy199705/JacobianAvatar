"""
SMPL-X parameter loader supporting both .pt (torch) and .json formats.

Usage:
    from utils.dataloader.smplx_param_loader import load_smplx_params

    # Auto-detect: tries .pt first, falls back to .json
    params_list = load_smplx_params('/path/to/smplx_folder', device='cuda')

    # Force specific format
    params_list = load_smplx_params('/path/to/pt_folder', fmt='pt', device='cuda')
    params_list = load_smplx_params('/path/to/json_folder', fmt='json', device='cuda')

    # JSON with coordinate flip (X,Z negation for fitting-tool convention mismatch)
    params_list = load_smplx_params('/path/to/json_folder', fmt='json', flip_xz=True, device='cuda')

Returns:
    list[dict[str, torch.Tensor]] — each dict has the same keys as the .pt format:
        global_orient (1,3), body_pose (1,63), jaw_pose (1,3),
        left_eye_pose (1,3), right_eye_pose (1,3),
        left_hand_pose (1,45), right_hand_pose (1,45),
        betas (1,10), expression (1,10), transl (1,3), scale (1,1)
"""

import os
import glob
import json
import torch
import numpy as np


# JSON key → canonical .pt key mapping
_JSON_KEY_MAP = {
    'leye_pose': 'left_eye_pose',
    'reye_pose': 'right_eye_pose',
}

# Keys to skip when loading JSON (metadata, not model params)
_JSON_SKIP_KEYS = {'fit_status'}


def _flip_xz_param(param: dict) -> dict:
    """
    Flip X and Z axes for global_orient and transl.
    This handles the coordinate convention difference between some
    fitting tools (e.g., EasyMocap-style) and the pipeline's camera space.

    Transform: R_flip = diag(-1, 1, -1)  (180° rotation around Y axis)
    - transl:       negate X and Z components
    - global_orient: apply R_flip @ R (axis-angle → matrix → flip → axis-angle)
    """
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    # Flip transl X, Z
    if 'transl' in param:
        param['transl'][..., 0] *= -1
        param['transl'][..., 2] *= -1

    # Flip global_orient: R_new = R_flip @ R_old
    if 'global_orient' in param:
        device = param['global_orient'].device
        R = axis_angle_to_matrix(param['global_orient'])
        R_flip = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device))
        R_new = R_flip @ R
        param['global_orient'] = matrix_to_axis_angle(R_new)

    return param


def _load_single_json(json_path: str, device: str = 'cpu') -> dict:
    """Load a single JSON SMPL-X param file and return a dict of tensors."""
    with open(json_path) as f:
        raw = json.load(f)

    param = {}
    for k, v in raw.items():
        if k in _JSON_SKIP_KEYS:
            continue
        key = _JSON_KEY_MAP.get(k, k)
        param[key] = torch.tensor(np.array(v), dtype=torch.float32, device=device)
    return param


def _load_single_pt(pt_path: str, device: str = 'cpu') -> dict:
    """Load a single .pt SMPL-X param file and return a dict of tensors."""
    param = torch.load(pt_path, map_location=device)
    for key in param:
        if isinstance(param[key], torch.Tensor):
            param[key] = param[key].to(device)
    return param


def load_smplx_params(folder: str, fmt: str = 'auto', flip_xz: bool = False,
                      device: str = 'cpu') -> list:
    """
    Load SMPL-X parameters from a folder.

    Args:
        folder: path containing .pt or .json files
        fmt: 'pt', 'json', or 'auto' (tries .pt first, then .json)
        flip_xz: if True, negate X/Z in transl and global_orient
                  (for fitting tools that use a different coordinate convention)
        device: torch device string

    Returns:
        list of per-frame dicts, each mapping param name → tensor (1, D)
    """
    if fmt == 'auto':
        pt_files = sorted(glob.glob(os.path.join(folder, '*.pt')))
        json_files = sorted(glob.glob(os.path.join(folder, '*.json')))
        if pt_files:
            fmt = 'pt'
        elif json_files:
            fmt = 'json'
        else:
            raise FileNotFoundError(f"No .pt or .json files found in {folder}")

    if fmt == 'pt':
        files = sorted(glob.glob(os.path.join(folder, '*.pt')))
        loader = _load_single_pt
    elif fmt == 'json':
        files = sorted(glob.glob(os.path.join(folder, '*.json')))
        loader = _load_single_json
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'pt', 'json', or 'auto'.")

    if not files:
        raise FileNotFoundError(f"No .{fmt} files found in {folder}")

    params = [loader(f, device=device) for f in files]

    if flip_xz:
        params = [_flip_xz_param(p) for p in params]
        print(f"[smplx_param_loader] Applied X,Z flip to {len(params)} frames")

    print(f"[smplx_param_loader] Loaded {len(params)} frames from {folder} (fmt={fmt})")
    return params

import torch
import numpy as np
from scipy.signal import savgol_filter

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle

#from exavatar code  : https://github.com/mks0601/ExAvatar_RELEASE/blob/main/fitting/tools/smooth_smplx_params.py
def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4

    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result

def smoothen_poses(poses, window_length): #window length is 9 in exavatar.
    """Smooth joint angles. Poses and global_root_orient should be given as rotation vectors."""
    n_joints = poses.shape[1] // 3

    # Convert poses to quaternions.
    qs = matrix_to_quaternion(axis_angle_to_matrix(torch.FloatTensor(poses).view(-1,3))).numpy()
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)

    # Smooth the quaternions.
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)

    ps_clean = matrix_to_axis_angle(quaternion_to_matrix(torch.FloatTensor(qs_clean).view(-1,4))).numpy()
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])
    return ps_clean

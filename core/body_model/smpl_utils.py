# from pysdf import SDF
import numpy as np
import torch
import trimesh
import torch.nn.functional as F
from sklearn.neighbors import KDTree
# transform_mat and the forward batch_rigid_transform come from the smplx package
# (MPI license, not redistributed here).
from smplx.lbs import transform_mat
from smplx.lbs import batch_rigid_transform as smplx_batch_rigid_transform


def to_torch(x, dtype=torch.float32, device=None):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x).to(dtype=dtype)

    if device is not None:
        x = x.to(device)
    return x

def lbs_transfer_nearest(V1, V2, W, device='cuda:0'):
    kdtree = KDTree(V1, leaf_size=30)
    kd_idx = kdtree.query(V2, k=1, return_distance=False)
    W_new = W[:, kd_idx.squeeze(), :]
    return to_torch(W_new, device=device)


def transform_homogeneous(tform, verts):
    homogen_coord = torch.ones([verts.shape[0], verts.shape[1], 1], dtype=torch.float32).to(verts.device)
    verts_homo = torch.cat([verts, homogen_coord], dim=2)
    transformed_homo = torch.matmul(tform, torch.unsqueeze(verts_homo, dim=-1))

    return transformed_homo[:, :, :3, 0]


def batch_rigid_transform(rot_mats, joints, parents,
                          inverse=True,
                          dtype=torch.float32):
    """Rigid joint transformations for SMPL(-X) skinning.

    inverse=False: forward transformation, i.e. smplx.lbs.batch_rigid_transform
                   (posed_joints: BxNx3).
    inverse=True:  inverse (posed -> rest) transformations (posed_joints: BxNx3x1).
    Returns (posed_joints, rel_transforms), rel_transforms: BxNx4x4.
    """
    posed_joints, rel_transforms = smplx_batch_rigid_transform(rot_mats, joints, parents, dtype=dtype)
    if inverse is not True:
        return posed_joints, rel_transforms

    joints = torch.unsqueeze(joints, dim=-1)
    posed_joints = torch.unsqueeze(posed_joints, dim=-1)
    rel_joints = posed_joints.clone()
    rel_joints[:, 1:] -= posed_joints[:, parents[1:]]
    transforms_mat_inv = transform_mat(
        rot_mats.view(-1, 3, 3),
        torch.zeros_like(rel_joints.view(-1, 3, 1))).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat_inv[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat_inv[:, i])
        transform_chain.append(curr_res)

    for i in range(len(transform_chain)):
        transform_chain[i] = torch.inverse(transform_chain[i])
        transform_chain[i][:, :3, 3] = joints[:, i, :, :].view(-1, 3)

    transforms = torch.stack(transform_chain, dim=1)
    joints_homogen = F.pad(posed_joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms

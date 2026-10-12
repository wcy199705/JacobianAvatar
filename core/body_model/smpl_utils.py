# from pysdf import SDF
import numpy as np
import torch
import trimesh
import torch.nn.functional as F
from sklearn.neighbors import KDTree


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


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents,
                          inverse=True,
                          dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    inverse : bool
        inverse/forward warping.
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the libs rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    B = joints.shape[0]

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest libs
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    if inverse is True:
        posed_joints = torch.unsqueeze(posed_joints, dim=-1)
        rel_joints = posed_joints.clone()
        rel_joints[:, 1:] -= posed_joints[:, parents[1:]]
        # rot_inv = torch.transpose(rot_mats.view(-1, 3, 3), dim0=1, dim1=2)
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
    else:
        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms
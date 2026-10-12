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




def replace_hands(V1, F1, V2, F2, v_label):
    v_margin = 10
    scale = 100 #100  # to set signed distance volume in the millimeter space
    smpl_mesh_scaled = trimesh.Trimesh(V1 * scale, F1, process=False)
    v_min = np.floor(np.min((smpl_mesh_scaled.bounds[0], smpl_mesh_scaled.bounds[0]), axis=0)).astype(int)
    v_max = np.ceil(np.max((smpl_mesh_scaled.bounds[1], smpl_mesh_scaled.bounds[1]), axis=0)).astype(int)
    v_min -= v_margin
    v_max += v_margin
    res = (v_max - v_min) * 2

    scan_mesh_scaled = trimesh.Trimesh(V2 * scale, F2, process=False)
    from utils.geometry.camera_utils import get_grid_coord
    from pysdf import SDF
    query_pts, grid_coord = get_grid_coord(v_min, v_max, res=res)
    sdf_scan = SDF(scan_mesh_scaled.vertices, scan_mesh_scaled.faces)
    sdf_smpl = SDF(smpl_mesh_scaled.vertices, smpl_mesh_scaled.faces)
    volume_smpl = sdf_smpl(query_pts)
    target_volume = sdf_scan(query_pts)

    volume_smpl = volume_smpl.reshape((res[0], res[1], res[2]))
    target_volume = target_volume.reshape((res[0], res[1], res[2]))

    avg_left = np.mean(smpl_mesh_scaled.vertices[v_label['left_wrist_idx'], :], axis=0)
    avg_right = np.mean(smpl_mesh_scaled.vertices[v_label['right_wrist_idx'], :], axis=0)

    delta = np.abs(grid_coord[0, 0, 0, 1] - grid_coord[0, 0, 0, 0])
    left_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_left[0]) <= delta)
    right_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_right[0]) <= delta)

    offset_left = left_idx[0][-1]  # right-most value
    offset_right = right_idx[0][0]  # left-most value

    med_idx = target_volume.shape[1] // 2

    def volume_filter(volume, iter=1):
        if isinstance(volume, np.ndarray):
            volume = torch.FloatTensor(volume).unsqueeze(0)
        filters = torch.ones(1, 1, 3, 3, 3) / 27  # average filter
        for _ in range(iter):
            volume = F.conv3d(volume, filters, padding=1)
        return volume.squeeze().numpy()

    # linearly blending two volumes near the wrist
    # fill in near smpl surface regions (inside is positive)
    # target_volume[volume_smpl > 2] = volume_smpl[volume_smpl > 2]
    b_range = 20
    for k in range(b_range):
        # linear increase
        alpha = (1 - k / (b_range - 1))
        # negative direction for left
        target_volume[offset_left - k, med_idx:, :] = \
            target_volume[offset_left - k, med_idx:, :] * (1 - alpha) + \
            volume_smpl[offset_left - k, med_idx:, :] * alpha
        target_volume[offset_right + k, med_idx:, :] = \
            target_volume[offset_right + k, med_idx:, :] * (1 - alpha) \
            + volume_smpl[offset_right + k, med_idx:, :] * alpha

    target_volume[offset_left:, med_idx:, :] = volume_smpl[offset_left:, med_idx:, :]
    target_volume[:offset_right, med_idx:, :] = volume_smpl[:offset_right, med_idx:, :]
    filtered = volume_filter(target_volume[offset_right:offset_left, :, :])
    target_volume[offset_right:offset_left, :, :] = filtered  # exclude fingers

    def get_mesh(volume, grid_coord, scale_factor=1.0):
        from torchmcubes import marching_cubes, grid_interp
        if isinstance(volume, np.ndarray):
            volume = torch.Tensor(volume)
        vertices, faces = marching_cubes(volume, 0.0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
        new_mesh = trimesh.Trimesh(new_vertices / scale_factor, faces)
        return new_mesh

    mesh_merged = get_mesh(target_volume, grid_coord, scale_factor=scale)
    #mesh_merged.export('merged_mesh.ply')
    hands_verts = smpl_mesh_scaled.vertices[v_label['leftHand'] + v_label['rightHand']] / scale
    hands_verts = hands_verts[[np.argmin(hands_verts[:, 0]), np.argmax(hands_verts[:, 0])], :]  # finger tips

    kdtree = KDTree(hands_verts, leaf_size=30)
    dist, kd_idx = kdtree.query(mesh_merged.vertices, k=1, return_distance=True)
    v_label['hands'], _ = np.where(dist < 0.1)
    mesh_merged.visual.vertex_colors[v_label['hands'], :] = [255, 0, 0, 255]

    # mesh_merged.show()
    return mesh_merged




def lbs_transfer_robust(V1, F1, N1, V2, F2, N2, W, device='cuda:0',smoothing=True):
    """
    Robust Skin Weights Transfer via Weight Inpainting
    in SIGGRAPH Asia 2023 Technical Communications
    by Epic Games
    Args:
        V1: source vertices (V1 x 3)
        F1: source faces (F1 x 3)
        N1: source vertex normals (V1 x 3)
        V2: target vertices (V2 x 3)
        F2: target faces (F2 x 3)
        N2: target vertex normals (V2 x 3)
        W: linear blend skinning weights (V1 x J) J: number of joints
    Returns:
        W2: transferred lbs weights for V2 (target mesh vertices)
    """
    from utils.geometry.robust_lbs_transfer import find_matches_closest_surface, inpaint, smooth
    dDISTANCE_THRESHOLD = 0.05  # threshold distance D
    dDISTANCE_THRESHOLD_SQRD = dDISTANCE_THRESHOLD * dDISTANCE_THRESHOLD
    dANGLE_THRESHOLD_DEGREES = 30  # threshold angle theta in degrees

    # for every vertex on the target mesh find the closest point on the source mesh and copy weights over
    matched, lbs_interpolated = find_matches_closest_surface(V1.astype(np.float64), F1.astype(np.int64),
                                                             N1.astype(np.float64),
                                                             V2.astype(np.float64), F2.astype(np.int64),
                                                             N2.astype(np.float64),
                                                             W.astype(np.float64),
                                                             dDISTANCE_THRESHOLD_SQRD,
                                                             dANGLE_THRESHOLD_DEGREES)
    if smoothing:
        success, lbs_weights = inpaint(V2.astype(np.float64), F2.astype(np.int64), lbs_interpolated, matched) #makes singular values
        if not success:
            print("inpainting failed, applying smoothing only")
            lbs_weights = lbs_interpolated

        lbs_weights, _ = smooth(V2.astype(np.float64), F2.astype(np.int64), lbs_weights, matched, dDISTANCE_THRESHOLD,
                                num_smooth_iter_steps=10, smooth_alpha=0.2)
    else:
        lbs_weights = lbs_interpolated
    # np.sum(matched == False) / (np.sum(matched == True) + np.sum(matched == False)) # % of false matches
    return to_torch(lbs_weights, device=device).unsqueeze(0)


def decimate(mesh, num_faces=5000, canonical=True):
    from igl import decimate
    device = mesh.device
    if canonical:
        V1 = mesh.v_canon.squeeze(0).detach().cpu().numpy().astype(np.double)
    else:
        V1 = mesh.xyz[0].detach().cpu().numpy().astype(np.double)
    F1 = mesh.face.detach().cpu().numpy().astype(np.int32)
    try: # NOTE: igl version issue; behavior differs in the latest version (implemented with pip install libigl==2.5.1)
        _, V2, F2, _, J = decimate(V1, F1, max_m=num_faces)  # from igl import decimate
    except:
        V2, F2, _, J = decimate(V1, F1, max_m=num_faces)  # from igl import decimate

    if canonical:
        mesh._v_canon = mesh.to_torch(V2, device=device).unsqueeze(0)
    else:
        mesh.xyz = V2[None, ...]

    mesh.face = F2
    J = torch.IntTensor(J).to(device)
    mesh.lbs_weights = mesh.lbs_weights[:, J, :]
    if mesh.rgb is not None:
        mesh.rgb = mesh.rgb[:, J, :]
    if mesh.disp is not None:
        mesh.disp = mesh.disp[:, J, :]
    else:
        mesh.disp = torch.zeros_like(mesh.v_canon).to(device)
    mesh.set_connectivity()
    return mesh, J

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
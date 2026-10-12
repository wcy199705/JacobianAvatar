import numpy as np
import torch


def make_rotate_euler(yaw=0.0, pitch=0.0, roll=0.0, is_radian=False):
    if not is_radian:
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
        roll = np.deg2rad(roll)

    sin_x = np.sin(pitch)
    sin_y = np.sin(yaw)
    sin_z = np.sin(roll)

    cos_x = np.cos(pitch)
    cos_y = np.cos(yaw)
    cos_z = np.cos(roll)

    Rx = np.array([[1.0,   0,      0],
                   [0, cos_x, -sin_x],
                   [0, sin_x,  cos_x]])

    Ry = np.array([[ cos_y, 0, sin_y],
                   [ 0,   1.0,     0],
                   [-sin_y, 0, cos_y]])

    Rz = np.array([[cos_z, -sin_z, 0],
                   [sin_z,  cos_z, 0],
                   [   0,    0,  1.0]])

    return np.matmul(np.matmul(Rz,Ry),Rx)


def projection(points3d, camera_intri, R=None, T=None, distance=None):
    """ project the 3d points to camera coordinate

    Arguments:
        points3d {Tensor} -- (bn, N, 3)
        camera_intri {Tensor} -- (bn, 3, 3)
        distance {Tensor} -- (bn, 1, 1)
        R: bn, 3, 3
        T: bn, 3, 1
    Returns:
        points2d -- (bn, N, 2)
    """
    if R is not None:
        Rt = torch.transpose(R, 1, 2)
        if T.shape[-1] == 1:
            Tt = torch.transpose(T, 1, 2)
            points3d = torch.matmul(points3d, Rt) + Tt
        else:
            points3d = torch.matmul(points3d, Rt) + T

    if distance is None:
        img_points = torch.div(points3d[:, :, :2],
                               points3d[:, :, 2:3])
    else:
        img_points = torch.div(points3d[:, :, :2],
                               distance)
    camera_mat = camera_intri[:, :2, :2]
    center = torch.transpose(camera_intri[:, :2, 2:3], 1, 2)
    img_points = torch.matmul(img_points, camera_mat.transpose(1, 2)) + center
    # img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
    # + center
    return img_points

def projection_nvdiffrast(points3d, cameras, pad_one=True):
    """
    Projection compatible with the cameras used in render_batch
    points3d : (B, N, 3)  or (N, 3), both supported
    
    cameras : list of camera objects, each with .mvp or .RT, .K, etc.
    """
    B = len(cameras) if isinstance(cameras, (list, tuple)) else points3d.shape[0]
    
    # Expand if points3d has no batch dimension
    if points3d.dim() == 2:
        points3d = points3d[None].expand(B, -1, -1)
    
    # Get MVP matrices (same as render_batch)
    mvp_batch = torch.stack([c.mvp.squeeze(0) for c in cameras], dim=0)   # (B, 4, 4)
    
    # Convert to homogeneous coordinates
    if pad_one:
        ones = torch.ones(*points3d.shape[:2], 1, device=points3d.device)
        points_hom = torch.cat([points3d, ones], dim=-1)          # (B, N, 4)
    else:
        points_hom = points3d
    
    # Transform to clip space
    points_clip = torch.matmul(points_hom, mvp_batch.transpose(1, 2))   # (B, N, 4)
    
    # perspective divide
    xyz = points_clip[..., :3]
    w   = points_clip[..., 3:4].clamp(min=1e-6)
    uv   = xyz[..., :2] / w                # (B, N, 2)   → range [-1, +1]
    
    # NDC → pixel coordinates (assumed same as render_batch)
    # uses cameras[0].image_width, image_height
    w_img = cameras[0].image_width
    h_img = cameras[0].image_height
    
    x_pix = (uv[..., 0] + 1) * 0.5 * w_img
    y_pix = (-uv[..., 1] + 1) * 0.5 * h_img #flip
    
    pixels = torch.stack([x_pix, y_pix], dim=-1)   # (B, N, 2)
    
    return pixels


def projection_from_intrinsics(K, image_size, near=0.1, far=10):
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
    Args:
        K: Intrinsic matrix, (N, 3, 3)
            K = [[
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1],
                ]
            ]
        image_size: (height, width)
    Output:
        proj = [[
                [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                [0.0,    0.0,     -(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     -1.0,                     0.0                     ]
            ]
        ]
    """
    B = K.shape[0]
    h, w = image_size

    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = torch.zeros([B, 4, 4], device=K.device)
    proj[:, 0, 0] = fx * 2 / w
    proj[:, 1, 1] = fy * 2 / h
    proj[:, 0, 2] = (w - 2 * cx) / w
    proj[:, 1, 2] = ( 2 * cy - h) / h #???
    proj[:, 2, 2] = -(far + near) / (far - near)
    proj[:, 2, 3] = -2 * far * near / (far - near)
    proj[:, 3, 2] = -1
    return proj

def mvp_from_camera_param(RT, K, image_size,near,far):
    # projection matrix
    proj = projection_from_intrinsics(K, image_size,near,far)

    # Modelview and modelview + projection matrices.
    if RT.shape[-2] == 3:
        mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
        mv[..., 3, 3] = 1
    elif RT.shape[-2] == 4:
        mv = RT
    mvp = torch.bmm(proj, mv)
    return mvp

def ortho_mvp_from_camera_praram(RT, K, image_size,near,far):
    # projection matrix
    B = RT.shape[0]
    proj = projection_from_intrinsics_ortho(K, image_size,near,far)

    # Modelview and modelview + projection matrices.
    if RT.shape[-2] == 3:
        mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
        mv[..., 3, 3] = 1
    elif RT.shape[-2] == 4:
        mv = RT
    mvp = torch.bmm(proj, mv)
    return mvp


def projection_from_intrinsics_ortho(K, image_size, near=0.1, far=10):  # TODO: verify
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
    using an orthographic projection, matching the y-orientation of the perspective projection.
    Args:
        K: Intrinsic matrix, (N, 3, 3) or (N, 4)
            If (N, 3, 3): K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            If (N, 4): K = [fx, fy, cx, cy]
        image_size: (height, width)
        near: Near clipping plane (default: 0.1)
        far: Far clipping plane (default: 10)
    Output:
        proj: Orthographic projection matrix, (N, 4, 4)
            proj = [
                [2*fx/w, 0,       0,                    (w - 2*cx)/w],
                [0,      2*fy/h,  0,                    (h - 2*cy)/h],
                [0,      0,       -2/(far-near),       -(far+near)/(far-near)],
                [0,      0,       0,                    1]
            ]
    """
    B = K.shape[0]
    h, w = image_size

    # Extract intrinsics
    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        fx, fy, cx, cy = fx.squeeze(-1), fy.squeeze(-1), cx.squeeze(-1), cy.squeeze(-1)
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    # Initialize projection matrix
    proj = torch.zeros([B, 4, 4], device=K.device)

    # x: Map [0, w] to [-1, 1], adjusted by cx
    proj[:, 0, 0] = 2 * fx / w  # Scale x
    proj[:, 0, 3] = (w - 2 * cx) / w  # Translate x

    # y: Map [0, h] to [-1, 1], matching perspective projection's y-convention
    proj[:, 1, 1] = 2 * fy / h  # Scale y (positive, like perspective)
    proj[:, 1, 3] = (h - 2 * cy) / h  # Translate y (same as perspective)

    # z: Map [near, far] to [-1, 1], flip z (camera z: out, clip z: in)
    proj[:, 2, 2] = -2 / (far - near)  # Scale z
    proj[:, 2, 3] = -(far + near) / (far - near)  # Translate z

    # w: Set to 1 (no perspective division)
    proj[:, 3, 3] = 1
    return proj


def homogenous_perspective_from_intriniscs(K, height,width):
    #classpytorch3d.renderer.cameras.PerspectiveCameras -CY: it has errors.
    """
    fx = focal_length[:, 0]
    fy = focal_length[:, 1]
    px = principal_point[:, 0]
    py = principal_point[:, 1]

    proj = [
            [fx,   0,   px,   0],
            [0,   fy,   py,   0],
            [0,    0,    0,   1],
            [0,    0,    1,   0],
    ]

    """

    B = K.shape[0]
    proj = torch.zeros([B, 4, 4], device=K.device)
    proj[:,0,0] = K[:,0,0]*2/width
    proj[:,1,1] = K[:,1,1]*2/height
    proj[:,0,2] = (width-K[:,0,2]*2)/width
    #proj[:,1,2] = #(height-K[:,1,2]*2)/height
    #proj[:,0,2] = 2 * K[:,0,2] / width - 1     # cx
    proj[:,1,2] = 2 * K[:,1,2] / height - 1    # cy

    proj[:,2,3] = -1.0
    proj[:,3,2] = -1.0
    return proj

def homogenous_mvp_from_camera_param(RT, K, height,width):
    # projection matrix
    proj = homogenous_perspective_from_intriniscs(K, height, width)

    # Modelview and modelview + projection matrices.
    if RT.shape[-2] == 3:
        mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
        mv[..., 3, 3] = 1
    elif RT.shape[-2] == 4:
        mv = RT
    mvp = torch.bmm(proj, mv)
    #mvp[1:] = -mvp[1:]
    return mvp


def update_intrinsic_after_crop(K, bbox):
    """
    Args:
        K (torch.Tensor): (3, 3) original intrinsic matrix.
        bbox (tuple): (xmin, ymin, width, height) crop bounding box.

    Returns:
        new_K (torch.Tensor): updated intrinsic matrix after cropping.
    """

    assert K.shape == (3, 3), "K must be a 3x3 matrix"
    xmin, ymin, crop_w, crop_h = bbox

    # Copy the original K
    new_K = K.clone()

    # Adjust the principal point
    new_K[0, 2] = K[0, 2] - xmin  # new cx = old cx - left
    new_K[1, 2] = K[1, 2] - ymin  # new cy = old cy - top

    return new_K

def update_intrinsic_after_downsampling(K, w_ratio, h_ratio):
    """
    Args:
        K (torch.Tensor): (3, 3) original intrinsic matrix.
        bbox (tuple): (xmin, ymin, width, height) crop bounding box.

    Returns:
        new_K (torch.Tensor): updated intrinsic matrix after cropping.
    """

    assert K.shape == (3, 3), "K must be a 3x3 matrix"    # Copy the original K
    new_K = K.clone()

    # Adjust the principal point
    new_K[0, :] = new_K[0, :]*w_ratio  # new cx = old cx - left
    new_K[1, :] = new_K[1, :]*h_ratio  # new cx = old cx - left

    return new_K


def compute_face_laplacian(faces: torch.Tensor) -> torch.Tensor:
    F = faces.shape[0]
    device = faces.device

    # 1. Define edges of every face (3 per face)
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)

    # 2. Normalize edges (sort vertex order)
    edges, _ = torch.sort(edges, dim=1)

    # 3. Find duplicate edges
    unique_edges, inverse_indices = torch.unique(edges, dim=0, return_inverse=True)
    
    # [modified] use repeat instead of repeat_interleave!
    face_ids = torch.arange(F, device=device).repeat(3)

    perm = torch.argsort(inverse_indices)
    sorted_inverse = inverse_indices[perm]
    sorted_face_ids = face_ids[perm]

    shared_mask = sorted_inverse[:-1] == sorted_inverse[1:]
    
    src_faces = sorted_face_ids[:-1][shared_mask]
    dst_faces = sorted_face_ids[1:][shared_mask]

    # 4. Build sparse Laplacian matrix
    idx_row = torch.cat([src_faces, dst_faces])
    idx_col = torch.cat([dst_faces, src_faces])
    values = torch.full((idx_row.shape[0],), -1.0, device=device)

    # [modified] use torch.ones for readability
    degrees = torch.zeros(F, device=device)
    degrees.index_add_(0, idx_row, torch.ones(idx_row.shape[0], device=device)) 

    diag_row = torch.arange(F, device=device)
    diag_col = torch.arange(F, device=device)
    
    final_indices = torch.stack([
        torch.cat([idx_row, diag_row]),
        torch.cat([idx_col, diag_col])
    ])
    final_values = torch.cat([values, degrees])

    # 5. Create sparse tensor
    L = torch.sparse_coo_tensor(final_indices, final_values, (F, F))
    
    return L.coalesce()
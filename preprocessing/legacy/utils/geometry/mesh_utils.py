import numpy as np
import torch
import open3d as o3d
import trimesh
from typing import Tuple
# from diff_renderer.normal_nds.nds.core import Mesh, View
# from diff_renderer.normal_nds.nds.utils.images import sample



def poisson_trimesh(mesh, depth=9):
    pass

    mesh.vertex_normals.flags.writeable =True
    mesh_p = poisson_o3d(mesh.vertices,
                            vertex_normals=mesh.vertex_normals,
                            depth=depth)
    return trimesh.Trimesh(np.asarray(mesh_p.vertices), np.asarray(mesh_p.triangles), process=False)

def poisson_o3d(vertices, vertex_colors=None, vertex_normals=None, depth=9):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(vertices)
    if vertex_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(vertex_colors/255)
    if vertex_normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(vertex_normals)  # slower
    else:
        pcd.estimate_normals()

    # o3d.visualization.draw_geometries([pcd])
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    # o3d.visualization.draw_geometries([mesh])
    return mesh

#NOTE: convert mesh -> sdf -> mesh to make it uniform

def postprocess_mesh( mesh, num_faces=None):
    """Post processing mesh by removing small isolated pieces.

    Args:
        mesh (trimesh.Trimesh): input mesh to be processed
        num_faces (int, optional): min face num threshold. Defaults to 4096.
    """
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 100
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=3)
    mask = np.zeros(total_num_faces, dtype=bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)

    return mesh

def to_numpy(data):
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data

def get_normalization_params(vertices, height=1.8):
    # vertices = m.vertices
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1
    center = np.median(vertices, 0)
    center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

    scale = height / (vmax[up_axis] - vmin[up_axis])
    return scale, center



def normalize_trimesh(m, height=1.8):
    # for standing io (general)
    vertices = m.vertices
    scale, center = get_normalization_params(vertices, height=height)

    # without library (comment below lines for non-agisoft results)
    vertices = (vertices - center) * scale
    return vertices, m.faces

def remesh_botsch(vertices, faces):
    from gpytoolbox import remesh_botsch
    iters = 20
    target_length = 0.02  # target length

    V2, F2 = remesh_botsch(vertices.astype(np.double),
                           faces.astype(np.int32),
                           i=iters, h=target_length, project=True)
    return V2, F2


def perspective_projection(points, rotation, translation,
                           focal_length=0., camera_center=0., K=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    if K is None:
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K.float(), projected_points.float())

    return projected_points[:, :, :-1]


def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0, 1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1, 2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2, 0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges


def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    # originally, assert counts.max()
    if counts.max() == 2:
        return None

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0)  # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64, device=indices.device)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei]
        face_correspondences_indices[ei_unique] += 1

    return face_correspondences[counts == 2].to(device=indices.device)


class AABB:
    def __init__(self, points):
        """ Construct the axis-aligned bounding box from a set of points.

        Args:
            points (tensor): Set of points (N x 3).
        """
        self.min_p, self.max_p = np.amin(points, axis=0), np.amax(points, axis=0)

    @classmethod
    def load(cls, path):
        points = np.loadtxt(path)
        return cls(points.astype(np.float32))

    def save(self, path):
        np.savetxt(path, np.array(self.minmax))

    @property
    def minmax(self):
        return [self.min_p, self.max_p]

    @property
    def center(self):
        return 0.5 * (self.max_p + self.min_p)

    @property
    def corners(self):
        return np.array([
            [self.min_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.max_p[1], self.max_p[2]],
            [self.min_p[0], self.max_p[1], self.max_p[2]]
        ])


def compute_laplacian_cot(V,F,device='cpu'):
    import igl
    # 1. Compute L using libigl (CPU, Scipy CSC format)
    # .squeeze() handles the (1, N, 3) -> (N, 3) case
    L_scipy = igl.cotmatrix(
        np.ascontiguousarray(V.squeeze().cpu().numpy()).astype(np.double), 
        np.ascontiguousarray(F.cpu().numpy()).astype(np.int64)
    )
    L_data = L_scipy.data
    is_nan = np.isnan(L_data)

    if is_nan.any():
        #print("NaN found! Applying Identity Trick...")
        
        # Convert L_scipy to a mutable format (LIL) (modifying CSR is slow)
        L_lil = L_scipy.tolil()
        
        # Find indices of rows that contain any NaN
        # (finding row indices depends on the sparse structure; the following is a straightforward approach)
        nan_indices = np.where(is_nan)[0]
        
        # Recover actual row indices from the sparse matrix's internal structure
        # The simple way is to loop or scan the whole matrix and check.
        # This shows a conceptual implementation.
        
        for i in range(L_lil.shape[0]):
            row_data = L_lil.data[i]
            if np.any(np.isnan(row_data)):
                # 2. Reset the entire row to 0
                L_lil.rows[i] = [i]  # keep only its own column index
                L_lil.data[i] = [1.0] # set itself (diagonal entry) to 1
                
                # (optional) to preserve symmetry the corresponding column should also be handled,
                # but handling only the row is usually enough to keep the solver from blowing up.

        # Convert back to CSR
        L_fixed = L_lil.tocsr()
        #print("Fixed Max:", L_fixed.max())

    else:
        #print("No NaN found.")
        L_fixed = L_scipy
    # 2. Convert Scipy CSC -> Scipy COO (Coordinate format)
    # PyTorch prefers COO format for construction
    L_coo = L_fixed.tocoo()

    # 3. Convert to PyTorch Sparse Tensor
    # We stack the row and col indices to create a (2, NNZ) tensor
    indices = torch.from_numpy(np.vstack((L_coo.row, L_coo.col))).long()
    values = torch.from_numpy(L_coo.data).float() # Cast to float32 for PyTorch training
    size = torch.Size(L_coo.shape)

    L = torch.sparse.FloatTensor(indices, values, size).to(device) 
    return L
def compute_laplacian_uniform2(V, edges, device='cpu'):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This src is adapted from from PyTorch3D
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    # if hasattr(io, 'vertices') and io.vertices is not None:
    #     V = io.vertices.shape[0]
    # else:
    #     V = io.v_posed.shape[0]
    # V = vertices.shape[0]

    # verts_packed = io.vertices # (sum(V_n), 3)
    edges_packed = edges  # (sum(E_n), 2)
    # V = io.vertices.shape[0]

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L





def marching_cubes(voxel_grid: torch.tensor, voxel_occupancy: torch.tensor, level: float = 0.5, **kwargs) -> Tuple[
    torch.tensor, torch.IntTensor]:
    """ Compute the marching cubes surface from an occupancy grid.

    Args:
        voxel_grid: Coordinates of the voxels with shape (HxWxDx3)
        voxel_occupancy: Occupancy of the voxels with shape (HxWxD), where 1 means occupied and 0 means not occupied.
        level: Occupancy value that marks the surface.

    Returns:
        Array of vertices (Nx3) and face indices (Nx3) of the marching cubes surface.
    """

    from skimage import measure
    spacing = (voxel_grid[1, 1, 1] - voxel_grid[0, 0, 0]).cpu().numpy()
    vertices, faces, normals, values = measure.marching_cubes_lewiner(voxel_occupancy.cpu().numpy(), level=0.5,
                                                                      spacing=spacing, **kwargs)

    # Re-center vertices
    vertices += voxel_grid[0, 0, 0].cpu().numpy()

    vertices = torch.from_numpy(vertices.copy()).to(voxel_grid.device)
    faces = torch.from_numpy(faces.copy()).to(voxel_grid.device)

    return vertices, faces







mesh_generator_names = [
    'vh16',
    'vh32',
    'vh64',
    'sphere16'
]


# def generate_mesh(generator_name: str, views: List[View], aabb: AABB, mask_adjusting_rate: float, device: torch.device = 'cpu'):
#     # Set up the io generators
#     mesh_generators = {
#         'vh16': (lambda: compute_visual_hull(views, aabb, grid_size=16, device=device, mask_adjusting_rate=mask_adjusting_rate)),
#         'vh32': (lambda: compute_visual_hull(views, aabb, grid_size=32, device=device, mask_adjusting_rate=mask_adjusting_rate)),
#         'vh64': (lambda: compute_visual_hull(views, aabb, grid_size=64, device=device, mask_adjusting_rate=mask_adjusting_rate)),
#         'sphere16': (lambda: generate_sphere(aabb, 16)),
#     }
#
#     v, f = mesh_generators[generator_name]()
#     return Mesh(v, f, device=device)
# def mesh2open3d(io, texture=None):
#     vertices = io.vertices.detach().cpu().numpy() if type(io.vertices) == torch.Tensor else io.vertices
#     faces = io.faces.detach().cpu().numpy() if type(io.faces) == torch.Tensor else io.faces
#     o3d_mesh = o3d.geometry.TriangleMesh()
#     o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
#     if texture is not None:
#         uvs = io.visual.uv
#         uvs[:, 1] = 1 - uvs[:, 1]
#
#         triangles_uvs = []
#         for i in range(3):
#             triangles_uvs.append(uvs[faces[:, i]].reshape(-1, 1, 2))
#         triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)
#
#         o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
#         o3d_mesh.textures = [o3d.io.read_image(texture)]
#     o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
#     return o3d_mesh


def vertex_to_face_color_batch(vertex_colors, faces):
    """
    Vertex Colors (B, V, C) -> Face Colors (B, F, C)
    Assumes the topology (faces) is shared by all samples in the batch.
    """
    # vertex_colors: [Batch, V, C]
    # faces: [F, 3]
    
    # 1. Gather the colors of the 3 vertices of each face via fancy indexing.
    # vertex_colors[:, faces] has shape [Batch, F, 3, C].
    # (the batch dimension is kept; the V dimension is indexed by faces)
    gathered_colors = vertex_colors[:, faces]
    
    # 2. Average over the 3 vertices (dim=2).
    # Result shape: [Batch, F, C]
    face_colors = torch.mean(gathered_colors, dim=2)
    
    return face_colors

def face_to_vertex_color_batch(face_colors, faces, num_vertices=None):
    """
    Face Colors (B, F, C) -> Vertex Colors (B, V, C)
    Assumes the topology (faces) is shared by all samples in the batch.
    """
    # face_colors: [Batch, F, C]
    # faces: [F, 3]
    
    B, F, C = face_colors.shape
    device = face_colors.device
    
    if num_vertices is None:
        num_vertices = faces.max().item() + 1
        
    # 1. Initialize the output tensor (including batch)
    vertex_colors = torch.zeros((B, num_vertices, C), device=device, dtype=face_colors.dtype)
    
    # Tensor counting how many faces each vertex belongs to (topology is shared, so no batch dimension needed)
    # Divided later via broadcasting
    counts = torch.zeros((num_vertices, 1), device=device, dtype=face_colors.dtype)
    ones = torch.ones(F, 1, device=device, dtype=face_colors.dtype) # value 1 to accumulate
    
    # 2. Loop over the 3 corners of each face (only 3 iterations)
    for i in range(3):
        idx = faces[:, i].long() # Shape: (F,)
        
        # [key] index_add_ supports batched inputs.
        # Add face_colors at idx positions along dim=1 (vertex dimension)
        # vertex_colors: (B, V, C), idx: (F,), source: (B, F, C) -> dimensions match
        vertex_colors.index_add_(1, idx, face_colors)
        
        # Counts depend only on topology, not batch, so compute them separately
        counts.index_add_(0, idx, ones)
        
    # 3. Compute the mean (avoid division by zero)
    counts = torch.clamp(counts, min=1.0)
    
    # counts is (V, 1) and vertex_colors is (B, V, C), so
    # reshape to (1, V, 1) for broadcast division
    vertex_colors = vertex_colors / counts.view(1, num_vertices, 1)
    
    return vertex_colors
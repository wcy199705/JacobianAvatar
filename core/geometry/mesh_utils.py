import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import trimesh
from typing import Tuple
# from diff_renderer.normal_nds.nds.core import Mesh, View
# from diff_renderer.normal_nds.nds.utils.images import sample


# Persistent store for Poisson remesh results.
#
# open3d's `create_from_point_cloud_poisson` uses multi-threaded sampling that
# gives slightly different output on each call (e.g. face count off by a few).
# That non-determinism propagates into `Cascade_Avatar.set_hand_idxs`, whose
# output is saved in the ckpt as `avatar_indices.rhand` etc. When a later
# launch produces a slightly different mesh, the saved buffer no longer matches
# and `ckpt_load.py` silently drops the keys — wrong region indices, splotchy
# colors.
#
# To fix this, the first run for a given input mesh + depth produces a
# canonical Poisson result and writes it to disk under
# `<project_root>/precomputed/poisson/<hash>.npz`. Subsequent runs (for any
# experiment on the same scene) read that file and skip the call entirely.
# Files are intended to be permanent — they are NOT temporary cache.
_POISSON_STORE_DIR = Path(__file__).resolve().parents[2] / 'precomputed' / 'poisson'


def _poisson_index_key(vertices: np.ndarray, depth: int) -> str:
    """SHA1 over (vertex bytes, depth) — stable across launches of the
    same scene."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(str(depth).encode())
    return h.hexdigest()[:16]


def poisson_trimesh(mesh, depth=9):
    """Deterministic Poisson remesh.

    Persists each result to `precomputed/poisson/<hash>.npz` so the next
    launch for the same scene returns the identical mesh (and therefore
    the identical `avatar_indices` after Cascade_Avatar setup).
    """
    mesh.vertex_normals.flags.writeable = True

    key = _poisson_index_key(np.asarray(mesh.vertices), depth)
    store_file = _POISSON_STORE_DIR / f'{key}.npz'
    if store_file.exists():
        data = np.load(store_file)
        return trimesh.Trimesh(data['vertices'], data['faces'], process=False)

    mesh_p = poisson_o3d(mesh.vertices,
                         vertex_normals=mesh.vertex_normals,
                         depth=depth)
    verts = np.asarray(mesh_p.vertices)
    faces = np.asarray(mesh_p.triangles)
    _POISSON_STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(store_file, vertices=verts, faces=faces)
    print(f'[poisson_remesh] persisted new mesh → {store_file.name} '
          f'({verts.shape[0]} verts, {faces.shape[0]} faces)')
    return trimesh.Trimesh(verts, faces, process=False)

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

# NOTE: convert mesh -> sdf -> mesh to make it uniform

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


def get_normalization_params(vertices, height=1.8):
    # vertices = m.vertices
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1
    center = np.median(vertices, 0)
    center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

    scale = height / (vmax[up_axis] - vmin[up_axis])
    return scale, center


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
        
        # Find indices of rows containing any NaN
        # (finding row indices can depend on the sparse structure; below is a straightforward approach)
        nan_indices = np.where(is_nan)[0]
        
        # Trace back the actual row indices from the sparse matrix internals
        # In the simplest form this requires looping or scanning everything.
        # A conceptual implementation is shown here.
        
        for i in range(L_lil.shape[0]):
            row_data = L_lil.data[i]
            if np.any(np.isnan(row_data)):
                # 2. Reset the entire row to zero
                L_lil.rows[i] = [i]  # keep only its own column index
                L_lil.data[i] = [1.0] # set its own (diagonal) entry to 1
                
                # (optional) To preserve symmetry the corresponding column should also be handled,
                # but handling rows alone is usually enough to keep the solver from blowing up.

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
    Assumes topology (faces) is shared by all samples in the batch.
    """
    # vertex_colors: [Batch, V, C]
    # faces: [F, 3]
    
    # 1. Fetch the colors of the 3 vertices of each face via fancy indexing.
    # vertex_colors[:, faces] has shape [Batch, F, 3, C].
    # (Batch dimension is kept; along V, entries are gathered by the faces indices)
    gathered_colors = vertex_colors[:, faces]
    
    # 2. Average over the 3 vertices (dim=2).
    # Result shape: [Batch, F, C]
    face_colors = torch.mean(gathered_colors, dim=2)
    
    return face_colors

def face_to_vertex_color_batch(face_colors, faces, num_vertices=None):
    """
    Face Colors (B, F, C) -> Vertex Colors (B, V, C)
    Assumes topology (faces) is shared by all samples in the batch.
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
    ones = torch.ones(F, 1, device=device, dtype=face_colors.dtype) # value of 1 to add
    
    # 2. Loop over the 3 corners of each face (only 3 iterations)
    for i in range(3):
        idx = faces[:, i].long() # Shape: (F,)
        
        # [Key] index_add_ supports batched inputs.
        # Add face_colors at idx positions along dim=1 (vertex dimension)
        # vertex_colors: (B, V, C), idx: (F,), source: (B, F, C) -> dimensions match exactly
        vertex_colors.index_add_(1, idx, face_colors)
        
        # Counts depend only on topology, not batch, so computed separately
        counts.index_add_(0, idx, ones)
        
    # 3. Compute mean (avoid division by zero)
    counts = torch.clamp(counts, min=1.0)
    
    # counts is (V, 1) and vertex_colors is (B, V, C), so
    # reshape to (1, V, 1) for a broadcast division
    vertex_colors = vertex_colors / counts.view(1, num_vertices, 1)
    
    return vertex_colors
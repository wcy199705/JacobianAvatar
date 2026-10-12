
#
# Adopted from NDS.core.
# Three essential primitive data structures are defined
# PointCloud (PCD)
# L Mesh with vertex colors
#   L Mesh with texture map (also UV coordinates)
# L GaussianModel also extends PCD
#
import torch
import numpy as np
import trimesh


class PointCloud:
    """ PointCloud with vertex normals and vertex colors.
    Args:
        vertices (tensor): Vertex buffer (Vx3)
        colors (tensor): Color buffer (Vx3)
        normals (tensor): Normal buffer (Vx3)
        device (torch.device): Device where the io buffers are stored
    """
    def __init__(self, vertices, normals=None, colors=None, device='cpu', detach=False):
        self.device = device

        # these variables are initialized when input is not None
        self._xyz = self.to_torch(vertices, device, detach=detach)
        self._rgb = self.to_torch(colors, device, detach=detach)
        self._norm_xyz = self.to_torch(normals, device, detach=detach)

    @property
    def xyz(self):
        return self._xyz  # set xyz as used in gaussian splats

    @property
    def normal(self):
        return self._norm_xyz  # to be distinguished from norm_face

    @property
    def rgb(self):
        return self._rgb

    @staticmethod
    def to_torch(data, device, dtype=torch.float32, detach=False):
        if data is None:
            return None
        elif torch.is_tensor(data):
            if detach:
                torch_tensor = data.detach()
            else:
                torch_tensor = data.to(device, dtype=dtype)
        else:
            torch_tensor = torch.tensor(data.copy().astype(np.float32), dtype=dtype, device=device)
        return torch_tensor

    def to(self, device):
        mesh = PointCloud(self._xyz, self._norm_xyz, self._rgb, device=device)
        return mesh

    def detach(self):
        mesh = PointCloud(self._xyz, self._norm_xyz, self._rgb, detach=True)
        return mesh


class Mesh(PointCloud):
    """ Triangle io defined by an indexed vertex buffer.
    Vertices have color values.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        colors (tensor) : Color buffer (Vx3)
        device (torch.device): Device where the io buffers are stored
        detach (bool): whether to detach variables or not, when the variables vertices, indices, colors are tensors
    """
    def __init__(self, vertices, indices, colors=None, device='cpu', detach=False):
        super(Mesh, self).__init__(vertices, colors=colors, device=device, detach=detach)

        self._face = self.to_torch(indices, device, detach=detach).type(torch.int64)

        # define and set internal variables for Mesh
        self._edge = None
        self._norm_xyz = None
        self._norm_face = None
        self._laplacian = None
        self._connected_face = None
        self._avg_edge = None  # the length of average edge
        self.set_connectivity()

    def set_connectivity(self):
        # set internal variables with setter functions
        """
        Set internal variables
        normal vectors, edges, laplacian, and connected faces
        Returns: Nothing
        """
        self.norm_face = self._xyz
        self.norm_xyz = self._xyz
        self.edge = self._face
        # self.connected_face = self._face  #
        self.laplacian = None

    def subdivide(self):
        vertices, faces = trimesh.remesh.subdivide(
            vertices=np.hstack((self.xyz.squeeze().detach().cpu().numpy(),
                                self.rgb.squeeze().detach().cpu().numpy())),
            faces=self.face.detach().cpu().numpy())

        # set subdivided smplx mesh with uv coordinates.
        self.xyz = vertices[:, 0:3]
        self.rgb = vertices[:, 3:6]
        self.face = faces
        self.set_connectivity()

    def to(self, device):
        mesh = Mesh(self._xyz.to(device), self._face.to(device), self._rgb.to(device), device=device)
        mesh._edge = self._edge.to(device) if self._edge is not None else None
        mesh._connected_face = self._connected_face.to(device) if self._connected_face is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        return mesh

    def detach(self):
        mesh = Mesh(self._xyz.detach(), self._face.detach(), self._rgb.detach(), device=self.device)
        mesh._norm_face = self._norm_face.detach()
        mesh._norm_xyz = self._norm_xyz.detach()
        mesh._edges = self._edge.detach() if self._edge is not None else None
        mesh._connected_faces = self._connected_face.detach() if self._connected_face is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh

    def to_trimesh(self):
        vertices = self._xyz.cpu().detach().numpy()
        color = self._rgb.cpu().detach().numpy()
        faces = self._face.cpu().detach().numpy().astype(np.int32)
        return trimesh.Trimesh(vertices, faces, vertex_colors=color, process=False)

    # -------------------------- essential functions for mesh --------------------------
    @property
    def xyz(self):
        return self._xyz

    @property
    def face(self):
        return self._face

    @property
    def rgb(self):
        return self._rgb

    @property
    def edge(self):
        return self._edge

    @property
    def connected_face(self):
        return self._connected_face

    @property
    def laplacian(self):
        return self._laplacian

    @property
    def norm_face(self):
        return self._norm_face

    @property
    def norm_xyz(self):
        return self._norm_xyz

    @xyz.setter
    def xyz(self, xyz):
        self._xyz = self.to_torch(xyz, self.device)

    @face.setter
    def face(self, face):
        self._face = self.to_torch(face, self.device).type(torch.int64)

    @rgb.setter
    def rgb(self, rgb):
        self._rgb = self.to_torch(rgb, self.device)

    @edge.setter
    def edge(self, face):
        from utils.geometry.mesh_utils import find_edges
        self._edge = find_edges(face).to(self.device)

    @connected_face.setter
    def connected_face(self, face):
        from utils.geometry.mesh_utils import find_connected_faces
        self._connected_face = find_connected_faces(face).to(self.device)

    @laplacian.setter
    def laplacian(self, _):
        from utils.geometry.mesh_utils import compute_laplacian_uniform2
        self._laplacian = compute_laplacian_uniform2(self._xyz.shape[-2], self._edge, device=self.device)

    @property
    def avg_edge(self):
        if self._avg_edge is None:
            self.avg_edge(self._xyz)
        return self._avg_edge

    @avg_edge.setter
    def avg_edge(self, xyz):
        a = xyz[self._face][:, 0, :]
        b = xyz[self._face][:, 1, :]
        c = xyz[self._face][:, 2, :]
        e0 = (a - b).norm(dim=1, p=2)
        e1 = (b - c).norm(dim=1, p=2)
        e2 = (c - a).norm(dim=1, p=2)

        # average length to set various threshold value
        self._avg_edge = torch.mean(e0 + e1 + e2) / 3

    @norm_face.setter
    def norm_face(self, xyz):
        if len(xyz.shape) == 2:
            a = xyz[self._face][:, 0, :]
            b = xyz[self._face][:, 1, :]
            c = xyz[self._face][:, 2, :]
        else:
            a = xyz[:, self._face][:, :, 0, :]
            b = xyz[:, self._face][:, :, 1, :]
            c = xyz[:, self._face][:, :, 2, :]
        self._norm_face = torch.nn.functional.normalize(torch.cross(b - a, c - a, dim=-1), p=2, dim=-1).to(self.device)

    @norm_xyz.setter
    def norm_xyz(self, xyz):
        if self._norm_face is None:
            self.norm_face(xyz)

        vertex_normals = torch.zeros_like(xyz).to(self.device)
        if len(xyz.shape) == 2:
            vertex_normals = vertex_normals.index_add(0, self._face[:, 0], self._norm_face)
            vertex_normals = vertex_normals.index_add(0, self._face[:, 1], self._norm_face)
            vertex_normals = vertex_normals.index_add(0, self._face[:, 2], self._norm_face)
        else:
            vertex_normals = vertex_normals.index_add(1, self._face[:, 0], self._norm_face)
            vertex_normals = vertex_normals.index_add(1, self._face[:, 1], self._norm_face)
            vertex_normals = vertex_normals.index_add(1, self._face[:, 2], self._norm_face)
        self._norm_xyz = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)


class TexturedMesh(Mesh):
    """ Triangle io defined by an indexed vertex buffer.
    __init__, to, detach functions are overrode
    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        uv_vts (tensor) : UV coordinate for vertices (Vx2)
        tex (tensor) : texture map (HxWx3), initialized as a grey image if None
        device (torch.device): Device where the io buffers are stored
    """
    def __init__(self, vertices, indices, uv_vts, tex=None, tex_res=None, device='cuda:0', detach=False):
        # initialize Mesh.__init__()
        super(TexturedMesh, self).__init__(vertices, indices, device=device, detach=detach)

        self._uv = self.to_torch(uv_vts, device, detach=detach)
        if tex is not None:
            self._tex = self.to_torch(tex, device)
            if torch.max(self._tex) > 1.0:
                self._tex = self._tex / 255.0
        else:
            if tex_res is None:
                tex_res = [1024, 1024]
            self._tex = torch.ones((tex_res[0], tex_res[1], 3),
                                  dtype=torch.float32,
                                  requires_grad=True).to(device) * 0.2

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = self.uv_sampling()
        return self._rgb

    @property
    def uv(self):
        return self._uv

    @uv.setter
    def uv(self, uv):
        self._uv = self.to_torch(uv, device=self.device)

    @property  # no setter for uv texture map
    def tex(self):
        return self._tex

    # sampling vertex colors and displacement vectors in the uv space.
    def uv_sampling(self):
        '''
        :param displacement: [B, C, H, W] image features
        :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
        :return: [B, C, N] image features at the uv coordinates
        '''
        # uv is in [0, 1] so we need to convert to be in [-1, 1]
        uv = self._uv.unsqueeze(0).unsqueeze(2) * 2 - 1
        tex = self._tex.permute(2, 0, 1).unsqueeze(0)
        samples = torch.nn.functional.grid_sample(tex, uv, align_corners=True)  # [B, C, N, 1]
        return samples.squeeze().transpose(1, 0).contiguous()

    def subdivide(self):
        vertices, faces = trimesh.remesh.subdivide(
            vertices=np.hstack((self.xyz.detach().cpu().numpy(),
                                self.rgb.detach().cpu().numpy(),
                                self.uv.detach().cpu().numpy())),
            faces=self.face.detach().cpu().numpy())

        # set subdivided smplx mesh with uv coordinates.
        self.xyz = vertices[:, 0:3]
        self.rgb = vertices[:, 3:6]
        self.uv = vertices[:, 6:8]
        self.face = faces
        self.set_connectivity()

    # ----------------------- overrode functions ----------------------- #
    def to_trimesh(self, with_texture=False):
        import cv2
        from PIL import Image
        vertices = self._xyz.cpu().detach().numpy()
        uv = self._uv.cpu().detach().numpy()
        faces = self._face.cpu().detach().numpy().astype(np.int32)
        if with_texture:
            texture = np.uint8(self._tex.cpu().detach().numpy() * 255)
            texture_image = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            texture_image = np.flip(texture_image, axis=0)
            tex_pil = Image.fromarray(texture_image)
        else:
            tex_pil = None
        visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_pil)
        return trimesh.Trimesh(vertices, faces, visual=visual, process=False)

    def to(self, device):
        self.device = device
        mesh = TexturedMesh(self._xyz.to(device),
                            self._face.to(device),
                            self._uv.to(device),
                            tex=self._tex.to(device),
                            device=device)
        mesh._edge = self._edge.to(device) if self._edge is not None else None
        mesh._norm_xyz = self._norm_xyz.to(device) if self._norm_xyz is not None else None
        mesh._norm_face = self._norm_face.to(device) if self._norm_face is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        mesh._connected_face = self._connected_face.to(device) if self._connected_face is not None else None
        return mesh

    def detach(self):
        mesh = TexturedMesh(self._xyz.detach(),
                            self._face.detach(),
                            self._uv.detach(),
                            tex=self._tex.detach(),
                            device=self.device,
                            detach=True)

        mesh._edges = self._edge.detach() if self._edge is not None else None
        mesh._norm_xyz = self._norm_xyz.detach() if self._norm_xyz is not None else None
        mesh._norm_face = self._norm_face.detach() if self._norm_face is not None else None
        mesh._connected_faces = self._connected_face.detach() if self._connected_face is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh
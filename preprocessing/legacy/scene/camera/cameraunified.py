import torch
import numpy as np
import torch.nn.functional as F
from utils.graphics.graphics_utils import *
from utils.geometry.camera_utils import update_intrinsic_after_downsampling, make_rotate_euler, mvp_from_camera_param, homogenous_mvp_from_camera_param, homogenous_perspective_from_intriniscs , update_intrinsic_after_crop, ortho_mvp_from_camera_praram


class UnifiedCamera:
    """ Camera class compatible with GaussianSplat rasterizer
    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """
    def __init__(self, K, R, t, fovx, fovy, znear, zfar, width, height,
                 timestep=None, orthographic=False, homogenous=False, device=torch.device('cuda:0')):
        self.device = device
        self.K = self.to_torch(K, self.device)
        self.R = self.to_torch(R, self.device)
        self.t = self.to_torch(t, self.device)
        self.orthographic = orthographic
        self.gl_coordinate = True
        # compatible with gaussian rasterizer
        self.image_width = width
        self.image_height = height
        self.image_size = [height, width]
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear = znear
        self.zfar = zfar
        self.homogenous=homogenous
        #import pdb;pdb.set_trace()
        if homogenous:
            import pdb;pdb.set_trace()
            print("CY: error exists (should change to pytorc3d renderer/cameras)")
            
        #    assert len(self.K.shape) == 2
        #    self.mvp = homogenous_mvp_from_camera_param(self.RT.unsqueeze(0),self.K.unsqueeze(0), self.image_height,self.image_width)
        #    self.projection_matrix = homogenous_perspective_from_intriniscs(self.K.unsqueeze(0), self.image_height,self.image_width) [0]
        ##else:
        #    self.homogenous = False
        
        if orthographic:
            self.mvp = ortho_mvp_from_camera_praram(self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear, far=self.zfar) #not use intrinsic
        else:
            self.mvp = mvp_from_camera_param(self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear, far=self.zfar)

        self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, fovX=fovx, fovY=fovy, focal_x=self.K[0,0], 
            focal_y=self.K[1,1], cx=self.K[0,2],cy=self.K[1,2], width=self.image_width, height=self.image_height,).transpose(0, 1).to(self.device)

                                 #                       fovY=self.FoVy)
        
        
        #getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                 #                       fovY=self.FoVy).transpose(0, 1).to(self.device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.timestep = timestep

        # Camera coordinate system (up-vector and viewing direction)
        self.direction = np.array([0, 0, -1])
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])

    @staticmethod
    def to_torch(data, device, dtype=torch.float32):
        if data is None:
            return None
        elif torch.is_tensor(data):
            torch_tensor = data.to(device, dtype=dtype)
        else:
            torch_tensor = torch.tensor(data.copy(), dtype=dtype, device=device)
        return torch_tensor

    @property
    def center(self):
        return -self.R.t() @ self.t

    @property
    def P(self):
        return self.K @ torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)

    @property
    def gl_base(self):
        return torch.FloatTensor(np.array([[1., 0,  0],
                                           [0, -1., 0],
                                           [0,  0,-1.]])).to(self.device)

    @property
    def RT(self):
        return torch.cat([self.gl_base @ self.R, self.t.unsqueeze(-1)], dim=-1)
        # return torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)

    @property
    def world_view_transform(self):
        RT = torch.cat([self.R, self.gl_base @self.t.unsqueeze(-1)], dim=-1)
        RT_new = F.pad(RT, (0, 0, 0, 1), mode='constant', value=0)
        RT_new[3, 3] = 1.0
        return RT_new.transpose(0, 1)

    @staticmethod
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    def get_rotation_matrix(self):
        rot_mat = np.eye(3)
        s = self.right
        s = self.normalize_vector(s)
        rot_mat[0, :] = s
        u = self.up
        u = self.normalize_vector(u)
        rot_mat[1, :] = -u
        rot_mat[2, :] = self.normalize_vector(self.direction)
        return rot_mat

    def project_points(self, x):
        pass

    @staticmethod
    def intrinsic_from_fov(fov_x, fov_y, width, height):
        f_x = fov2focal(np.deg2rad(fov_x), width)
        f_y = fov2focal(np.deg2rad(fov_y), height)
        return np.array([[f_x, 0, width/2.0],
                         [0, f_y,  height/2.0],
                         [0,   0,       1.0]])

    @staticmethod
    def intrinsic_from_param(cam_param):
        return np.array([[cam_param['fx'], 0, cam_param['width']/2.0],
                         [0, cam_param['fy'],  cam_param['height']/2.0],
                         [0,               0,                    1.0]])
    @staticmethod
    def intrinsic_from_param_each(fx,fy,cx,cy):
        return np.array([[fx, 0, cx],
                         [0, fy,  cy],
                         [0,               0,                    1.0]])
    

    @classmethod
    def perspective_camera_from_K(cls, K, width, height, znear, zfar, R=None, T=None, orthographic=False, device=torch.device('cpu')):

        K = K.to(device)

        if R==None and T==None:
            #static camera case.
            R= torch.eye(3,device=device)
            #due to glbase
            #R[1,1] = -1
            #R[2,2] = -1
            t= [0,0,0]
        else:
            R = R.to(device)
            t = T.to(device)

        fovx = focal2fov(focal=K[0,0],pixels=width) 
        fovy = focal2fov(focal=K[1,1],pixels=height)
        

        camera = UnifiedCamera(K, R, t, fovx, fovy, znear, zfar, width, height, orthographic=orthographic, homogenous=False, device=device)

        return camera




    def K_update_bbox(self, bbox):
        #bbox = [x1,y1,w,h]
        #update K
        
        self.K = update_intrinsic_after_crop(self.K,bbox=bbox)
        self.image_height, self.image_width = min(self.image_height,bbox[3]),min(self.image_height,bbox[2])
        self.image_size = [self.image_height, self.image_width]
        fovx = focal2fov(focal=self.K[0,0],pixels=self.image_width)
        fovy = focal2fov(focal=self.K[1,1],pixels=self.image_height)
        self.FoVx = fovx
        self.FoVy = fovy
        if self.homogenous:
            self.mvp = homogenous_mvp_from_camera_param(self.RT.unsqueeze(0),self.K.unsqueeze(0), self.image_height,self.image_width)
            self.projection_matrix = homogenous_perspective_from_intriniscs(self.K.unsqueeze(0), self.image_height,self.image_width)
        else:
            self.homogenous = False
            self.mvp = mvp_from_camera_param(self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear,far=self.zfar)
            self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, fovX=fovx,    focal_x=self.K[0,0], 
                focal_y=self.K[1,1], cx=self.K[0,2],cy=self.K[1,2], width=self.image_width, height=self.image_height,
                                                                    fovY=fovy).transpose(0, 1).to(self.device)

            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.device)

    def downsampling_camera(self, downsampling):
        #bbox = [x1,y1,w,h]
        #update K
        w_ratio = (self.image_width//downsampling)/self.image_width
        h_ratio = (self.image_height//downsampling)/self.image_height

        self.K = update_intrinsic_after_downsampling(self.K,w_ratio, h_ratio)
        self.image_height, self.image_width = self.image_height//downsampling, self.image_width//downsampling
        self.image_size = [self.image_height, self.image_width]
        if self.homogenous:
            self.mvp = homogenous_mvp_from_camera_param(self.RT.unsqueeze(0),self.K.unsqueeze(0), self.image_height,self.image_width)
            self.projection_matrix = homogenous_perspective_from_intriniscs(self.K.unsqueeze(0), self.image_height,self.image_width)
        else:
            self.homogenous = False
            self.mvp = mvp_from_camera_param(self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear,far=self.zfar)
            fovx = focal2fov(focal=self.K[0,0],pixels=self.image_width)
            fovy = focal2fov(focal=self.K[1,1],pixels=self.image_height)
            self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, fovX=fovx, fovY=fovy, focal_x=self.K[0,0], 
                    focal_y=self.K[1,1], cx=self.K[0,2],cy=self.K[1,2], width=self.image_width, height=self.image_height,).transpose(0, 1).to(self.device)
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.device)


    def add_rotation_noise(self, noise_scale=0.0, is_radian=False):
            """
            
            """
            # 1. Generate random noise for yaw, pitch, roll using randn
            # torch.randn(3) draws 3 samples from a standard normal distribution (mean 0, std 1).
            random_noises = torch.randn(3) * noise_scale
            yaw_noise, pitch_noise, roll_noise = random_noises.tolist()
            
            # 2. Build the noise rotation matrix
            noise_R = make_rotate_euler(yaw=yaw_noise, pitch=pitch_noise, roll=roll_noise, is_radian=is_radian)
            noise_R = self.to_torch(noise_R, self.device)
            
            # 3. Apply the noise to the existing rotation matrix (matrix product)
            self.R = torch.matmul(noise_R, self.R)
            
            # 4. Recompute and update dependent (extrinsic-related) quantities after the rotation change
            if self.homogenous:
                self.mvp = homogenous_mvp_from_camera_param(
                    self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_height, self.image_width
                )
            elif self.orthographic:
                self.mvp = ortho_mvp_from_camera_praram(
                    self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear, far=self.zfar
                )
            else:
                self.mvp = mvp_from_camera_param(
                    self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear, far=self.zfar
                )

            # Update the full projection transform (full_proj_transform) and camera center (camera_center)
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
            ).squeeze(0).to(self.device)
            
            self.camera_center = self.world_view_transform.inverse()[3, :3]
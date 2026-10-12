import torch
import numpy as np
from utils.graphics.graphics_utils import *
from utils.geometry.camera_utils import mvp_from_camera_param, update_intrinsic_after_crop, ortho_mvp_from_camera_praram
from scene.camera.cameraunified import UnifiedCamera

class Cam_w_minicam(UnifiedCamera):
    """ Camera class compatible with GaussianSplat rasterizer : minicam version
    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """
    def __init__(self, K, R, t, fovx, fovy, znear, zfar, width, height, minicam_bbox,
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
        if orthographic:
            self.mvp = ortho_mvp_from_camera_praram(self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear, far=self.zfar) #not use intrinsic
        else:
            self.mvp = mvp_from_camera_param(self.RT.unsqueeze(0), self.K.unsqueeze(0), self.image_size, near=self.znear, far=self.zfar)

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                        fovY=self.FoVy).transpose(0, 1).to(self.device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.timestep = timestep

        # Camera coordinate system (up-vector and viewing direction)
        self.direction = np.array([0, 0, -1])
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])
    
        self.get_minicam(minicam_bbox)


    
    def get_minicam(self, bbox):
        #bbox = [x1,y1,w,h]
        #update K
        minicam_K = update_intrinsic_after_crop(self.K,bbox=bbox)
        minicam_width, minicam_height=  bbox[2], bbox[3]

        self.minicam = UnifiedCamera.perspective_camera_from_K(minicam_K,minicam_width,minicam_height,self.znear,self.zfar,R=self.R,T=self.t, device=self.device)
    

    @classmethod
    def perspective_camera_from_K(cls, K, minicam_bbox, width, height, znear, zfar, R=None, T=None, orthographic=False, device=torch.device('cpu')):

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
        

        camera = Cam_w_minicam(K, R, t, fovx, fovy, znear, zfar, width, height, orthographic=orthographic, minicam_bbox=minicam_bbox, homogenous=False, device=device)

        return camera
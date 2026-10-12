
import cv2
import os
import torch  # necessary to import simple_knn._C (libc.so)
# import numpy as np
from torch import nn
from random import shuffle
from typing import Optional
from plyfile import PlyElement
from core.mesh import PointCloud
from core.gaussian_utils import *
from core.graphics.sh_utils import RGB2SH, SH2RGB
from core.camera import UnifiedCamera
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel(PointCloud):
    """
    GaussianModel extends PointCloud by adding important properties to each point,
    such as opacity, spherical harmonics, scale, and rotation
    """
    def __init__(self, sh_degree=3, optimizer_type="default", device='cuda:0'):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device = device

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    

    @property
    def scaling(self):
        #import pdb;pdb.set_trace()
        return self.scaling_activation(self._scaling)
    
    @property
    def rotation(self):
        #import pdb;pdb.set_trace()

        return self.rotation_activation(self._rotation)
    
    @property
    def xyz(self):
        #import pdb;pdb.set_trace()
        return self._xyz

    @property
    def normal(self):
        print("not test. Smallest scale -> surface of gaussian -> normal of gaussian ")
        _, scale_idx = torch.min(self.scaling)
        rot_mat = build_rotation(self.rotation)
        normal = rot_mat[torch.arange(0,rot_mat.shape[0]),scale_idx,torch.arange(0,3)]
        return normal
    
    @property
    def rgb(self):
        return SH2RGB(self.features_dc)

    @property
    def features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def features_dc(self):
        return self._features_dc
    
    @property
    def features_rest(self):
        return self._features_rest
    
    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    
    @xyz.setter
    def xyz(self, xyz):
        #import pdb;pdb.set_trace()

        self._xyz = nn.Parameter(xyz.requires_grad_(False))
    
    @features_dc.setter
    def features_dc(self,features_dc):#, direct=False):
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))

    @features_rest.setter
    def features_rest(self, features_rest):# direct=False):
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))

    @rotation.setter
    def rotation(self,quaternion):#, direct=False):
        #import pdb;pdb.set_trace()
        self._rotation = nn.Parameter(quaternion.requires_grad_(True))
        
    @opacity.setter
    def opacity(self,target_opacity):
        """
        inverse_opacity_activation is inverse function of sigmoid
        """
        self._opacity = nn.Parameter(self.inverse_opacity_activation(target_opacity).requires_grad_(True))
        # if direct:
        #     self._opacity = target_opacity
        # else:

    @scaling.setter
    def scaling(self, target_scaling):
        """
        scaling_inverse_activation is tor.log ( activation of scaling is exponential function.)

        """
        self._scaling = nn.Parameter(self.scaling_inverse_activation(target_scaling).requires_grad_(True))
        # self._scaling = nn.Parameter(target_scaling.requires_grad_(True))
        # if direct:
        #     self._scaling = target_scaling
        # else:

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    
    


    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._rotation = self._rotation.to(device)
        self._scaling = self._scaling.to(device)
        self.max_radii2D = self.max_radii2D.to(device)


        
        #self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
        #                                                lr_delay_steps=training_args.exposure_lr_delay_steps,
        #                                                lr_delay_mult=training_args.exposure_lr_delay_mult,
        #                                                max_steps=training_args.iterations)


            #validation step            

            
    

        
        #Naive version: test results are saved in model_path folder

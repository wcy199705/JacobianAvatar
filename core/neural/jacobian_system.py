# Differential-operator setup adapted from Neural Jacobian Fields (https://github.com/ThibaultGROUEIX/NeuralJacobianFields),
# source_njf/MeshProcessor.py. Copyright (c) Adobe Inc. and its licensors. Adobe Research License (noncommercial
# research use only), see third_party_licenses/NeuralJacobianFields/license.txt.
# Network design based on ExAvatar (https://github.com/mks0601/ExAvatar_RELEASE), MIT License,
# see third_party_licenses/ExAvatar/LICENSE.

import torch
import torch.nn as nn
import numpy as np
#    triplane_shape_3d = (2, 2, 2)
#    triplane_face_shape_3d = (0.3, 0.3, 0.3)
#    triplane_shape = (32, 128, 128)
#     lr = 1e-3 
from copy import deepcopy
# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

#from utils.graphics.sh_utils import RGB2SH, eval_sh_mesh, SH2RGB
#import torch_scatter

import os
"""
implement ezmesh based approaches.
"""
from easydict import EasyDict

from core.optim.poisson import screened_poisson_system_matrices_from_mesh
import numpy as np


class jacobian_system(nn.Module):
    def __init__(self, jacob_config, init_da_mesh, device='cuda'):
        super(jacobian_system, self).__init__()    


        self.device = device
        self.jacob_config = jacob_config
        self.dump = self.jacob_config['dump']

        #init_mesh = Mesh(init_vts,ez_mesh.face,colors=0.3*torch.ones_like(init_vts))
        self.adj_pair = None
        self.init_mesh = init_da_mesh
        self.init_vts = init_da_mesh.xyz.squeeze()
        self.init_face = init_da_mesh.face
        self.init_poisson_solver(self.jacob_config['jacobian_solver'])
        self.gen_poisson_solver()


    def init_poisson_solver(self,poisson_cfg):
        self.diff_ops = EasyDict()
        self.diff_ops.splu = EasyDict()
        self.diff_ops.splu.L = None
        self.diff_ops.splu.U = None
        self.diff_ops.splu.perm_c = None
        self.diff_ops.splu.perm_r = None
        self.diff_ops.frames = None
        self.diff_ops.rhs = None
        self.diff_ops.grad = None
        self.diff_ops.poisson_sys_mat = None
        self.faces_wks = None
        self.vert_wks = None
        self.diff_ops.poisson = None
        self.compute_splu= poisson_cfg['compute_splu']
        self.ttype = torch.float32


    @torch.no_grad()
    def gen_poisson_solver(self): #,solver_save_path):
        init_da_verts = self.init_vts.clone()
        init_da_mean = torch.mean(init_da_verts, dim=0)[None, :].to(self.device)
        
        if 'init_da_verts' not in self._buffers:
            self.register_buffer('init_da_verts', init_da_verts)
        else:
            self.init_da_verts = init_da_verts

        if 'init_da_mean' not in self._buffers:
            self.register_buffer('init_da_mean', init_da_mean)
        else:
            self.init_da_mean = init_da_mean


        poisson_sys_mat = screened_poisson_system_matrices_from_mesh(V= self.init_vts.detach().cpu().numpy(), F=self.init_face.cpu().numpy(),  cpuonly=False, dump=self.dump, da_vts=init_da_verts-self.init_da_mean )#,screen_damp=1)
        self.diff_ops.grad = poisson_sys_mat.igl_grad
        self.diff_ops.rhs = poisson_sys_mat.rhs
        self.diff_ops.laplacian = poisson_sys_mat.lap
        self.diff_ops.frames = poisson_sys_mat.w
        self.diff_ops.poisson_sys_mat = poisson_sys_mat
        
        diff_ops = self.diff_ops
        #if self.diff_ops.poisson_sys_mat is None: # not created if loaded from disk the diff ops
        #    diff_ops.poisson_sys_mat = PoissonSystemMatrices(da_verts.cpu().numpy(), faces.cpu().numpy(), diff_ops.grad, diff_ops.rhs, diff_ops.frames, self.ttype, lap = diff_ops.laplacian,  cpuonly=self.cpuonly)
        self.diff_ops.poisson_solver = diff_ops.poisson_sys_mat.create_poisson_solver().to(self.device) # call 2

    def vertices_from_jacobians(self, d):
        return self.diff_ops.poisson_solver.solve_poisson(d)


    def get_optimizable_params(self):
        all_params = []
        if not len(self.optimizable_modules)==0:
            for module in self.optimizable_modules:
                module.toggle_trainable_params()
                all_params.extend(module.get_optimizable_params())
        return all_params
    #save
    #load separatly.
        #add ezmesh


    #load mlps

        #        self.jacobians_from_vertices( self.vertices_from_jacobians(torch.eye(3,3)[None].repeat(self.face.shape[0],1,1)[None].cuda()))

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
    



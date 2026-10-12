
import cv2
import os
import torch  # necessary to import simple_knn._C (libc.so)
# import numpy as np
from torch import nn
from random import shuffle
from typing import Optional
from plyfile import PlyElement
from scene.object.mesh import PointCloud
from scene.object.gaussian_utils import *
from utils.graphics.sh_utils import RGB2SH, SH2RGB
from scene.camera.cameraunified import UnifiedCamera
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

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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

    @property
    def exposure(self):
        return self._exposure
    
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

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def get_xyz_nearest_distance(self):
        from simple_knn._C import distCUDA2
        eps = 0.0000001
        distance = torch.clamp_min(distCUDA2(self.xyz), eps)
        return distance
    
    def normal2rot_scale_for_GS_init(self, normal:torch.Tensor):
        eps = 1e-10
        def compute_closest_distances(points):
            from scipy.spatial import cKDTree
            # Convert torch tensor to numpy array
            points_np = points.detach().cpu().numpy()
            
            # Build KDTree
            tree = cKDTree(points_np)
            
            # Find the nearest neighbor of each point
            closest_distances = torch.full((points.shape[0],), float('inf'))
            closest_vectors = torch.zeros((points.shape[0], 3))
            batch_size = 8192
            for i in range(0,points.shape[0],batch_size):
                # Query the KDTree for the nearest point (excluding the point itself)
                dist, idx = tree.query(points_np[i:i+batch_size], k=2, workers=4)  # k=2 includes the point itself, so take the 2nd nearest
                closest_distances[i:i+batch_size] = torch.tensor(dist[:,1]) + torch.sqrt(3*torch.tensor(eps**2)) # minimum distance excluding the point itself
                closest_vectors[i:i+batch_size] = torch.tensor(points_np[idx[:,1]] - points_np[i:i+batch_size] +eps )  # vector to the nearest point

            return closest_distances.to(self.device), closest_vectors.to(self.device)

        distance, vector = compute_closest_distances(self.xyz)
        unitvector2nearest = vector/distance[:,None]
        rotation = torch.stack([normal, unitvector2nearest, torch.cross(normal, unitvector2nearest,dim=-1)], dim=-1) # check n's value
        return rotation, distance

    def init_gs_from_rgb(self, rgb):
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        return (features[:, :, 0:1].transpose(1, 2).contiguous(),
                features[:, :, 1:].transpose(1, 2).contiguous())

    def set_GS_from_PCD(self, PCD: PointCloud ):
        """
        Convert PointCloud class to Gaussian Class
        """
        fused_point_cloud = PCD.xyz
        fused_color = RGB2SH(PCD.rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self.xyz = fused_point_cloud

        if PCD.normal != None:
            rots, dist2 = self.normal2rot_scale_for_GS_init(PCD.normal)
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1 # quaternion's w = 1
            self.rotation = rots
            dist2 = self.get_xyz_nearest_distance()
        #set GS parameters
        self.features_dc= features[:,:,0:1].transpose(1, 2).contiguous()
        self.features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self.scaling =torch.sqrt(dist2*1e-2)[...,None].repeat(1,3)
        self.max_radii2D = torch.zeros((self.xyz.shape[0])).to(self.device)
        self.opacity = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32).to(self.device)
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._rotation = self._rotation.to(device)
        self._scaling = self._scaling.to(device)
        self.max_radii2D = self.max_radii2D.to(device)

    def save_gs_ply(self, path):
        #mkdir_p(os.path.dirname(path))
       # import pdb;pdb.set_trace()
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale =  self.scaling_inverse_activation(self.scaling.detach()).cpu().numpy() 
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.opacity, torch.ones_like(self.opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        #change to set function
        """
        self.xyz =optimizable_tensors["xyz"]
        self.features_dc ( optimizable_tensors["f_dc"] , direct=True)
        self.features_rest ( optimizable_tensors["f_rest"] , direct=True)
        self.opacity ( optimizable_tensors["opacity"] , direct=True)
        self.scaling ( optimizable_tensors["scaling"] , direct=True)
        self.rotation ( optimizable_tensors["rotation"] , direct=True)
        """
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def select_densification(self,grads,grad_threshold):
        """
        Densification in Algorithm 1 of original 3D Gaussian Splatting paper (if delta p L > tau_p)
        """
        n_init_points = self.xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        return selected_pts_mask

    def select_over_reconstruction(self,grads,grad_threshold,scene_extent):
        selected_pts_mask = self.select_densification(grads,grad_threshold)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling, dim=1).values > self.percent_dense*scene_extent)
        return selected_pts_mask

    def select_under_reconstruction(self,grads,grad_threshold,scene_extent):
        selected_pts_mask = self.select_densification(grads,grad_threshold)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling, dim=1).values <= self.percent_dense*scene_extent)
        return selected_pts_mask

    def select_pts(self,selected_pts_mask):
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_pts = {"xyz": new_xyz,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "opacity": new_opacity,
                "scaling" : new_scaling,
                "rotation" : new_rotation,
                "tmp_radii": new_tmp_radii}
        return new_pts

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask = self.select_over_reconstruction(grads, grad_threshold, scene_extent)

        stds = self.scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds) #random
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_pts = self.select_pts(selected_pts_mask)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + new_pts['xyz'].repeat(N, 1)
        new_scaling = new_pts['scaling'].repeat(N,1) / (0.8*N)
        new_rotation = new_pts['rotation'].repeat(N,1)
        new_features_dc = new_pts['f_dc'].repeat(N,1,1)
        new_features_rest = new_pts['f_rest'].repeat(N,1,1)
        new_opacity = new_pts['opacity'].repeat(N,1)
        new_tmp_radii = new_pts['tmp_radii'].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = self.select_under_reconstruction(grads,grad_threshold,scene_extent)
        new_pts = self.select_pts(selected_pts_mask)
        new_xyz = new_pts['xyz']
        new_features_dc = new_pts['f_dc']
        new_features_rest = new_pts['f_rest']
        new_opacity = new_pts['opacity']
        new_scaling = new_pts['scaling']
        new_rotation = new_pts['rotation']
        new_tmp_radii = new_pts['tmp_radii']

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    def update_densification_stats(self,
                                train_render_pkg,
                                ):
        viewspace_point_tensor = train_render_pkg['viewspace_points']
        visibility_filter = train_render_pkg['visibility_filter']
        radii = train_render_pkg['radii']
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
        self.add_densification_stats(viewspace_point_tensor, visibility_filter)

    def training_setup(self, training_args, train_cameras):
        self.set_gs_spatial_lr_scale(train_cameras)
        self.percent_dense = training_args['percent_dense']
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        lr_args = training_args['lr']
        print("Warning!!!!!!!!!!!!!!!! I set lr of xyz and features as  0")
        l = [
            {'params': [self._xyz], 'lr': lr_args['xyz_init']* self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': lr_args['feature'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_args['feature'] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': lr_args['opacity'], "name": "opacity"},
            {'params': [self._scaling], 'lr': lr_args['scale'], "name": "scaling"},
            {'params': [self._rotation], 'lr': lr_args['rotation'], "name": "rotation"}
        ]
        #import pdb;pdb.set_trace()
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        #self.exposure_optimizer = torch.optim.Adam([self._exposure])
        #import pdb;pdb.set_trace()
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=lr_args['xyz_init']* self.spatial_lr_scale,
                                                    lr_final=lr_args['xyz_final']* self.spatial_lr_scale,
                                                    lr_delay_mult=lr_args['xyz_decay'],
                                                    max_steps=lr_args['xyz_max_steps'])
        
        #self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
        #                                                lr_delay_steps=training_args.exposure_lr_delay_steps,
        #                                                lr_delay_mult=training_args.exposure_lr_delay_mult,
        #                                                max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        #if self.pretrained_exposures is None:
        #    for param_group in self.exposure_optimizer.param_groups:
        #        param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                #import pdb;pdb.set_trace()
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def set_gs_spatial_lr_scale(self, train_cams:list[UnifiedCamera]):

        def get_diag(cam_centers):
            try:
                cam_centers = torch.stack(cam_centers,dim=0) #np.hstack(cam_centers)
            except:
                cam_centers = torch.stack(cam_centers,dim=0) #np.hstack(cam_centers)

            avg_cam_center = torch.mean(cam_centers,dim=0, keepdims=True)#np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = torch.linalg.norm(cam_centers - center, dim=1, keepdims=True)
            max_dist = torch.max(dist)
            return  max_dist
        cam_centers = []
        for cam in train_cams:
            cam_centers.append(cam.camera_center)

        max_dist = get_diag(cam_centers)
        spatial_lr_scale = max_dist * 1.1
        self.spatial_lr_scale = spatial_lr_scale.clone().cpu()
        self.cameras_extent = spatial_lr_scale.clone()

    def update_optimization_option(self, iteration):
        if iteration % 1000 == 0:
            self.oneupSHdegree()
        self.update_learning_rate(iteration)

    @torch.no_grad()
    def optimizer_step(self, train_render_pkg):
        #import pdb;pdb.set_trace()
        if self.optimizer_type =='sparse_adam':
            radii = train_render_pkg['radii']
            visible = radii > 0
            self.optimizer.step(visible, radii.shape[0])

        elif self.optimizer_type == 'default':
            self.optimizer.step()

    @torch.no_grad()
    def adaptive_density_control(self, 
                                train_render_pkg, 
                                iteration,
                                training_args):
        if iteration < training_args['densify_until_iter']:
            self.update_densification_stats(train_render_pkg)
            if iteration > training_args['densify_from_iter'] and iteration % training_args['densification_interval'] == 0:

                size_threshold = 20 if iteration >  training_args['opacity_reset_interval'] else None
                self.densify_and_prune( training_args['densify_grad_threshold'], 0.005, self.cameras_extent, size_threshold, train_render_pkg['radii'])
    
                if iteration % training_args['opacity_reset_interval'] == 0 or (training_args['white_background'] and iteration == training_args['densify_from_iter']): #or ( and iteration == training_args['densify_grad_threshold'] opt.densify_from_iter):
                    self.reset_opacity()

            #validation step            

    def get_camera_image(self,
                         current_train_camera:Optional[list[UnifiedCamera]],
                         current_train_image:Optional[list[torch.Tensor]],
                         current_train_masks:Optional[list[torch.Tensor]],
                         current_indices:Optional[list[int]],
                         train_cams,
                         train_images,
                         train_masks
                         ):
        if len(current_indices) == 0:
            current_train_camera = train_cams.copy()
            current_train_image = train_images.copy()
            current_train_masks = train_masks.copy()

            current_indices = list(range(len(train_cams)))
            shuffle(current_indices)
        rand_idx = current_indices.pop()
        train_cam = current_train_camera[rand_idx]
        train_image = current_train_image[rand_idx]
        train_mask = current_train_masks[rand_idx]

        return train_cam, train_image, train_mask
            
    
    def training_step(self,
                train_cam,
                train_image,
                train_mask,
                render_fn,
                training_args
                ):
        from utils.graphics.loss_utils import l1_loss, ssim
        try:
            from fused_ssim import fused_ssim
            FUSED_SSIM_AVAILABLE = True
        except:
            FUSED_SSIM_AVAILABLE = False
        bg = torch.rand(3,1,1).float().cuda()

        render_pkg = render_fn(self, train_cam, bg_color= bg)
        image= render_pkg["render"]
        #cv2.imwrite('test.png',image.permute(1,2,0).detach().cpu().numpy()*255)
        #cv2.imwrite('train_image.png',train_image.permute(1,2,0).detach().cpu().numpy()*255)

        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        l1_wo_mask = l1_loss(image,train_image)
        train_mask = train_mask *1.0
        #image = image*train_mask + (1-train_mask) *bg
        train_image = train_image*train_mask + (1-train_mask) *bg
        Ll1 = l1_loss(image, train_image)  + l1_wo_mask
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), train_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, train_image)

        loss = (1.0 - training_args['lambda_dssim']) * Ll1 \
            + training_args['lambda_dssim'] * (1.0 - ssim_value)
        #import pdb;pdb.set_trace()

        return render_pkg, loss

    @torch.no_grad()
    def validation_step(self,
                idx,
                valid_cam,
                valid_image,
                render_fn,
                save_folder
                ):

        render_pkg = render_fn(self,valid_cam)
        render_image = render_pkg['render']
        if valid_image != None:
            psnr_val =  psnr(render_image, valid_image)
        else:
            psnr_val = -1
        if save_folder !=None:
            render_image_numpy = torch2cv2(render_image)
            image_name =  os.path.join(save_folder,str(idx).zfill(4)+'.png')
            cv2.imwrite(image_name,render_image_numpy)
        return psnr_val
        
        #Naive version: test results are saved in model_path folder

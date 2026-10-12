# Avatar optimizer: SMPL fitting (preprocessing) + avatar training
import os
import copy
import torch
import numpy as np
import cv2
from torch import nn
from torch.nn.functional import l1_loss
from tqdm import tqdm
from typing import List
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix
from utils.geometry.camera_utils import projection, projection_nvdiffrast
from utils.geometry.mesh_utils import face_to_vertex_color_batch
from scene.object.regularization.sdf_volume import GridSDFGrad
from scene.object.mesh import Mesh
from scene.optimizer.loss_functions import *
from scene.optimizer.opt_utils import *


class AvatarOptimizer(nn.Module):
    def __init__(self,
                 # mesh: Union[Mesh | SMPLMesh],
                 mesh_renderer,
                 gs_renderer = None,
                 opt_params = None,
                 device=torch.device('cuda:0')):
        super(AvatarOptimizer, self).__init__()

        self.device = device
        self.mesh_renderer = mesh_renderer
        self.gs_renderer = gs_renderer

    def list2batch(self, params):
        # works for batch size of 1 (parametric models)
        batch_params = {}
        for key in params[0].keys():
            batch_params[key] = [e[key].detach() for e in params]
            batch_params[key] = torch.concatenate(batch_params[key], dim=0)

        return batch_params

    def fitting_smplx(self, taget_keypoints, smpl_mesh, smpl_params, camera, opt_options, weight, kpt_idx,
                           keys: List,
                           max_iter=500,loss_options={}, decay=0.95, images=None): #by CY

        #if isinstance(camera,list):
       #     import copy
        #    camera = 
        #    camera_stack = copy.deepcopy(camera[0])
        ##    camera_stack.K = torch.stack([cam.K for cam in camera],dim=0).to(self.device)
         #   camera_stack.R = torch.stack([cam.R for cam in camera],dim=0).to(self.device)
         #   camera_stack.t = torch.stack([cam.t for cam in camera],dim=0).to(self.device)
         #   camera = camera_stack

        body_idx = [int(i) for i in range(0,25)]
        body_idx = body_idx + [66,67]
        B = smpl_params['scale'].shape[0]
        smpl_params['betas'] = smpl_params['betas'].mean(dim=0,keepdim=True)
        #smpl_params['scale'] = smpl_params['scale'].mean(dim=0,keepdim=True)

        group_params = []
        for key in opt_options.keys():
            group_params.append(set_opt_group(smpl_params[key], opt_options[key]))

        # default lr is n
        optimizer, scheduler = set_adam_optimizer(group_params, lr=1e-02, decay=decay)
        scheduler_interval = 100

        #mask = self.get_joint_mask(smpl_params)
        import copy
        if 'init_body' in loss_options.keys():
            init_body_pose = copy.deepcopy(smpl_params['body_pose']).detach()
        if 'init_jaw' in loss_options.keys():
            init_jaw = copy.deepcopy(smpl_params['jaw_pose']).detach()
        if 'init_head' in loss_options.keys():
            init_head = copy.deepcopy(init_body_pose[:,15*3:15*3+3]).detach()
        if 'init_neck' in loss_options.keys():
            init_neck = copy.deepcopy(init_body_pose[:,12*3:12*3+3]).detach()

        l2_loss = nn.MSELoss()

        # body_pose_ori = smpl_params['body_pose'].detach().clone()
        pbar = tqdm(range(max_iter))
        #taget_keypoints[:,:,0:1] = taget_keypoints[:,:,0:1]#/ camera.image_width 
        #taget_keypoints[:,:,1:2] = taget_keypoints[:,:,1:2]#/ camera.image_height 
        taget_keypoints_norm = copy.deepcopy(taget_keypoints)
        height, width = camera[0].image_size
        taget_keypoints_norm[...,0] = taget_keypoints_norm[...,0]/width 
        taget_keypoints_norm[...,1] = taget_keypoints_norm[...,1]/height 

        for k in pbar:
            loss = 0.0

            #if isinstance(smpl_mesh, EzMocapMesh):
            #    smpl_params['poses'] = smpl_params['poses']# * mask
            #elif isinstance(smpl_mesh, SMPLMesh):
            #    smpl_params['body_pose'] = smpl_params['body_pose']# * mask
            smpl_output = smpl_mesh.predict(smpl_params)
            est_joint = smpl_output.joints #3d joint
            #proj_est_joint = projection(est_joint[:,:144], camera.K.expand(B, -1, -1),
            #                            camera.R.expand(B, -1, -1),
            #                            -camera.t.expand(B, -1).unsqueeze(-1))
            proj_est_joint = projection_nvdiffrast(est_joint[:, :144],
                                                   cameras =camera )
            #proj_est_joint[:,:,0:1] = proj_est_joint[:,:,0:1]#/ camera.image_width 
            #proj_est_joint[:,:,1:2] = proj_est_joint[:,:,1:2]#/ camera.image_height 
            conf = taget_keypoints_norm[:, :, 2:]
            tmp_kpt_est_joint = proj_est_joint[:,kpt_idx,:]
            #from utils.io.keypoints import visualize_2d_points_on_numpy_images

            #visualize_2d_points_on_numpy_images(taget_keypoints[...,:2],tmp_kpt_est_joint.detach(),images=images,output_dir = 'vis_proj_result')

            if 'keypoint' in loss_options.keys():
                tmp_kpt_est_joint[...,0] = tmp_kpt_est_joint[...,0]/width
                tmp_kpt_est_joint[...,1] = tmp_kpt_est_joint[...,1]/height 
                if weight['body']!=0:
                    body_part_loss = l2_loss(tmp_kpt_est_joint[:,body_idx,:]*conf[:,body_idx], taget_keypoints_norm[:, body_idx, :2]*conf[:,body_idx]) *weight['body']
                    #body_part_loss = torch.sum(torch.pow(tmp_kpt_est_joint[:,body_idx,:]*conf[:,body_idx]- taget_keypoints_norm[:, body_idx, :2]*conf[:,body_idx],1/2),dim=-1)
                    #body_part_loss = torch.sum(body_part_loss,dim=-1)/(torch.sum(body_part_loss>0.0 ,dim=-1)+1e-8)
                    #body_part_loss = torch.mean(body_part_loss)

                    #l2_loss(tmp_kpt_est_joint[:,body_idx,:]*conf[:,body_idx], taget_keypoints_norm[:, body_idx, :2]*conf[:,body_idx])  
                else:
                    body_part_loss = 0.0
                #whole body fine tuning
                if weight['hand']!=0:
                    hand_part_loss = l2_loss(tmp_kpt_est_joint[:,25:65,:]*conf[:,25:65], taget_keypoints_norm[:, 25:65, :2]*conf[:,25:65]) *weight['hand']

                   # hand_part_loss = torch.sum(torch.pow(tmp_kpt_est_joint[:,25:65,:]*conf[:,25:65]- taget_keypoints_norm[:, 25:65, :2]*conf[:,25:65],1/2),dim=-1)
                    #hand_part_loss[hand_part_loss>0.003] = 0.0
                   # hand_part_loss = torch.sum(hand_part_loss,dim=-1)/(torch.sum(hand_part_loss>0.0 ,dim=-1)+1e-8)
                   # hand_part_loss = torch.mean(hand_part_loss)
                else:
                    hand_part_loss = 0.0

                
                #face_part_loss = l2_loss(tmp_kpt_est_joint[:,67:,:]*conf[:,67:], taget_keypoints[:, 67:, :2]*conf[:,67:])  
                if weight['face']!=0:
                    face_part_loss = l2_loss(tmp_kpt_est_joint[:,67:,:]*conf[:,67:], taget_keypoints_norm[:, 67:, :2]*conf[:,67:]) *weight['face']

                    #face_part_loss = torch.sum(torch.pow(tmp_kpt_est_joint[:,67:,:]*conf[:,67:]- taget_keypoints_norm[:, 67:, :2]*conf[:,67:],1/2),dim=-1)
                    #face_part_loss[face_part_loss>0.003] = 0.0
                    #face_part_loss = torch.sum(face_part_loss,dim=-1)/(torch.sum(face_part_loss>0.0 ,dim=-1)+1e-8)
                    #face_part_loss = torch.mean(face_part_loss)
                else:
                    face_part_loss = 0.0
                #jaw loss
                #not forwarding loss : 


                loss +=  loss_options['keypoint']*(body_part_loss*weight['body'] + hand_part_loss*weight['hand'] + face_part_loss*weight['face'])
            if 'init_body' in loss_options.keys():
                loss += loss_options['init_body']*l1_loss(smpl_params['body_pose'],init_body_pose)
            if 'init_foot' in loss_options.keys():

                foot_idx = [i for i in range(27,33)] # foot 9 and 10  on body pose (wo hand pose and pelvis).
                foot_idx = foot_idx + [i for i in range(18,24)] #ankle: 6,7 on body pose (wo hand pose and pelvis)
                loss += loss_options['init_foot']*l1_loss(smpl_params['body_pose'][:,foot_idx],init_body_pose[:,foot_idx])
            if 'init_jaw' in loss_options.keys():       
                loss += loss_options['init_jaw']*l1_loss(smpl_params['jaw_pose'],init_jaw)

            if 'init_head' in loss_options.keys():       
                loss += loss_options['init_head']*l1_loss(smpl_params['body_pose'][:,15*3:15*3+3],init_head)
            if 'init_neck' in loss_options.keys():       
                loss += loss_options['init_neck']*l1_loss(smpl_params['body_pose'][:,12*3:12*3+3],init_neck)

            if 'global_smooth' in loss_options.keys():
                #check 3? or 3*3... should be rotation matrix? or 6D.. - to do list.
                global_rotation_mat = axis_angle_to_matrix(smpl_params['global_orient'])
                loss += loss_options['global_smooth']*(l2_loss(global_rotation_mat[1:-1], global_rotation_mat[:-2]) +
                                                        l2_loss(global_rotation_mat[1:-1], global_rotation_mat[2:]))
                loss += loss_options['global_smooth']*(l2_loss(smpl_params['transl'][1:-1], smpl_params['transl'][:-2]) +
                                                        l2_loss(smpl_params['transl'][1:-1], smpl_params['transl'][2:]))

            #l2_loss(proj_est_joint[:,kpt_idx,:]*conf, taget_keypoints[:, :, :2]*conf) 
            #check invisible joint positions - add...?


            #loss = chamfer_distance_pytorch3d(smpl_output.vertices, mesh.xyz.unsqueeze(0))
            #loss += torch.mean(smpl_params['poses'][:, 36:39] ** 2) * 0.1  # neck constraints

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            pbar.set_description(f"loss: {loss:.8f}")
            if (k % scheduler_interval) == 0:
                scheduler.step()

        for key in keys:
            smpl_params[key] = smpl_params[key].detach()
        return smpl_params

    def toggle_gradients(self, params, grad=False):
        for key in params.keys():
            if isinstance(params[key], torch.Tensor):
                if grad:
                    params[key].requires_grad_(grad)
                else:
                    params[key].detach().requires_grad_(grad)

        return params

    def batch2list(self, batch_params):
        params_new = []
        if 'poses' in batch_params:
            batch = batch_params['poses'].shape[0]
        elif 'body_pose' in batch_params:
            batch = batch_params['body_pose'].shape[0]
        elif 'full_pose' in batch_params:
            batch = batch_params['full_pose'].shape[0]
        else:
            assert "invalid template model parameters."

        for k in range(batch):
            params = {}
            for key in batch_params.keys():
                if batch_params[key].shape[0] == batch:
                    params[key] = batch_params[key][k:k+1].detach().clone()
                elif batch_params[key].shape[0] == 1:  # merged betas
                    params[key] = batch_params[key].detach().clone()
            params_new.append(params)
        return params_new

    def pose_update_only(self, opt_conf, loss_conf, smpl_mesh, da_verts,  fullpose_dict, pose_conf, batch_size=16,test_idx=0,  use_weight=False,  use_6d_rot=True,
                                              cameras=None, normals=None, masks=None, max_iters=100, debug_folder=None, taget_keypoints=None,  kpt_idx=None, images=None, downsampling=1.0):
        full_pose_update_idx=None
        if debug_folder is not None:
            os.makedirs(debug_folder ,exist_ok=True)
        import copy
        #edges,face_to_edge = calc_edges(faces) #E,2 F,3

        #edge_length = calc_edge_length(vertices.squeeze(),edges) #E
        if downsampling !=1.0:
            #update 
            print("Downsampling: ",downsampling)
            original_height,original_width,c = images[0].shape

            downsampling_width = original_width//downsampling
            downsampling_height = original_height//downsampling

        if images is not None:
            if downsampling >1.0:
                images = [torch.from_numpy(cv2.resize(k,(downsampling_width,downsampling_height),cv2.INTER_LANCZOS4)).unsqueeze(0)  for k in images]
            else:
                images = [torch.from_numpy(k).unsqueeze(0)  for k in images]

            images = torch.concatenate(images, dim=0)
            if torch.max(images) > 128:
                images = images/255.0
        if normals is not None:
            if downsampling >1.0:
                normals = [torch.from_numpy(resize_normal_map(k,(downsampling_height,downsampling_width))).unsqueeze(0) for k in normals]

            else:
                normals = [torch.from_numpy(k).unsqueeze(0) for k in normals]
            normals = torch.concatenate(normals, dim=0)
        
            if torch.max(normals) > 128:
                normals = normals/255.0


        if masks is not None:
            if downsampling >1.0:
                masks = [1.0*torch.from_numpy(cv2.resize(k.astype(np.float32)/255,(downsampling_width,downsampling_height),cv2.INTER_NEAREST)).unsqueeze(0).unsqueeze(-1) for k in masks]

            else:
                masks = [torch.from_numpy(k).unsqueeze(0).unsqueeze(-1) for k in masks]

            masks = torch.concatenate(masks, dim=0)
            if torch.max(masks) > 128:
                masks = masks/255.0
            for idx, mask in enumerate(masks):
                mask[mask>0.5] = 1.0
                mask[mask<=0.5]=0.0
                masks[idx] = mask


        batch_params = {}


        valid_keys = ['full_pose', 'transl',  'betas'] #, 'betas', 'bias']
        if isinstance(fullpose_dict, list):
            for key in fullpose_dict[0].keys():
                if key in valid_keys:
                    if key =='full_pose':
                        if use_6d_rot:
                            batch_params[key] = [smpl_mesh.full_pose_to_sixrot(e[key].view(fullpose_dict[key].shape[0],-1,3)).detach() for e in fullpose_dict] 
                            batch_params[key] = torch.concatenate(batch_params[key], dim=0)

                        else:
                            batch_params[key] = [e[key].detach() for e in fullpose_dict]
                            batch_params[key] = torch.concatenate(batch_params[key], dim=0)

                    else:
                        batch_params[key] = [e[key].detach() for e in fullpose_dict]
                        batch_params[key] = torch.concatenate(batch_params[key], dim=0)
        else:
            for key in fullpose_dict.keys():
                if key =='full_pose':
                    if use_6d_rot:
                        batch_params[key] = smpl_mesh.full_pose_to_sixrot(fullpose_dict['full_pose'].view(fullpose_dict['full_pose'].shape[0],-1,3))
                    else:
                        batch_params[key] = fullpose_dict[key].detach()
                    batch_params[key] = batch_params[key].view(fullpose_dict['full_pose'].shape[0],-1)

                else:
                    batch_params[key] = fullpose_dict[key].detach()

        if 'init_full_pose' in loss_conf.keys():
            init_full_poses = copy.deepcopy(batch_params['full_pose'].detach())


        copy_cameras = copy.deepcopy(cameras)
        if isinstance(copy_cameras,list):
            #update cameras
            
            camera = copy_cameras[0]

            for idx, camera in enumerate(copy_cameras):
                camera.downsampling_camera(downsampling)
                camera.image_size = int(camera.image_height), int(camera.image_width)
                copy_cameras[idx] = camera


            import copy
            camera_stack = copy.deepcopy(copy_cameras[0])
            camera_stack.K = torch.stack([cam.K for cam in copy_cameras],dim=0).to(self.device)
            camera_stack.R = torch.stack([cam.R for cam in cameras],dim=0).to(self.device)
            camera_stack.t = torch.stack([cam.t for cam in copy_cameras],dim=0).to(self.device)
            camera = camera_stack
        else:
            camera = copy_cameras[0]
        da_verts= da_verts.squeeze().detach()
        
        #nrm_opt = MeshOptimizer(da_verts.detach(), ez_mesh.face.detach(), lr=lr, edge_len_lims=[0.005, 0.001]) #[0.001, 0.01]
        #vertices, faces = nrm_opt.vertices, nrm_opt.faces

        #batchwise rendering
        from scene.optimizer.batchify import ClusteredBalancedSampler,RandomSampler
        seed=20250903
        if batch_size <6:
            sampler = RandomSampler(num_samples=len(masks),batch_size=batch_size,seed=seed)
        else:
            if use_6d_rot:
                global_orient = smpl_mesh.sixrot_to_full_pose(batch_params['full_pose'][...,:6]).detach().cpu().numpy()
            else:
                global_orient=batch_params['full_pose'][...,:3].detach().cpu().numpy()
            sampler = ClusteredBalancedSampler.initialize(global_orientations=global_orient,
                                        num_groups=6,batch_size=batch_size,
                                            seed=seed)


        #lpips_loss = LPIPS(img_size=camera.image_size)
        pbar = tqdm(range(max_iters))
        sampler_iterator = iter(sampler)
        epoch_count=0
        if len(masks)%batch_size==0:
            num_iter = len(masks)//batch_size
        else:
            num_iter = len(masks)//batch_size +1

        group_params = []
        with torch.no_grad():
            if batch_params['betas'].shape[0] != 1:
                batch_params['betas'] = batch_params['betas'].mean(0,keepdim=True)
        batch_params = self.toggle_gradients(batch_params, grad=True)
        
        for i, key in enumerate(opt_conf.keys()):
            if key in batch_params:
                group_params.append(set_opt_group(batch_params[key], opt_conf[key]))

        # default lr is n
        decay = 0.95
        optimizer, scheduler = set_sgd_optimizer(group_params, lr=1e-3, decay=decay,momentum=0)
        scheduler_interval = 100//num_iter

        smpl_mesh.T_da_inv = None
        #sampling weights
        if use_weight:
            with torch.no_grad():
                from pytorch3d.transforms import matrix_to_rotation_6d,  axis_angle_to_matrix
                import torch.nn.functional as F
                #getting full da pose.
                da_pose = smpl_mesh.get_da_fullpose()

                #da_pose = ez_mesh.get_dapose_param(betas=ez_params['betas'])['poses']
                #convert26d.
                #current poses.
                sampling_pose_idxs = [0,1,2,3,4,5,8,12,13,15,16,17,18]
                if len(batch_params['full_pose'].shape)==2:
                    if use_6d_rot:

                        batch_poses = batch_params['full_pose'].detach().view(batch_params['full_pose'].shape[0],-1,6)
                    else:
                        batch_poses = batch_params['full_pose'].detach().view(batch_params['full_pose'].shape[0],-1,3)
                        batch_poses = axis_angle_to_matrix(batch_poses)
                        batch_poses = matrix_to_rotation_6d(batch_poses)

                if len(da_pose.shape)==2:
                    da_pose = da_pose.view(da_pose.shape[0],-1,3)
                da_pose = axis_angle_to_matrix(da_pose)
                da_pose = matrix_to_rotation_6d(da_pose)

                #measure similarity between poses
                distance_pose = torch.pow(batch_poses - da_pose,2)[:,sampling_pose_idxs,:]
                distance_pose = distance_pose.sum(dim=-1)
                distance_pose = distance_pose.view(distance_pose.shape[0],-1)
                feature_dim = distance_pose.shape[-1]
                scale_factor = (feature_dim ** 0.5) * 1/3
                distance_pose = -distance_pose/scale_factor#torch.exp(-torch.sum(distance_pose,dim=-1))
                distance_pose = distance_pose.sum(dim=-1)
                loss_weight = F.softmax(distance_pose, dim=0).detach()
                #cv2.imwrite('argmin.png',images[loss_weight.argmin()].detach().cpu().numpy()*255)
                #cv2.imwrite('argmax.png',images[loss_weight.argmax()].detach().cpu().numpy()*255)
                loss_weight = (loss_weight/torch.max(loss_weight)).detach()
                #other poses.weights = F.softmax(logits, dim=-1)

            #da-pose


        taget_keypoints_norm = copy.deepcopy(taget_keypoints)
        height, width = camera.image_size
        taget_keypoints_norm[...,0] = taget_keypoints_norm[...,0]/width 
        taget_keypoints_norm[...,1] = taget_keypoints_norm[...,1]/height 
        l2_loss = nn.MSELoss()
        body_idx = [int(i) for i in range(0,25)]
        body_idx = body_idx + [66,67]

        hand_idxs = None
        for k in pbar:
            #update SDF volume?

            for i in range(num_iter): #tqdm(range(iters)):
                optimizer.zero_grad()

                try:
                    batch_idxs = next(sampler_iterator)
                except StopIteration:
                    # StopIteration raised: all batches of the sampler have been consumed
                    # print(f"\n--- All batches consumed, Epoch {epoch_count} finished. Starting next epoch ---\n")
                    epoch_count += 1
                    
                    # 1. Reshuffle the data order for the new epoch
                    sampler.reshuffle(seed=epoch_count+seed)
                    
                    # 2. Create a new iterator to restart from the beginning
                    sampler_iterator = iter(sampler)
                    
                    # 3. Fetch the first batch from the new iterator
                    batch_idxs = next(sampler_iterator)

                batch_cameras = [copy_cameras[i] for i in batch_idxs]                #side_index = side_idxs - start_idx
                #if isinstance(batch_cameras,list):
                B = len(batch_idxs)
                import copy
                #camera_stack = copy.deepcopy(cameras[0])
                tmp_camera_K = camera_stack.K[batch_idxs].to(self.device)
                tmp_camera_R = camera_stack.R[batch_idxs].to(self.device)
                tmp_camera_t = camera_stack.t[batch_idxs].to(self.device)
                if use_weight:
                    tmp_loss_weihgt = loss_weight[batch_idxs]
                else:
                    tmp_loss_weihgt = torch.ones(B).to(self.device)


                #keypoints loss.


                if normals!=None:
                    tmp_normals = normals[batch_idxs].to(self.device)

                tmp_masks = masks[batch_idxs].float().to(self.device)
                tmp_batch_params = {}
                for key in batch_params.keys():
                    if key !='betas':
                        tmp_batch_params[key] = batch_params[key][batch_idxs]
                tmp_batch_params['betas'] = batch_params['betas'] #shared in all images.
                
                
                if use_6d_rot:
                    tmp_batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(tmp_batch_params['full_pose']).view(B,-1)
                posed_verts= smpl_mesh.FS_fullposedict(full_pose_dict=tmp_batch_params,v_canon=da_verts[None],return_J=False) #, T_da_inv=T_da_inv)

                tmp_mesh = Mesh(vertices=posed_verts,indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)

                tmp_smplx_poses = smpl_mesh.full_pose_dict2smplx(tmp_batch_params)
                smpl_output = smpl_mesh.predict(tmp_smplx_poses)
                est_joint = smpl_output.joints

                #norm_v =  2.0 * (v[None] - SDF_volume.center) / SDF_volume.span
                #DA_mesh = Mesh(vertices=da_verts[None],indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)

                #forward skinning ezmocap
                #smpl_output = smpl_mesh.predict(tmp_batch_params)
                #est_joint = smpl_output.joints #3d joint
                #proj_est_joint = projection(est_joint[:,:144], camera.K.expand(B, -1, -1),
                #                            camera.R.expand(B, -1, -1),
                #                            -camera.t.expand(B, -1).unsqueeze(-1))
                proj_est_joint = projection_nvdiffrast(est_joint[:, :144],
                                                    cameras =batch_cameras )

                tmp_target_kpts = taget_keypoints_norm[batch_idxs]
                conf = tmp_target_kpts[:, :, 2:]
                conf[conf<0.90] = 0.0
                tmp_kpt_est_joint = proj_est_joint[:,kpt_idx,:]


                rendered = self.mesh_renderer.render_SH_batch_w_depth(tmp_mesh, batch_cameras,w_albedo=False) 
                #render_normals = rendered['normal']
                # if i < 800:
                
                #align normal 
                #addign laplacian loss.
                alpha = rendered['mask']

                masks_overlap = tmp_masks *alpha
                masks_overlap = masks_overlap.detach()
                #masks_overlap = masks_overlap.view(B,-1,1)


                loss = 0.0

                if 'keypoint' in loss_conf.keys():
                    tmp_kpt_est_joint[...,0] = tmp_kpt_est_joint[...,0]/width
                    tmp_kpt_est_joint[...,1] = tmp_kpt_est_joint[...,1]/height 
                    #tmp_kpt_est_joint[...,0] = 1- tmp_kpt_est_joint[...,0]
                    body_part_loss = l1_loss(tmp_kpt_est_joint[:,body_idx,:]*conf[:,body_idx], tmp_target_kpts[:, body_idx, :2]*conf[:,body_idx]) # *weight['body']
                    hand_part_loss = l1_loss(tmp_kpt_est_joint[:,25:65,:]*conf[:,25:65], tmp_target_kpts[:, 25:65, :2]*conf[:,25:65]) # *weight['hand']
                    face_part_loss = l1_loss(tmp_kpt_est_joint[:,67:,:]*conf[:,67:], tmp_target_kpts[:, 67:, :2]*conf[:,67:]) #*weight['face']

                    kpt_loss = body_part_loss +  hand_part_loss + face_part_loss
                    loss += kpt_loss*loss_conf['keypoint']


                if "normal_all_cosine" in loss_conf.keys():
                    render_normal_vec = rendered['normal']*2.0 -1.0
                    
                    render_normal_vec = render_normal_vec/(torch.norm(render_normal_vec,dim=-1)[...,None]+1e-8)

                    render_normal_vec = render_normal_vec  * masks_overlap
                    #cv2.imwrite('test_normal.png',127.5*(render_normal_vec[0].detach().cpu().numpy()+1.0))
                    tmp_normals = tmp_normals * masks_overlap
                    normal_loss = masks_overlap - masks_overlap*torch.sum(render_normal_vec*tmp_normals,dim=-1,keepdim=True)
                    normal_loss = torch.sum(normal_loss,dim=(1,2,3))/torch.sum(masks_overlap,dim=(1,2,3))
                    loss += torch.sum(tmp_loss_weihgt*loss_conf['normal_all_cosine']*normal_loss)/torch.sum(tmp_loss_weihgt) #(torch.sum(masks_overlap - masks_overlap*(torch.sum(render_normal_vec*tmp_normals,dim=-1)[...,None]))/torch.sum(masks_overlap,dim=(1,2,3))))

                    

                if "masks" in loss_conf.keys():
                    mask_nonoverlap = torch.abs(alpha- tmp_masks)
                    #mask_nonoverlap = mask_nonoverlap.view(B,-1,3)
                    mask_loss = torch.sum(mask_nonoverlap,dim=(1,2,3))/torch.sum(tmp_masks,dim=(1,2,3))

                    loss += torch.sum(tmp_loss_weihgt*loss_conf['masks']*mask_loss)/torch.sum(tmp_loss_weihgt) #torch.sum(tmp_loss_weihgt*loss_conf['masks']* torch.sum(mask_nonoverlap,dim=-1)/ torch.sum(tmp_masks,dim=-1))/torch.sum(tmp_loss_weihgt)

                if 'init_full_pose' in loss_conf.keys():
                    if use_6d_rot:
                        loss += loss_conf['init_full_pose'] *l1_loss( smpl_mesh.full_pose_to_sixrot(tmp_batch_params['full_pose']),init_full_poses[batch_idxs])
                    else:
                        loss += loss_conf['init_full_pose'] *l1_loss( tmp_batch_params['full_pose'],init_full_poses[batch_idxs])


                loss.backward()


                if 'full_pose' in opt_conf.keys():
                    with torch.no_grad():
                        batch_idxs_tensor = torch.tensor(batch_idxs, device=self.device, dtype=torch.long)
                        if full_pose_update_idx is None:
                            full_pose_update_idx = []
                            if use_6d_rot:
                                last_dim =6
                            else:
                                last_dim =3

                            if 'hand' in pose_conf and pose_conf['hand']>0:
                                # Joint 30~54 -> Parameter (30*3) ~ (55*3)
                                # corresponds to indices 90 ~ 164
                                hand_params = np.arange(25 * last_dim, 55 * last_dim)
                                full_pose_update_idx.append(hand_params)

                            if 'body' in pose_conf and pose_conf['body']>0:
                                # Joint 1~21 -> Parameter (1*3) ~ (22*3)
                                # corresponds to indices 3 ~ 65
                                body_params = np.arange(1 * last_dim, 22 * last_dim)

                                full_pose_update_idx.append(body_params)

                            if 'eye' in pose_conf and pose_conf['eye']>0:
                                body_params = np.arange(23 * last_dim, 25 * last_dim)
                                full_pose_update_idx.append(body_params)

                            if 'jaw' in pose_conf and pose_conf['jaw']>0:
                                body_params = np.arange(22 * last_dim, 23 * last_dim)

                                full_pose_update_idx.append(body_params)

                            elif 'essential_body' in pose_conf and pose_conf['essential_body'] :
                                body_idxs = np.arange(1, 22)
                                except_idxs = [7, 8, 10, 11, 12, 15, 20, 21]
                                
                                # Get the remaining Joint indices after removing excluded ones
                                valid_mask = ~np.isin(body_idxs, except_idxs)
                                essential_body_idxs = body_idxs[valid_mask] 
                                
                                # Expand Joint indices by 3x (3 axes per Joint)
                                # e.g. Joint [1, 2] -> [3, 4, 5, 6, 7, 8]
                                essential_params = []
                                for idx in essential_body_idxs:
                                    essential_params.extend([idx*last_dim, idx*last_dim+1, idx*last_dim+2])
                                essential_params = np.array(essential_params)
                                full_pose_update_idx.append(essential_params)

                            if 'global_orient' in pose_conf and pose_conf['global_orient'] :
                                # Joint 0 -> Parameter 0, 1, 2
                                if use_6d_rot:
                                    global_params = np.array([0, 1, 2,3,4,5])

                                else:
                                    global_params = np.array([0, 1, 2])
                                full_pose_update_idx.append(global_params)

                            if len(full_pose_update_idx) > 0:
                                full_pose_update_idx = np.concatenate(full_pose_update_idx)
                                full_pose_update_idx = np.unique(full_pose_update_idx)
                            else:
                                full_pose_update_idx = np.array([], dtype=int)

                            # 2. Build the full index range (0 ~ 164)
                            total_params = 55*last_dim # 55 joints * 3
                            all_idxs = np.arange(total_params)

                            # 3. Get the non-updated (Frozen) indices
                            # Subtract the update indices from the full index set (set difference)
                            full_pose_frozen_idx = np.setdiff1d(all_idxs, full_pose_update_idx)
                    
                        if 'hand' in pose_conf and pose_conf['hand']>0:
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],hand_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],hand_params] * pose_conf['hand']  # 

                        if 'body' in pose_conf and pose_conf['body']>0:
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],body_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],body_params] * pose_conf['body']  #

                        elif 'essential_body' in pose_conf and pose_conf['essential_body']>0 :
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],essential_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],essential_params] * pose_conf['essential_body']  # 
                        if 'global_orient' in pose_conf and pose_conf['global_orient']>0:
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],global_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],global_params] * pose_conf['global_orient']  # 
                        batch_params['full_pose'].grad[batch_idxs_tensor[:,None],full_pose_frozen_idx] =  batch_params['full_pose'].grad[batch_idxs_tensor[:,None],full_pose_frozen_idx] *0.0


                if len(group_params) !=0:
                    optimizer.step()
                    pbar.set_description(f"Loss: {loss.item():.8f}")
                    if (k % scheduler_interval) == 0:
                        scheduler.step()



                with torch.no_grad():
                    _preproc_save_every = int(os.environ.get('PREPROC_SAVE_EVERY', 200))
                    if debug_folder is not None and k%_preproc_save_every==0 and i==0 or (k==(max_iters-1) and i==0):

                        os.makedirs(f'{debug_folder}/overlay_normal', exist_ok=True)
                        os.makedirs(f'{debug_folder}/normals', exist_ok=True)
                        os.makedirs(f'{debug_folder}/smpl', exist_ok=True)
                        debug_batch_params = {}
                        for key in batch_params.keys():
                            debug_batch_params[key] = batch_params[key].detach()

                        if use_6d_rot:
                            debug_batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(debug_batch_params['full_pose'])
                        
                        batch_param_list= self.batch2list(debug_batch_params)

                        for render_idx in range(len(masks)):

                            batch_param_test = batch_param_list[render_idx]
                            posed_verts_save = smpl_mesh.FS_fullposedict(batch_param_test,v_canon=da_verts[None]) #, T_da_inv=T_da_inv)
                            save_mesh = Mesh(vertices=posed_verts_save.squeeze(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)
                            rendered_save = self.mesh_renderer.render_SH_batch_w_depth(save_mesh, [copy_cameras[render_idx]], w_albedo=True) 

                            os.makedirs(f'{debug_folder}/normals/{str(k).zfill(4)}', exist_ok=True)
                            cv2.imwrite(f'{debug_folder}/normals/{str(k).zfill(4)}/{render_idx:04d}.png',(rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
                            #save da-verts and posed verts.
                            if images is not None:
                                os.makedirs(f'{debug_folder}/overlay_normal/{str(k).zfill(4)}', exist_ok=True)
                                cv2.imwrite(f'{debug_folder}/overlay_normal/{str(k).zfill(4)}/{render_idx:04d}.png',((rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy()/2 + images[render_idx].cpu().numpy()*127.5))

                        os.makedirs(f'{debug_folder}/posed_mesh', exist_ok=True)

                        os.makedirs(f'{debug_folder}/da_mesh', exist_ok=True)

                        batch_param_test = batch_param_list[test_idx]
                        posed_verts_save = smpl_mesh.FS_fullposedict(batch_param_test,v_canon=da_verts[None]) #, T_da_inv=T_da_inv)
                        save_mesh = Mesh(vertices=posed_verts_save.squeeze(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)
                        save_mesh.to_trimesh().export(f'{debug_folder}/posed_mesh/{str(k).zfill(5)}_posed.ply')
                        Mesh(vertices=da_verts.detach(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device).to_trimesh().export(f'{debug_folder}/da_mesh/{str(k).zfill(5)}_daverts.ply')
                        from utils.geometry.avatar_utils import save_render_smplmesh_params

                        save_render_smplmesh_params( smpl_mesh, self.mesh_renderer, camera=copy_cameras,
                                                                        save_root=os.path.join(debug_folder,'smpl',str(k).zfill(5)),
                                                                        smpl_param_batch= smpl_mesh.full_pose_dict2smplx(debug_batch_params),
                                                                        images=images)
                        #overlay version.

                        del debug_batch_params
                    #os.makedirs(f'{debug_folder}/rgba', exist_ok=True)
                    #cv2.imwrite(f'{debug_folder}/rgba/{k:04d}.png',(rendered['rgba'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())

                torch.cuda.empty_cache() 
        os.makedirs(f'{debug_folder}/render_all_normals', exist_ok=True)
        os.makedirs(f'{debug_folder}/overlay_all_normal', exist_ok=True)

        batch_param_list= smpl_mesh.batch2list(batch_params)
        vertices = da_verts.detach()
        faces = smpl_mesh.face
        with torch.no_grad():
            for i in range(len(masks)):
                batch_param_test = copy.deepcopy(batch_param_list[i])
                if use_6d_rot:
                    batch_param_test['full_pose'] = smpl_mesh.sixrot_to_full_pose(batch_param_test['full_pose'])

                posed_verts_save = smpl_mesh.FS_fullposedict(batch_param_test,v_canon=vertices[None])  #smpl_mesh.FS_fullposedict(Namespace(**batch_param_test),da_verts=vertices[None]) #, T_da_inv=T_da_inv)

                #posed_verts_save, joints = smpl_mesh.forward_skinning_da_verts(Namespace(**batch_param_test),da_verts=vertices[None], T_da_inv=T_da_inv)
                save_mesh = Mesh(vertices=posed_verts_save.squeeze(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)

                rendered_save = self.mesh_renderer.render_SH_batch_w_depth(save_mesh, [copy_cameras[i]], w_albedo=True) 


                cv2.imwrite(f'{debug_folder}/render_all_normals/{i:04d}.png',(rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
                cv2.imwrite(f'{debug_folder}/overlay_all_normal/{i:04d}.png',((rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy()/2 + images[i].cpu().numpy()*127.5))


        with torch.no_grad():
            torch.cuda.empty_cache()
            B = len(masks)
            batch_params = self.toggle_gradients(batch_params, grad=False) 
            if use_6d_rot:
                batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(batch_params['full_pose'])

        return  batch_params

    def ECCVavatar_w_split_datasets_w_split_level(self,  FineAvatar, full_pose_dict, optimizer=None, scheduler=None, novel_view_dataset=None, novel_pose_dataset=None, extra_dataset=None, sdf_volume=None,
                                                   opt_flag={}, opt_conf={},all_loss_conf=None, pose_conf={}, downsampling=1,is_evaluation=False, camera_noise=0,
                                        cameras=None, test_step=4, warm_up=None,segments=None, debug_da_camera=None,pose_noise_dict=None, use_6d_rot=False,
                                        keypoints=None, images=None, masks=None, normals=None, depths=None, max_iters=1000,
                                         waft_of=None, vposer_sampling=None, head_imgs= None, head_masks=None, head_bboxs=None,
                                        invalid_sample=None, paired_next_idxs=None, paired_prev_idxs=None, paired_curr_idxs=None, pose_update_only=False,
                                        batch_size=16, debug_folder=None,OF_model=None,waft_features=None,
                                        dino_conf=None):
                training_levels = opt_conf['training_levels']
                opt_pose = 'update_pose' in opt_flag.keys() and opt_flag['update_pose'] 
                full_pose_update_idx =None
                from utils.geometry.avatar_utils import  forward_cascadeavatar_split_use_gs, forward_cascadeavatar_split_use_gs_w_OF, save_render_cascade_general_version_split_gs, benchmarking_EaxAvatar_version

                #propagate coarse avatar to fine avatar.
                if novel_view_dataset is not None:
                    novel_view_data = None
                if novel_pose_dataset is not None:
                    novel_pose_data = None
                if extra_dataset is not None:
                    extra_data = None
                
                import copy
                from scene.optimizer.batchify import TripleClusteredPairedSampler_w_except
                #tmp_ez_mesh = copy.deepcopy(MLP.ez_mesh)
                copy_cameras = copy.deepcopy(cameras)
                if downsampling !=1.0:
                    #update 
                    print("Downsampling: ",downsampling)
                    original_height,original_width,c = images[0].shape

                    downsampling_width = original_width//downsampling
                    downsampling_height = original_height//downsampling
                    img_size = (downsampling_height,downsampling_width,c)
                else:
                    original_height,original_width,c = images[0].shape

                    img_size = (original_height,original_width,c)

                if waft_of is not None:
                    if downsampling >1.0:
                        waft_of = torch.nn.functional.interpolate(
                            waft_of,
                            size=(downsampling_height, downsampling_width),
                            mode='bilinear',
                            align_corners=False 
                        )
                        waft_of = waft_of/downsampling
                    
                
                if waft_features is not None:
                    if downsampling >1.0:
                        B,F,H_feat,W_feat = waft_features.shape
                        #waft_features = waft_features.permute(0, 3, 1, 2)
                        waft_features = torch.nn.functional.interpolate(
                            waft_features.float(),
                            size=(H_feat//downsampling, W_feat//downsampling),
                            mode='bilinear',
                            align_corners=False 
                        )
                        #waft_features = waft_features.permute(0, 2, 3, 1)Cascade_avatar_reuse_2DGS_w_split_datasets


                if keypoints is not None:
                    keypoints = [k.to(self.device).unsqueeze(0) for k in keypoints]
                    keypoints = torch.concatenate(keypoints, dim=0)
                    if downsampling >1.0:
                        keypoints[...,:2] = keypoints[...,:2]/downsampling
                if images is not None:
                    if downsampling >1.0:
                        images = [torch.from_numpy(cv2.resize(k,(downsampling_width,downsampling_height),cv2.INTER_LANCZOS4)).unsqueeze(0)  for k in images]
                        images = torch.concatenate(images, dim=0)

                    else:
                        if not isinstance(images, np.ndarray):
                            images = np.stack(list(images), axis=0)
                        images = torch.from_numpy(images)
                            #images = [torch.from_numpy(k).unsqueeze(0)  for k in images]
                    with torch.no_grad():
                        if torch.max(images) > 128:
                            images = images/255.0


                if head_imgs is not None:
                    if not isinstance(head_imgs, np.ndarray):
                        head_imgs = np.stack(list(head_imgs), axis=0)
                    head_imgs = torch.from_numpy(head_imgs)
                            #images = [torch.from_numpy(k).unsqueeze(0)  for k in images]
                    with torch.no_grad():
                        if torch.max(head_imgs) > 128:
                            head_imgs = head_imgs/255.0
                if head_masks is not None:
                    if not isinstance(head_masks, np.ndarray):
                        head_masks = np.stack(list(head_masks), axis=0)
                    head_masks = torch.from_numpy(head_masks)
                    if torch.max(head_masks) > 128:
                        head_masks = head_masks/255.0
                    for idx, head_mask in enumerate(head_masks):
                        head_mask[head_mask>0.5] = 1.0
                        head_mask[head_mask<=0.5]=0.0
                        head_masks[idx] = head_mask
                if head_bboxs is not None:
                    head_bboxs = torch.tensor(head_bboxs)#.to(self.device)
                    


                if normals is not None:
                    if downsampling >1.0:
                        normals = [torch.from_numpy(resize_normal_map(k,(downsampling_height,downsampling_width))).unsqueeze(0) for k in normals]
                        normals = torch.concatenate(normals, dim=0)

                    else:
                        if isinstance(normals, np.ndarray):
                            normals = torch.from_numpy(normals)
                        else:
                            normals = torch.stack([torch.from_numpy(k) for k in normals], dim=0)

                    if torch.max(normals) > 128:
                        normals = normals/255.0
                if segments is not None:
                    segments = [torch.from_numpy(k.astype(np.float32)).unsqueeze(0) for k in segments]
                    segments = torch.concatenate(segments, dim=0)
                    if torch.max(segments) > 128:
                        segments = segments/255.0
                    if len(segments.shape) == 3:
                        segments = segments.unsqueeze(dim=-1)


                if depths is not None:
                    if downsampling >1.0:

                        depths = [torch.from_numpy(resize_depth_map(k, (downsampling_height,downsampling_width))).unsqueeze(0) for k in depths]
                    else:
                        depths = torch.concatenate(depths, dim=0)

                if masks is not None:
                    if downsampling >1.0:
                        masks = [1.0*torch.from_numpy(cv2.resize(k.astype(np.float32)/255,(downsampling_width,downsampling_height),cv2.INTER_NEAREST)).unsqueeze(0).unsqueeze(-1) for k in masks]

                    else:
                        masks = [torch.from_numpy(k).unsqueeze(0).unsqueeze(-1) for k in masks]

                    masks = torch.concatenate(masks, dim=0)
                    if torch.max(masks) > 128:
                        masks = masks/255.0
                    for idx, mask in enumerate(masks):
                        mask[mask>0.5] = 1.0
                        mask[mask<=0.5]=0.0
                        masks[idx] = mask

                if isinstance(copy_cameras,list):
                    #update cameras
                    
                    camera = copy_cameras[0]

                    for idx, camera in enumerate(copy_cameras):
                        camera.downsampling_camera(downsampling)
                        copy_cameras[idx] = camera


                    import copy
                    camera_stack = copy.deepcopy(copy_cameras[0])
                    camera_stack.K = torch.stack([cam.K for cam in copy_cameras],dim=0).to(self.device)
                    camera_stack.R = torch.stack([cam.R for cam in copy_cameras],dim=0).to(self.device)
                    camera_stack.t = torch.stack([cam.t for cam in copy_cameras],dim=0).to(self.device)
                    camera = camera_stack
                else:
                    camera = copy_cameras[0]

                smpl_mesh = copy.deepcopy(FineAvatar.mesh_avatars[-1])
                batch_params = {}
                valid_keys = ['full_pose', 'transl',  'betas'] #, 'betas', 'bias']
                if isinstance(full_pose_dict, list):
                    for key in full_pose_dict[0].keys():
                        if key in valid_keys:
                            if key =='full_pose':
                                if use_6d_rot:
                                    batch_params[key] = [smpl_mesh.full_pose_to_sixrot(e[key].view(full_pose_dict[key].shape[0],-1,3)).detach() for e in full_pose_dict] 
                                    batch_params[key] = torch.concatenate(batch_params[key], dim=0)

                                else:
                                    batch_params[key] = [e[key].detach() for e in full_pose_dict]
                                    batch_params[key] = torch.concatenate(batch_params[key], dim=0)

                            else:
                                batch_params[key] = [e[key].detach() for e in full_pose_dict]
                                batch_params[key] = torch.concatenate(batch_params[key], dim=0)
                else:
                    for key in full_pose_dict.keys():
                        if key =='full_pose':
                            if use_6d_rot:
                                batch_params[key] = smpl_mesh.full_pose_to_sixrot(full_pose_dict['full_pose'].view(full_pose_dict['full_pose'].shape[0],-1,3))
                            else:
                                batch_params[key] = full_pose_dict[key].detach()
                            batch_params[key] = batch_params[key].view(full_pose_dict['full_pose'].shape[0],-1)

                        else:
                            batch_params[key] = full_pose_dict[key].detach()
                seed=20250903

                #sampler_temp = TripleRandomSampler(num_samples=batch_params['Rh'].shape[0], batch_size=batch_size_pair//3,seed=seed)
                if use_6d_rot:
                    global_orient = batch_params['full_pose'][:,:3]

                else:
                    global_orient = batch_params['full_pose'][:,:3]
    
                """
                if invalid_sample is not None:
                    sampler = TripleClusteredPairedSampler_w_except.initialize(global_orient.detach().cpu().numpy(),
                    num_groups=6,
                    batch_size=batch_size//3,
                    invalid_indices=invalid_sample,
                    custom_curr_indices=paired_curr_idxs,
                    custom_prev_indices =paired_prev_idxs,
                    custom_next_indices =paired_next_idxs,
                    seed=seed )
                else:
                """
                from scene.optimizer.batchify import RandomSampler
                seed=20250903
                # num_valid_centers: triple sampler filters valid centers, so limit num_groups
                # to avoid KMeans k > n_samples (custom_curr has the center count)
                _n_centers = len(paired_curr_idxs) if paired_curr_idxs is not None else len(copy_cameras)
                _num_groups = min(6, len(copy_cameras), max(1, _n_centers))
                sampler = TripleClusteredPairedSampler_w_except.initialize(global_orient.detach().cpu().numpy(),
                num_groups=_num_groups,
                batch_size=batch_size//3,
                invalid_indices=invalid_sample,
                custom_curr_indices=paired_curr_idxs,
                custom_prev_indices =paired_prev_idxs,
                custom_next_indices =paired_next_idxs,
                seed=seed )
                #    print("error. this optimizer for triple sets")                
                #(num_samples=batch_params['Rh'].shape[0], batch_size=batch_size_pair//2,seed=seed)
                # trainable parameters
                pose_group_params = []
                if batch_params['betas'].shape[0] != 1:
                    batch_params['betas'] = batch_params['betas'].mean(0,keepdim=True)
                batch_params = self.toggle_gradients(batch_params, grad=True)
                for i, key in enumerate(opt_conf.keys()):
                    if key in batch_params:
                        pose_group_params.append(set_opt_group(batch_params[key], opt_conf[key]))
                
                if warm_up is None:
                    warm_up =  -1
                # optimizers
                if pose_update_only:
                    pose_optimizer, pose_scheduler = set_adam_optimizer(pose_group_params, lr=1e-3, decay=0.95)
                elif opt_pose:
                    pose_optimizer, pose_scheduler = set_sgd_optimizer(pose_group_params, lr=1e-3, decay=0.8, momentum=0.0)
                if not pose_update_only:
                    freeze_decoders = opt_flag.get('freeze_decoders', False)
                    FineAvatar.set_grad_params_groups(training_levels, freeze_decoders=freeze_decoders)
                else:
                    FineAvatar.set_all_grad_false()

                # loss and training parameters
                l2_loss = nn.MSELoss()
                bce_loss = nn.BCEWithLogitsLoss()
                ssim_loss  =SSIM()
                huber_match_loss = None
                scheduler_interval = 100
                # Exp K v14: optional adaptive LR scheduler fix.
                # Default False preserves the existing (k*num_iter)%100 behavior.
                # When True, steps scheduler every `lr_scheduler_interval` OUTER iters (in k),
                # decoupled from num_iter (fixes silent dataset-size dependence).
                _sch_adaptive = opt_conf.get('lr_scheduler_adaptive', False) if isinstance(opt_conf, dict) else False
                if _sch_adaptive:
                    scheduler_interval_k = int(opt_conf.get('lr_scheduler_interval_k',
                                                            max(1, max_iters // 10)))
                else:
                    scheduler_interval_k = None
                all_keys = set()
                for conf in all_loss_conf.values():
                    all_keys.update(conf.keys())

                # 2. Sort for readability and convert to a list
                all_loss_keys = sorted(list(all_keys))
                pbar = tqdm(range(max_iters))
                if len(masks)%batch_size==0:
                    num_iter = len(masks)//batch_size
                else:
                    num_iter = len(masks)//batch_size +1

                lpips_loss = LPIPS(img_size=camera.image_size, downsample_factor=2)
                head_lpips_loss = LPIPS(img_size=camera.image_size, downsample_factor=1)
                if vposer_sampling is not None:
                    vposer_loss_fn = vposer_loss(vposer_sampling)

                if 'init_full_pose' in all_loss_keys:
                    init_full_poses = copy.deepcopy(batch_params['full_pose'].detach())



                if 'init_body' in all_loss_keys:
                    body_idx = [i for i in range(3,66)]
                    init_body_pose = full_pose_dict['full_pose'][:,body_idx].detach()
                if 'init_foot' in all_loss_keys:
                    foot_idx = [i for i in range(30,36)] # foot 9 and 10  on body pose (wo hand pose and pelvis).
                    foot_idx = foot_idx + [i for i in range(21,27)] #ankle: 6,7 on body pose (wo hand pose and pelvis                
                    init_foot_pose = full_pose_dict['poses'][:,foot_idx].detach()


                if OF_model is not None:
                    waft_loss = WAFTLoss(model=OF_model)
                sampler_iterator = iter(sampler)
                #if use_smoothing_prop:
                #    loss_vals = torch.zeros((batch_params['Rh'].shape[0]),device=self.device)
                epoch_count=0
                #level = opt_flag['level']
                #if CoarseAvatar is not None:
                #    C_consistency_loss = FaceConsistencyLoss(faces=CoarseAvatar.mesh_avatar.face)
                            
                if sdf_volume is not None:
                    P_sdf_volume = copy.deepcopy(sdf_volume)

                if 'acc_reg' in all_loss_keys:
                    mass_weights = []
                    #based on rest verts -> use rest mesh for jacobian system
                    for level in range(FineAvatar.max_level):

                        mass_weights.append(calculate_vertex_areas(FineAvatar.jacobian_systems[level].init_da_verts.squeeze(), FineAvatar.jacobian_systems[level].init_face))

                for k in pbar:
                    #if with_jacobian_update is not None:
                    #    if k !=0 and k%with_jacobian_update==0:
                    #        MLP.update_jacobian_field_from_current_da_verts()

                    #smooth_loss_weight= torch.ones((batch_params['Rh'].shape[0]),device=self.device)
                    for idx in range(num_iter):
                        if 'update_jacobian' in opt_flag.keys():
                            if idx==0 and k!=0 and k % opt_flag['update_jacobian']==0:
                                FineAvatar.update_jacobian_system(
                                    target_pose = full_pose_dict,
                                    batch_size=batch_size)

                                #print('done')
                            if sdf_volume is not None:
                                P_sdf_volume = GridSDFGrad(verts=FineAvatar.P_jacobian_system.init_da_verts.squeeze(),faces=FineAvatar.P_jacobian_system.init_face,resolution=256,padding=0.2)

                        normal_loss=0.0
                        import random
                        bg = [random.random() for _ in range(3)]

                        try:
                            # Try to fetch the next batch from the iterator
                            batch_idxs_pairs = next(sampler_iterator)
                        except StopIteration:
                            # StopIteration raised: all batches of the sampler have been consumed
                        # print(f"\n--- All batches consumed, Epoch {epoch_count} finished. Starting next epoch ---\n")
                            epoch_count += 1
                            # 1. Reshuffle the data order for the new epoch
                            sampler_iterator.reshuffle(seed=epoch_count+seed)

                            # 2. Create a new iterator to restart from the beginning
                            sampler_iterator = iter(sampler)

                            # 3. Fetch the first batch from the new iterator
                            batch_idxs_pairs = next(sampler_iterator)

                        all_batch_idxs = list(batch_idxs_pairs.flatten())
                        B_pairs = len(batch_idxs_pairs)
                        B = len(all_batch_idxs)
                        #index
                    # prev_idxs = B_org + 3*np.arange(B_pairs)
                        if batch_size>1:
                            prev_idxs =  3*np.arange(B_pairs)
                            curr_idxs =  3*np.arange(B_pairs) + 1
                            next_idxs =  3*np.arange(B_pairs) + 2

                        batch_cameras = [copy.deepcopy(copy_cameras[i]) for i in all_batch_idxs]                #side_index = side_idxs - start_idx

                        if camera_noise > 0:
                            for cam in batch_cameras:
                                cam.add_rotation_noise(noise_scale=camera_noise, is_radian=False) 
                        curr_batch_size= len(all_batch_idxs)
                        #if isinstance(batch_cameras,list):
                        import copy
                        #camera_stack = copy.deepcopy(cameras[0])
                        tmp_camera_K = camera_stack.K[all_batch_idxs].to(self.device)
                        tmp_camera_R = camera_stack.R[all_batch_idxs].to(self.device)
                        tmp_camera_t = camera_stack.t[all_batch_idxs].to(self.device)
                        # = camera_stack

                        if waft_features !=None:
                            tmp_waft_feature = waft_features[all_batch_idxs]
                            tmp_waft_feature = tmp_waft_feature.to(self.device)
                        if waft_of !=None:
                            tmp_waft_of = waft_of[all_batch_idxs]
                            tmp_waft_of = tmp_waft_of.to(self.device)

                            #resize to original size?  torch.max(tmp_waft_of.view(tmp_waft_of.shape[0],-1),dim=0)
                        # _,feat_H,feat_W,feat_C = dino_feature.shape
                            #feat_K = tmp_camera_K.expand(B, -1, -1)

                        if normals!=None:
                            tmp_normals = normals[all_batch_idxs].to(self.device)
                        if depths!=None:
                            tmp_depths = depths[all_batch_idxs].to(self.device)
                        if images != None:
                            tmp_images = images[all_batch_idxs].to(self.device)

                        if keypoints!=None:
                            tmp_keypoints = copy.deepcopy(keypoints[all_batch_idxs])
                            tmp_keypoints[...,0] = tmp_keypoints[...,0]*10 /camera.image_width
                            tmp_keypoints[...,1]  = tmp_keypoints[...,1]*10 /camera.image_height

                        tmp_masks = masks[all_batch_idxs].float().to(self.device)

                        if head_masks is not None:
                            tmp_head_masks = head_masks[all_batch_idxs].float().to(self.device)
                            if tmp_head_masks.ndim==3:
                                tmp_head_masks = tmp_head_masks[...,None]
                        if head_bboxs is not None:
                            tmp_head_bbox = head_bboxs[all_batch_idxs].float().to(self.device)

                        if images is not None:
                            bg_tensor = torch.zeros_like(tmp_images)

                            bg_tensor[...,:3] = torch.tensor(bg, device=self.device)

                            masked_gt_img = torch.where(tmp_masks.bool(), tmp_images, bg_tensor)

                        if head_imgs is not None:
                            tmp_head_images = head_imgs[all_batch_idxs].to(self.device)

                            bg_head_tensor = torch.zeros_like(tmp_head_images)

                            bg_head_tensor[...,:3] = torch.tensor(bg, device=self.device)
                            #cv2.imwrite('tmp_head_mask.png',tmp_head_masks[0].detach().cpu().numpy()*255)
                            tmp_masked_head_gt = torch.where(tmp_head_masks.bool(), tmp_head_images, bg_head_tensor)

                        if segments is not None:
                            tmp_segments = segments[all_batch_idxs].to(self.device)
                        tmp_batch_params = {}
                        for key in batch_params.keys():
                            if key !='betas':
                                tmp_batch_params[key] = batch_params[key][all_batch_idxs] 
                                                


                        tmp_batch_params['betas'] = batch_params['betas'] #shared in all images.

                        if use_6d_rot:
                            tmp_batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(tmp_batch_params['full_pose']).view(B,-1)
                            #if 'mesh_OF_reg' in loss_keys or  'mesh_OF_init_reg' in loss_keys: #not yet used.
                        of_cur_idxs = np.array( curr_idxs.tolist()+ prev_idxs.tolist() )# + np.array( prev_idxs.tolist()) 
                        of_next_idxs = np.array( next_idxs.tolist() + curr_idxs.tolist() ) # +  np.array( curr_idxs.tolist()) 

                        render_GS_OF = 'GS_OF_reg' in all_loss_keys  or 'GS_OF_init_reg' in all_loss_keys 
                        if 'DoubleAvatar' in opt_conf.keys():
                            #base_eye = torch.eye(3, device=self.device)[None, None, :, :]
                            
                            if pose_noise_dict is not None:
                                noise_size = pose_noise_dict['noise_scale'] * (1-  k/max_iters  ) + pose_noise_dict['min_noise'] #max_iters
                            else:
                                noise_size = None
                            if not render_GS_OF:
                                F_outputs_all, gs_out_all = forward_cascadeavatar_split_use_gs(single_avatar=FineAvatar,   
                                                batch_params= tmp_batch_params,
                                                cameras = batch_cameras,
                                                bg_color=bg,
                                                render_out_levels=training_levels,
                                                noise_size=noise_size,
                                                )
                            else:
                                F_outputs_all, gs_out_all = forward_cascadeavatar_split_use_gs_w_OF(single_avatar=FineAvatar,   
                                                batch_params= tmp_batch_params,
                                                cameras = batch_cameras,
                                                bg_color=bg,
                                                of_cur_idxs=of_cur_idxs,
                                                of_next_idxs=of_next_idxs,
                                                tmp_camera_K=tmp_camera_K,
                                                tmp_camera_R=tmp_camera_R,
                                                tmp_camera_t=tmp_camera_t,
                                                render_out_levels=training_levels,
                                                noise_size=noise_size,
                                                )
                                                                


                        total_loss = 0.0
                        loss_dict = {} # dict storing unweighted raw losses

                        for level_idx in training_levels:

                            loss_conf = all_loss_conf[f'level_{str(level_idx)}']
                            # Exp G — residual curriculum: linearly scale delta_*_l2 weights
                            # from w_init at t_start to w_final at t_end over the outer step k.
                            # Absence of 'residual_curriculum' key → behavior unchanged.
                            _rc = loss_conf.get('residual_curriculum', None) if isinstance(loss_conf, dict) else None
                            if _rc is not None:
                                _t_start = _rc.get('t_start', 0)
                                _t_end = _rc.get('t_end', 1)
                                _w_init = _rc.get('w_init', 1.0)
                                _w_final = _rc.get('w_final', 1.0)
                                _denom = max(1, _t_end - _t_start)
                                _ratio = min(max((k - _t_start) / _denom, 0.0), 1.0)
                                _rc_scale = _w_init + (_w_final - _w_init) * _ratio
                                # shallow copy and overwrite delta_* keys with scaled values
                                loss_conf = dict(loss_conf)
                                for _k_curric in ('delta_rot_l2', 'delta_scale_l2', 'delta_shear_l2'):
                                    if _k_curric in loss_conf:
                                        loss_conf[_k_curric] = float(_rc_scale) * loss_conf[_k_curric]

                            loss_keys = loss_conf.keys()
                            if len(loss_keys) > 0:
                                avatar_labels = FineAvatar.avatar_indices[level_idx]
                                lhand_labels = avatar_labels.lhand
                                rhand_labels = avatar_labels.rhand
                                head_labels = avatar_labels.head
                                head_vts_labels, lhand_vts_labels, rhand_vts_labels = avatar_labels.head_vts,  avatar_labels.lhand_vts, avatar_labels.rhand_vts
                                loss = 0
                                F_outputs = F_outputs_all['levels'][level_idx]

                                F_P_base_vts = F_outputs['posed_vts'] # (B, V, 3)
                                F_P_davts = F_outputs['Da_vts']    # (B, V, 3)
                                if FineAvatar.use_gs_geo:
                                    GS2D_output = gs_out_all[level_idx]

                                # Jacobian offset magnitude (Δv_jac) — ALWAYS log (diagnostic for update magnitude)
                                # Δv_jac = F_P_davts (canonical after Poisson) − init_vts
                                try:
                                    _init_vts_for_diag = FineAvatar.jacobian_systems[level_idx].init_vts
                                    if _init_vts_for_diag.dim() == 2:
                                        _init_vts_for_diag = _init_vts_for_diag.unsqueeze(0)
                                    _delta_v_jac_diag = (F_P_davts - _init_vts_for_diag).detach()
                                    loss_dict[f"L{level_idx}_delta_v_jac_norm"] = _delta_v_jac_diag.norm(dim=-1).mean().item()
                                    loss_dict[f"L{level_idx}_delta_v_jac_max"] = _delta_v_jac_diag.norm(dim=-1).max().item()
                                except Exception:
                                    pass



                                F_face = FineAvatar.mesh_avatars[level_idx].face
                                num_faces = F_face.squeeze().shape[0]
                                F_rgb = FineAvatar.mesh_avatars[level_idx].rgb
                                face_albedo = F_outputs['gs']['rgb']
                                face_rgb = F_outputs['gs']['rgb_refine']

                                vts_albedo = face_to_vertex_color_batch(face_albedo, F_face )
                                vts_rgb = face_to_vertex_color_batch(face_rgb, F_face)

                                F_P_damesh = Mesh(F_P_davts ,F_face,colors=F_rgb,device=self.device)
                                #face_lap = FineAvatar.face_laps[-1] #s[-1]# Cache this if slow

                                #F_num_vts = FineAvatar.mesh_avatars[level_idx].v_canon.squeeze().shape[0]
                                curr_face_lap = FineAvatar.face_laps[level_idx] #s[-1]# Cache this if slow

                                curr_faces = FineAvatar.mesh_avatars[level_idx].face
                                curr_rgb = FineAvatar.mesh_avatars[level_idx].rgb # or None                            F_P_davts = F_outputs['P_da_vts']

                                F_P_base_mesh = Mesh(F_P_base_vts, curr_faces, colors=curr_rgb, device=self.device)
                                level_w = 1.0 #loss_conf.get('level_weights', [1.0] * (max_level))[level_idx]
                                prefix = f"L{level_idx}_"

                                if 'mesh_OF_reg' in loss_keys or  'mesh_OF_init_reg' in loss_keys: #not yet used.

                                    current_vts = torch.concatenate([F_P_base_vts[of_cur_idxs]],dim=0)
                                    next_vts = torch.concatenate([F_P_base_vts[of_next_idxs]],dim=0)
                                    curr_uv = projection(current_vts, tmp_camera_K[of_cur_idxs],
                                                tmp_camera_R[of_cur_idxs],
                                                    -tmp_camera_t[of_cur_idxs].unsqueeze(-1))

                                    next_uv = projection(next_vts, tmp_camera_K[of_next_idxs],
                                                tmp_camera_R[of_next_idxs],
                                                    -tmp_camera_t[of_next_idxs].unsqueeze(-1))
                                    displacement = next_uv -curr_uv
                                    displacement_dict = {}
                                    displacement_dict['disp'] = displacement
                                    displacement_dict['curr_idxs'] = of_cur_idxs
                                    displacement_dict['next_idxs'] = of_next_idxs

                                else:
                                    displacement_dict = None

                                
                                F_P_mesh_rendered = self.mesh_renderer.render_SH_batch_w_albedo_refine_rgb(F_P_base_mesh, 
                                                                        rgb= vts_albedo, refine_rgb=vts_rgb,
                                                                        cameras=batch_cameras,
                                                                        bg = bg, disp_dict=displacement_dict) 
                                mesh_rgb = F_P_mesh_rendered['rgba'][...,:3]
                                mesh_coarse_rgb =F_P_mesh_rendered['rgba_coarse'][...,:3]


                                F_P_mesh_mask = F_P_mesh_rendered['mask']
                                F_P_mesh_masks_overlap = tmp_masks * F_P_mesh_mask
                                F_P_mesh_masks_overlap = F_P_mesh_masks_overlap.detach()

                                # ============================================================
                                # MLP displacement dual-branch (guarded by loss_conf flag)
                                # Sequential render: MLP mesh rendered, data loss + laplacian,
                                # plus one-way distillation ‖Δv_jac − Δv_mlp.detach()‖².
                                # ============================================================
                                if loss_conf.get('use_mlp_displacement', False) and 'delta_v_mlp' in F_outputs:
                                    delta_v_mlp = F_outputs['delta_v_mlp']                    # (B, N_verts, 3)
                                    mesh_mlp_canon_vts = F_outputs['mesh_mlp_canon_vts']      # (B, N_verts, 3)
                                    mesh_mlp_posed_vts = F_outputs['mesh_mlp_posed_vts']      # (B, N_verts, 3)

                                    # MLP mesh (posed) for rendering
                                    F_P_mesh_mlp = Mesh(mesh_mlp_posed_vts, curr_faces, colors=curr_rgb, device=self.device)

                                    # Render MLP mesh — uses SAME albedo/refine_rgb as Jac path (colors are shared via vts_albedo/vts_rgb)
                                    F_P_mesh_mlp_rendered = self.mesh_renderer.render_SH_batch_w_albedo_refine_rgb(
                                        F_P_mesh_mlp,
                                        rgb=vts_albedo, refine_rgb=vts_rgb,
                                        cameras=batch_cameras,
                                        bg=bg, disp_dict=None,
                                    )
                                    mesh_mlp_rgb = F_P_mesh_mlp_rendered['rgba'][..., :3]
                                    F_P_mesh_mlp_mask = F_P_mesh_mlp_rendered['mask']

                                    # MLP data loss — RGB L1 + SSIM + LPIPS + albedo L1 + mask (full data supervision on MLP mesh)
                                    _w_mlp_rgb = float(loss_conf.get('MLP_Mesh_rgb_l1', 10.0))
                                    _w_mlp_ssim = float(loss_conf.get('MLP_Mesh_rgb_ssim', 0.0))
                                    _w_mlp_lpips = float(loss_conf.get('MLP_Mesh_rgb_lpips', 0.0))
                                    _w_mlp_albedo_l1 = float(loss_conf.get('MLP_Mesh_albedo_l1', 0.0))
                                    _w_mlp_mask = float(loss_conf.get('MLP_P_mesh_masks', 1.0))
                                    if _w_mlp_rgb > 0:
                                        _rgb_l1_mlp = l1loss(mesh_mlp_rgb, masked_gt_img)
                                        loss_dict[prefix + 'MLP_Mesh_rgb_l1'] = _rgb_l1_mlp.item()
                                        loss = loss + _w_mlp_rgb * _rgb_l1_mlp
                                    if _w_mlp_ssim > 0:
                                        _ssim_mlp = torch.mean(1 - ssim_loss.forward(mesh_mlp_rgb, masked_gt_img))
                                        loss_dict[prefix + 'MLP_Mesh_rgb_ssim'] = _ssim_mlp.item()
                                        loss = loss + _w_mlp_ssim * _ssim_mlp
                                    if _w_mlp_lpips > 0:
                                        _lpips_mlp = lpips_loss(mesh_mlp_rgb, masked_gt_img, F_P_mesh_mlp_rendered['mask']).mean()
                                        loss_dict[prefix + 'MLP_Mesh_rgb_lpips'] = _lpips_mlp.item()
                                        loss = loss + _w_mlp_lpips * _lpips_mlp
                                    if _w_mlp_albedo_l1 > 0:
                                        _mlp_coarse_rgb = F_P_mesh_mlp_rendered['rgba_coarse'][..., :3]
                                        _albedo_l1_mlp = l1loss(_mlp_coarse_rgb, masked_gt_img)
                                        loss_dict[prefix + 'MLP_Mesh_albedo_l1'] = _albedo_l1_mlp.item()
                                        loss = loss + _w_mlp_albedo_l1 * _albedo_l1_mlp
                                    if _w_mlp_mask > 0:
                                        _mask_mlp = mesh_mask_loss(F_P_mesh_mlp_mask, tmp_masks)
                                        loss_dict[prefix + 'MLP_P_mesh_masks'] = _mask_mlp.item()
                                        loss = loss + _w_mlp_mask * _mask_mlp

                                    # Δv_mlp displacement-field Laplacian (‖L · Δv_mlp‖²)
                                    _w_mlp_lap = float(loss_conf.get('MLP_laplacian', 10000.0))
                                    if _w_mlp_lap > 0:
                                        _base_verts_for_mlp = FineAvatar.jacobian_systems[level_idx].init_vts.detach()
                                        _mesh_mlp_canon_obj = Mesh(mesh_mlp_canon_vts, curr_faces, colors=None, device=self.device)
                                        _lap_mlp = laplacian_offset_loss(_mesh_mlp_canon_obj, _base_verts_for_mlp)
                                        loss_dict[prefix + 'MLP_laplacian'] = _lap_mlp.item()
                                        loss = loss + _w_mlp_lap * _lap_mlp

                                    # Distillation loss: ‖Δv_jac − Δv_mlp.detach()‖²  (one-way, ramp-up)
                                    _w_distill_target = float(loss_conf.get('MLP_distill_weight', 1.0))
                                    _ramp_steps = int(loss_conf.get('MLP_distill_ramp_steps', 100))
                                    _ramp_w = min(1.0, (pbar.n + 1) / float(max(1, _ramp_steps)))
                                    _w_distill = _ramp_w * _w_distill_target
                                    if _w_distill > 0:
                                        _init_vts_jac = FineAvatar.jacobian_systems[level_idx].init_vts
                                        if _init_vts_jac.dim() == 2:
                                            _init_vts_jac = _init_vts_jac.unsqueeze(0)
                                        _delta_v_jac = F_P_davts - _init_vts_jac
                                        _distill = ((_delta_v_jac - delta_v_mlp.detach()) ** 2).mean()
                                        loss_dict[prefix + 'MLP_distill'] = _distill.item()
                                        # Relative ratio (non-integrability floor diagnostic)
                                        _mlp_norm_sq = (delta_v_mlp.detach() ** 2).mean().item() + 1e-12
                                        loss_dict[prefix + 'MLP_distill_ratio'] = _distill.item() / _mlp_norm_sq
                                        loss_dict[prefix + 'MLP_distill_ramp_w'] = float(_ramp_w)
                                        loss = loss + _w_distill * _distill

                                    # Diagnostic: mean ‖Δv_mlp‖ (sanity gate)
                                    loss_dict[prefix + 'MLP_delta_v_norm'] = delta_v_mlp.detach().norm(dim=-1).mean().item()
                                # ============================================================
                                # end MLP displacement block
                                # ============================================================

                                # ============================================================
                                # Adjacent-frame RGB / albedo consistency
                                # Penalizes per-vertex color change between (prev,curr) and (curr,next)
                                # frame pairs within the OF triplet batch.
                                # L0: strong (e.g. 10); L1/L2: weaker (e.g. 1).
                                # ============================================================
                                _w_adj_rgb = float(loss_conf.get('adjacent_rgb_consistency', 0.0))
                                _w_adj_albedo = float(loss_conf.get('adjacent_albedo_consistency', 0.0))
                                if (_w_adj_rgb > 0 or _w_adj_albedo > 0) and batch_size > 1 and B_pairs > 0:
                                    import numpy as _np
                                    _pi = torch.as_tensor(prev_idxs, dtype=torch.long, device=self.device)
                                    _ci = torch.as_tensor(curr_idxs, dtype=torch.long, device=self.device)
                                    _ni = torch.as_tensor(next_idxs, dtype=torch.long, device=self.device)
                                    if _w_adj_rgb > 0:
                                        _r_prev = vts_rgb.index_select(0, _pi)
                                        _r_curr = vts_rgb.index_select(0, _ci)
                                        _r_next = vts_rgb.index_select(0, _ni)
                                        _d_pc = (_r_prev - _r_curr).abs().mean()
                                        _d_cn = (_r_curr - _r_next).abs().mean()
                                        _adj_rgb = 0.5 * (_d_pc + _d_cn)
                                        loss_dict[prefix + 'adjacent_rgb_consistency'] = _adj_rgb.item()
                                        loss = loss + _w_adj_rgb * _adj_rgb
                                    if _w_adj_albedo > 0:
                                        _a_prev = vts_albedo.index_select(0, _pi)
                                        _a_curr = vts_albedo.index_select(0, _ci)
                                        _a_next = vts_albedo.index_select(0, _ni)
                                        _d_a_pc = (_a_prev - _a_curr).abs().mean()
                                        _d_a_cn = (_a_curr - _a_next).abs().mean()
                                        _adj_a = 0.5 * (_d_a_pc + _d_a_cn)
                                        loss_dict[prefix + 'adjacent_albedo_consistency'] = _adj_a.item()
                                        loss = loss + _w_adj_albedo * _adj_a
                                # ============================================================

                                # === Adjacent canonical-mesh consistency (v23+ — temporal stretch fix) ===
                                # Penalizes per-vertex change across (prev,curr,next) frames in canonical
                                # space (F_P_davts has same canonical mesh + per-frame J residual).
                                # Targets "stretching at specific timestamps" — frames where pose-conditioned
                                # residual J spikes locally. Triggers only when key present + B_pairs > 0.
                                _w_adj_canon = float(loss_conf.get('adjacent_canon_consistency', 0.0))
                                if _w_adj_canon > 0 and batch_size > 1 and B_pairs > 0:
                                    _pi2 = torch.as_tensor(prev_idxs, dtype=torch.long, device=self.device)
                                    _ci2 = torch.as_tensor(curr_idxs, dtype=torch.long, device=self.device)
                                    _ni2 = torch.as_tensor(next_idxs, dtype=torch.long, device=self.device)
                                    _v_prev = F_P_davts.index_select(0, _pi2)
                                    _v_curr = F_P_davts.index_select(0, _ci2)
                                    _v_next = F_P_davts.index_select(0, _ni2)
                                    _d_v_pc = (_v_prev - _v_curr).abs().mean()
                                    _d_v_cn = (_v_curr - _v_next).abs().mean()
                                    _adj_v = 0.5 * (_d_v_pc + _d_v_cn)
                                    loss_dict[prefix + 'adjacent_canon_consistency'] = _adj_v.item()
                                    loss = loss + _w_adj_canon * _adj_v

                                # === Adjacent GS mean_offset consistency ===
                                # Penalizes per-face mean_offset change across (prev,curr,next) frames.
                                # Complements adjacent_canon (mesh canonical) — mean_offset is GS-specific
                                # per-frame Gaussian displacement, not covered by canon consistency.
                                # GS stage only (mean_offset present in F_outputs['gs'] when use_gs_geo=True).
                                _w_adj_mo = float(loss_conf.get('adjacent_mean_offset', 0.0))
                                if _w_adj_mo > 0 and batch_size > 1 and B_pairs > 0:
                                    _gs_out = F_outputs.get('gs', None)
                                    if _gs_out is not None and 'mean_offset' in _gs_out:
                                        _mo = _gs_out['mean_offset']  # (B, F, 3)
                                        _pi3 = torch.as_tensor(prev_idxs, dtype=torch.long, device=self.device)
                                        _ci3 = torch.as_tensor(curr_idxs, dtype=torch.long, device=self.device)
                                        _ni3 = torch.as_tensor(next_idxs, dtype=torch.long, device=self.device)
                                        _mo_prev = _mo.index_select(0, _pi3)
                                        _mo_curr = _mo.index_select(0, _ci3)
                                        _mo_next = _mo.index_select(0, _ni3)
                                        _d_mo_pc = (_mo_prev - _mo_curr).abs().mean()
                                        _d_mo_cn = (_mo_curr - _mo_next).abs().mean()
                                        _adj_mo = 0.5 * (_d_mo_pc + _d_mo_cn)
                                        loss_dict[prefix + 'adjacent_mean_offset'] = _adj_mo.item()
                                        loss = loss + _w_adj_mo * _adj_mo
                                # ============================================================


                                if 'init_full_pose' in loss_keys  or  'init_full_pose' in loss_keys:

                                    if use_6d_rot:
                                        init_loss = l1_loss( smpl_mesh.full_pose_to_sixrot(tmp_batch_params['full_pose']),init_full_poses[all_batch_idxs])
                                    else:                                    
                                        init_loss = l1_loss( tmp_batch_params['full_pose'],init_full_poses[all_batch_idxs])

                                    loss_dict['init_full_pose'] = init_loss.item() # store raw loss
                                    loss += loss_conf['init_full_pose'] *init_loss

                                if 'mesh_OF_reg' in loss_keys  or  'mesh_OF_init_reg' in loss_keys:
                                    mesh_init_flow = F_P_mesh_rendered['displacement']
                                    # --- OF visualization (every 10 steps + first step) ---
                                    if debug_folder is not None and (k == 0 or (k + 1) % 10 == 0):
                                        try:
                                            from utils.debug.visualize_of import visualize_optical_flow
                                            visualize_optical_flow(
                                                save_dir=debug_folder, step=k,
                                                mesh_init_flow=mesh_init_flow,
                                                gt_waft_of=tmp_waft_of if waft_of is not None else None,
                                                masks=tmp_masks,
                                                rendered_mask=F_P_mesh_rendered['mask'],
                                                of_cur_idxs=of_cur_idxs,
                                                of_next_idxs=of_next_idxs,
                                                waft_loss=waft_loss if OF_model is not None else None,
                                                waft_features=tmp_waft_feature if waft_features is not None else None,
                                                images=tmp_images if images is not None else None,
                                            )
                                        except Exception as e:
                                            print(f"[OF_VIS] Warning: {e}")
                                    #fisrt_displacement = F_P_mesh_rendered['displacement'][0:1].permute(0,3,1,2)
                                    # curr_img = mesh_rgb[of_cur_idxs][0].detach()
                                    # next_img = mesh_rgb[of_next_idxs][0].detach()
                                    """
                                    from libs.WAFT.utils.utils import  coords_grid, bilinear_sampler
                                    def warp_with_flow(image, flow):
                                        N, _, H, W = image.shape
                                        coords2 = coords_grid(N, H, W, device=image.device).permute(0, 2, 3, 1)
                                        coords2 = coords2 + flow.permute(0, 2, 3, 1)
                                        warped_image = bilinear_sampler(image, coords2)
                                        return warped_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

                                    warped_img = warp_with_flow(next_img.permute(2,0,1)[None,:],fisrt_displacement.detach())
                                    cv2.imwrite('train_warped_img.png',warped_img*255)
                                    warped_img = warp_with_flow(next_img.permute(2,0,1)[None,:],tmp_waft_of[0:1].detach())

                                    cv2.imwrite('train_warped_gt_flow_img.png',warped_img*255)

                                    cv2.imwrite('train_tgt_img.png',curr_img.detach().cpu().numpy()*255)
                                    cv2.imwrite('train_next_img.png',next_img.detach().cpu().numpy()*255)
                                    """
                                    H,W = camera.image_size
                            
                                of_loss = None
                            
                                if 'acc_reg' in loss_keys and loss_conf['acc_reg']>0:
                                    prev_vts = F_P_base_vts[prev_idxs]
                                    curr_vts = F_P_base_vts[curr_idxs]
                                    next_vts = F_P_base_vts[next_idxs]
                                    acc_reg_loss = acc_loss(prev_vts,curr_vts,next_vts,mass_weights=mass_weights[level_idx])

                                    loss_dict['acc_reg'] = acc_reg_loss.item() # store raw loss
                                    loss += acc_reg_loss*loss_conf['acc_reg']

                                if 'mesh_OF_reg' in loss_keys and loss_conf['mesh_OF_reg']>0:
                                    if waft_features is not None:

                                        of_loss = waft_loss.loss_w_init_of_w_waft_feature(
                                            tmp_masks,
                                            F_P_mesh_rendered['mask'],
                                            of_cur_idxs,
                                            of_next_idxs,
                                            mesh_init_flow,
                                            tmp_waft_feature,
                                            no_grad=True,
                                        )
                                        loss_dict['mesh_OF_reg'] = of_loss.item() # store raw loss

                                    else:
                                        
                                        of_loss = waft_loss.loss_w_images(
                                            tmp_masks,
                                            F_P_mesh_rendered['mask'],
                                            of_cur_idxs,
                                            of_next_idxs,
                                            mesh_init_flow,
                                            tmp_images
                                        )
                                        loss_dict['mesh_OF_reg'] = of_loss.item() # store raw loss

                                    loss += of_loss*loss_conf['mesh_OF_reg']
                                #from libs.WAFT.utils.flow_viz import flow_to_image
                                if 'mesh_OF_init_reg' in loss_keys and loss_conf['mesh_OF_init_reg']>0:
                                    of_loss = waft_loss.loss_w_gt_of(
                                        tmp_masks,
                                        F_P_mesh_rendered['mask'],
                                        of_cur_idxs,
                                        of_next_idxs,
                                        mesh_init_flow,
                                        tmp_waft_of
                                    )

                                    loss_dict[prefix+'mesh_OF_init_reg'] = of_loss.item() # store raw loss
                                    loss += of_loss*loss_conf['mesh_OF_init_reg']

                                if 'GS_OF_reg' in loss_keys and loss_conf['GS_OF_reg']>0:
                                    if waft_features is not None:
                                        of_loss = waft_loss.loss_w_init_of_w_waft_feature(
                                            tmp_masks,
                                            GS2D_output['render_alpha'],
                                            of_cur_idxs,
                                            of_next_idxs,
                                            GS2D_output['render_flow'],
                                            tmp_waft_feature,
                                            no_grad=True,
                                        )
                                        loss_dict['GS_OF_reg'] = of_loss.item() # store raw loss

                                    else:
                                        
                                        of_loss = waft_loss.loss_w_images(
                                            tmp_masks,
                                            GS2D_output['render_alpha'],
                                            of_cur_idxs,
                                            of_next_idxs,
                                            GS2D_output['render_flow'],
                                            tmp_images
                                        )
                                        loss_dict['GS_OF_reg'] = of_loss.item() # store raw loss


                                if 'GS_OF_init_reg' in loss_keys and loss_conf['GS_OF_init_reg']>0:
                                    of_loss = waft_loss.loss_w_gt_of(
                                        tmp_masks,
                                        GS2D_output['render_alpha'],
                                        of_cur_idxs,
                                        of_next_idxs,
                                        GS2D_output['render_flow'],
                                        tmp_waft_of
                                    )

                                    loss_dict[prefix+'GS_OF_init_reg'] = of_loss.item() # store raw loss
                                    loss += of_loss*loss_conf['GS_OF_init_reg']
                                if isinstance(loss, torch.Tensor):
                                    if torch.isnan(loss):
                                        import pdb;pdb.set_trace()


                                if "P_mesh_masks" in loss_keys and loss_conf['P_mesh_masks'] >0:
                                    mask_loss = mesh_mask_loss(F_P_mesh_rendered['mask'],tmp_masks) #+ mesh_mask_loss(C_P_mesh_rendered['mask'],tmp_masks)
                                    loss_dict[prefix+'P_mesh_masks'] = mask_loss.item() # store raw loss
                                    loss += mask_loss*loss_conf['P_mesh_masks']

                                if "P_mesh_masks_boundary" in loss_keys and loss_conf['P_mesh_masks_boundary'] > 0:
                                    _bw_sigma = float(loss_conf.get('P_mesh_masks_boundary_sigma', 3.0))
                                    _bw_alpha = float(loss_conf.get('P_mesh_masks_boundary_alpha', 3.0))
                                    bw_loss = boundary_weighted_mask_loss(F_P_mesh_rendered['mask'], tmp_masks, sigma=_bw_sigma, alpha=_bw_alpha)
                                    loss_dict[prefix+'P_mesh_masks_boundary'] = bw_loss.item()
                                    loss += bw_loss * loss_conf['P_mesh_masks_boundary']

                                if "P_mesh_masks_out" in loss_keys and loss_conf['P_mesh_masks_out'] >0:
                                    mask_loss = mesh_mask_loss(torch.abs(F_P_mesh_rendered['mask'] - tmp_masks),tmp_masks) #+mesh_mask_loss(torch.abs(C_P_mesh_rendered['mask'] - tmp_masks),tmp_masks)
                                    loss_dict[prefix+'P_mesh_masks_out'] = mask_loss.item() # store raw loss
                                    loss += mask_loss*loss_conf['P_mesh_masks_out'] 



                                if "laplacian" in loss_keys  and loss_conf['laplacian'] >0  :
                                    # Relative laplacian: penalize change in laplacian from base mesh
                                    # base_verts = posed_vts - offset (init mesh with LBS applied)
                                    base_verts = (F_P_base_vts - F_outputs['offset']).detach()
                                    lap_loss = laplacian_offset_loss(F_P_base_mesh, base_verts)
                                    loss += lap_loss * loss_conf["laplacian"]
                                    loss_dict[prefix+'laplacian'] = lap_loss.item() # store raw loss

                                if "canon_vts_l2" in loss_keys and loss_conf["canon_vts_l2"] > 0:
                                    _V_init_canon = FineAvatar.jacobian_systems[level_idx].init_vts.detach()
                                    if _V_init_canon.dim() == 3:
                                        _V_init_canon = _V_init_canon.squeeze(0)
                                    canon_l2 = ((F_P_davts - _V_init_canon) ** 2).sum(-1).mean()
                                    loss += canon_l2 * loss_conf["canon_vts_l2"]
                                    loss_dict[prefix+'canon_vts_l2'] = canon_l2.item()

                                if "canon_vts_l1" in loss_keys and loss_conf["canon_vts_l1"] > 0:
                                    _V_init_canon = FineAvatar.jacobian_systems[level_idx].init_vts.detach()
                                    if _V_init_canon.dim() == 3:
                                        _V_init_canon = _V_init_canon.squeeze(0)
                                    canon_l1 = (F_P_davts - _V_init_canon).norm(dim=-1).mean()
                                    loss += canon_l1 * loss_conf["canon_vts_l1"]
                                    loss_dict[prefix+'canon_vts_l1'] = canon_l1.item()

                                if "canon_offset_lap" in loss_keys and loss_conf["canon_offset_lap"] > 0:
                                    _V_init_canon = FineAvatar.jacobian_systems[level_idx].init_vts.detach()
                                    if _V_init_canon.dim() == 3:
                                        _V_init_canon = _V_init_canon.squeeze(0)
                                    canon_lap = laplacian_offset_loss(F_P_damesh, _V_init_canon)
                                    loss += canon_lap * loss_conf["canon_offset_lap"]
                                    loss_dict[prefix+'canon_offset_lap'] = canon_lap.item()

                                if "canon_edge_length" in loss_keys and loss_conf["canon_edge_length"] > 0:
                                    _V_init_canon = FineAvatar.jacobian_systems[level_idx].init_vts.detach()
                                    if _V_init_canon.dim() == 3:
                                        _V_init_canon = _V_init_canon.squeeze(0)
                                    canon_el = edge_length_preservation_loss(F_P_davts, _V_init_canon, curr_faces)
                                    loss += canon_el * loss_conf["canon_edge_length"]
                                    loss_dict[prefix+'canon_edge_length'] = canon_el.item()

                                if "edge_length" in loss_keys and loss_conf["edge_length"] > 0:
                                    base_verts_el = (F_P_base_vts - F_outputs['offset']).detach()
                                    el_loss = edge_length_preservation_loss(F_P_base_vts, base_verts_el[0] if base_verts_el.dim() == 3 else base_verts_el, curr_faces)
                                    loss += el_loss * loss_conf["edge_length"]
                                    loss_dict[prefix+'edge_length'] = el_loss.item()

                                of_loss = None


                                if "P_mesh_normal_cosine" in loss_keys and loss_conf["P_mesh_normal_cosine"] > 0:
                                    #normal_loss= masked_normal_cosine_loss(C_P_mesh_rendered['normal'], tmp_normals,C_P_mesh_masks_overlap,True) + \
                                    normal_loss=     masked_normal_cosine_loss(F_P_mesh_rendered['normal'], tmp_normals,F_P_mesh_masks_overlap,True)
                                    loss_dict[prefix+'P_mesh_normal_cosine'] = normal_loss.item() # store raw loss
                                    loss += normal_loss * loss_conf['P_mesh_normal_cosine']

                                if "P_mesh_normal_ssim" in loss_keys and loss_conf["P_mesh_normal_ssim"] > 0:
                                    render_normal_vec_ssim =F_P_mesh_rendered['normal']
                                    tmp_normal_ssim = (tmp_normals+1.0)/2.0
                                    normal_ssim_loss  = torch.mean(1-ssim_loss.forward(render_normal_vec_ssim, tmp_normal_ssim.float())) # torch.sum(mesh_masks_overlap - mesh_masks_overlap*(torch.sum(render_normal_vec*tmp_normals,dim=-1)[...,None]))/torch.sum(mesh_masks_overlap)
                                    loss_dict[prefix+'P_mesh_normal_ssim'] = normal_ssim_loss.item() # store raw loss
                                    loss += normal_ssim_loss * loss_conf['P_mesh_normal_ssim']

                                if "P_normal_z_cosine" in loss_keys and loss_conf["P_normal_z_cosine"] > 0:
                                    from scene.optimizer.loss_functions import masked_normal_z_cosine_loss
                                    nz_cos = masked_normal_z_cosine_loss(F_P_mesh_rendered['normal'], tmp_normals, F_P_mesh_masks_overlap, True)
                                    loss_dict[prefix+'P_normal_z_cosine'] = nz_cos.item()
                                    loss += nz_cos * loss_conf['P_normal_z_cosine']

                                if "P_normal_z_l1" in loss_keys and loss_conf["P_normal_z_l1"] > 0:
                                    from scene.optimizer.loss_functions import masked_normal_z_l1_loss
                                    nz_l1 = masked_normal_z_l1_loss(F_P_mesh_rendered['normal'], tmp_normals, F_P_mesh_masks_overlap, True)
                                    loss_dict[prefix+'P_normal_z_l1'] = nz_l1.item()
                                    loss += nz_l1 * loss_conf['P_normal_z_l1']

                                if "normal_tv" in loss_keys and loss_conf["normal_tv"] > 0:
                                    rn = (F_P_mesh_rendered['normal'] * 2.0 - 1.0) * F_P_mesh_masks_overlap
                                    normal_tv_val = tv_loss(rn)
                                    loss_dict[prefix+'normal_tv'] = normal_tv_val.item()
                                    loss += normal_tv_val * loss_conf['normal_tv']

                                # Sobel-based normal edge matching loss.
                                # Compare Sobel(pred_normal) vs Sobel(gt_normal) — encourages predicted
                                # normal map edges to align with GT normal edges (edge-aware supervision).
                                # Inspired by 3DGS/NeRF edge-aware normal regularizers.
                                if "normal_sobel_edge_l1" in loss_conf.keys() and loss_conf['normal_sobel_edge_l1'] > 0:
                                    # pred is rendered in [0,1] → convert to [-1,1] like GT (tmp_normals already unit-norm in [-1,1] FG, 0 BG via mask in _preprocess_normal).
                                    _pred_n = (F_P_mesh_rendered['normal'] * 2.0 - 1.0)         # (B, H, W, 3) in [-1,1]
                                    _gt_n = tmp_normals                                          # (B, H, W, 3): FG [-1,1] unit, BG 0
                                    _mask_se = F_P_mesh_masks_overlap                            # (B, H, W, 1) intersection of GT mask & rendered mask

                                    # to (B, C, H, W)
                                    _p = _pred_n.permute(0, 3, 1, 2)
                                    _g = _gt_n.permute(0, 3, 1, 2)
                                    if _mask_se.dim() == 4 and _mask_se.shape[-1] == 1:
                                        _m = _mask_se.permute(0, 3, 1, 2)                        # (B, 1, H, W)
                                    else:
                                        _m = _mask_se.unsqueeze(1) if _mask_se.dim() == 3 else _mask_se

                                    # Pre-mask BOTH pred and GT by the same overlap mask (matches existing
                                    # normal_all_cosine convention). Ensures pred BG and GT BG both = 0 so the
                                    # silhouette step in conv2d is identical on both sides.
                                    _p = _p * _m
                                    _g = _g * _m

                                    _kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                                       device=_p.device, dtype=_p.dtype).view(1, 1, 3, 3) / 8.0
                                    _ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                                       device=_p.device, dtype=_p.dtype).view(1, 1, 3, 3) / 8.0
                                    import torch.nn.functional as _F_nn
                                    _kx_3 = _kx.repeat(3, 1, 1, 1)
                                    _ky_3 = _ky.repeat(3, 1, 1, 1)
                                    _pgx = _F_nn.conv2d(_p, _kx_3, padding=1, groups=3)
                                    _pgy = _F_nn.conv2d(_p, _ky_3, padding=1, groups=3)
                                    _ggx = _F_nn.conv2d(_g, _kx_3, padding=1, groups=3)
                                    _ggy = _F_nn.conv2d(_g, _ky_3, padding=1, groups=3)
                                    _diff = (_pgx - _ggx).abs() + (_pgy - _ggy).abs()           # (B, 3, H, W)

                                    _msum = _m.sum() * 3.0 + 1e-6
                                    _se_loss = (_diff * _m).sum() / _msum
                                    loss_dict[prefix + 'normal_sobel_edge_l1'] = _se_loss.item()
                                    loss = loss + _se_loss * loss_conf['normal_sobel_edge_l1']


                                if "P_iso_GS_norm_hinge" in loss_conf.keys() and loss_conf['P_iso_GS_norm_hinge'] >0:
                                    #update vec
                                    #sampling vertex normal of current da_mesh.
                                    norm_loss = compute_hinge_sdf_loss(F_P_damesh.norm_xyz,F_P_damesh.xyz,P_sdf_volume,option='vts') # compute_sdf_loss(C_P_damesh.norm_xyz,C_P_damesh.xyz,P_sdf_volume) \
                                        #+compute_sdf_loss(F_P_damesh.norm_xyz,F_P_damesh.xyz,P_sdf_volume) 
                                    loss_dict[prefix+'P_iso_GS_norm_hinge'] = norm_loss.item() # store raw loss

                                    loss+=norm_loss*loss_conf['P_iso_GS_norm_hinge']  #torch.mean(v_norm_init)*loss_conf['iso_vec']


                                if "P_iso_sdf_vts_norm_hinge" in loss_conf.keys() and loss_conf['P_iso_sdf_vts_norm_hinge'] >0:
                                    #update vec
                                    #sampling vertex normal of current da_mesh.
                                    norm_loss = compute_hinge_sdf_loss(F_P_damesh.norm_xyz,F_P_damesh.xyz,P_sdf_volume,option='vts') # compute_sdf_loss(C_P_damesh.norm_xyz,C_P_damesh.xyz,P_sdf_volume) \
                                        #+compute_sdf_loss(F_P_damesh.norm_xyz,F_P_damesh.xyz,P_sdf_volume) 
                                    loss_dict[prefix+'P_iso_sdf_vts_norm_hinge'] = norm_loss.item() # store raw loss

                                    loss+=norm_loss*loss_conf['P_iso_sdf_vts_norm_hinge']  #torch.mean(v_norm_init)*loss_conf['iso_vec']



                                if "P_iso_sdf_face_norm_hinge" in loss_conf.keys() and loss_conf['P_iso_sdf_face_norm_hinge'] >0:
                                    #update vec
                                    F_P_damesh.norm_xyz[:,F_P_damesh.face,:].mean(dim=-2)
                                    #sampling vertex normal of current da_mesh.

                                    norm_loss =compute_hinge_sdf_loss(F_P_damesh.norm_face,(F_P_damesh.xyz[:,F_P_damesh.face,:]).mean(dim=-2) ,P_sdf_volume,option='face') #compute_sdf_loss(C_P_damesh.norm_face,(C_P_damesh.xyz[:,C_P_damesh.face,:]).mean(dim=-2) ,P_sdf_volume) \
                                    # +compute_sdf_loss(F_P_damesh.norm_face,(F_P_damesh.xyz[:,F_P_damesh.face,:]).mean(dim=-2) ,P_sdf_volume)
                                    loss_dict[prefix+'P_iso_sdf_face_norm_hinge'] = norm_loss.item() # store raw loss

                                    loss+=norm_loss*loss_conf['P_iso_sdf_face_norm_hinge']  #torch.mean(v_norm_init)*loss_conf['iso_vec']



                                #if "sdf_reg"  in loss_keys and loss_conf['sdf_reg']>0:
                                if "P_iso_sdf_vts_norm" in loss_conf.keys() and loss_conf['P_iso_sdf_vts_norm'] >0:
                                    #update vec
                                    #sampling vertex normal of current da_mesh.
                                    norm_loss = compute_sdf_loss(F_P_damesh.norm_xyz,F_P_damesh.xyz,P_sdf_volume) # compute_sdf_loss(C_P_damesh.norm_xyz,C_P_damesh.xyz,P_sdf_volume) \
                                        #+compute_sdf_loss(F_P_damesh.norm_xyz,F_P_damesh.xyz,P_sdf_volume) 
                                    loss_dict[prefix+'P_iso_sdf_vts_norm'] = norm_loss.item() # store raw loss

                                    loss+=norm_loss*loss_conf['P_iso_sdf_vts_norm']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "P_iso_sdf_face_norm" in loss_conf.keys() and loss_conf['P_iso_sdf_vts_norm'] >0:
                                    #update vec
                                    F_P_damesh.norm_xyz[:,F_P_damesh.face,:].mean(dim=-2)
                                    #sampling vertex normal of current da_mesh.
                                    norm_loss =compute_sdf_loss(F_P_damesh.norm_face,(F_P_damesh.xyz[:,F_P_damesh.face,:]).mean(dim=-2) ,P_sdf_volume) #compute_sdf_loss(C_P_damesh.norm_face,(C_P_damesh.xyz[:,C_P_damesh.face,:]).mean(dim=-2) ,P_sdf_volume) \
                                    # +compute_sdf_loss(F_P_damesh.norm_face,(F_P_damesh.xyz[:,F_P_damesh.face,:]).mean(dim=-2) ,P_sdf_volume)
                                    loss_dict[prefix+'P_iso_sdf_face_norm'] = norm_loss.item() # store raw loss

                                    loss+=norm_loss*loss_conf['P_iso_sdf_face_norm']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                #normal loss
                                if "GS_normal_error" in loss_conf.keys() and loss_conf['GS_normal_error'] >0:
                                    normal_loss = masked_normal_cosine_loss(GS2D_output['rend_normal'], GS2D_output['surf_normal'],tmp_masks,normalize=False)

                                    #cv2.imwrite('rend_normal.png',(test_normal.detach().cpu().numpy()[0]+1.0)*127.5)
                                    loss_dict[prefix+'GS_normal_error'] = normal_loss.item() # store raw loss
                                    loss+=normal_loss*loss_conf['GS_normal_error']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                                
                                if "GS_normal_render" in loss_conf.keys() and loss_conf['GS_normal_render'] >0:
                                    normal_loss = masked_normal_cosine_loss(GS2D_output['rend_normal'], tmp_normals, tmp_masks,normalize=False)
                                    loss_dict[prefix+'GS_normal_render'] = normal_loss.item() # store raw loss
                                    loss+=normal_loss*loss_conf['GS_normal_render']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                                
                                if "GS_normal_surf" in loss_conf.keys() and loss_conf['GS_normal_surf'] >0:
                                    normal_loss = masked_normal_cosine_loss(GS2D_output['surf_normal'], tmp_normals, tmp_masks,normalize=False)
                                    loss_dict[prefix+'GS_normal_surf'] = normal_loss.item() # store raw loss
                                    loss+=normal_loss*loss_conf['GS_normal_surf']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "GS_mask_l1" in loss_conf.keys() and loss_conf['GS_mask_l1'] >0:
                                    mask_loss = mesh_mask_loss(GS2D_output['render_alpha'], tmp_masks)
                                    loss_dict[prefix+'GS_mask_l1'] = mask_loss.item() # store raw loss
                                    loss+=mask_loss*loss_conf['GS_mask_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                            


                                if "GS_normal_render_ssim" in loss_keys and loss_conf["GS_normal_render_ssim"] > 0:
                                    #render_normal_vec_ssim = (F_P_mesh_rendered['normal']+1.0)/2.0
                                    render_normal_vec_ssim = (GS2D_output['rend_normal']+1.0)/2.0
                                    tmp_normal_ssim = (tmp_normals+1.0)/2.0
                                    normal_ssim_loss  = torch.mean(1-ssim_loss.forward(render_normal_vec_ssim, tmp_normal_ssim.float())) # torch.sum(mesh_masks_overlap - mesh_masks_overlap*(torch.sum(render_normal_vec*tmp_normals,dim=-1)[...,None]))/torch.sum(mesh_masks_overlap)
                                    loss_dict[prefix+'GS_normal_render_ssim'] = normal_ssim_loss.item() # store raw loss
                                    loss += normal_ssim_loss * loss_conf['GS_normal_render_ssim']

                                if "GS_surf_render_ssim" in loss_keys and loss_conf["GS_surf_render_ssim"] > 0:
                                    #render_normal_vec_ssim = (F_P_mesh_rendered['normal']+1.0)/2.0
                                    render_normal_vec_ssim = (GS2D_output['surf_normal']+1.0)/2.0
                                    tmp_normal_ssim = (tmp_normals+1.0)/2.0
                                    normal_ssim_loss  = torch.mean(1-ssim_loss.forward(render_normal_vec_ssim, tmp_normal_ssim.float())) # torch.sum(mesh_masks_overlap - mesh_masks_overlap*(torch.sum(render_normal_vec*tmp_normals,dim=-1)[...,None]))/torch.sum(mesh_masks_overlap)
                                    loss_dict[prefix+'GS_surf_render_ssim'] = normal_ssim_loss.item() # store raw loss
                                    loss += normal_ssim_loss * loss_conf['GS_surf_render_ssim']

                                if "sqrt_area_offset" in loss_keys and loss_conf["sqrt_area_offset"] > 0:
                                    hand_except_mask = F_outputs['gs']['hand_except_mask']
                                    loss_val = torch.clamp( F_outputs['gs']['sqrt_area']*0.7 - F_outputs['gs']['scaling'], min=0)[:,hand_except_mask,:]
                                    diff = (loss_val**2).mean()
                                    loss_dict[prefix+'sqrt_area_offset'] = diff.item() # store raw loss
                                    loss += diff * loss_conf['sqrt_area_offset']

                                if 'use_rgb' in opt_flag and opt_flag['use_rgb']:
                                
                                    if "GS_rgb_l1" in loss_conf.keys() and loss_conf['GS_rgb_l1'] >0:
                                        rgb_l1 = l1loss(GS2D_output['render'], masked_gt_img)
                                        loss_dict[prefix+'GS_rgb_l1'] = rgb_l1.item() # store raw loss
                                        loss+=rgb_l1*loss_conf['GS_rgb_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                    if "GS_rgb_ssim" in loss_conf.keys() and loss_conf['GS_rgb_ssim'] >0:
                                        rgb_ssim = torch.mean(1- ssim_loss.forward(GS2D_output['render'], masked_gt_img))
                                        loss_dict[prefix+'GS_rgb_ssim'] = rgb_ssim.item() # store raw loss
                                        loss+=rgb_ssim*loss_conf['GS_rgb_ssim']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                    if "GS_rgb_lpips" in loss_conf.keys() and loss_conf['GS_rgb_lpips'] >0:
                                        rgb_lpips = lpips_loss(GS2D_output['render'],masked_gt_img,GS2D_output['render_alpha']).mean()
                                        loss_dict[prefix+'GS_rgb_lpips'] = rgb_lpips.item() # store raw loss
                                        loss+=rgb_lpips*loss_conf['GS_rgb_lpips']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                                    
                                    if "Mesh_GS_rgb" in loss_conf.keys() and loss_conf['Mesh_GS_rgb'] >0:
                                        rgb_l1 = l1loss(GS2D_output['render'], mesh_rgb)
                                        loss_dict[prefix+'Mesh_GS_rgb'] = rgb_l1.item() # store raw loss
                                        loss+=rgb_l1*loss_conf['Mesh_GS_rgb']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                    # GS init distillation (curriculum decay).
                                    # Teacher = mesh_rgb (detached). Student = GS render at same training pose.
                                    # Strong at iter 0 → linear decay to 0 over `GS_init_distill_decay_steps` iters.
                                    _w_init_d = float(loss_conf.get('GS_init_distill_l1', 0.0))
                                    _w_init_d_ssim = float(loss_conf.get('GS_init_distill_ssim', 0.0))
                                    if _w_init_d > 0 or _w_init_d_ssim > 0:
                                        _decay_steps = float(loss_conf.get('GS_init_distill_decay_steps', 25.0))
                                        _decay = max(0.0, 1.0 - float(k) / max(_decay_steps, 1.0))
                                        _w_l1_eff = _w_init_d * _decay
                                        _w_ssim_eff = _w_init_d_ssim * _decay
                                        if _w_l1_eff > 0 or _w_ssim_eff > 0:
                                            _mesh_target = mesh_rgb.detach()
                                            _gs_pred = GS2D_output['render']
                                            if _w_l1_eff > 0:
                                                _l1 = l1loss(_gs_pred, _mesh_target)
                                                loss_dict[prefix + 'GS_init_distill_l1'] = _l1.item()
                                                loss = loss + _w_l1_eff * _l1
                                            if _w_ssim_eff > 0:
                                                _ssim_v = torch.mean(1 - ssim_loss.forward(_gs_pred, _mesh_target))
                                                loss_dict[prefix + 'GS_init_distill_ssim'] = _ssim_v.item()
                                                loss = loss + _w_ssim_eff * _ssim_v
                                    
                                    if "Mesh_rgb_l1" in loss_conf.keys() and loss_conf['Mesh_rgb_l1'] >0:
                                        rgb_l1 = l1loss(mesh_rgb, masked_gt_img)
                                        loss_dict[prefix+'Mesh_rgb_l1'] = rgb_l1.item() # store raw loss
                                        loss+=rgb_l1*loss_conf['Mesh_rgb_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                    if "Mesh_rgb_ssim" in loss_conf.keys() and loss_conf['Mesh_rgb_ssim'] >0:
                                        rgb_ssim = torch.mean(1- ssim_loss.forward(mesh_rgb, masked_gt_img))
                                        loss_dict[prefix+'Mesh_rgb_ssim'] = rgb_ssim.item() # store raw loss
                                        loss+=rgb_ssim*loss_conf['Mesh_rgb_ssim']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                    if "Mesh_rgb_lpips" in loss_conf.keys() and loss_conf['Mesh_rgb_lpips'] >0:
                                        rgb_lpips = lpips_loss(mesh_rgb,masked_gt_img,F_P_mesh_rendered['mask']).mean()
                                        loss_dict[prefix+'Mesh_rgb_lpips'] = rgb_lpips.item() # store raw loss
                                        loss+=rgb_lpips*loss_conf['Mesh_rgb_lpips']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                # L1 regularizer on rgb_residual (Refine_RGB_MLP output, pre-sigmoid).
                                # Penalizes residual magnitude → anchors refined_rgb to albedo, prevents
                                # Refine_RGB_MLP from over-fitting visible-view shading at expense of unseen views.
                                # Active in mesh + GS stages (gated by loss_conf key).
                                if "rgb_refine_reg" in loss_conf.keys() and loss_conf['rgb_refine_reg'] > 0:
                                    _residual = F_outputs['gs'].get('rgb_residual', None)
                                    if _residual is not None:
                                        _res_loss = _residual.abs().mean()
                                        loss_dict[prefix + 'rgb_refine_reg'] = _res_loss.item()
                                        loss = loss + _res_loss * loss_conf['rgb_refine_reg']

                                if 'Mesh_head_rgb_lpips' in loss_conf.keys() and loss_conf['Mesh_head_rgb_lpips'] >0:
                                        rgb_lpips = head_lpips(head_lpips_loss,mesh_rgb,
                                                                F_P_mesh_mask,
                                                                tmp_masked_head_gt,
                                                                tmp_head_bbox,
                                                                ).mean()
                                        loss_dict[prefix+'Mesh_head_rgb_lpips'] = rgb_lpips.item() # store raw loss
                                        loss+=rgb_lpips*loss_conf['Mesh_head_rgb_lpips']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                # Mesh head RGB L1 (mirrors GS_head_rgb_l1, mesh stage version)
                                if 'Mesh_head_rgb_l1' in loss_conf.keys() and loss_conf['Mesh_head_rgb_l1'] > 0:
                                        rgb_l1 = head_l1(l1_loss, mesh_rgb,
                                                            F_P_mesh_mask,
                                                            tmp_masked_head_gt,
                                                            tmp_head_bbox,
                                                            ).mean()
                                        loss_dict[prefix + 'Mesh_head_rgb_l1'] = rgb_l1.item()
                                        loss = loss + rgb_l1 * loss_conf['Mesh_head_rgb_l1']

                                if "Mesh_GS_albedo" in loss_conf.keys() and loss_conf['Mesh_GS_albedo'] >0:
                                    rgb_l1 = l1loss(GS2D_output['render_albedo'], mesh_coarse_rgb)
                                    loss_dict[prefix+'Mesh_GS_albedo'] = rgb_l1.item() # store raw loss
                                    loss+=rgb_l1*loss_conf['Mesh_GS_albedo']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "Mesh_albedo_l1" in loss_conf.keys() and loss_conf['Mesh_albedo_l1'] >0:
                                    rgb_l1 = l1loss(mesh_coarse_rgb, masked_gt_img)
                                    loss_dict[prefix+'Mesh_albedo_l1'] = rgb_l1.item() # store raw loss
                                    loss+=rgb_l1*loss_conf['Mesh_albedo_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "Mesh_albedo_ssim" in loss_conf.keys() and loss_conf['Mesh_albedo_ssim'] >0:
                                    rgb_ssim = torch.mean(1- ssim_loss.forward(mesh_coarse_rgb, masked_gt_img))
                                    loss_dict[prefix+'Mesh_albedo_ssim'] = rgb_ssim.item() # store raw loss
                                    loss+=rgb_ssim*loss_conf['Mesh_albedo_ssim']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "Mesh_albedo_lpips" in loss_conf.keys() and loss_conf['Mesh_albedo_lpips'] >0:
                                    rgb_lpips = lpips_loss(mesh_coarse_rgb,masked_gt_img,F_P_mesh_rendered['mask']).mean()
                                    loss_dict[prefix+'Mesh_albedo_lpips'] = rgb_lpips.item() # store raw loss
                                    loss+=rgb_lpips*loss_conf['Mesh_albedo_lpips']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if 'GS_head_rgb_lpips' in loss_conf.keys() and loss_conf['GS_head_rgb_lpips'] >0:
                                        rgb_lpips = head_lpips(head_lpips_loss,GS2D_output['render'],
                                                                GS2D_output['render_alpha'],
                                                                tmp_masked_head_gt,
                                                                tmp_head_bbox,
                                                                ).mean()
                                        loss_dict[prefix+'GS_head_rgb_lpips'] = rgb_lpips.item() # store raw loss
                                        loss+=rgb_lpips*loss_conf['GS_head_rgb_lpips']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                                if 'GS_head_rgb_l1' in loss_conf.keys() and loss_conf['GS_head_rgb_l1'] >0:
                                        rgb_lpips = head_l1(l1_loss,GS2D_output['render'],
                                                                GS2D_output['render_alpha'],
                                                                tmp_masked_head_gt,
                                                                tmp_head_bbox,
                                                                ).mean()
                                        loss_dict[prefix+'GS_head_rgb_l1'] = rgb_lpips.item() # store raw loss
                                        loss+=rgb_lpips*loss_conf['GS_head_rgb_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']


                                if "GS_albedo_l1" in loss_conf.keys() and loss_conf['GS_albedo_l1'] >0:
                                    rgb_l1 = l1loss(GS2D_output['render_albedo'], masked_gt_img)
                                    loss_dict[prefix+'GS_albedo_l1'] = rgb_l1.item() # store raw loss
                                    loss+=rgb_l1*loss_conf['GS_albedo_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "GS_albedo_ssim" in loss_conf.keys() and loss_conf['GS_albedo_ssim'] >0:
                                    rgb_ssim = torch.mean(1- ssim_loss.forward(GS2D_output['render_albedo'], masked_gt_img))
                                    loss_dict[prefix+'GS_albedo_ssim'] = rgb_ssim.item() # store raw loss
                                    loss+=rgb_ssim*loss_conf['GS_albedo_ssim']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "GS_albedo_lpips" in loss_conf.keys() and loss_conf['GS_albedo_lpips'] >0:
                                    rgb_lpips = lpips_loss(GS2D_output['render_albedo'],masked_gt_img,GS2D_output['render_alpha']).mean()
                                    loss_dict[prefix+'GS_albedo_lpips'] = rgb_lpips.item() # store raw loss
                                    loss+=rgb_lpips*loss_conf['GS_albedo_lpips']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "global_scale_reg" in loss_conf.keys() and loss_conf['global_scale_reg'] >0:
                                    #rgb_lpips = lpips_loss(GS2D_output['render_albedo'],masked_gt_img,GS2D_output['rend_alpha'])
                                    global_scale = torch.exp(FineAvatar.global_scale)
                                    global_scale_loss = torch.mean(global_scale)
                                    loss_dict[prefix+'global_scale_reg'] = global_scale_loss.item() # store raw loss
                                    loss+=global_scale_loss*loss_conf['global_scale_reg']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                                if "global_scale_update_loss" in loss_conf.keys() and loss_conf['global_scale_update_loss'] >0:
                                    #rgb_lpips = lpips_loss(GS2D_output['render_albedo'],masked_gt_img,GS2D_output['rend_alpha'])
                                    global_scale = torch.exp(FineAvatar.global_scale)
                                    global_scale_loss = torch.mean(torch.clamp(torch.abs(global_scale-1.0)-0.2,min=0.0))
                                    loss_dict[prefix+'global_scale_update_loss'] = global_scale_loss.item() # store raw loss
                                    loss+=global_scale_loss*loss_conf['global_scale_update_loss']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                """
                                if "P_jacob_scale_smooth_reg" in loss_keys and loss_conf["P_jacob_scale_smooth_reg"] > 0:
                                    laplacian_loss_val =  F_consistency_loss(F_outputs['pose_scale']) # C_consistency_loss(C_outputs['pose_scale'])
                                    loss_dict[prefix+'P_jacob_scale_smooth_reg'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val* loss_conf["P_jacob_scale_smooth_reg"]   

                                if "P_jacob_shear_smooth_reg" in loss_keys and loss_conf["P_jacob_shear_smooth_reg"] > 0:
                                    laplacian_loss_val =F_consistency_loss(F_outputs['pose_shear'])  #C_consistency_loss(C_outputs['pose_shear']) + F_consistency_loss(F_outputs['pose_shear'])
                                    loss_dict[prefix+'P_jacob_shear_smooth_reg'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val* loss_conf["P_jacob_shear_smooth_reg"]   
                                """
                                if "Jacob_shear_l1" in loss_conf.keys() and loss_conf['Jacob_shear_l1'] >0:
                                    #rgb_lpips = lpips_loss(GS2D_output['render_albedo'],masked_gt_img,GS2D_output['rend_alpha'])
                                    shear_loss = torch.mean(torch.abs(F_outputs['pose_shear']))
                                    loss_dict[prefix+'Jacob_shear_l1'] = shear_loss.item() # store raw loss
                                    loss+=shear_loss*loss_conf['Jacob_shear_l1']  #torch.mean(v_norm_init)*loss_conf['iso_vec']

                                if "Jacob_scale_l1" in loss_conf.keys() and loss_conf['Jacob_scale_l1'] >0:
                                    # L1 on (scale - 1.0): anchor each face's scale toward identity
                                    scale_l1_loss = torch.mean(torch.abs(F_outputs['pose_scale'] - 1.0))
                                    loss_dict[prefix+'Jacob_scale_l1'] = scale_l1_loss.item()
                                    loss += scale_l1_loss * loss_conf['Jacob_scale_l1']



                                # === DEBUG: save mesh .pt at target steps for self-intersection analysis ===
                                if not hasattr(FineAvatar, '_dbg_saved_steps'):
                                    FineAvatar._dbg_saved_steps = set()
                                _target_save = {5, 6, 7, 8, 9, 10, 11, 12}
                                _cur_step = iter_idx if 'iter_idx' in dir() else -1
                                if _cur_step in _target_save and _cur_step not in FineAvatar._dbg_saved_steps:
                                    try:
                                        import os as _os
                                        _save_dir = _os.environ.get('DBG_MESH_SAVE_DIR', '/tmp/dbg_mesh')
                                        _os.makedirs(_save_dir, exist_ok=True)
                                        _fn = f'{_save_dir}/L{level}_step{_cur_step:04d}.pt'
                                        torch.save({
                                            'posed_vts': F_outputs['posed_vts'].detach().cpu(),
                                            'faces': curr_faces.detach().cpu(),
                                            'pose_scale': F_outputs['pose_scale'].detach().cpu(),
                                            'pose_shear': F_outputs['pose_shear'].detach().cpu(),
                                            'step': _cur_step,
                                        }, _fn)
                                        FineAvatar._dbg_saved_steps.add(_cur_step)
                                        print(f'[DBG] Saved mesh snapshot: {_fn}')
                                    except Exception as e:
                                        pass

                                # === DEBUG: per-step decoder output stats (always log to loss_dict) ===
                                with torch.no_grad():
                                    s_ = F_outputs['pose_scale']
                                    sh_ = F_outputs['pose_shear']
                                    loss_dict[prefix+'_dbg_scale_max'] = s_.abs().max().item()
                                    loss_dict[prefix+'_dbg_scale_std'] = s_.std().item()
                                    loss_dict[prefix+'_dbg_shear_max'] = sh_.abs().max().item()
                                    loss_dict[prefix+'_dbg_shear_std'] = sh_.std().item()
                                    # which face has max scale (track if same face spikes)
                                    s_abs = s_.abs().reshape(-1, s_.shape[-1]).max(dim=-1).values  # [B*F]
                                    loss_dict[prefix+'_dbg_face_idx_max_scale'] = int(s_abs.argmax().item() % s_.shape[-2])
                                    if 'pose_rot' in F_outputs:
                                        r_ = F_outputs['pose_rot']  # 6D rep, identity = [1,0,0,0,1,0]
                                        identity_6d = r_.new_tensor([1,0,0,0,1,0])
                                        rdev = (r_ - identity_6d).abs()
                                        loss_dict[prefix+'_dbg_rot_devmax'] = rdev.max().item()
                                        loss_dict[prefix+'_dbg_rot_devstd'] = rdev.std().item()
                                    # mesh face area stats (detect degenerate mesh)
                                    if 'posed_vts' in F_outputs and 'curr_faces' in dir():
                                        try:
                                            v_ = F_outputs['posed_vts']
                                            if v_.dim() == 3: v_ = v_[0]
                                            f_ = curr_faces.long()
                                            fv = v_[f_]
                                            e1 = fv[:, 1] - fv[:, 0]
                                            e2 = fv[:, 2] - fv[:, 0]
                                            n_ = torch.cross(e1, e2, dim=-1)
                                            face_area = 0.5 * n_.norm(dim=-1)
                                            face_normals = n_ / (n_.norm(dim=-1, keepdim=True) + 1e-12)
                                            loss_dict[prefix+'_dbg_area_min'] = face_area.min().item()
                                            loss_dict[prefix+'_dbg_area_max'] = face_area.max().item()
                                            loss_dict[prefix+'_dbg_area_med'] = face_area.median().item()
                                            # det(J) = scale[0]*scale[1]*scale[2] (approx, since rotation is proper det=1)
                                            det_J = s_[..., 0] * s_[..., 1] * s_[..., 2]
                                            loss_dict[prefix+'_dbg_detJ_neg_count'] = int((det_J < 0).sum().item())
                                            loss_dict[prefix+'_dbg_detJ_min'] = float(det_J.min().item())
                                            # Face normal flip count vs canonical mesh's face normals
                                            if hasattr(FineAvatar, '_canonical_face_normals') and FineAvatar._canonical_face_normals.get(level, None) is not None:
                                                canon_fn = FineAvatar._canonical_face_normals[level]
                                                dot = (face_normals * canon_fn).sum(dim=-1)
                                                loss_dict[prefix+'_dbg_normal_flipped'] = int((dot < 0).sum().item())
                                            else:
                                                # cache on first call: base mesh normals (identity Jacobian → canonical posed)
                                                if not hasattr(FineAvatar, '_canonical_face_normals'):
                                                    FineAvatar._canonical_face_normals = {}
                                                FineAvatar._canonical_face_normals[level] = face_normals.detach().clone()
                                                loss_dict[prefix+'_dbg_normal_flipped'] = 0
                                        except Exception:
                                            pass

                                if "Jacob_scale_lap_reg" in loss_keys and loss_conf["Jacob_scale_lap_reg"] > 0:
                                    laplacian_loss_val =  face_disp_propagation_loss(F_outputs['pose_scale'], curr_face_lap) # C_consistency_loss(C_outputs['pose_scale'])
                                    loss_dict[prefix+'Jacob_scale_lap_reg'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val* loss_conf["Jacob_scale_lap_reg"]

                                if "Jacob_arap_scale" in loss_keys and loss_conf["Jacob_arap_scale"] > 0:
                                    # ARAP: penalize scale deviation from identity (1.0)
                                    arap_scale_loss = torch.mean((F_outputs['pose_scale'] - 1.0) ** 2)
                                    loss_dict[prefix+'Jacob_arap_scale'] = arap_scale_loss.item()
                                    loss += arap_scale_loss * loss_conf["Jacob_arap_scale"]

                                if "Jacob_det_reg" in loss_keys and loss_conf["Jacob_det_reg"] > 0:
                                    # Volume preservation: det(J) = product of scales ≈ 1
                                    # Form: 'l1' = |det-1| (constant gradient, tighter anchor)
                                    #       'l2' = (det-1)² (legacy, default)
                                    s = F_outputs['pose_scale']
                                    s_prod = s[..., 0] * s[..., 1] * s[..., 2]
                                    _form = str(loss_conf.get('Jacob_det_reg_form', 'l2')).lower()
                                    if _form == 'l1':
                                        det_loss = torch.mean(torch.abs(s_prod - 1.0))
                                    else:
                                        det_loss = torch.mean((s_prod - 1.0) ** 2)
                                    loss_dict[prefix+'Jacob_det_reg'] = det_loss.item()
                                    loss += det_loss * loss_conf["Jacob_det_reg"]

                                if "Jacob_shear_lap_reg" in loss_keys and loss_conf["Jacob_shear_lap_reg"] > 0:
                                    laplacian_loss_val =  face_disp_propagation_loss(F_outputs['pose_shear'], curr_face_lap) # C_consistency_loss(C_outputs['pose_scale'])
                                    loss_dict[prefix+'Jacob_shear_lap_reg'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val* loss_conf["Jacob_shear_lap_reg"]

                                if "Jacob_arap_shear" in loss_keys and loss_conf["Jacob_arap_shear"] > 0:
                                    arap_shear_loss = torch.mean(F_outputs['pose_shear'] ** 2)
                                    loss_dict[prefix+'Jacob_arap_shear'] = arap_shear_loss.item()
                                    loss += arap_shear_loss * loss_conf["Jacob_arap_shear"]

                                # Temporal Jacobian smoothness (polar-decomp variant).
                                # Decomposed J = R · U where U = scale + shear. Penalize:
                                #   - rotation: ‖R_curr · R_prev^T - I‖²  (transformation-aware, not linear diff)
                                #   - scale:    ‖scale_curr - scale_prev‖²
                                #   - shear:    ‖shear_curr - shear_prev‖²
                                # Avoids inverse instability on degenerate faces (det J near 0 / negative).
                                # Batch layout: triplet (prev, curr, next), stride-3 groups.
                                if "Jacob_temporal_smooth" in loss_keys and loss_conf["Jacob_temporal_smooth"] > 0:
                                    try:
                                        rot_field = F_outputs.get('pose_rot', None)        # (B, F, 6)
                                        scale_field = F_outputs.get('pose_scale', None)    # (B, F, 3)
                                        shear_field = F_outputs.get('pose_shear', None)    # (B, F, 3)
                                        if (rot_field is not None and scale_field is not None and shear_field is not None
                                                and rot_field.dim() == 3 and rot_field.shape[0] >= 3 and rot_field.shape[0] % 3 == 0):
                                            G_grp = rot_field.shape[0] // 3
                                            F_count = rot_field.shape[1]
                                            R_mat = rotation_6d_to_matrix(rot_field.view(-1, 6)).view(G_grp, 3, F_count, 3, 3)
                                            S_vec = scale_field.view(G_grp, 3, F_count, 3)
                                            Sh_vec = shear_field.view(G_grp, 3, F_count, 3)
                                            R_p, R_c, R_n = R_mat[:, 0], R_mat[:, 1], R_mat[:, 2]
                                            I3 = torch.eye(3, device=R_mat.device, dtype=R_mat.dtype).expand_as(R_p)
                                            # Relative rotation: R_curr @ R_prev^T should ≈ I for smooth motion
                                            R_rel_pc = torch.matmul(R_c, R_p.transpose(-2, -1)) - I3
                                            R_rel_cn = torch.matmul(R_n, R_c.transpose(-2, -1)) - I3
                                            rot_loss = R_rel_pc.pow(2).mean() + R_rel_cn.pow(2).mean()
                                            scale_loss = (S_vec[:, 1] - S_vec[:, 0]).pow(2).mean() + (S_vec[:, 2] - S_vec[:, 1]).pow(2).mean()
                                            shear_loss = (Sh_vec[:, 1] - Sh_vec[:, 0]).pow(2).mean() + (Sh_vec[:, 2] - Sh_vec[:, 1]).pow(2).mean()
                                            temp_loss = rot_loss + scale_loss + shear_loss
                                            loss += temp_loss * loss_conf["Jacob_temporal_smooth"]
                                            loss_dict[prefix + 'Jacob_temporal_smooth'] = temp_loss.item()
                                            loss_dict[prefix + 'Jacob_temporal_smooth_rot'] = rot_loss.item()
                                            loss_dict[prefix + 'Jacob_temporal_smooth_scale'] = scale_loss.item()
                                            loss_dict[prefix + 'Jacob_temporal_smooth_shear'] = shear_loss.item()
                                    except Exception as _e:
                                        pass

                                # Stretch regularizers (A/B/C/D variants for ablation).
                                # All gated by weight > 0 — default 0 preserves baseline.
                                #   A: edge length asymmetric hinge — only penalize stretching edges
                                #   B: per-axis scale hinge — only penalize scale > 1+tol
                                #   C: Cauchy-Green tensor — ||J^T·J - I||² (rotation-invariant strain penalty)
                                #   D: SVD max — largest principal stretch > threshold
                                _w_str_A = float(loss_conf.get('stretch_edge_hinge', 0.0))
                                _w_str_B = float(loss_conf.get('stretch_scale_hinge', 0.0))
                                _w_str_C = float(loss_conf.get('stretch_cauchy_green', 0.0))
                                _w_str_D = float(loss_conf.get('stretch_svd_max', 0.0))
                                _str_tol = float(loss_conf.get('stretch_tol', 0.1))     # B threshold (1.0 + tol)
                                _str_a_tol = float(loss_conf.get('stretch_a_tol', 0.1)) # A threshold (edge ratio)
                                _str_d_tol = float(loss_conf.get('stretch_d_tol', 0.5)) # D threshold (1.0 + tol = 1.5)

                                # B: scale hinge (cheapest — pose_scale already computed)
                                if _w_str_B > 0:
                                    _s_field = F_outputs.get('pose_scale', None)
                                    if _s_field is not None:
                                        _s_overstretch = (_s_field - (1.0 + _str_tol)).clamp(min=0.0)
                                        _b_loss = _s_overstretch.pow(2).mean()
                                        loss = loss + _w_str_B * _b_loss
                                        loss_dict[prefix + 'stretch_scale_hinge'] = _b_loss.item()

                                # C: Cauchy-Green tensor (J^T J - I)^2
                                if _w_str_C > 0:
                                    _J_field = F_outputs.get('jacobian', None)
                                    if _J_field is not None and _J_field.shape[-1] == 9:
                                        _J_mat = _J_field.view(*_J_field.shape[:-1], 3, 3)
                                        _C_mat = torch.matmul(_J_mat.transpose(-2, -1), _J_mat)
                                        _I3 = torch.eye(3, device=_C_mat.device, dtype=_C_mat.dtype)
                                        _c_loss = (_C_mat - _I3).pow(2).mean()
                                        loss = loss + _w_str_C * _c_loss
                                        loss_dict[prefix + 'stretch_cauchy_green'] = _c_loss.item()

                                # D: SVD max — largest singular value of U (= scale*shear matrix)
                                if _w_str_D > 0:
                                    _s_field = F_outputs.get('pose_scale', None)
                                    _sh_field = F_outputs.get('pose_shear', None)
                                    if _s_field is not None and _sh_field is not None:
                                        # Construct U upper-triangular matrix per face
                                        _Bf, _Ff = _s_field.shape[0], _s_field.shape[1]
                                        _U = torch.zeros(_Bf, _Ff, 3, 3, device=_s_field.device, dtype=_s_field.dtype)
                                        _U[..., 0, 0] = _s_field[..., 0]
                                        _U[..., 1, 1] = _s_field[..., 1]
                                        _U[..., 2, 2] = _s_field[..., 2]
                                        _U[..., 0, 1] = _sh_field[..., 0]
                                        _U[..., 0, 2] = _sh_field[..., 1]
                                        _U[..., 1, 2] = _sh_field[..., 2]
                                        _sigma = torch.linalg.svdvals(_U)              # (B, F, 3)
                                        _sigma_max = _sigma.max(dim=-1).values         # (B, F)
                                        _d_overstretch = (_sigma_max - (1.0 + _str_d_tol)).clamp(min=0.0)
                                        _d_loss = _d_overstretch.pow(2).mean()
                                        loss = loss + _w_str_D * _d_loss
                                        loss_dict[prefix + 'stretch_svd_max'] = _d_loss.item()

                                # A: edge length hinge — needs precomputed canonical edges per level (cached lazy)
                                if _w_str_A > 0:
                                    _cache_key = f'_canon_stretch_edges_lv{level_idx}'
                                    if not hasattr(FineAvatar, _cache_key):
                                        _face_l = FineAvatar.mesh_avatars[level_idx].face  # (F, 3)
                                        _e1 = torch.stack([_face_l[:, 0], _face_l[:, 1]], dim=-1)
                                        _e2 = torch.stack([_face_l[:, 1], _face_l[:, 2]], dim=-1)
                                        _e3 = torch.stack([_face_l[:, 2], _face_l[:, 0]], dim=-1)
                                        _all_e = torch.cat([_e1, _e2, _e3], dim=0)
                                        _all_e_sorted = torch.sort(_all_e, dim=-1).values
                                        _u_e = torch.unique(_all_e_sorted, dim=0)
                                        _init_v = FineAvatar.jacobian_systems[level_idx].init_vts.detach()
                                        if _init_v.dim() == 3:
                                            _init_v = _init_v.squeeze(0)
                                        _canon_lens = (_init_v[_u_e[:, 0]] - _init_v[_u_e[:, 1]]).norm(dim=-1)
                                        setattr(FineAvatar, _cache_key, (_u_e, _canon_lens))
                                    _u_edges, _c_lens = getattr(FineAvatar, _cache_key)
                                    _v0 = F_P_davts.index_select(1, _u_edges[:, 0])
                                    _v1 = F_P_davts.index_select(1, _u_edges[:, 1])
                                    _curr_lens = (_v0 - _v1).norm(dim=-1)
                                    _ratio = _curr_lens / (_c_lens.unsqueeze(0) + 1e-8)
                                    _a_overstretch = (_ratio - (1.0 + _str_a_tol)).clamp(min=0.0)
                                    _a_loss = _a_overstretch.pow(2).mean()
                                    loss = loss + _w_str_A * _a_loss
                                    loss_dict[prefix + 'stretch_edge_hinge'] = _a_loss.item()

                                # Two-head Jacobian residual regularizers (Exp D v6).
                                # Only fire when (a) loss_conf has the key AND (b) decoder is TwoHead variant
                                # (exposes last_delta_* buffers). Existing single-head runs are unaffected.
                                _th_dec = None
                                try:
                                    _th_dec = FineAvatar.pose_base_decoders[level_idx]
                                except Exception:
                                    _th_dec = None
                                if _th_dec is not None and hasattr(_th_dec, 'last_delta_rot_6d'):
                                    if "delta_rot_l2" in loss_keys and loss_conf["delta_rot_l2"] > 0 \
                                            and _th_dec.last_delta_rot_6d is not None:
                                        d_rot = _th_dec.last_delta_rot_6d
                                        d_rot_loss = torch.mean(d_rot ** 2)
                                        loss += d_rot_loss * loss_conf["delta_rot_l2"]
                                        loss_dict[prefix + 'delta_rot_l2'] = d_rot_loss.item()
                                    if "delta_scale_l2" in loss_keys and loss_conf["delta_scale_l2"] > 0 \
                                            and _th_dec.last_delta_scale is not None:
                                        d_sc = _th_dec.last_delta_scale
                                        d_sc_loss = torch.mean(d_sc ** 2)
                                        loss += d_sc_loss * loss_conf["delta_scale_l2"]
                                        loss_dict[prefix + 'delta_scale_l2'] = d_sc_loss.item()
                                    if "delta_shear_l2" in loss_keys and loss_conf["delta_shear_l2"] > 0 \
                                            and _th_dec.last_delta_shear is not None:
                                        d_sh = _th_dec.last_delta_shear
                                        d_sh_loss = torch.mean(d_sh ** 2)
                                        loss += d_sh_loss * loss_conf["delta_shear_l2"]
                                        loss_dict[prefix + 'delta_shear_l2'] = d_sh_loss.item()

                                # Exp v18 — Pose-Lipschitz smoothness (pose_smooth).
                                # Perturb body_pose by ε ~ N(0, σ²), re-forward the decoder with
                                # detached triplane feat, penalize ||J_pert − J_orig||² / σ².
                                # Enforces Lipschitz smoothness of body_pose → J mapping so that
                                # frames with similar pose produce similar Jacobians — attacks
                                # pose-overfit that causes view-dependent deformation.
                                # Cost: ~+15% (one extra decoder forward, pose encoder + decoder).
                                if "pose_smooth" in loss_keys and loss_conf["pose_smooth"] > 0 \
                                        and F_outputs.get('jac_feat', None) is not None \
                                        and 'jacobian' in F_outputs and tmp_batch_params is not None:
                                    try:
                                        _ps_sigma_deg = float(loss_conf.get("pose_smooth_sigma_deg", 3.0))
                                        _ps_sigma = _ps_sigma_deg * np.pi / 180.0
                                        _ps_pose = {k: (v.detach() if torch.is_tensor(v) else v)
                                                    for k, v in tmp_batch_params.items()}
                                        _ps_pose = copy.deepcopy(_ps_pose)
                                        _ps_eps = torch.zeros_like(_ps_pose['full_pose'])
                                        _ps_eps[..., 3:66] = torch.randn_like(_ps_eps[..., 3:66]) * _ps_sigma
                                        _ps_pose['full_pose'] = _ps_pose['full_pose'] + _ps_eps
                                        _ps_emb = FineAvatar.pose_base_encoder(_ps_pose, face_number=1)
                                        _ps_jac_feat = F_outputs['jac_feat'].detach()
                                        _ps_B, _ps_Fc, _ = _ps_jac_feat.shape
                                        _ps_emb_ex = _ps_emb.view(_ps_B, 1, -1).expand(-1, _ps_Fc, -1)
                                        _ps_dec_in = torch.cat([_ps_jac_feat, _ps_emb_ex], dim=-1)
                                        _ps_decoder = FineAvatar.pose_base_decoders[level_idx]
                                        _ps_out = _ps_decoder(_ps_dec_in, decompose=True)
                                        _ps_J_pert = _ps_out[-1] if isinstance(_ps_out, tuple) else _ps_out
                                        _ps_J_orig = F_outputs['jacobian'].detach()  # one-sided: only J_pert is optimized toward J_orig, saves ~800MB grad memory at L1
                                        if _ps_J_orig.shape == _ps_J_pert.shape:
                                            _ps_loss = ((_ps_J_pert - _ps_J_orig) ** 2).mean() / (_ps_sigma ** 2 + 1e-8)
                                            loss_dict[prefix + 'pose_smooth'] = _ps_loss.item()
                                            loss += _ps_loss * loss_conf["pose_smooth"]
                                    except Exception as _ps_exc:
                                        if k == 0:
                                            print(f"[WARN] pose_smooth skipped at iter {k}: {_ps_exc}")

                                # Exp v18 — Pmesh_ARAP on RESIDUAL J's mesh-level deformation.
                                # Reference = posed mesh from canon-head J only (pose-free part).
                                # Penalizes edge deviation caused by the pose-conditioned residual
                                # head → visible vertices can still deform coherently, invisible
                                # vertices fall back to canon-only (pose-free) geometry — killing
                                # the supervision null-space that drives view overfitting.
                                # Cost: one extra Poisson solve per iter (factorization cached,
                                # ~5-10% overhead). No extra decoder forward.
                                if "Pmesh_arap" in loss_keys and loss_conf["Pmesh_arap"] > 0:
                                    _th_dec_arap = None
                                    try:
                                        _th_dec_arap = FineAvatar.pose_base_decoders[level_idx]
                                    except Exception:
                                        _th_dec_arap = None
                                    if _th_dec_arap is not None and hasattr(_th_dec_arap, 'build_canon_only_jacobian'):
                                        # Canon reference is detached — no grad needed for the
                                        # anchor side. Wrap in no_grad to avoid building a graph
                                        # for the Poisson solve (saves ~1.5 GB at L1).
                                        with torch.no_grad():
                                            _J_canon = _th_dec_arap.build_canon_only_jacobian()
                                            if _J_canon is not None:
                                                _sys = FineAvatar.jacobian_systems[level_idx]
                                                _V_canon_only = _sys.vertices_from_jacobians(
                                                    _J_canon.view(_J_canon.shape[0], -1, 3, 3)
                                                ) + _sys.init_da_mean
                                            else:
                                                _V_canon_only = None
                                        if _V_canon_only is not None:
                                            _cosw = float(loss_conf.get("Pmesh_arap_cos_weight", 0.5))
                                            pm_arap = pmesh_arap_loss(F_P_davts, _V_canon_only, curr_faces, cos_weight=_cosw)
                                            loss_dict[prefix + 'Pmesh_arap'] = pm_arap.item()
                                            loss += pm_arap * loss_conf["Pmesh_arap"]

                                # === Physical shading δ regularizers (gated by config) =========
                                # Triggers ONLY when:
                                #   1. loss_conf has 'use_physical_shading' = True, AND
                                #   2. FineAvatar has 'physical_shading' attribute (the subclass
                                #      Cascade_Avatar_TwoHead_Ring_BodyAug_Shading).
                                # When gate is False or attr missing, this block is a no-op
                                # — existing experiments behave identically.
                                # Read from F_outputs['gs'] (per-Gaussian dict from avatar.forward),
                                # NOT gs_out_all[level_idx] (which is the rasterized image).
                                if (loss_conf.get('use_physical_shading', False)
                                        and hasattr(FineAvatar, 'physical_shading')):
                                    try:
                                        _gs_lvl = F_outputs.get('gs', None)
                                        if _gs_lvl is not None and 'shading_delta' in _gs_lvl:
                                            _delta = _gs_lvl['shading_delta']           # (B, F, 3)
                                            # 1) δ L1 magnitude
                                            if 'shading_delta_l1' in loss_keys and loss_conf.get('shading_delta_l1', 0) > 0:
                                                _d_l1 = torch.abs(_delta).mean()
                                                loss_dict[prefix + 'shading_delta_l1'] = _d_l1.item()
                                                loss += _d_l1 * loss_conf['shading_delta_l1']
                                            # 2) T-pose anchor: extra δ_MLP forward at canonical pose
                                            if 'shading_delta_anchor' in loss_keys and loss_conf.get('shading_delta_anchor', 0) > 0:
                                                _normal = _gs_lvl.get('normal_world', None)
                                                if _normal is not None:
                                                    _B = _normal.shape[0]
                                                    _delta_canon = FineAvatar.compute_canonical_delta(_normal, _B)
                                                    _d_anchor = (_delta_canon ** 2).mean()
                                                    loss_dict[prefix + 'shading_delta_anchor'] = _d_anchor.item()
                                                    loss += _d_anchor * loss_conf['shading_delta_anchor']
                                            # 3) Diagnostic — log shading + δ stats
                                            if 'shading_scalar' in _gs_lvl:
                                                _s = _gs_lvl['shading_scalar']
                                                loss_dict[prefix + '_dbg_shading_mean'] = float(_s.mean().item())
                                                loss_dict[prefix + '_dbg_shading_std'] = float(_s.std().item())
                                                loss_dict[prefix + '_dbg_delta_absmean'] = float(torch.abs(_delta).mean().item())
                                                loss_dict[prefix + '_dbg_delta_max'] = float(torch.abs(_delta).max().item())
                                    except Exception as _shading_exc:
                                        if k == 0:
                                            print(f"[WARN] shading δ reg skipped at iter {k}: {_shading_exc}")

                                if "Jacob_face_normal_consist" in loss_keys and loss_conf["Jacob_face_normal_consist"] > 0:
                                    # Adjacent face-normal consistency (mesh-space smoothness)
                                    if not hasattr(FineAvatar, '_edge_face_pairs_cache'):
                                        FineAvatar._edge_face_pairs_cache = {}
                                    cache = FineAvatar._edge_face_pairs_cache
                                    if level not in cache:
                                        cache[level] = build_edge_face_pairs(curr_faces)
                                    edge_pairs = cache[level]
                                    if edge_pairs.shape[0] > 0:
                                        fnc_loss = face_normal_consistency_loss(F_P_base_vts[0] if F_P_base_vts.dim()==3 else F_P_base_vts, curr_faces, edge_pairs)
                                        loss_dict[prefix+'Jacob_face_normal_consist'] = fnc_loss.item()
                                        loss += fnc_loss * loss_conf["Jacob_face_normal_consist"]

                                # === Local Jacobian regularizers (world → local projection) ===
                                # All keys gate with > 0; missing or 0 weight → skip entirely.
                                _local_keys = ['Local_Jac_arap', 'Local_Jac_tilt',
                                               'Local_Jac_area_preserve',
                                               'Local_Jac_smooth',
                                               'Local_Jac_shared_edge']
                                if any(loss_conf.get(k, 0) > 0 for k in _local_keys):
                                    from scene.object.neural.local_jacobian_utils import (
                                        build_local_frame_tensors,
                                        build_shared_edge_info,
                                        j_world_to_local,
                                    )
                                    from scene.optimizer.loss_functions import (
                                        local_jac_arap_loss,
                                        local_jac_tilt_loss,
                                        local_jac_area_preserve_loss,
                                        local_jac_arap_laplacian_loss,
                                        local_jac_shared_edge_loss,
                                    )
                                    # Lazy-cache canonical-mesh geometry buffers per level.
                                    # IMPORTANT: J_world from decoder is at the SUBDIVIDED mesh
                                    # resolution (Poisson solver's mesh), NOT the base
                                    # mesh_avatars[level].face.  Use jacobian_systems[level].init_*
                                    # which holds the subdivided init_vts / init_face.
                                    if not hasattr(FineAvatar, '_local_jac_cache'):
                                        FineAvatar._local_jac_cache = {}
                                    if level_idx not in FineAvatar._local_jac_cache:
                                        _jsys = FineAvatar.jacobian_systems[level_idx]
                                        V_canon = _jsys.init_vts.detach().cpu().numpy()
                                        F_canon = _jsys.init_face.detach().cpu().numpy()
                                        dev = F_outputs['jacobian'].device
                                        _g = build_local_frame_tensors(
                                            V_canon, F_canon,
                                            device=dev,
                                            dtype=F_outputs['jacobian'].dtype)
                                        _s = build_shared_edge_info(
                                            V_canon, F_canon,
                                            device=dev,
                                            dtype=F_outputs['jacobian'].dtype)
                                        FineAvatar._local_jac_cache[level_idx] = {**_g, **_s}
                                    _lg = FineAvatar._local_jac_cache[level_idx]
                                    # J_world (B, F, 3, 3) from decoder output flattened as (B, F, 9)
                                    _Jw = F_outputs['jacobian']
                                    if _Jw.dim() == 3:
                                        _B, _F = _Jw.shape[0], _Jw.shape[1]
                                        _Jw = _Jw.view(_B, _F, 3, 3)
                                    assert _Jw.shape[1] == _lg['F_frame'].shape[0], \
                                        f"face count mismatch: J_world {_Jw.shape[1]} vs F_frame {_lg['F_frame'].shape[0]}"
                                    _J_local = j_world_to_local(_Jw, _lg['F_frame'], _lg['W'])

                                    if loss_conf.get('Local_Jac_arap', 0) > 0:
                                        _l = local_jac_arap_loss(_J_local)
                                        loss_dict[prefix+'Local_Jac_arap'] = _l.item()
                                        loss += _l * loss_conf['Local_Jac_arap']
                                    if loss_conf.get('Local_Jac_tilt', 0) > 0:
                                        _l = local_jac_tilt_loss(_J_local)
                                        loss_dict[prefix+'Local_Jac_tilt'] = _l.item()
                                        loss += _l * loss_conf['Local_Jac_tilt']
                                    if loss_conf.get('Local_Jac_area_preserve', 0) > 0:
                                        _l = local_jac_area_preserve_loss(_J_local)
                                        loss_dict[prefix+'Local_Jac_area_preserve'] = _l.item()
                                        loss += _l * loss_conf['Local_Jac_area_preserve']
                                    _face_pairs = _lg.get('face_pairs', None)
                                    if _face_pairs is not None and _face_pairs.shape[0] > 0:
                                        if loss_conf.get('Local_Jac_smooth', 0) > 0:
                                            _l = local_jac_arap_laplacian_loss(_J_local, _face_pairs)
                                            loss_dict[prefix+'Local_Jac_smooth'] = _l.item()
                                            loss += _l * loss_conf['Local_Jac_smooth']
                                        if loss_conf.get('Local_Jac_shared_edge', 0) > 0:
                                            _J_restricted = torch.einsum(
                                                "fij,bfjk->bfik", _lg['F_frame'], _J_local)
                                            _l = local_jac_shared_edge_loss(
                                                _J_restricted, _face_pairs,
                                                _lg['tangent_a'], _lg['tangent_b'])
                                            loss_dict[prefix+'Local_Jac_shared_edge'] = _l.item()
                                            loss += _l * loss_conf['Local_Jac_shared_edge']

                                if "albedo_lap_reg" in loss_keys and loss_conf["albedo_lap_reg"] > 0:
                                    laplacian_loss_val =  face_disp_propagation_loss(face_albedo, curr_face_lap) # C_consistency_loss(C_outputs['pose_scale'])
                                    loss_dict[prefix+'albedo_lap_reg'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val* loss_conf["albedo_lap_reg"]   
                                if 'use_rgb' in opt_flag and opt_flag['use_rgb']:
                                    if "rgb_lap_reg" in loss_keys and loss_conf["rgb_lap_reg"] > 0:
                                        laplacian_loss_val =  face_disp_propagation_loss(face_rgb, curr_face_lap) # C_consistency_loss(C_outputs['pose_scale'])
                                        loss_dict[prefix+'rgb_lap_reg'] = laplacian_loss_val.item() # store raw loss
                                        loss += laplacian_loss_val* loss_conf["rgb_lap_reg"]              
                                
                                if "GS_laplacian" in loss_keys and loss_conf["GS_laplacian"] > 0:
                                    laplacian_loss_val =  face_disp_propagation_loss(F_outputs['gs']['xyz'], curr_face_lap) # C_consistency_loss(C_outputs['pose_scale'])
                                    loss_dict[prefix+'GS_laplacian'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val* loss_conf["GS_laplacian"]   

                                #gs reguliarze
                                if "scale_iso_reg" in loss_keys and loss_conf["scale_iso_reg"] > 0:

                                    scale_iso_reg = (torch.abs(F_outputs['gs']['scale_u'] - F_outputs['gs']['scale_v'])).mean()
                                    loss_dict[prefix+'scale_iso_reg'] = scale_iso_reg.item() # store raw loss
                                    loss += scale_iso_reg* loss_conf["scale_iso_reg"]   
                    
                                if "scale_offset_reg" in loss_conf.keys() and loss_conf['scale_offset_reg'] >0:

                                    scale_offset_reg = torch.abs(F_outputs['gs']['scale_offset']).mean()
                                    loss_dict[prefix+'scale_offset_reg'] = scale_offset_reg.item() # store raw loss
                                    loss += scale_offset_reg* loss_conf["scale_offset_reg"]   

                                if "mean_offset_reg" in loss_conf.keys() and loss_conf['mean_offset_reg'] >0:

                                    mean_offset_reg = torch.abs(F_outputs['gs']['mean_offset']).mean()
                                    loss_dict[prefix+'mean_offset_reg'] = mean_offset_reg.item() # store raw loss
                                    loss += mean_offset_reg* loss_conf["mean_offset_reg"]   


                                if 'gs_face_norm_reg' in loss_conf.keys() and loss_conf['gs_face_norm_reg'] >0:
                                    #GS mesh get
                                    gs_norm =  torch.nn.functional.normalize(F_outputs['gs']['mean_offset']+1e-8, p=2, dim=-1)
                                    align_face_loss = 1.0 - torch.abs(torch.sum(gs_norm * F_P_damesh.norm_face, dim=-1, keepdim=True) ) 
                                    align_face_loss= align_face_loss.mean()
                                    loss_dict[prefix+'gs_face_norm_reg'] = align_face_loss.item() # store raw loss
                                    loss += align_face_loss* loss_conf["gs_face_norm_reg"]   

                                    #MEsh facenorm


                                if "semantic_mesh_offset" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_mesh_offset']
                                    loss_val = compute_semantic_base_loss(torch.abs(F_outputs['offset']),
                                                                    label_indices = avatar_labels,
                                                                    weight_config= loss_weights,
                                                                    vts=True)
                                                                    
                                    loss_dict[prefix+'semantic_mesh_offset'] = loss_val.item() # store raw loss
                                    loss += loss_val            


                                if 'use_rgb' in opt_flag and opt_flag['use_rgb']:

                                    if "semantic_rgb_lap" in loss_conf.keys():
                                        loss_weights = loss_conf['semantic_rgb_lap']
                                        laplacian_loss_val = compute_semantic_laplacian_loss(face_rgb,
                                                                        face_laplacian = curr_face_lap,
                                                                        label_indices = avatar_labels,
                                                                        weight_config= loss_weights)
                                        
                                        loss_dict[prefix+'semantic_rgb_lap'] = laplacian_loss_val.item() # store raw loss
                                        loss += laplacian_loss_val            

                                if "semantic_albedo_lap" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_albedo_lap']
                                    laplacian_loss_val = compute_semantic_laplacian_loss(face_albedo,
                                                                    face_laplacian = curr_face_lap,
                                                                    label_indices = avatar_labels,
                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_albedo_lap'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val            


                                if "semantic_mean_offset_lap" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_mean_offset_lap']
                                    laplacian_loss_val = compute_semantic_laplacian_loss(F_outputs['gs']['mean_offset'],
                                                                    face_laplacian = curr_face_lap,
                                                                    label_indices = avatar_labels,
                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_mean_offset_lap'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val            


                                if "semantic_mean_offset_norm_lap" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_mean_offset_norm_lap']
                                    laplacian_loss_val = compute_semantic_laplacian_loss(torch.norm(torch.abs(F_outputs['gs']['mean_offset']),dim=-1, keepdim=True),
                                                                    face_laplacian = curr_face_lap,
                                                                    label_indices = avatar_labels,
                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_mean_offset_norm_lap'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val            




                                if "semantic_scale_offset_lap" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_scale_offset_lap']
                                    laplacian_loss_val = compute_semantic_laplacian_loss(F_outputs['gs']['scale_offset'],
                                                                    face_laplacian = curr_face_lap,
                                                                    label_indices = avatar_labels,
                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_scale_offset_lap'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val            




                                if "semantic_scale_lap" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_scale_lap']
                                    laplacian_loss_val = compute_semantic_laplacian_loss(F_outputs['gs']['scaling'],
                                                                    face_laplacian = curr_face_lap,
                                                                    label_indices = avatar_labels,
                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_scale_lap'] = laplacian_loss_val.item() # store raw loss
                                    loss += laplacian_loss_val            



                                if "semantic_mean_offset_l1" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_mean_offset_l1']
                                    loss_ = torch.abs(F_outputs['gs']['mean_offset'])
                                    loss_val = compute_semantic_base_loss(loss_,
                                                                    label_indices = avatar_labels,

                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_mean_offset_l1'] = loss_val.item() # store raw loss
                                    loss += loss_val            

                                if "semantic_mean_offset_l2" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_mean_offset_l2']
                                    loss_ = torch.pow(F_outputs['gs']['mean_offset'],2)
                                    loss_val = compute_semantic_base_loss(loss_,
                                                                    label_indices = avatar_labels,

                                                                    weight_config= loss_weights)
                                    
                                    loss_dict[prefix+'semantic_mean_offset_l2'] = loss_val.item() # store raw loss
                                    loss += loss_val            

                                if "semantic_scale_l1" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_scale_l1']
                                    loss_ = torch.abs(F_outputs['gs']['scaling'])

                                    loss_val = compute_semantic_base_loss(loss_,
                                                                    label_indices = avatar_labels,

                                                                    weight_config= loss_weights)
                                    loss_dict[prefix+'semantic_scale_l1'] = loss_val.item() # store raw loss
                                    loss += loss_val            
                                if "semantic_scale_l2" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_scale_l2']
                                    loss_ = torch.pow(F_outputs['gs']['scaling'],2)

                                    loss_val = compute_semantic_base_loss(loss_,
                                                                    label_indices = avatar_labels,

                                                                    weight_config= loss_weights)
                                    loss_dict[prefix+'semantic_scale_l2'] = loss_val.item() # store raw loss
                                    loss += loss_val            

                                if "semantic_scale_offset_l1" in loss_conf.keys():
                                    loss_weights = loss_conf['semantic_scale_offset_l1']
                                    loss_ = torch.abs(F_outputs['gs']['scale_offset'])

                                    loss_val = compute_semantic_base_loss(loss_,
                                                                    label_indices = avatar_labels,

                                                                    weight_config= loss_weights)
                                    loss_dict[prefix+'semantic_scale_offset_l1'] = loss_val.item() # store raw loss
                                    loss += loss_val

                                # Triplane Total Variation loss for spatial smoothness
                                if "triplane_tv" in loss_keys and loss_conf["triplane_tv"] > 0:
                                    if hasattr(FineAvatar, 'jacobian_encoders') and level_idx < len(FineAvatar.jacobian_encoders):
                                        encoder = FineAvatar.jacobian_encoders[level_idx]
                                        if hasattr(encoder, 'planes'):
                                            tv_loss_val = triplane_tv_loss(encoder.planes)
                                            loss_dict[prefix+'triplane_tv'] = tv_loss_val.item()
                                            loss += tv_loss_val * loss_conf["triplane_tv"]

                                total_loss = total_loss+ loss * level_w

                        # DINOv2 projection loss (optional)
                        if dino_conf is not None and not pose_update_only:
                            dino_conf['_current_step'] = k
                            dino_loss = self._compute_dino_projection_loss(
                                FineAvatar, F_outputs_all, gs_out_all,
                                all_batch_idxs, batch_cameras, tmp_masks,
                                dino_conf, loss_dict, training_levels
                            )
                            if dino_loss is not None:
                                total_loss = total_loss + dino_loss

                        # GS rendering distillation on blended training poses
                        _active_lc = all_loss_conf.get('level_1', {})
                        _w_distill_l1 = float(_active_lc.get('GS_distill_mesh_l1', 0.0))
                        _w_distill_ssim = float(_active_lc.get('GS_distill_mesh_ssim', 0.0))
                        if (_w_distill_l1 > 0 or _w_distill_ssim > 0) and not pose_update_only and FineAvatar.use_gs_geo:
                            try:
                                N_total = batch_params['full_pose'].shape[0]
                                B_local = tmp_batch_params['full_pose'].shape[0]
                                idx1 = torch.randint(0, N_total, (B_local,), device=self.device)
                                idx2 = torch.randint(0, N_total, (B_local,), device=self.device)
                                a_min = float(_active_lc.get('GS_distill_alpha_min', 0.2))
                                a_max = float(_active_lc.get('GS_distill_alpha_max', 0.8))
                                alpha = torch.rand(B_local, device=self.device) * (a_max - a_min) + a_min

                                blend_params = {}
                                for _key, _v in batch_params.items():
                                    if _key == 'betas':
                                        blend_params[_key] = _v
                                    else:
                                        _v_dev = _v.to(self.device) if _v.device != torch.device(self.device) else _v
                                        if _v_dev.shape[0] == N_total:
                                            _p1 = _v_dev[idx1]
                                            _p2 = _v_dev[idx2]
                                            _a_view = alpha.view([B_local] + [1] * (_p1.dim() - 1))
                                            blend_params[_key] = _a_view * _p1 + (1 - _a_view) * _p2
                                        else:
                                            blend_params[_key] = _v_dev

                                if use_6d_rot and 'full_pose' in blend_params:
                                    blend_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(
                                        blend_params['full_pose']).view(B_local, -1)

                                F_blend, gs_blend_out = forward_cascadeavatar_split_use_gs(
                                    single_avatar=FineAvatar,
                                    batch_params=blend_params,
                                    cameras=batch_cameras,
                                    bg_color=bg,
                                    render_out_levels=training_levels,
                                    noise_size=None,
                                )

                                finest_lv = max(training_levels)
                                F_blend_lv = F_blend['levels'][finest_lv]
                                F_face_lv = FineAvatar.mesh_avatars[finest_lv].face
                                _curr_rgb_lv = FineAvatar.mesh_avatars[finest_lv].rgb

                                with torch.no_grad():
                                    _blend_face_albedo = F_blend_lv['gs']['rgb'].detach()
                                    _blend_face_rgb = F_blend_lv['gs']['rgb_refine'].detach()
                                    _blend_vts_albedo = face_to_vertex_color_batch(_blend_face_albedo, F_face_lv)
                                    _blend_vts_rgb = face_to_vertex_color_batch(_blend_face_rgb, F_face_lv)
                                    _blend_posed_vts_det = F_blend_lv['posed_vts'].detach()
                                    _blend_mesh = Mesh(_blend_posed_vts_det, F_face_lv,
                                                       colors=_curr_rgb_lv, device=self.device)
                                    _blend_mesh_rendered = self.mesh_renderer.render_SH_batch_w_albedo_refine_rgb(
                                        _blend_mesh, rgb=_blend_vts_albedo, refine_rgb=_blend_vts_rgb,
                                        cameras=batch_cameras, bg=bg, disp_dict=None,
                                    )
                                _blend_mesh_rgb = _blend_mesh_rendered['rgba'][..., :3].detach()
                                _blend_mesh_mask = _blend_mesh_rendered['mask'].detach()

                                _blend_gs_rgb = gs_blend_out[finest_lv]['render']
                                _blend_gs_alpha = gs_blend_out[finest_lv]['render_alpha']

                                _mask_overlap = (_blend_mesh_mask * _blend_gs_alpha.detach()).clamp(0, 1)

                                if _w_distill_l1 > 0:
                                    _per_pix = (_blend_gs_rgb - _blend_mesh_rgb).abs().sum(dim=-1, keepdim=True)
                                    _l1 = (_per_pix * _mask_overlap).sum() / (_mask_overlap.sum() * 3 + 1e-6)
                                    loss_dict['GS_distill_mesh_l1'] = _l1.item()
                                    total_loss = total_loss + _w_distill_l1 * _l1
                                if _w_distill_ssim > 0:
                                    _ssim_v = 1 - ssim_loss.forward(_blend_gs_rgb, _blend_mesh_rgb).mean()
                                    loss_dict['GS_distill_mesh_ssim'] = _ssim_v.item()
                                    total_loss = total_loss + _w_distill_ssim * _ssim_v
                            except Exception as _de:
                                if k == 0:
                                    print(f"[GS_DISTILL] WARN: distill failed: {_de}")
                                    import traceback; traceback.print_exc()

                        if max_iters>0: #and k!=0:
                            test_interval = max(max_iters//test_step, 1)
                            if ( (k%(test_interval)==0 and idx==0 ) or (k==(max_iters-1) and idx==0 )): #and k!=0 :  #and
                                # Exp v18 — free fragmented VRAM before eval-render block.
                                # Extra caches from Pmesh_arap (canon_rot_6d etc.) + pose_smooth
                                # decoder re-forward push eval rendering over 24GB otherwise.
                                torch.cuda.empty_cache()

                                with torch.no_grad():

                                    import os
                                    if len(masks) >500:
                                        train_sampling = 10

                                    else:
                                        train_sampling = 5
                                    if pose_update_only:
                                        train_sampling = 1
                                    os.makedirs(debug_folder,exist_ok=True)
                                    save_folder = os.path.join(debug_folder,str(k).zfill(4))
                                    os.makedirs(save_folder,exist_ok=True)
                                    sampled_camera= copy_cameras[::train_sampling]
                                    debug_batch_params = {}
                                    for key in batch_params.keys():
                                        debug_batch_params[key] = batch_params[key].detach()
                                    if use_6d_rot:
                                        debug_batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(debug_batch_params['full_pose'])


                                    sampled_ez_param_list =  self.batch2list(debug_batch_params)[::train_sampling]
                                    sampled_images =images[::train_sampling]


                                    sampled_masks =masks[::train_sampling]
                                    sampled_data = {}
                                    sampled_data['cameras']=sampled_camera
                                    sampled_data['test_images']=sampled_images.cpu().numpy()
                                    sampled_data['test_masks']=sampled_masks.cpu().numpy()
                                    
                                    sampled_data['smpl_params'] = sampled_ez_param_list                                    
                                    if normals is not None:
                                        sampled_normals =normals[::train_sampling]
                                        sampled_data['test_normals']=sampled_normals.cpu().numpy()

                                    else:
                                        sampled_normals=None

                                    if debug_da_camera is not None:
                                            debug_data = {}
                                            num_samples = len(sampled_images)
                                            debug_data['cameras']=[debug_da_camera[0] for i in range(num_samples)]
                                            debug_data['test_images']=sampled_images.cpu().numpy()
                                            debug_data['test_masks']=sampled_masks.cpu().numpy()
                                            if sampled_normals is not None:
                                                debug_data['test_normals']=sampled_normals.cpu().numpy()
                                            else:
                                                debug_data['test_normals'] = None
                                            debug_data['smpl_params'] = sampled_ez_param_list

                                            save_render_cascade_general_version_split_gs(FineAvatar, self.mesh_renderer,
                                                                            render_out_levels = training_levels,

                                                                save_root=os.path.join(save_folder,'fine_posed_front_canon'), canon=True,
                                                                benchmark_data=debug_data )
                                            debug_data['cameras']=[debug_da_camera[1] for i in range(num_samples)]

                                            save_render_cascade_general_version_split_gs(FineAvatar, self.mesh_renderer,
                                                                            render_out_levels = training_levels,
                                                                save_root=os.path.join(save_folder,'fine_posed_back_canon'), canon=True,
                                                                benchmark_data=debug_data )


                                    save_render_cascade_general_version_split_gs(FineAvatar,  self.mesh_renderer, benchmark_data=sampled_data,
                                                                            render_out_levels = training_levels,
                                                                        save_root=os.path.join(save_folder,'fine_train_view'),
                                                                        canon=False)

                                    # =========================================================
                                    # Chimera-pose generality viz (optional, per-scene)
                                    # Path taken from opt_conf['chimera_poses_path'] if present.
                                    # Skips silently if not configured or file missing.
                                    # =========================================================
                                    try:
                                        _chimera_path = opt_conf.get('chimera_poses_path', None) if isinstance(opt_conf, dict) else None
                                        if _chimera_path and os.path.exists(_chimera_path) and debug_da_camera is not None:
                                            from utils.geometry.chimera_pose import load_chimera_poses
                                            from utils.geometry.chimera_eval import eval_and_save_chimera
                                            _meta, _chimeras = load_chimera_poses(_chimera_path, device=self.device)
                                            _chimera_save = os.path.join(save_folder, 'chimera_view')
                                            _summary = eval_and_save_chimera(
                                                FineAvatar, self.mesh_renderer,
                                                _chimeras, debug_da_camera[0],
                                                _chimera_save,
                                            )
                                            print(f"[chimera_view] {len(_chimeras)} poses -> {_chimera_save}")
                                            print(f"[chimera_view] summary: {_summary}")
                                    except Exception as _ce:
                                        print(f"[chimera_view] skipped: {_ce}")


                                    if novel_view_dataset is not None:
                                        if novel_view_data is None:
                                            novel_view_data = novel_view_dataset.get_all_data()
                                        bg = novel_view_dataset.bgcolor_list

                                        save_render_cascade_general_version_split_gs(FineAvatar,  self.mesh_renderer, benchmark_data=novel_view_data,
                                                                            render_out_levels = training_levels, is_benchmark=True,
                                                                            save_root=os.path.join(save_folder,'novel_view'), bg=bg,
                                                                            canon=False)


                                    if novel_pose_dataset is not None:
                                        if novel_pose_data is None:
                                            novel_pose_data = novel_pose_dataset.get_all_data()
                                        bg = novel_pose_dataset.bgcolor_list

                                        benchmarking_EaxAvatar_version(FineAvatar,  self.mesh_renderer, benchmark_data=novel_pose_data,
                                                                            render_out_levels = training_levels,is_benchmark=True,
                                                                            save_root=os.path.join(save_folder,'novel_pose'), bg=bg,
                                                                            canon=False)

                                        # Render test/novel poses from front/back canonical camera
                                        # (mirrors fine_posed_front_canon / fine_posed_back_canon for TRAINING poses).
                                        if debug_da_camera is not None:
                                            try:
                                                _np_smpl = novel_pose_data.get('smpl_params', None)
                                                if _np_smpl is not None:
                                                    # smpl_params is a DICT of batched tensors; frame count comes from cameras list (or any tensor's shape[0]).
                                                    _np_cams = novel_pose_data.get('cameras', [])
                                                    _np_n = len(_np_cams) if _np_cams is not None else 0
                                                    if _np_n == 0:
                                                        # fallback: use full_pose tensor shape if cameras missing
                                                        if isinstance(_np_smpl, dict) and 'full_pose' in _np_smpl:
                                                            _np_n = _np_smpl['full_pose'].shape[0]
                                                        else:
                                                            _np_n = len(_np_smpl)
                                                    _np_front_data = {
                                                        'cameras': [debug_da_camera[0] for _ in range(_np_n)],
                                                        'test_images': novel_pose_data.get('test_images', None),
                                                        'test_masks': novel_pose_data.get('test_masks', None),
                                                        'test_normals': novel_pose_data.get('test_normals', None),
                                                        'smpl_params': _np_smpl,
                                                    }
                                                    save_render_cascade_general_version_split_gs(
                                                        FineAvatar, self.mesh_renderer,
                                                        render_out_levels=training_levels,
                                                        save_root=os.path.join(save_folder, 'novel_posed_front_canon'),
                                                        canon=True,
                                                        benchmark_data=_np_front_data,
                                                    )
                                                    _np_back_data = dict(_np_front_data)
                                                    _np_back_data['cameras'] = [debug_da_camera[1] for _ in range(_np_n)]
                                                    save_render_cascade_general_version_split_gs(
                                                        FineAvatar, self.mesh_renderer,
                                                        render_out_levels=training_levels,
                                                        save_root=os.path.join(save_folder, 'novel_posed_back_canon'),
                                                        canon=True,
                                                        benchmark_data=_np_back_data,
                                                    )
                                            except Exception as _e:
                                                print(f"[novel_posed_canon] skipped: {_e}")
                                    if pose_update_only:
                                        bg = novel_pose_dataset.bgcolor_list

                                        benchmarking_EaxAvatar_version(FineAvatar,  self.mesh_renderer, benchmark_data=sampled_data,
                                                                            render_out_levels = training_levels,is_benchmark=True,
                                                                            save_root=os.path.join(save_folder,'novel_pose'), bg=bg,
                                                                            canon=False)


                                    if extra_dataset is not None:
                                        if extra_data is None:
                                            extra_data = extra_dataset.get_all_data()
                                        save_render_cascade_general_version_split_gs(FineAvatar,  self.mesh_renderer, benchmark_data=extra_data,
                                                                            render_out_levels = training_levels,
                                                                            save_root=os.path.join(save_folder,'extra_pose'), 
                                                                            canon=False)

                                    # DINOv2 feature visualization
                                    if dino_conf is not None:
                                        try:
                                            from utils.debug.visualize_dino import visualize_dino_features
                                            import nvdiffrast.torch as dr

                                            dino_vis_dir = os.path.join(debug_folder, 'dino_debug')
                                            os.makedirs(dino_vis_dir, exist_ok=True)

                                            vis_idx = all_batch_idxs[0]  # first frame in batch
                                            vis_cam = batch_cameras[0]

                                            # GT DINOv2 feature map
                                            gt_feat = dino_conf['dino_loader'].render_dino_feature_map(
                                                dino_conf['features'], vis_idx,
                                                dino_conf['image_height'], dino_conf['image_width']
                                            ).to(self.device)

                                            # Predicted: render triplane features via nvdiffrast
                                            finest_level = training_levels[-1]
                                            if hasattr(FineAvatar, 'dino_levels') and finest_level in FineAvatar.dino_levels:
                                                encoder = FineAvatar.jacobian_encoders[finest_level]
                                                fc = getattr(FineAvatar, f'face_center_{finest_level}').to(self.device).unsqueeze(0)
                                                tri_feat = encoder(fc)  # [1, Nf, C]
                                                faces_l = FineAvatar.mesh_avatars[finest_level].face
                                                vts_feat = face_to_vertex_color_batch(tri_feat, faces_l)  # [1, V, C]

                                                vis_posed_vts = F_outputs_all['levels'][finest_level]['posed_vts'][0:1]
                                                K = vis_cam.K.to(self.device)
                                                R = vis_cam.R.to(self.device)
                                                t = vis_cam.t.to(self.device)
                                                if t.dim() == 1: t = t.unsqueeze(1)
                                                H_img = dino_conf['image_height']
                                                W_img = dino_conf['image_width']

                                                vc = (R @ vis_posed_vts[0].T + t).T.unsqueeze(0)
                                                fx, fy = K[0,0], K[1,1]
                                                cx, cy = K[0,2], K[1,2]
                                                ndc_x = (2*fx*vc[...,0]/(vc[...,2]+1e-8)+2*cx-W_img)/W_img
                                                ndc_y = (2*fy*vc[...,1]/(vc[...,2]+1e-8)+2*cy-H_img)/H_img
                                                ndc_z = vc[...,2]/(vc[...,2].max()+1e-8)
                                                vc4 = torch.stack([ndc_x,-ndc_y,ndc_z,torch.ones_like(ndc_z)],dim=-1)

                                                rast_out, _ = dr.rasterize(dino_conf['renderer'].glctx, vc4, faces_l.int(), resolution=[H_img, W_img])
                                                rendered_feat, _ = dr.interpolate(vts_feat.contiguous(), rast_out, faces_l.int())
                                                # Flip to top-down (image convention) for visualization
                                                rendered_feat = rendered_feat.flip(1)
                                                vis_mask = (rast_out[0,:,:,3] > 0).float().flip(0)

                                                # Canonical features
                                                canon_feat = tri_feat[0].detach()  # [Nf, C]
                                                canon_centers = getattr(FineAvatar, f'face_center_{finest_level}').detach()

                                                # GT mask for the visualized frame
                                                vis_gt_mask = None
                                                if tmp_masks is not None:
                                                    m = tmp_masks[0]
                                                    if m.ndim == 3:
                                                        m = m[:, :, 0]
                                                    vis_gt_mask = m.to(self.device).float()

                                                # GT image for reference
                                                vis_gt_image = None
                                                if images is not None:
                                                    vis_gt_image = images[vis_idx]

                                                visualize_dino_features(
                                                    gt_feat_map=gt_feat,
                                                    pred_feat_map=rendered_feat[0].detach(),
                                                    canon_feat=canon_feat,
                                                    canon_face_centers=canon_centers,
                                                    mask=vis_mask,
                                                    gt_mask=vis_gt_mask,
                                                    projection_layer=dino_conf['projection_layer'],
                                                    save_path=os.path.join(dino_vis_dir, f'step{k:04d}_dino_summary.png'),
                                                    gt_image=vis_gt_image,
                                                )
                                        except Exception as e:
                                            import traceback
                                            print(f"[DINO_VIS] Warning: {e}")
                                            traceback.print_exc()

                                    import gc

                                    gc.collect()
                                    torch.cuda.empty_cache()
                        if not pose_update_only:
                            optimizer.zero_grad()
                        if opt_pose:
                            pose_optimizer.zero_grad()
                        total_loss.backward(retain_graph=False)
                        # GS RGB warmup freeze — zero out RGB_MLP/Refine_RGB_MLP grads
                        # for first N iters of GS#1 stage to prevent color corruption.
                        # Activated when FineAvatar.gs_freeze_rgb_iters > 0 (set from CLI).
                        _freeze_rgb_iters = float(getattr(FineAvatar, 'gs_freeze_rgb_iters', 0.0))
                        if (_freeze_rgb_iters > 0 and FineAvatar.use_gs_geo
                                and not pose_update_only and k < _freeze_rgb_iters):
                            for _p in FineAvatar.RGB_MLP.parameters():
                                _p.grad = None
                            for _p in FineAvatar.Refine_RGB_MLP.parameters():
                                _p.grad = None
                        loss_val = total_loss.item()
                        try:
                            del output
                            del nvdiff_mesh
                            del mesh_rendered
                            del mesh_rendered_img
                            del masked_gt_img
                            del loss
                            if 'fine_gs_init_flow' in locals(): del fine_gs_init_flow
                            if 'mesh_init_flow' in locals(): del mesh_init_flow
                        except NameError:
                            pass

                            
                        if 'full_pose' in opt_conf.keys():
                            with torch.no_grad():
                                batch_idxs_tensor = torch.tensor(all_batch_idxs, device=self.device, dtype=torch.long)
                                if full_pose_update_idx is None:
                                    full_pose_update_idx = []
                                    if use_6d_rot:
                                        last_dim =6
                                    else:
                                        last_dim =3

                                    if 'hand' in pose_conf and pose_conf['hand']>0:
                                        # Joint 30~54 -> Parameter (30*3) ~ (55*3)
                                        # corresponds to indices 90 ~ 164
                                        hand_params = np.arange(30 * last_dim, 55 * last_dim)
                                        full_pose_update_idx.append(hand_params)

                                    if 'body' in pose_conf and pose_conf['body']>0:
                                        # Joint 1~21 -> Parameter (1*3) ~ (22*3)
                                        # corresponds to indices 3 ~ 65
                                        body_params = np.arange(1 * last_dim, 22 * last_dim)

                                        full_pose_update_idx.append(body_params)

                                    elif 'essential_body' in pose_conf and pose_conf['essential_body'] :
                                        body_idxs = np.arange(1, 22)
                                        except_idxs = [7, 8, 10, 11, 12, 15, 20, 21]
                                        
                                        # Get the remaining Joint indices after removing excluded ones
                                        valid_mask = ~np.isin(body_idxs, except_idxs)
                                        essential_body_idxs = body_idxs[valid_mask] 
                                        
                                        # Expand Joint indices by 3x (3 axes per Joint)
                                        # e.g. Joint [1, 2] -> [3, 4, 5, 6, 7, 8]
                                        essential_params = []
                                        for idx in essential_body_idxs:
                                            essential_params.extend([idx*last_dim, idx*last_dim+1, idx*last_dim+2])
                                        essential_params = np.array(essential_params)
                                        full_pose_update_idx.append(essential_params)

                                    if 'global_orient' in pose_conf and pose_conf['global_orient'] :
                                        # Joint 0 -> Parameter 0, 1, 2
                                        if use_6d_rot:
                                            global_params = np.array([0, 1, 2,3,4,5])

                                        else:
                                            global_params = np.array([0, 1, 2])
                                        full_pose_update_idx.append(global_params)

                                    if len(full_pose_update_idx) > 0:
                                        full_pose_update_idx = np.concatenate(full_pose_update_idx)
                                        full_pose_update_idx = np.unique(full_pose_update_idx)
                                    else:
                                        full_pose_update_idx = np.array([], dtype=int)

                                    # 2. Build the full index range (0 ~ 164)
                                    total_params = 55*last_dim # 55 joints * 3
                                    all_idxs = np.arange(total_params)

                                    # 3. Get the non-updated (Frozen) indices
                                    # Subtract the update indices from the full index set (set difference)
                                    full_pose_frozen_idx = np.setdiff1d(all_idxs, full_pose_update_idx)
                            
                                if 'hand' in pose_conf and pose_conf['hand']>0:
                                    batch_params['full_pose'].grad[batch_idxs_tensor[:,None],hand_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],hand_params] * pose_conf['hand']  # 

                                if 'body' in pose_conf and pose_conf['body']>0:
                                    batch_params['full_pose'].grad[batch_idxs_tensor[:,None],body_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],body_params] * pose_conf['body']  #

                                elif 'essential_body' in pose_conf and pose_conf['essential_body']>0 :
                                    batch_params['full_pose'].grad[batch_idxs_tensor[:,None],essential_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],essential_params] * pose_conf['essential_body']  # 
                                if 'global_orient' in pose_conf and pose_conf['global_orient']>0:
                                    batch_params['full_pose'].grad[batch_idxs_tensor[:,None],global_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],global_params] * pose_conf['global_orient']  # 
                                batch_params['full_pose'].grad[batch_idxs_tensor[:,None],full_pose_frozen_idx] =  batch_params['full_pose'].grad[batch_idxs_tensor[:,None],full_pose_frozen_idx] *0.0
                        # Gradient clipping on Jacobian decoder (scale/shear explosion prevention)
                        if opt_flag.get('jac_grad_clip', 0) > 0:
                            clip_val = opt_flag['jac_grad_clip']
                            for lev_idx, dec in enumerate(FineAvatar.pose_base_decoders):
                                if hasattr(dec, 'scale_net'):
                                    gn_s = torch.nn.utils.clip_grad_norm_(dec.scale_net.parameters(), max_norm=clip_val)
                                    loss_dict[f'_dbg_L{lev_idx}_scale_grad_norm'] = float(gn_s)
                                if hasattr(dec, 'shear_net'):
                                    gn_sh = torch.nn.utils.clip_grad_norm_(dec.shear_net.parameters(), max_norm=clip_val)
                                    loss_dict[f'_dbg_L{lev_idx}_shear_grad_norm'] = float(gn_sh)
                        else:
                            # Also log grad norms when clip is 0 (for noreg/diagnosis)
                            with torch.no_grad():
                                for lev_idx, dec in enumerate(FineAvatar.pose_base_decoders):
                                    if hasattr(dec, 'scale_net'):
                                        gn_s = sum(p.grad.norm()**2 for p in dec.scale_net.parameters() if p.grad is not None)**0.5
                                        loss_dict[f'_dbg_L{lev_idx}_scale_grad_norm'] = float(gn_s)
                                    if hasattr(dec, 'shear_net'):
                                        gn_sh = sum(p.grad.norm()**2 for p in dec.shear_net.parameters() if p.grad is not None)**0.5
                                        loss_dict[f'_dbg_L{lev_idx}_shear_grad_norm'] = float(gn_sh)

                        if not pose_update_only:
                            optimizer.step()
                        if opt_pose:
                            pose_optimizer.step()

                        # EMA tracking — smooth per-frame variance to track the trend
                        # α=0.9 (effective window ~10 steps, same scale as the previous mean_last10 analysis)
                        if not hasattr(pbar, '_ema'):
                            pbar._ema = {}
                        _ema_alpha = 0.9
                        _lv = float(loss_val)
                        pbar._ema['total_loss'] = (
                            _ema_alpha * pbar._ema.get('total_loss', _lv) + (1 - _ema_alpha) * _lv
                        )
                        _ema_items = {'total_loss_ema': pbar._ema['total_loss']}
                        for _k, _v in list(loss_dict.items()):
                            if _k.startswith('_dbg_') or _k.endswith('_ema'):
                                continue
                            try:
                                _vf = float(_v)
                            except (TypeError, ValueError):
                                continue
                            pbar._ema[_k] = (
                                _ema_alpha * pbar._ema.get(_k, _vf) + (1 - _ema_alpha) * _vf
                            )
                            _ema_items[f'{_k}_ema'] = pbar._ema[_k]
                        loss_dict.update(_ema_items)

                        # 1. Main loss goes in set_description
                        pbar.set_description(f"Loss: {loss_val:.4f}")
                        # postfix: show normal_cosine EMA of the currently active level
                        _nc_postfix = {
                            _k.replace('_P_mesh_normal_cosine', '_nc'): f'{_v:.4f}'
                            for _k, _v in pbar._ema.items()
                            if _k.endswith('P_mesh_normal_cosine')
                        }
                        pbar.set_postfix(_nc_postfix)

                        # 2. (example) once every 10 iterations,
                        #    print all losses on the line 'above' the progress bar without disturbing it.
                        #    (pbar.n is the current 0-based iteration count)
                        if pbar.n % 1 == 0  and idx==0 :
                            log_str = f"[Step {pbar.n:04d}] Loss: {loss_val:.4f} (ema {pbar._ema['total_loss']:.4f})"

                            # Append every loss_dict entry to the string
                            for loss_name, loss_value in loss_dict.items():
                                    # Show up to 4 significant digits (small values auto-switch to e-05 notation)
                                    log_str += f", {loss_name}: {loss_value:.4g}"
                            pbar.write(log_str)

                            # Save loss_dict to file for analysis
                            _save_steps = {0, 100, 200, 300, 400, 499}
                            if debug_folder is not None and pbar.n in _save_steps:
                                import json as _json
                                _loss_log = {k: float(v) for k, v in loss_dict.items()}
                                _loss_log['total_loss'] = float(loss_val)
                                _loss_path = os.path.join(debug_folder, f'loss_step{pbar.n}.json')
                                os.makedirs(debug_folder, exist_ok=True)
                                with open(_loss_path, 'w') as _f:
                                    _json.dump(_loss_log, _f, indent=2)


                    if _sch_adaptive:
                        # Adaptive path: step at outer k boundary, skip k=0 to avoid PyTorch warning.
                        if k > 0 and (k % scheduler_interval_k) == 0:
                            scheduler.step()
                            if opt_pose:
                                pose_scheduler.step()
                    else:
                        if ((k*num_iter) % scheduler_interval) == 0:
                            scheduler.step()
                            if opt_pose:
                                pose_scheduler.step()
                    # LR log (always-on, harmless, helps spot silent regressions)
                    if (k % 10 == 0 or k == max_iters - 1) and optimizer is not None:
                        try:
                            _cur_lr = optimizer.param_groups[0]['lr']
                            print(f"[LR_LOG] step={k} lr={_cur_lr:.6g} (adaptive={_sch_adaptive})")
                        except Exception:
                            pass

                with torch.no_grad():
                    torch.cuda.empty_cache()
                    B = len(masks)
                    batch_params = self.toggle_gradients(batch_params, grad=False) 
                    if use_6d_rot:
                        batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(batch_params['full_pose'])


                #checkpoint = {
                #    'optimizer_state_dict': optimizer.state_dict(),
                #    'scheduler_state_dict': scheduler.state_dict(),
                #}
                FineAvatar.set_all_grad_false()
                return batch_params,  FineAvatar, optimizer, scheduler

    def largesteps_w_rgb(self, opt_conf, loss_conf, smpl_mesh, da_verts,  fullpose_dict, batch_size=16,test_idx=0, grad_clip_hand=False, head_idxs=None, lambda_=19, step_size=3e-2, use_weight=False, SDF_volume=None,
                                   pose_conf={}, use_6d_rot=False, cameras=None, normals=None, masks=None, max_iters=100, debug_folder=None, images=None, SDF_volume_update=-1,downsampling=1.0):
        full_pose_update_idx =None
        if debug_folder is not None:
            os.makedirs(debug_folder ,exist_ok=True)
        import copy
        #edges,face_to_edge = calc_edges(faces) #E,2 F,3

        #edge_length = calc_edge_length(vertices.squeeze(),edges) #E
        if downsampling !=1.0:
            #update 
            print("Downsampling: ",downsampling)
            original_height,original_width,c = images[0].shape

            downsampling_width = original_width//downsampling
            downsampling_height = original_height//downsampling

        if images is not None:
            if downsampling >1.0:
                images = [torch.from_numpy(cv2.resize(k,(downsampling_width,downsampling_height),cv2.INTER_LANCZOS4)).unsqueeze(0)  for k in images]
            else:
                images = [torch.from_numpy(k).unsqueeze(0)  for k in images]

            images = torch.concatenate(images, dim=0)
            if torch.max(images) > 128:
                images = images/255.0
        if normals is not None:
            if downsampling >1.0:
                normals = [torch.from_numpy(resize_normal_map(k,(downsampling_height,downsampling_width))).unsqueeze(0) for k in normals]

            else:
                normals = [torch.from_numpy(k).unsqueeze(0) for k in normals]
            normals = torch.concatenate(normals, dim=0)
        
            if torch.max(normals) > 128:
                normals = normals/255.0


        if masks is not None:
            if downsampling >1.0:
                masks = [1.0*torch.from_numpy(cv2.resize(k.astype(np.float32)/255,(downsampling_width,downsampling_height),cv2.INTER_NEAREST)).unsqueeze(0).unsqueeze(-1) for k in masks]

            else:
                masks = [torch.from_numpy(k).unsqueeze(0).unsqueeze(-1) for k in masks]

            masks = torch.concatenate(masks, dim=0)
            if torch.max(masks) > 128:
                masks = masks/255.0
            for idx, mask in enumerate(masks):
                mask[mask>0.5] = 1.0
                mask[mask<=0.5]=0.0
                masks[idx] = mask


        batch_params = {}


        valid_keys = ['full_pose', 'transl',  'betas'] #, 'betas', 'bias']
        if isinstance(fullpose_dict, list):
            for key in fullpose_dict[0].keys():
                if key in valid_keys:
                    if key =='full_pose':
                        if use_6d_rot:
                            batch_params[key] = [smpl_mesh.full_pose_to_sixrot(e[key].view(fullpose_dict[key].shape[0],-1,3)).detach() for e in fullpose_dict] 
                            batch_params[key] = torch.concatenate(batch_params[key], dim=0)

                        else:
                            batch_params[key] = [e[key].detach() for e in fullpose_dict]
                            batch_params[key] = torch.concatenate(batch_params[key], dim=0)

                    else:
                        batch_params[key] = [e[key].detach() for e in fullpose_dict]
                        batch_params[key] = torch.concatenate(batch_params[key], dim=0)
        else:
            for key in fullpose_dict.keys():
                if key =='full_pose':
                    if use_6d_rot:
                        batch_params[key] = smpl_mesh.full_pose_to_sixrot(fullpose_dict['full_pose'].view(fullpose_dict['full_pose'].shape[0],-1,3))
                    else:
                        batch_params[key] = fullpose_dict[key].detach()
                    batch_params[key] = batch_params[key].view(fullpose_dict['full_pose'].shape[0],-1)

                else:
                    batch_params[key] = fullpose_dict[key].detach()

        if 'init_full_pose' in loss_conf.keys():
            init_full_poses = copy.deepcopy(batch_params['full_pose'].detach())


        copy_cameras = copy.deepcopy(cameras)
        if isinstance(copy_cameras,list):
            #update cameras
            
            camera = copy_cameras[0]

            for idx, camera in enumerate(copy_cameras):
                camera.downsampling_camera(downsampling)
                camera.image_size = int(camera.image_height), int(camera.image_width)
                copy_cameras[idx] = camera


            import copy
            camera_stack = copy.deepcopy(copy_cameras[0])
            camera_stack.K = torch.stack([cam.K for cam in copy_cameras],dim=0).to(self.device)
            camera_stack.R = torch.stack([cam.R for cam in cameras],dim=0).to(self.device)
            camera_stack.t = torch.stack([cam.t for cam in copy_cameras],dim=0).to(self.device)
            camera = camera_stack
        else:
            camera = copy_cameras[0]
        da_verts= da_verts.squeeze()

        if "iso" in loss_conf.keys():
            with torch.no_grad():
                sdf_volume = SDF_volume #signed_distance.reshape(resolution, resolution, resolution)

        if "iso_vec" in loss_conf.keys():
            with torch.no_grad():
                initial_mesh = Mesh(smpl_mesh.xyz.squeeze(),smpl_mesh.face)
                v_norm_init = initial_mesh.norm_xyz.detach().cuda()

        #nrm_opt = MeshOptimizer(da_verts.detach(), ez_mesh.face.detach(), lr=lr, edge_len_lims=[0.005, 0.001]) #[0.001, 0.01]
        #vertices, faces = nrm_opt.vertices, nrm_opt.faces

        #batchwise rendering
        from scene.optimizer.batchify import ClusteredBalancedSampler,RandomSampler
        seed=20250903
        if batch_size <6:
            sampler = RandomSampler(num_samples=len(masks),batch_size=batch_size,seed=seed)
        else:
            if use_6d_rot:
                global_orient = smpl_mesh.sixrot_to_full_pose(batch_params['full_pose'][...,:6]).detach().cpu().numpy()
            else:
                global_orient=batch_params['full_pose'][...,:3].detach().cpu().numpy()
            sampler = ClusteredBalancedSampler.initialize(global_orientations=global_orient,
                                        num_groups=6,batch_size=batch_size,
                                            seed=seed)



        #lpips_loss = LPIPS(img_size=camera.image_size)
        pbar = tqdm(range(max_iters))
        sampler_iterator = iter(sampler)
        epoch_count=0
        if len(masks)%batch_size==0:
            num_iter = len(masks)//batch_size
        else:
            num_iter = len(masks)//batch_size +1

        assert 'Mesh' in opt_conf.keys()
        with torch.no_grad():
            if batch_params['betas'].shape[0] != 1:
                batch_params['betas'] = batch_params['betas'].mean(0,keepdim=True)
        batch_params = self.toggle_gradients(batch_params, grad=True)


        scheduler_interval = 100
        opt_group_params = []
        for i, key in enumerate(opt_conf.keys()):
            if key in batch_params:
                opt_group_params.append(set_opt_group(batch_params[key], opt_conf[key]))
        decay=0.95
        num_optimize_pose = len(opt_group_params)
        if 'mesh_rgb' in opt_conf:
            rgb_optimizer, rgb_scheduler = set_adam_optimizer([set_opt_group(smpl_mesh.rgb, opt_conf['mesh_rgb'])],lr=1e-3, decay=decay)
            #opt_group_params.append(set_opt_group(smpl_mesh.rgb, opt_conf['mesh_rgb']) )

        if num_optimize_pose>0 or 'mesh_rgb' in opt_conf:
            others_optimizer, others_scheduler = set_sgd_optimizer(opt_group_params, lr=1e-3, decay=decay, momentum=0.0)

        from largesteps.parameterize import from_differential, to_differential
        from largesteps.geometry import compute_matrix
        #assume v is canonical vertices, f is faces.
        original_da_verts = da_verts.detach()
        #with torch.no_grad():
        M = compute_matrix(original_da_verts, smpl_mesh.face, lambda_)
        u = to_differential(M, original_da_verts)
        from largesteps.optimize import AdamUniform
        u.requires_grad = True
        opt = AdamUniform([u], step_size)
        smpl_mesh.T_da_inv = None
        #sampling weights
        if use_weight:
            with torch.no_grad():
                from pytorch3d.transforms import matrix_to_rotation_6d,  axis_angle_to_matrix
                import torch.nn.functional as F
                #getting full da pose.
                da_pose = smpl_mesh.get_da_fullpose()

                #da_pose = ez_mesh.get_dapose_param(betas=ez_params['betas'])['poses']
                #convert26d.
                #current poses.
                sampling_pose_idxs = [0,1,2,3,4,5,8,12,13,15,16,17,18]
                if len(batch_params['full_pose'].shape)==2:
                    if use_6d_rot:
                        batch_poses = batch_params['full_pose'].detach().view(batch_params['full_pose'].shape[0],-1,6)
                    else:
                        batch_poses = batch_params['full_pose'].detach().view(batch_params['full_pose'].shape[0],-1,3)
                if not use_6d_rot:
                    batch_poses = axis_angle_to_matrix(batch_poses)
                    batch_poses = matrix_to_rotation_6d(batch_poses)
                if len(da_pose.shape)==2:
                    da_pose = da_pose.view(da_pose.shape[0],-1,3)
                da_pose = axis_angle_to_matrix(da_pose)
                da_pose = matrix_to_rotation_6d(da_pose)

                #measure similarity between poses
                distance_pose = torch.pow(batch_poses - da_pose,2)[:,sampling_pose_idxs,:]
                distance_pose = distance_pose.sum(dim=-1)
                distance_pose = distance_pose.view(distance_pose.shape[0],-1)
                feature_dim = distance_pose.shape[-1]
                scale_factor = (feature_dim ** 0.5) * 1/3
                distance_pose = -distance_pose/scale_factor#torch.exp(-torch.sum(distance_pose,dim=-1))
                distance_pose = distance_pose.sum(dim=-1)
                loss_weight = F.softmax(distance_pose, dim=0).detach()
                #cv2.imwrite('argmin.png',images[loss_weight.argmin()].detach().cpu().numpy()*255)
                #cv2.imwrite('argmax.png',images[loss_weight.argmax()].detach().cpu().numpy()*255)
                loss_weight = (loss_weight/torch.max(loss_weight)).detach()
                #other poses.weights = F.softmax(logits, dim=-1)

            #da-pose



        hand_idxs = None
        for k in pbar:
            #update SDF volume?
            for i in range(num_iter): #tqdm(range(iters)):
                if SDF_volume_update != -1:
                    if k%SDF_volume_update ==0 and i==0 and k!=0:
                        #update sdf_volume
                        with torch.no_grad():
                            v = from_differential(M, u, 'Cholesky').detach()
                            SDF_volume = GridSDFGrad(verts=v.squeeze(),faces=smpl_mesh.face,resolution=256,padding=0.2)

                try:
                    # Try to fetch the next batch from the iterator
                    batch_idxs = next(sampler_iterator)
                except StopIteration:
                    # StopIteration raised: all batches of the sampler have been consumed
                    # print(f"\n--- All batches consumed, Epoch {epoch_count} finished. Starting next epoch ---\n")
                    epoch_count += 1
                    
                    # 1. Reshuffle the data order for the new epoch
                    sampler.reshuffle(seed=epoch_count+seed)
                    
                    # 2. Create a new iterator to restart from the beginning
                    sampler_iterator = iter(sampler)
                    
                    # 3. Fetch the first batch from the new iterator
                    batch_idxs = next(sampler_iterator)

                batch_cameras = [copy_cameras[i] for i in batch_idxs]                #side_index = side_idxs - start_idx
                #if isinstance(batch_cameras,list):
                import copy
                #camera_stack = copy.deepcopy(cameras[0])
                tmp_camera_K = camera_stack.K[batch_idxs].to(self.device)
                tmp_camera_R = camera_stack.R[batch_idxs].to(self.device)
                tmp_camera_t = camera_stack.t[batch_idxs].to(self.device)

                tmp_loss_weihgt = loss_weight[batch_idxs]
                # = camera_stack
                import random
                bg = [random.random() for _ in range(3)]
                #side_index = side_index[side_index>=0]
                #side_index = side_index[side_index<batch_size]

                if normals!=None:
                    tmp_normals = normals[batch_idxs].to(self.device)
               # i#f depths!=None:
                 #   tmp_depths = depths[batch_idxs]


                #if keypoints!=None:
                #    tmp_keypoints = copy.deepcopy(keypoints[batch_idxs])
                #    tmp_keypoints[...,0] = tmp_keypoints[...,0]*10 /camera.image_width
                #    tmp_keypoints[...,1]  = tmp_keypoints[...,1]*10 /camera.image_height
                # pdb;pdb.set_trace()

                tmp_masks = masks[batch_idxs].float().to(self.device)
                if images != None:
                    tmp_images = images[batch_idxs].to(self.device)
                    bg_tensor = torch.zeros_like(tmp_images)

                    bg_tensor[...,:3] = torch.tensor(bg, device=self.device)

                    masked_gt_img = torch.where(tmp_masks.bool(), tmp_images, bg_tensor)

                tmp_batch_params = {}
                for key in batch_params.keys():
                    if key !='betas':
                        tmp_batch_params[key] = batch_params[key][batch_idxs] 
                tmp_batch_params['betas'] = batch_params['betas'] #shared in all images.
                B = tmp_masks.shape[0] #tmp_batch_params['poses'].shape[0]                    vertices, joints = MLP.ez_mesh.forward_skinning_da_verts(Namespace(**tmp_batch_params),da_verts=da_verts, T_da_inv=T_da_inv)
                if use_6d_rot:
                    tmp_batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(tmp_batch_params['full_pose']).view(B,-1)

                if not use_weight:
                    tmp_loss_weihgt = torch.ones(B).to(tmp_masks.device)

                v = from_differential(M, u, 'Cholesky')

                posed_verts = smpl_mesh.FS_fullposedict(full_pose_dict=tmp_batch_params,v_canon=v[None]) #, T_da_inv=T_da_inv)

                tmp_mesh = Mesh(vertices=posed_verts,indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)
                #norm_v =  2.0 * (v[None] - SDF_volume.center) / SDF_volume.span
                DA_mesh = Mesh(vertices=v[None],indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)

                #forward skinning ezmocap



                rendered = self.mesh_renderer.render_SH_batch(tmp_mesh,  batch_cameras,bg = bg, use_rgb=True, w_albedo=False) 
                #render_normals = rendered['normal']
                # if i < 800:
                
                #align normal 
                #addign laplacian loss.
                alpha = rendered['mask']

                masks_overlap = tmp_masks *alpha
                masks_overlap = masks_overlap.detach()
                #masks_overlap = masks_overlap.view(B,-1,1)
                

                loss = 0.0
                if "normal_all_cosine" in loss_conf.keys():
                    render_normal_vec = rendered['normal']*2.0 -1.0
                    
                    render_normal_vec = render_normal_vec/(torch.norm(render_normal_vec,dim=-1)[...,None]+1e-8)

                    render_normal_vec = render_normal_vec  * masks_overlap
                    #cv2.imwrite('test_normal.png',127.5*(render_normal_vec[0].detach().cpu().numpy()+1.0))
                    tmp_normals = tmp_normals * masks_overlap
                    normal_loss = masks_overlap - masks_overlap*torch.sum(render_normal_vec*tmp_normals,dim=-1,keepdim=True)
                    normal_loss = torch.sum(normal_loss,dim=(1,2,3))/torch.sum(masks_overlap,dim=(1,2,3))
                    loss += torch.sum(tmp_loss_weihgt*loss_conf['normal_all_cosine']*normal_loss)/torch.sum(tmp_loss_weihgt) #(torch.sum(masks_overlap - masks_overlap*(torch.sum(render_normal_vec*tmp_normals,dim=-1)[...,None]))/torch.sum(masks_overlap,dim=(1,2,3))))

                    
                    #cv2.imwrite('tmp_normals.png',127.5*(tmp_normals[0].detach().cpu().numpy()+1.0))
                #if "normal_smooth_cy" in loss_conf.keys():
                #    loss += tv_loss(rendered["normal"] * masks_overlap) * loss_conf['normal_smooth_cy']
                #    loss += mesh_edge_loss(vertices, smpl_mesh.face) * loss_conf['normal_smooth_cy']

                if "masks" in loss_conf.keys():
                    mask_nonoverlap = torch.abs(alpha- tmp_masks)
                    #mask_nonoverlap = mask_nonoverlap.view(B,-1,3)
                    mask_loss = torch.sum(mask_nonoverlap,dim=(1,2,3))/torch.sum(tmp_masks,dim=(1,2,3))

                    loss += torch.sum(tmp_loss_weihgt*loss_conf['masks']*mask_loss)/torch.sum(tmp_loss_weihgt) #torch.sum(tmp_loss_weihgt*loss_conf['masks']* torch.sum(mask_nonoverlap,dim=-1)/ torch.sum(tmp_masks,dim=-1))/torch.sum(tmp_loss_weihgt)
                
                if "rgb_l1" in loss_conf.keys():
                    loss += l1_loss(rendered['rgba'][...,:3], masked_gt_img)*loss_conf['rgb_l1']
                    #cv2.imwrite('mask_nonoverlap.png',127.5*(mask_nonoverlap[0].detach().cpu().numpy()+1.0))
                if "normal_tv" in loss_conf.keys() and loss_conf['normal_tv'] > 0:
                    # TV on normalized rendered normal ([-1,1]), foreground only
                    rn = (rendered['normal'] * 2.0 - 1.0) * masks_overlap
                    normal_tv_val = tv_loss(rn)
                    loss_dict['normal_tv'] = normal_tv_val.item()
                    loss += loss_conf['normal_tv'] * normal_tv_val

                # Sobel-based normal edge matching (preprocessing).
                # Identical to F_P stage code: pre-mask both pred/gt by overlap, conv2d Sobel, L1 diff.
                if "normal_sobel_edge_l1" in loss_conf.keys() and loss_conf['normal_sobel_edge_l1'] > 0:
                    _pred_n = (rendered['normal'] * 2.0 - 1.0)
                    _gt_n = tmp_normals
                    _mask_se = masks_overlap
                    _p = _pred_n.permute(0, 3, 1, 2)
                    _g = _gt_n.permute(0, 3, 1, 2)
                    if _mask_se.dim() == 4 and _mask_se.shape[-1] == 1:
                        _m = _mask_se.permute(0, 3, 1, 2)
                    else:
                        _m = _mask_se.unsqueeze(1) if _mask_se.dim() == 3 else _mask_se
                    _p = _p * _m
                    _g = _g * _m
                    _kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                       device=_p.device, dtype=_p.dtype).view(1, 1, 3, 3) / 8.0
                    _ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                       device=_p.device, dtype=_p.dtype).view(1, 1, 3, 3) / 8.0
                    import torch.nn.functional as _F_nn
                    _kx_3 = _kx.repeat(3, 1, 1, 1)
                    _ky_3 = _ky.repeat(3, 1, 1, 1)
                    _pgx = _F_nn.conv2d(_p, _kx_3, padding=1, groups=3)
                    _pgy = _F_nn.conv2d(_p, _ky_3, padding=1, groups=3)
                    _ggx = _F_nn.conv2d(_g, _kx_3, padding=1, groups=3)
                    _ggy = _F_nn.conv2d(_g, _ky_3, padding=1, groups=3)
                    _diff = (_pgx - _ggx).abs() + (_pgy - _ggy).abs()
                    _msum = _m.sum() * 3.0 + 1e-6
                    _se_loss = (_diff * _m).sum() / _msum
                    loss = loss + _se_loss * loss_conf['normal_sobel_edge_l1']

                # z-axis normal comparison (camera-facing component only). Camera +z is depth axis.
                if "P_normal_z_cosine" in loss_conf.keys() and loss_conf["P_normal_z_cosine"] > 0:
                    from scene.optimizer.loss_functions import masked_normal_z_cosine_loss
                    _nz_cos = masked_normal_z_cosine_loss(rendered['normal'], tmp_normals, masks_overlap, True)
                    loss = loss + _nz_cos * loss_conf['P_normal_z_cosine']
                if "P_normal_z_l1" in loss_conf.keys() and loss_conf["P_normal_z_l1"] > 0:
                    from scene.optimizer.loss_functions import masked_normal_z_l1_loss
                    _nz_l1 = masked_normal_z_l1_loss(rendered['normal'], tmp_normals, masks_overlap, True)
                    loss = loss + _nz_l1 * loss_conf['P_normal_z_l1']

                if "laplacian" in loss_conf.keys():
                    loss += laplacian_loss(tmp_mesh) * loss_conf["laplacian"]   
                if "rgb_laplacian" in loss_conf.keys():
                    loss += vertex_disp_propagation_loss(tmp_mesh, tmp_mesh.rgb) * loss_conf["rgb_laplacian"]   

                if "cotlaplacian" in loss_conf.keys() and loss_conf["cotlaplacian"] > 0:
                    loss += vertex_disp_propagation_loss(tmp_mesh, tmp_mesh.xyz,cot=True)* loss_conf["cotlaplacian"] 
                    #loss_dict['cotlaplacian'] = laplacian_loss_val.item() # store raw loss
                if "reg_l1" in loss_conf.keys() and loss_conf['reg_l1']>0:
                    loss +=  torch.mean(torch.abs(v-original_da_verts))
                if 'init_full_pose' in loss_conf.keys():
                    if use_6d_rot:
                        loss += loss_conf['init_full_pose'] *l1_loss( smpl_mesh.full_pose_to_sixrot(tmp_batch_params['full_pose']),init_full_poses[batch_idxs])
                    else:
                        loss += loss_conf['init_full_pose'] *l1_loss( tmp_batch_params['full_pose'],init_full_poses[batch_idxs])

                #loss += mesh_edge_loss(vertices, faces) * 0.1
                """
                if "iso" in loss_conf.keys() and loss_conf['iso'] >0:
                    sdf_values = sdf_volume(v.squeeze()) 
                    avg_sdf_values = sdf_values.mean(dim=0)
                    vts_mean = v.mean(dim=-1)
                    sdf_values_grad_trick = vts_mean-(vts_mean-sdf_values).detach()
                    loss+= torch.mean(torch.abs(sdf_values_grad_trick -avg_sdf_values))*loss_conf["iso"]
                
                if "iso_vec" in loss_conf.keys() and loss_conf['iso_vec'] >0:
                    #update vec
                    v_diff = v-original_da_verts
                    #v_diff_mask = torch.norm(v_diff,dim=-1) >1e-6
                    #if torch.sum(v_diff_mask)>0:
                    # sampled_v_diff = v_diff[v_diff_mask]
                    #update_vec = torch.nn.functional.normalize(v_diff)#/torch.norm(v_diff,dim=-1)
                    v_norm_dot = torch.sum(v_diff*v_norm_init,dim=-1,keepdim=True)
                    #v_norm_init = torch.abs(1-torch.abs(v_norm_dot))
                    
                    loss+=v_nor m_dot.var()*loss_conf['iso_vec']  #torch.mean(v_norm_init)*loss_conf['iso_vec']
                """

                if "iso_sdf_vts_norm" in loss_conf.keys() and loss_conf['iso_sdf_vts_norm'] >0:

                    DA_mesh_norm = DA_mesh.norm_xyz

   

                    sdf_norm = SDF_volume(v.squeeze())
                    min_bound = SDF_volume.vts_norm_low_bound
                    per_vert_error = torch.abs(1 - torch.sum(DA_mesh_norm * sdf_norm, dim=-1))
                    masked_error = F.relu(per_vert_error - min_bound)
                    if head_idxs is not None:
                        # Create a same-sized mask filled with ones
                        mask = torch.ones_like(masked_error)
                        # Set only the head_idxs entries to 0
                        mask[:, head_idxs] = 0.0
        
                        # Multiply by 0 to nullify the error and gradient there
                        masked_error = masked_error * mask
                    loss += torch.mean(masked_error) * loss_conf['iso_sdf_vts_norm']


                if "iso_sdf_face_norm" in loss_conf.keys() and loss_conf['iso_sdf_face_norm'] >0:
                    #update vec
                    #sampling vertex normal of current da_mesh.
                    DA_mesh_norm = DA_mesh.norm_face
                    #querying 
                    face_centers = (v[DA_mesh.face,:]).mean(dim=1)
                    sdf_norm = SDF_volume(face_centers.squeeze())
                    min_bound = SDF_volume.face_norm_low_bound
                    per_face_error = torch.abs(1 - torch.sum(DA_mesh_norm*sdf_norm, dim=-1))
                    masked_error = F.relu(per_face_error - min_bound)
                    if head_idxs is not None:
                        # Create a same-sized mask filled with ones
                        mask = torch.ones_like(masked_error)
                        # Set only the head_idxs entries to 0
                        mask[:, head_idxs] = 0.0    
                        # Multiply by 0 to nullify the error and gradient there
                        masked_error = masked_error * mask

                    loss += torch.mean(masked_error) * loss_conf['iso_sdf_face_norm']  
                opt.zero_grad()
                if num_optimize_pose>0:
                    others_optimizer.zero_grad()
                if 'mesh_rgb' in opt_conf:
                    rgb_optimizer.zero_grad()
                loss.backward()
                if grad_clip_hand:
                    with torch.no_grad():

                        if hand_idxs is None:
                            import copy
                            lbs_squeeze = copy.deepcopy(smpl_mesh._lbs_weights).squeeze()

                            hand_mask = torch.zeros_like(lbs_squeeze)[...,0]
                            hand_index = [20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
                            for hand_idx in hand_index:
                                hand_mask += torch.argmax(lbs_squeeze,dim=-1) == hand_idx
                            hand_idxs = torch.where( hand_mask)[0]

                        u.grad[hand_idxs,:] = 0

                #if not 'hand' in opt_conf.keys():
                #    full_pose_dict
                opt.step() 

                if 'full_pose' in opt_conf.keys():
                    with torch.no_grad():
                        batch_idxs_tensor = torch.tensor(batch_idxs, device=self.device, dtype=torch.long)
                        if full_pose_update_idx is None:
                            full_pose_update_idx = []
                            if use_6d_rot:
                                last_dim =6
                            else:
                                last_dim =3

                            if 'hand' in pose_conf and pose_conf['hand']>0:
                                # Joint 30~54 -> Parameter (30*3) ~ (55*3)
                                # corresponds to indices 90 ~ 164
                                hand_params = np.arange(25 * last_dim, 55 * last_dim)
                                full_pose_update_idx.append(hand_params)

                            if 'body' in pose_conf and pose_conf['body']>0:
                                # Joint 1~21 -> Parameter (1*3) ~ (22*3)
                                # corresponds to indices 3 ~ 65
                                body_params = np.arange(1 * last_dim, 22 * last_dim)

                                full_pose_update_idx.append(body_params)

                            if 'eye' in pose_conf and pose_conf['eye']>0:
                                body_params = np.arange(23 * last_dim, 25 * last_dim)
                                full_pose_update_idx.append(body_params)

                            if 'jaw' in pose_conf and pose_conf['jaw']>0:
                                body_params = np.arange(22 * last_dim, 23 * last_dim)

                                full_pose_update_idx.append(body_params)

                            elif 'essential_body' in pose_conf and pose_conf['essential_body'] :
                                body_idxs = np.arange(1, 22)
                                except_idxs = [7, 8, 10, 11, 12, 15, 20, 21]
                                
                                # Get the remaining Joint indices after removing excluded ones
                                valid_mask = ~np.isin(body_idxs, except_idxs)
                                essential_body_idxs = body_idxs[valid_mask] 
                                
                                # Expand Joint indices by 3x (3 axes per Joint)
                                # e.g. Joint [1, 2] -> [3, 4, 5, 6, 7, 8]
                                essential_params = []
                                for idx in essential_body_idxs:
                                    essential_params.extend([idx*last_dim, idx*last_dim+1, idx*last_dim+2])
                                essential_params = np.array(essential_params)
                                full_pose_update_idx.append(essential_params)

                            if 'global_orient' in pose_conf and pose_conf['global_orient'] :
                                # Joint 0 -> Parameter 0, 1, 2
                                if use_6d_rot:
                                    global_params = np.array([0, 1, 2,3,4,5])

                                else:
                                    global_params = np.array([0, 1, 2])
                                full_pose_update_idx.append(global_params)

                            if len(full_pose_update_idx) > 0:
                                full_pose_update_idx = np.concatenate(full_pose_update_idx)
                                full_pose_update_idx = np.unique(full_pose_update_idx)
                            else:
                                full_pose_update_idx = np.array([], dtype=int)

                            # 2. Build the full index range (0 ~ 164)
                            total_params = 55*last_dim # 55 joints * 3
                            all_idxs = np.arange(total_params)

                            # 3. Get the non-updated (Frozen) indices
                            # Subtract the update indices from the full index set (set difference)
                            full_pose_frozen_idx = np.setdiff1d(all_idxs, full_pose_update_idx)
                    
                        if 'hand' in pose_conf and pose_conf['hand']>0:
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],hand_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],hand_params] * pose_conf['hand']  # 

                        if 'body' in pose_conf and pose_conf['body']>0:
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],body_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],body_params] * pose_conf['body']  #

                        elif 'essential_body' in pose_conf and pose_conf['essential_body']>0 :
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],essential_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],essential_params] * pose_conf['essential_body']  # 
                        if 'global_orient' in pose_conf and pose_conf['global_orient']>0:
                            batch_params['full_pose'].grad[batch_idxs_tensor[:,None],global_params] = batch_params['full_pose'].grad[batch_idxs_tensor[:,None],global_params] * pose_conf['global_orient']  # 
                        batch_params['full_pose'].grad[batch_idxs_tensor[:,None],full_pose_frozen_idx] =  batch_params['full_pose'].grad[batch_idxs_tensor[:,None],full_pose_frozen_idx] *0.0



                if num_optimize_pose>0:
                    others_optimizer.step()
                    if (k % scheduler_interval) == 0:
                        others_scheduler.step()

                if 'mesh_rgb' in opt_conf:
                    rgb_optimizer.step()
                    if (k % scheduler_interval) == 0:
                        rgb_scheduler.step()




                pbar.set_description(f"Loss: {loss.item():.8f}")

 

                with torch.no_grad():
                    _preproc_save_every = int(os.environ.get('PREPROC_SAVE_EVERY', 200))
                    if debug_folder is not None and k%_preproc_save_every==0 and i==0 or (k==(max_iters-1) and i==0):
                        vertices = from_differential(M, u, 'Cholesky')
                        debug_batch_params = {}
                        for key in batch_params.keys():
                            debug_batch_params[key] = batch_params[key].detach()

                        if use_6d_rot:
                            debug_batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(debug_batch_params['full_pose'])

                        os.makedirs(f'{debug_folder}/overlay_normal', exist_ok=True)
                        os.makedirs(f'{debug_folder}/normals', exist_ok=True)
                        os.makedirs(f'{debug_folder}/smpl', exist_ok=True)
                        os.makedirs(f'{debug_folder}/normals/{str(k).zfill(4)}', exist_ok=True)
                        os.makedirs(f'{debug_folder}/rgb/{str(k).zfill(4)}', exist_ok=True)

                        batch_param_list= self.batch2list(debug_batch_params)

                        for render_idx in range(len(masks)):
                            batch_param_test = batch_param_list[render_idx]
                            posed_verts_save = smpl_mesh.FS_fullposedict(batch_param_test,v_canon=v[None]) #, T_da_inv=T_da_inv)
                            save_mesh = Mesh(vertices=posed_verts_save,indices=smpl_mesh.face,colors=smpl_mesh.rgb[None,...],device=smpl_mesh.device)
                            rendered_save = self.mesh_renderer.render_SH_batch(save_mesh, [copy_cameras[render_idx]], use_rgb=True,bg=bg) 

                            cv2.imwrite(f'{debug_folder}/normals/{str(k).zfill(4)}/{render_idx:04d}.png',(rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
                            cv2.imwrite(f'{debug_folder}/rgb/{str(k).zfill(4)}/{render_idx:04d}.png',(rendered_save['rgba'].detach()[0,:,:,:3]).cpu().numpy()*255)

                            #save da-verts and posed verts.
                            if images is not None:
                                os.makedirs(f'{debug_folder}/overlay_normal/{str(k).zfill(4)}', exist_ok=True)
                                cv2.imwrite(f'{debug_folder}/overlay_normal/{str(k).zfill(4)}/{render_idx:04d}.png',((rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy()/2 + images[render_idx].cpu().numpy()*127.5))

                        os.makedirs(f'{debug_folder}/posed_mesh', exist_ok=True)

                        os.makedirs(f'{debug_folder}/da_mesh', exist_ok=True)

                        batch_param_test = batch_param_list[test_idx]
                        posed_verts_save = smpl_mesh.FS_fullposedict(batch_param_test,v_canon=v[None]) #, T_da_inv=T_da_inv)
                        save_mesh = Mesh(vertices=posed_verts_save.squeeze(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)
                        save_mesh.to_trimesh().export(f'{debug_folder}/posed_mesh/{str(k).zfill(5)}_posed.ply')
                        Mesh(vertices=vertices.detach(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device).to_trimesh().export(f'{debug_folder}/da_mesh/{str(k).zfill(5)}_daverts.ply')
                        from utils.geometry.avatar_utils import save_render_smplmesh_params

                        save_render_smplmesh_params( smpl_mesh, self.mesh_renderer, camera=copy_cameras,
                                                                        save_root=os.path.join(debug_folder,'smpl',str(k).zfill(5)),
                                                                        smpl_param_batch= smpl_mesh.full_pose_dict2smplx(debug_batch_params),
                                                                        images=images)
                        #overlay version.


                    #os.makedirs(f'{debug_folder}/rgba', exist_ok=True)
                    #cv2.imwrite(f'{debug_folder}/rgba/{k:04d}.png',(rendered['rgba'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())

                torch.cuda.empty_cache() 
        os.makedirs(f'{debug_folder}/render_all_normals', exist_ok=True)
        os.makedirs(f'{debug_folder}/overlay_all_normal', exist_ok=True)
        if use_6d_rot:
            batch_params['full_pose'] = smpl_mesh.sixrot_to_full_pose(batch_params['full_pose'])

        batch_param_list= smpl_mesh.batch2list(batch_params)
        vertices = from_differential(M, u, 'Cholesky')
        vertices = vertices.detach()
        faces = smpl_mesh.face
        for i in range(len(masks)):
            batch_param_test = batch_param_list[i]
            posed_verts_save = smpl_mesh.FS_fullposedict(batch_param_test,v_canon=v[None])  #smpl_mesh.FS_fullposedict(Namespace(**batch_param_test),da_verts=vertices[None]) #, T_da_inv=T_da_inv)

            #posed_verts_save, joints = smpl_mesh.forward_skinning_da_verts(Namespace(**batch_param_test),da_verts=vertices[None], T_da_inv=T_da_inv)
            save_mesh = Mesh(vertices=posed_verts_save.squeeze(),indices=smpl_mesh.face,colors=smpl_mesh.rgb,device=smpl_mesh.device)

            rendered_save = self.mesh_renderer.render_SH_batch_w_depth(save_mesh, [copy_cameras[i]], w_albedo=True) 


            cv2.imwrite(f'{debug_folder}/render_all_normals/{i:04d}.png',(rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())
            cv2.imwrite(f'{debug_folder}/overlay_all_normal/{i:04d}.png',((rendered_save['normal'].detach()[0,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy()/2 + images[i].cpu().numpy()*127.5))


        with torch.no_grad():
            torch.cuda.empty_cache()
            B = len(masks)
            batch_params = self.toggle_gradients(batch_params, grad=False) 
            smpl_mesh.rgb.requires_grad =False
        return vertices, faces, batch_params, smpl_mesh

    def _compute_dino_projection_loss(self, FineAvatar, F_outputs_all, gs_out_all,
                                       batch_idxs, batch_cameras, masks,
                                       dino_conf, loss_dict, training_levels):
        """
        Compute DINOv2 projection loss using nvdiffrast rendering.

        dino_conf should contain:
            - 'features': [N, 768, H_patch, W_patch] cached DINOv2 features (CPU)
            - 'projection_layer': nn.Linear(768, C_proj)
            - 'weight': float, loss weight
            - 'renderer': nvdiffrast renderer for feature interpolation
            - 'dino_loader': DinoFeatureLoader instance (for render_dino_feature_map)
            - 'image_height', 'image_width': int
        """
        import torch.nn.functional as nnF
        import nvdiffrast.torch as dr

        weight = dino_conf.get('weight', 1.0)
        features = dino_conf['features']         # [N, 768, Hp, Wp] on CPU
        proj_layer = dino_conf['projection_layer']  # nn.Linear(768, C) or callable (e.g. PCA)
        renderer = dino_conf['renderer']
        dino_loader = dino_conf['dino_loader']
        H = dino_conf['image_height']
        W = dino_conf['image_width']

        device = self.device
        B = len(batch_idxs)

        # Use the finest training level
        level = training_levels[-1]
        faces = FineAvatar.mesh_avatars[level].face  # [F, 3]

        # Get triplane features for current batch faces
        # F_outputs_all['levels'][level] has posed_vts, Da_vts etc.
        level_output = F_outputs_all['levels'][level]
        posed_vts = level_output['posed_vts']  # [B, V, 3]

        # Access triplane encoder directly
        if hasattr(FineAvatar, 'dino_levels') and level in FineAvatar.dino_levels:
            encoder = FineAvatar.jacobian_encoders[level]
        else:
            return None

        # Get canonical face centers and query triplane
        face_centers = getattr(FineAvatar, f'face_center_{level}').to(device)
        if face_centers.ndim == 2:
            face_centers = face_centers.unsqueeze(0).expand(B, -1, -1)  # [B, Nf, 3]

        triplane_feat = encoder(face_centers)  # [B, Nf, C_proj]

        # Convert face features to vertex features for nvdiffrast
        vts_feat = face_to_vertex_color_batch(triplane_feat, faces)  # [B, V, C_proj]

        # Render feature map using nvdiffrast at DINOv2 patch resolution (fast)
        import nvdiffrast.torch as dr_
        C_proj = vts_feat.shape[-1]

        # DINOv2 patch resolution — rasterize at this size for speed + accuracy
        Hp, Wp = features.shape[2], features.shape[3]

        # 1. Batch rasterize at patch resolution
        verts_clip_list = []
        for b_idx in range(B):
            cam = batch_cameras[b_idx]
            K = cam.K.to(device)
            R = cam.R.to(device)
            t = cam.t.to(device)
            if t.dim() == 1:
                t = t.unsqueeze(1)

            verts_cam = (R @ posed_vts[b_idx].T + t).T  # [V, 3]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x, y, z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]

            # NDC coordinates (same projection, just rendered at lower resolution)
            ndc_x = (2.0 * fx * x / (z + 1e-8) + 2.0 * cx - W) / W
            ndc_y = (2.0 * fy * y / (z + 1e-8) + 2.0 * cy - H) / H
            ndc_z = z / (z.max() + 1e-8)

            vc4 = torch.stack([ndc_x, -ndc_y, ndc_z, torch.ones_like(z)], dim=-1)
            verts_clip_list.append(vc4)

        verts_clip_batch = torch.stack(verts_clip_list, dim=0)  # [B, V, 4]

        # Rasterize at patch resolution (Hp x Wp) — ~13x faster than full resolution
        rast_out, _ = dr_.rasterize(renderer.glctx, verts_clip_batch, faces.int(), resolution=[Hp, Wp])

        # Interpolate features at patch resolution
        rendered_feat, _ = dr_.interpolate(vts_feat.contiguous(), rast_out, faces.int())
        # rendered_feat: [B, Hp, Wp, C_proj]

        # 2. Compute loss at patch resolution
        total_dino_loss = 0.0
        n_valid = 0

        for b_idx in range(B):
            frame_idx = batch_idxs[b_idx]

            # GT at patch resolution — project 768→32 directly (no upsample needed)
            feat_patch = features[frame_idx].float().to(device)  # [768, Hp, Wp]
            gt_patch_hwc = feat_patch.permute(1, 2, 0)  # [Hp, Wp, 768]
            # Flip GT to match nvdiffrast bottom-up convention
            gt_patch_hwc = gt_patch_hwc.flip(0)
            gt_projected = proj_layer(gt_patch_hwc)  # [Hp, Wp, C_proj]

            # Mask at patch resolution (rast_out is already bottom-up)
            mask = (rast_out[b_idx, :, :, 3] > 0).float()  # [Hp, Wp]

            if masks is not None:
                gt_mask = masks[b_idx, :, :, 0].to(device).float() if masks.ndim == 4 else masks[b_idx].to(device).float()
                # Downsample GT mask to patch resolution and flip to bottom-up
                gt_mask_down = nnF.interpolate(
                    gt_mask.unsqueeze(0).unsqueeze(0), size=(Hp, Wp), mode='nearest'
                )[0, 0]
                mask = mask * gt_mask_down.flip(0)

            # Loss at patch resolution
            pred_feat = rendered_feat[b_idx]  # [Hp, Wp, C_proj]
            loss_type = dino_conf.get('loss_type', 'mse')

            if loss_type == 'cosine':
                # Cosine similarity loss: 1 - cos_sim (per pixel, masked)
                pred_flat = pred_feat.reshape(-1, C_proj)  # [Hp*Wp, C]
                gt_flat = gt_projected.reshape(-1, C_proj)
                mask_flat = mask.reshape(-1)  # [Hp*Wp]
                cos_sim = nnF.cosine_similarity(pred_flat, gt_flat, dim=-1)  # [Hp*Wp]
                cos_loss = (1.0 - cos_sim) * mask_flat
                n_pixels = mask_flat.sum().clamp(min=1.0)
                batch_loss = cos_loss.sum() / n_pixels
            else:
                # MSE loss (default)
                diff = (pred_feat - gt_projected) ** 2
                diff = diff * mask.unsqueeze(-1)
                n_pixels = mask.sum().clamp(min=1.0)
                batch_loss = diff.sum() / (n_pixels * C_proj)

            total_dino_loss += batch_loss
            n_valid += 1

            # Debug visualization: save actual loss inputs for first batch item
            if b_idx == 0 and dino_conf.get('debug_dir') is not None:
                step = dino_conf.get('_current_step', -1)
                debug_interval = dino_conf.get('debug_interval', 10)
                if step >= 0 and step % debug_interval == 0:
                    try:
                        import cv2 as _cv2
                        from utils.debug.visualize_dino import pca_to_rgb
                        _ddir = dino_conf['debug_dir']
                        os.makedirs(_ddir, exist_ok=True)

                        # Flip everything back to top-down for display
                        _gt = gt_projected.detach().flip(0)  # [Hp, Wp, C_proj] top-down
                        _pred = pred_feat.detach().flip(0)   # [Hp, Wp, C_proj] top-down
                        _mask = mask.detach().flip(0)         # [Hp, Wp] top-down

                        _gt_m = _gt * _mask.unsqueeze(-1)
                        _pred_m = _pred * _mask.unsqueeze(-1)

                        # Shared PCA from GT
                        _gt_rgb, _basis = pca_to_rgb(_gt_m.cpu())
                        _pred_rgb, _ = pca_to_rgb(_pred_m.cpu(), pca_basis=_basis)

                        # Cosine similarity map
                        _cos = nnF.cosine_similarity(_gt.reshape(-1, C_proj), _pred.reshape(-1, C_proj), dim=-1)
                        _cos_map = (_cos.reshape(Hp, Wp) * _mask).flip(0).flip(0).cpu()  # already top-down after first flip
                        _cos_map = _cos.reshape(Hp, Wp).detach().flip(0).cpu()
                        _cos_map_masked = _cos_map * _mask.cpu()

                        import matplotlib; matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

                        axes[0].imshow(_mask.cpu().numpy(), cmap='gray')
                        axes[0].set_title(f'Mask (step {step})'); axes[0].axis('off')

                        axes[1].imshow(_gt_rgb.numpy())
                        axes[1].set_title('GT Projected (in loss)'); axes[1].axis('off')

                        axes[2].imshow(_pred_rgb.numpy())
                        axes[2].set_title('Rendered (in loss)'); axes[2].axis('off')

                        # Error
                        _err = (_gt_m.cpu().float() - _pred_m.cpu().float()).norm(dim=-1)
                        _err_max = _err[_err > 0].quantile(0.95) if (_err > 0).any() else 1.0
                        _err_n = (_err / (_err_max + 1e-8)).clamp(0, 1).numpy()
                        _err_cm = _cv2.applyColorMap((_err_n * 255).astype(np.uint8), _cv2.COLORMAP_JET)
                        _err_cm = _cv2.cvtColor(_err_cm, _cv2.COLOR_BGR2RGB)
                        _err_cm[_mask.cpu().numpy() == 0] = 0
                        axes[3].imshow(_err_cm)
                        axes[3].set_title('|GT - Pred| L2'); axes[3].axis('off')

                        # Cosine similarity
                        _cos_vis = _cos_map_masked.numpy()
                        axes[4].imshow(_cos_vis, cmap='RdYlGn', vmin=0, vmax=1)
                        valid_cos = _cos_vis[_mask.cpu().numpy() > 0]
                        axes[4].set_title(f'Cosine Sim (mean={valid_cos.mean():.3f})'); axes[4].axis('off')

                        fig.suptitle(f'DINO Loss Debug — Step {step}, Frame {frame_idx}', fontsize=14)
                        fig.tight_layout()
                        fig.savefig(os.path.join(_ddir, f'loss_debug_step{step:04d}.png'), dpi=120)
                        plt.close(fig)
                    except Exception as _e:
                        print(f'[DINO_LOSS_DEBUG] {_e}')

        if n_valid > 0:
            dino_loss = (total_dino_loss / n_valid) * weight
            loss_dict['dino_projection'] = dino_loss.item()
            return dino_loss

        return None

    def _compute_dino_temporal_loss(self, rendered_feat_map, batch_idxs, masks,
                                     of_cur_idxs, of_next_idxs, waft_of, dino_conf, loss_dict):
        """
        DINO temporal correspondence loss.
        Warp current frame's rendered DINO features using WAFT flow → compare with next frame's rendered features.

        Args:
            rendered_feat_map: [B, Hp, Wp, C_proj] rendered DINO features from rasterization
            batch_idxs: frame indices for each batch element
            masks: [B, H, W, 1] GT masks
            of_cur_idxs, of_next_idxs: indices for current/next frame pairs
            waft_of: [B, 2, H, W] WAFT optical flow
            dino_conf: dict with 'dino_temporal_weight'
        """
        import torch.nn.functional as nnF

        weight = dino_conf.get('dino_temporal_weight', 0.0)
        if weight <= 0 or rendered_feat_map is None or waft_of is None:
            return None

        device = self.device
        Hp, Wp = rendered_feat_map.shape[1], rendered_feat_map.shape[2]
        C = rendered_feat_map.shape[3]
        H_img = dino_conf['image_height']
        W_img = dino_conf['image_width']

        total_loss = 0.0
        n_pairs = 0

        for ci, ni in zip(of_cur_idxs, of_next_idxs):
            if ci >= rendered_feat_map.shape[0] or ni >= rendered_feat_map.shape[0]:
                continue

            feat_curr = rendered_feat_map[ci]  # [Hp, Wp, C] bottom-up
            feat_next = rendered_feat_map[ni]

            # Get flow for this pair and resize to patch resolution
            flow = waft_of[ci:ci+1].to(device)  # [1, 2, H, W]
            flow_patch = nnF.interpolate(flow, size=(Hp, Wp), mode='bilinear', align_corners=True)
            flow_patch[:, 0] *= Wp / W_img
            flow_patch[:, 1] *= Hp / H_img

            # Warp curr features to next frame
            grid_y, grid_x = torch.meshgrid(
                torch.arange(Hp, device=device, dtype=torch.float32),
                torch.arange(Wp, device=device, dtype=torch.float32), indexing='ij')
            wx = grid_x + flow_patch[0, 0]
            wy = grid_y + flow_patch[0, 1]
            wx = 2 * wx / (Wp - 1) - 1
            wy = 2 * wy / (Hp - 1) - 1
            grid = torch.stack([wx, wy], dim=-1).unsqueeze(0)  # [1, Hp, Wp, 2]

            feat_curr_chw = feat_curr.permute(2, 0, 1).unsqueeze(0)  # [1, C, Hp, Wp]
            warped = nnF.grid_sample(feat_curr_chw, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            warped = warped[0].permute(1, 2, 0)  # [Hp, Wp, C]

            # Simple mask: both frames have valid rasterization
            mask_curr = (feat_curr.norm(dim=-1) > 0.01).float()
            mask_next = (feat_next.norm(dim=-1) > 0.01).float()
            warped_mask = nnF.grid_sample(mask_curr[None, None], grid, mode='nearest',
                                           padding_mode='zeros', align_corners=True)[0, 0]
            valid = warped_mask * mask_next

            if valid.sum() < 10:
                continue

            diff = ((warped - feat_next) ** 2) * valid.unsqueeze(-1)
            pair_loss = diff.sum() / (valid.sum() * C)
            total_loss += pair_loss
            n_pairs += 1

        if n_pairs > 0:
            dino_temp_loss = (total_loss / n_pairs) * weight
            loss_dict['dino_temporal'] = dino_temp_loss.item()
            return dino_temp_loss

        return None

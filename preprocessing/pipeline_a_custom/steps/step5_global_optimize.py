import torch
import numpy as np
import glob
import os
import json

import tqdm
import os
import numpy as np
import cv2
import yaml
import json
import torch
#from scene.camera.cameraunified import UnifiedCamera
from scene.camera.cameraunified_minicam import Cam_w_minicam
from scene.data_loader import DataLoader
from scene.object.mesh import Mesh
from scene.object.body_model.mesh import SMPLMesh
from render.render_unified import UnifiedRenderer
from scene.optimizer.avatar_optimizer import AvatarOptimizer as GPOOptimizer
import glob
import torch
import numpy as np
import glob
import os
import json

#run droid slam in prompt hmr for camera calibration.
#based on samurai result
#original prompt hmr:
#1. bbox detection -> sam2
#2. cam motion estimation
#3. kpt deteciton
#4. video hmr
#5. based on gravitiy axis(Network output), change to world coordinate
#6. post processing on translation.


#ours
#1. kpts / bbox deteciotn baed on dwpose
#2. hmr in whole images + samurai using kpts
#3. cam motion estimation
#but hmr input is camera intrinsic -> previous run samurai and cam motion estimation before hmr,
#-.

#4. based on gravity axis - fit floor hegiht ( ransac used in prompt hmr)
#5. post processing based on kpts, masks, segmentation.

from preprocessing.pipeline_a_custom.utils.prompthmr_setup import load_phmr
from prompt_hmr.models.inference import prepare_batch
from pipeline.tools import est_calib
from preprocessing.pipeline_a_custom.utils.run_samurai import run_samurai
from pytorch3d.transforms import rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle

def rot6d2axis_angle(rot_6d_pose):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot_6d_pose.view(-1,6)))
def transform_smpl_params(root_orient, transl, R, t, smpl_t_pose_pelvis): #from prompthmr
    '''
    Input:
        root_orient: [B, 3, 3]
        transl: [B, 3]
        R: [B, 3, 3]
        t: [B, 3]
        smpl_t_pose_pelvis: [3]
    
    Output: 
        root_orient: [B, 3, 3]
        transl: [B, 3]
    '''
    assert smpl_t_pose_pelvis.shape == (3,)
    smpl_t_pose_pelvis = smpl_t_pose_pelvis[None, : , None]
    transl = transl.unsqueeze(-1).float()
    t = t.unsqueeze(-1)

    transl = R @ (smpl_t_pose_pelvis + transl) + t - smpl_t_pose_pelvis
    root_orient = R @ root_orient
    transl = transl.squeeze()
    return root_orient, transl

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration and set up device.")
    
    # 2. Add the --config argument
    #    also usable via the short name -c
    #    default: path used as the default value
    path2config =  os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','scarf.yaml')
    with open(path2config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda')
    Path_params = params['Path']
    rndr = UnifiedRenderer(device=device)

    data_loader = DataLoader(model_type=params['SMPL']['model_type'])
    gpo_optimizer = GPOOptimizer(mesh_renderer=rndr.mesh_render, gs_renderer=None)
        # file names
    debug = False

    phmr = load_phmr(params['SMPL']['smpl_root'])
    if params['use_hamer']:  # HaMeR hand refinement — off in the release configs (needs libs/hamer + MANO)
        from libs.hamer.hamer_wrapping import run_hamer_model
        from libs.hamer.hamer.models import load_hamer
        hamer_model, hamer_model_cfg = load_hamer('libs/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt')

    """load images/ masks"""


    data_folder = Path_params['training_folder']


    save_root = data_folder # Path_params['save_root']
    image_root = os.path.join(data_folder,'images') #Path_params['orig_image_folder']

    #slam_params = params['SLAM']
    static_cam = True #slam_params['static_cam']
    cam_int_path = None#slam_params['cam_intrinsic_path']

    skeleton_style='mmpose'
    mmpose_folder= os.path.join(save_root,'mmpose_result_'+skeleton_style)
    crop_bbox_folder = os.path.join(mmpose_folder, 'cropped_bboxes')
    keypoints_folder = os.path.join(mmpose_folder, 'keypoints_whole_body')

    samurai_mask_root = os.path.join(save_root,'masks')
    crop_mask_folder = os.path.join(save_root,'mask_crop')

    images, img_list = data_loader.load_images(image_root, return_mask=False)
    crop_bboxes= []
    if not os.path.exists(samurai_mask_root) and not os.path.exists(crop_mask_folder):
        print('run samurai')
        first_image_name = os.path.basename(img_list[0])
        masks_out = run_samurai(save_folder=samurai_mask_root,bbox_path=os.path.join(save_root,'first_bbox.txt'),image_folder=image_root)
    mask_images = sorted(glob.glob(os.path.join(samurai_mask_root,'*.png')))
    masks = []
    #crop_images_list = []
    crop_images_list = []


    os.makedirs(crop_mask_folder,exist_ok=True)
    for mask_image_path in mask_images:
        mask_name = os.path.basename(mask_image_path)
        mask = cv2.imread(mask_image_path,cv2.IMREAD_UNCHANGED)
        crop_bbox_path = os.path.join(crop_bbox_folder,mask_name.split('.')[-2]+'.txt')
        with open(crop_bbox_path, 'r') as file:
            content = file.read()
            values = [int(item.strip()) for item in content.split(',') if item.strip()]
            crop_bboxes.append(values)
        mask = mask[values[1]:values[1]+values[3], values[0]:values[0] + values[2]]
        masks.append(mask)
        cv2.imwrite(os.path.join(crop_mask_folder,mask_name),mask)
        crop_image_path = os.path.join(save_root,'image_crop',mask_name)
        crop_image = cv2.imread(os.path.join(crop_image_path))
        crop_images_list.append(crop_image)

    crop_images = np.stack(crop_images_list,axis=0)

    original_height,original_width,c = images[0].shape
    B= len(images)
    try:
        cam_int = torch.Tensor(np.load(cam_int_path))
    except:
        cam_int_list = est_calib(images)
        cam_int = Cam_w_minicam.intrinsic_from_param_each(fx=cam_int_list[0],fy=cam_int_list[1],cx=cam_int_list[2],cy=cam_int_list[3])

    if cam_int_path !=None:
        opt_intr = False
    else:
        opt_intr = True
    if static_cam:
        cameras = []
        for i in range(B):
            camera = Cam_w_minicam.perspective_camera_from_K(K=torch.Tensor(cam_int),minicam_bbox=crop_bboxes[i],width=original_width,height=original_height, znear=0.01, zfar=1000.0, device=torch.device('cuda:0') )
            cameras.append(camera)
    else:
        from pipeline.camera import run_metric_slam
        images_np = np.stack(images,axis=0)
        masks_torch = torch.Tensor(np.stack(masks,axis=0)).to(device)/255.0
        cam_R, cam_T, cam_int_out = run_metric_slam( #following prompt hmr's config file.
            images_np, 
            droid_weights='libs/PromptHMR/data/pretrain/droidcalib.pth',
            masks=masks_torch, 
            calib=cam_int_list, 
            monodepth_method='metric3d_vit_large', 
            use_depth_inp=False, #self.cfg.use_depth,
            stride=1,
            opt_intr=opt_intr,
            save_depth=False,
            keyframes=None,
        )
        cameras = []
        #check types.
        for i in range(B):
            cam_int = Cam_w_minicam.intrinsic_from_param_each(fx=cam_int_out[0],fy=cam_int_out[1],cx=cam_int_out[2],cy=cam_int_out[3])

            camera = Cam_w_minicam.perspective_camera_from_K(K=torch.Tensor(cam_int),minicam_bbox=crop_bboxes[i],width=original_width,height=original_height, R=cam_R[i],T=cam_T[i],
                                                             znear=0.01, zfar=1000.0, device=torch.device('cuda:0') )
            cameras.append(camera)


    #assign camera parameters and save the camera paramters..
    skeleton_style='mmpose'
    mmpose_folder= os.path.join(save_root,'mmpose_result_'+skeleton_style)
    crop_bbox_folder = os.path.join(mmpose_folder, 'cropped_bboxes')
    keypoints_folder = os.path.join(mmpose_folder, 'keypoints_whole_body')


    #load bbox from dwpose results
    #bboxes = 
    bbox_files = sorted(glob.glob(os.path.join(crop_bbox_folder,'*.txt')))
    #load bboxes.


    smpl_mesh = SMPLMesh(smpl_config=params['SMPL']) #132: (22,6) #need to change.
    

    kpt_path_list = sorted(glob.glob(os.path.join(keypoints_folder, '*.json')))# dwpose results.
    kpts = {}
    dw_kpts = np.zeros((len(kpt_path_list),133,3))



    for frame_idx, kpt_path in enumerate(kpt_path_list):
        with open(kpt_path) as f:
            kpt = np.array(json.load(f), dtype=np.float32)
        dw_kpts[frame_idx] =kpt
    hmr_folder = os.path.join(save_root,'hmr_results')
    os.makedirs(hmr_folder,exist_ok=True)
    if debug:
        debugging_folder = os.path.join(hmr_folder,'debugging')
        os.makedirs(debugging_folder,exist_ok=True)
        debugging_mini_folder = os.path.join(hmr_folder,'debugging_minicam')
        os.makedirs(debugging_mini_folder,exist_ok=True)

    hamer_mesh_folder = os.path.join(hmr_folder,'hamer_mesh')
    os.makedirs(hamer_mesh_folder,exist_ok=True)

    smpl_save_folder = os.path.join(hmr_folder,'smplx_params')
    os.makedirs(smpl_save_folder,exist_ok=True)

    for idx, image in enumerate(tqdm.tqdm(images)):
        bbox_file = bbox_files[idx]
        with open(bbox_file, 'r') as file:
            content = file.read()
            values = [int(item.strip()) for item in content.split(',') if item.strip()]
        bboxes_sam = torch.Tensor(values)[None]
        bboxes_sam[...,2] = bboxes_sam[...,0] + bboxes_sam[...,2] #coordinate.
        bboxes_sam[...,3] = bboxes_sam[...,1] + bboxes_sam[...,3] #coordinate.
        #bboxes_sam: x1,y1,x2,y2
        cam_int_torch = torch.Tensor(cam_int)[None]

        phmr_input= [{'image_cv': image, 'boxes': bboxes_sam, 'text': None, 'masks': None, 'cam_int':cam_int_torch }] #camera intrinsic used in prompt hmr.
        batch = prepare_batch(phmr_input, img_size=896, interaction=False)
        with torch.no_grad():
            output = phmr(batch, use_mean_hands=True)[0]    
        smpl_t_pose_pelvis = smpl_mesh.smpl_model(
                global_orient=torch.zeros(1, 3).to(device), 
        body_pose=torch.zeros(1, 21*3).to(device), 
        betas=output['betas'].detach().clone()
        ).joints[0, 0]

        keys = ['pose', 'betas', 'transl', 'rotmat', 'vertices', 'body_joints', 'cam_int']
    #output = {k:output[k].detach().cpu() for k in keys}
        smpl_param = SMPLMesh.get_empty_params(pca_dim=45, device=device)
        """
        'global_orient': torch.zeros(1, 3).to(device),
                'expression': torch.zeros(1, 10).to(device),
                'betas': torch.zeros(1, 10).to(device),
                'jaw_pose': torch.zeros(1, 3).to(device),
                'body_pose': torch.zeros(1, 63).to(device),
                'left_hand_pose': torch.zeros(1, pca_dim).to(device),
                'right_hand_pose': torch.zeros(1, pca_dim).to(device),
                'transl': torch.zeros(1, 3).to(device),
                'scale': torch.FloatTensor([[1.0]]).to(device)
        
        """
        #output pose 6d -> 3d
        cam = cameras[idx]

        current_pose = rot6d2axis_angle(output['pose'].view(-1,6)).view(1,-1)
        smpl_param['global_orient'] = current_pose[...,:3].detach().clone()
        R_smpl = axis_angle_to_matrix(smpl_param['global_orient'])
        #R_global_new=torch.bmm(cam.R.T.unsqueeze(0),axis_angle_to_matrix(smpl_param['global_orient']))
        #smpl_param['global_orient'] = matrix_to_axis_angle(R_global_new)

        smpl_param['transl'] = output['transl'].detach().clone()

        #smpl_param['transl'] = torch.bmm(cam.R.T.unsqueeze(0), smpl_param['transl'].unsqueeze(-1) )
        #smpl_param['transl'] = smpl_param['transl'].squeeze(-1)+cam.t.unsqueeze(0)


        smpl_param['body_pose'] = current_pose[...,3:].detach().clone()
        #root_orient, pred_trans = transform_smpl_params(
        #    R_smpl , smpl_param['transl'], cam.R.unsqueeze(0), cam.t.unsqueeze(0), smpl_t_pose_pelvis)

        #smpl_param['transl']  = torch.unsqueeze(pred_trans,0)
        #smpl_param['global_orient']  = matrix_to_axis_angle(root_orient)

        #smpl_param['jaw_pose'] = nlf_params['jaw_pose'].detach().clone()
        #smpl_param['left_eye_pose'] = nlf_params['left_eye_pose'].detach().clone()
        #smpl_param['right_eye_pose'] = nlf_params['right_eye_pose'].detach().clone()
        

        #run hamer -> save hand mesh.
        #os.makedirs(os.path.join)
        if params['use_hamer']:
            with torch.no_grad(): #left_result,  right_result, left_mesh, right_mesh
                left_result,  right_result, left_mesh, right_mesh = run_hamer_model(
                    image,
                    dw_kpts[idx],
                    hamer_model,
                    focal_length=cam_int[0,0],
                    model_cfg=hamer_model_cfg, 
                    save_mesh=False,
                    mesh_save_name=os.path.join(hamer_mesh_folder,str(idx).zfill(4)))
                
                #align hamer and save it.

                #hamer_model_cfg,
                #focal_length=cam_int[0,0],
                #mesh_save_name=os.path.join(hamer_mesh_folder,str(idx).zfill(3)),
                #save_mesh=True)

            #load hamer pose save mesh..

            #load kpts of dwpose (original: vit pose was used.)
            
            
            smpl_param['left_hand_pose'] = left_result['hand_pose'].view(-1,45).detach().clone()
            smpl_param['right_hand_pose'] = right_result['hand_pose'].view(-1,45).detach().clone()
        else:

            smpl_param['left_hand_pose'] = torch.zeros((1,45)).to(device)
            smpl_param['right_hand_pose'] = torch.zeros((1,45)).to(device)

        smpl_param['left_eye_pose'] = torch.zeros((1,3)).to(device)
        smpl_param['right_eye_pose'] = torch.zeros((1,3)).to(device)

        smpl_param['betas'] = output['betas'].detach().clone()
        #smpl_param['vertices'] = output['vertices'].detach().clone()/1000
        #smpl_param['joints'] = output['body_joints'].detach().clone()/1000

        #smpl_output, mesh = smpl_mesh.predict(smpl_param, return_mesh=True)
        #mesh.export(os.path.join(hamer_mesh_folder,str(idx).zfill(4)+'_smpl.ply'))
        
        #debugging smpl_mesh
        #save smplmesh.
        debug = False
        if debug:
            smpl_output, mesh = smpl_mesh.predict(smpl_param, return_mesh=True)

            render_mesh = Mesh(mesh.vertices,mesh.faces,np.ones_like(mesh.vertices),device=device)

            render_out = rndr.mesh_render.render_batch(render_mesh, [cam],bg=[0,0,0])
            cv2.imwrite(os.path.join(debugging_folder,str(idx).zfill(5)+'.png'),image/2 + 255*render_out['normal'][0,...,:3].cpu().numpy()/2)
            cam = cameras[idx].minicam

            render_out = rndr.mesh_render.render_batch(render_mesh, [cam],bg=[0,0,0])
            cv2.imwrite(os.path.join(debugging_mini_folder,str(idx).zfill(5)+'.png'),crop_images[idx]/2 + 255*render_out['normal'][0,...,:3].cpu().numpy()/2)
            
        smpl_param_cpu = {k: v.detach().cpu() for k, v in smpl_param.items()}
        torch.save(smpl_param_cpu, os.path.join(smpl_save_folder,os.path.basename(img_list[idx]).split('.')[-2]+'.pt'))



            #spec calib -> add geometry constratint.

            #align with smpl with gravity.


            #and add prompt hmr to estimate 

            #todo..
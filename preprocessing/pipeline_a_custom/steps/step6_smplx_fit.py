import torch
import torch.nn as nn
import numpy as np
import glob
import os
import json
import math

import os
import numpy as np
import cv2
import yaml
import json
import torch
from scene.camera.cameraunified import UnifiedCamera
from scene.data_loader import DataLoader
from scene.object.mesh import Mesh
from scene.object.body_model.mesh import SMPLMesh
from render.render_unified import UnifiedRenderer
from scene.optimizer.avatar_optimizer import AvatarOptimizer as GPOOptimizer
import glob
import torch
import torch.nn as nn
import numpy as np
import glob
import os
import json
import copy


try:
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
assert has_mmdet, 'Please install mmdet to run the demo.'



smplx_joint = {  # 144
        'num': 55, # 22 (body joints) + 3 (face joints) + 30 (hand joints)
        'name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
                'Jaw', 'L_Eye', 'R_Eye', # face joints
                'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints 15
                'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints 15
                )
                }
smplx_root_idx = smplx_joint['name'].index('Pelvis')
smplx_part_idx = {'body': range(smplx_joint['name'].index('Pelvis'), smplx_joint['name'].index('R_Wrist')+1),
                        'face': range(smplx_joint['name'].index('Jaw'), smplx_joint['name'].index('R_Eye')+1),
                        'lhand': range(smplx_joint['name'].index('L_Index_1'), smplx_joint['name'].index('L_Thumb_3')+1),
                        'rhand': range(smplx_joint['name'].index('R_Index_1'), smplx_joint['name'].index('R_Thumb_3')+1)
                        }
# keypoint
tmp_kpt = { 
        'num': 135, # 25 (body joints) + 40 (hand joints) + 70 (face keypoints)  CY: hand joints: match to openpose ( without first one ) - why remove L/R_Wrist_Hand..? #0~24,66,67: body(full body without hand details and ...)
        'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints == openpose but different order.
                    'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
                    'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
                    'Head', 'Jaw', *['Face_' + str(i) for i in range(1,69)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
                ),
                # idx means: joint -> kpt (idx of joint) e.g. idx in 4: 'L_Knee' of smplx_joint == 'L_Knee' in self.kpt. 
        'idx': (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints- too many hand joints compare to smplx
            37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints- too many hand joints compare to smplx
            52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
            15,22, # head, jaw
            76,77,78,79,80,81,82,83,84,85, # eyebrow
            86,87,88,89, # nose
            90,91,92,93,94, # below nose
            95,96,97,98,99,100,101,102,103,104,105,106, # eyes
            107, # right mouth
            108,109,110,111,112, # upper mouth
            113, # left mouth
            114,115,116,117,118, # lower mouth
            119, # right lip
            120,121,122, # upper lip
            123, # left lip
            124,125,126, # lower lip
            127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
            ),


        }
tmp_kpt['root_idx'] = tmp_kpt['name'].index('Pelvis')
tmp_kpt['part_idx'] = {
        'body': range(tmp_kpt['name'].index('Pelvis'), tmp_kpt['name'].index('Nose')+1),
        'lhand': range(tmp_kpt['name'].index('L_Thumb_1'), tmp_kpt['name'].index('L_Pinky_4')+1),
        'rhand': range(tmp_kpt['name'].index('R_Thumb_1'), tmp_kpt['name'].index('R_Pinky_4')+1),
        'face': [tmp_kpt['name'].index('Neck'), tmp_kpt['name'].index('Head'), tmp_kpt['name'].index('Jaw'), tmp_kpt['name'].index('L_Eye'), tmp_kpt['name'].index('R_Eye')] + list(range(tmp_kpt['name'].index('Face_1'), tmp_kpt['name'].index('Face_68')+1)) + [tmp_kpt['name'].index('L_Ear'), tmp_kpt['name'].index('R_Ear')]}


dw_pose_kpt = {
            'num': 133,
            'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                *['Face_' + str(i) for i in range(52,69)], # face contour
                *['Face_' + str(i) for i in range(1,52)], # face - same to openpose order.
                'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand 21
                'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand 21
            }

def est_calib(images):
    """ Roughly estimate intrinsics by image dimensions """
    # imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = images[0] # cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = w0/2., h0/2.
    calib = [focal, focal, cx, cy]
    return calib


def visualize_2d_points_on_numpy_images(gt_points: torch.Tensor, 
                                        pred_points: torch.Tensor, 
                                        images: list, 
                                        output_dir: str):
    """
    Draw GT and Predicted 2D points on top of numpy images and save the result.

    Args:
        gt_points (torch.Tensor): GT points, shape [N, K, 2]
        pred_points (torch.Tensor): Pred points, shape [N, K, 2]
        images (List[np.ndarray]): List of images as numpy arrays (uint8, shape [H, W, 3])
        output_dir (str): Directory to save output images
    """
    assert gt_points.shape == pred_points.shape
    assert len(images) == gt_points.shape[0], "Mismatch between number of images and point sets"
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    N, K, _ = gt_points.shape

    for i in range(N):
        img = images[i]
        assert isinstance(img, np.ndarray), f"Image {i} is not a numpy array"
        assert img.ndim == 3 and img.shape[2] == 3, f"Image {i} must be HxWx3"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)

        gt = gt_points[i].cpu().numpy()
        pred = pred_points[i].cpu().numpy()

        # Draw points
        ax.scatter(gt[:, 0], gt[:, 1], c='blue', label='GT', marker='o')
        ax.scatter(pred[:, 0], pred[:, 1], c='red', label='Prediction', marker='x')

        # Annotate indices
        for j in range(K):
            ax.text(gt[j, 0]+2, gt[j, 1]+2, str(j), color='blue', fontsize=6)
            ax.text(pred[j, 0]+2, pred[j, 1]-4, str(j), color='red', fontsize=6)

        ax.axis('off')
        ax.set_title(f'Image {i}')
        ax.legend()

        save_path = os.path.join(output_dir, f'image_{i}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)


def dwpose2openpose(dw_kpts):
    neck = np.mean(dw_kpts[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(
                dw_kpts[:, 5, 2:4] > 0.3,
                dw_kpts[:, 6, 2:4] > 0.3).astype(int)
    
    B,_,_ = dw_kpts.shape
    openpose_kpts = torch.zeros((B,118,3))
    openpose_kpts[:,1,:] = torch.Tensor(neck)

    #new_keypoints_info = np.insert(
    #        dw_kpts, 17, neck, axis=1)
    #add foot part based on https://github.com/vchoutas/smplify-x/issues/106
    
    mmpose_idx = [
        0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2,   1,  4,  3, 17, 18, 19, 20, 21, 22
    ]#neck 
    openpose_idx = [
        0, 2, 3, 4,  5, 6, 7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ]#nose = 0, 
    #R_Ankle 17 R_heel 23/ R_small toe 22 R_big_toe 20
    #
    #openpose_body= torch.zeros((B,24,3))
    #openpose_flame= torch.zeros((B,51,3))
    #openpose_left_hand= torch.zeros((B,21,3))
    #openpose_right_hand= torch.zeros((B,21,3))

    #openpose_kpts[:,1:8] = torch.Tensor(new_keypoints_info[:,1:8]) #but center hip unknown.. ()
    #openpose_kpts[:,8:18] = torch.Tensor(new_keypoints_info[:,8:18]) #but center hip unknown.. ()
    openpose_kpts[:,openpose_idx] = torch.Tensor(dw_kpts[:,mmpose_idx])
    openpose_kpts[:,25:46]= torch.Tensor(dw_kpts[:,91:112])
    openpose_kpts[:,46:67]= torch.Tensor(dw_kpts[:,112:])
    openpose_kpts[:,67:118]= torch.Tensor(dw_kpts[:,40:91])# face_1~51
    openpose_kpts = openpose_kpts.cuda()
    return openpose_kpts


def change_kpt_name(src_kpt, src_name, dst_name):
    src_kpt_num = len(src_name)
    dst_kpt_num = len(dst_name)

    new_kpt = np.zeros(((dst_kpt_num,) + src_kpt.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_kpt[dst_idx] = src_kpt[src_idx]

    return new_kpt


def check_best_hunyuan3d_candidate(smplx_params, masks):
    def rotation_matrix_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance (in radians) between two 3x3 rotation matrices using PyTorch.

        Args:
            R1 (torch.Tensor): A tensor of shape (3, 3), a rotation matrix.
            R2 (torch.Tensor): A tensor of shape (3, 3), another rotation matrix.

        Returns:
            torch.Tensor: A scalar tensor representing the rotation angle in radians.
        """
        R = R1.T @ R2
        trace = torch.trace(R)
        cos_theta = (trace - 1) / 2.0

        # Clamp to valid range to avoid NaNs due to floating point errors
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        return theta
    def euler_to_matrix(euler):
        """Convert Euler angles to a rotation matrix in ZYX order."""
        x, y, z = euler
        cx, cy, cz = torch.cos(torch.tensor([x, y, z],device=euler.device))
        sx, sy, sz = torch.sin(torch.tensor([x, y, z],device=euler.device))

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ],device=euler.device)

        Ry = torch.tensor([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ],device=euler.device)

        Rz = torch.tensor([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ],device=euler.device)

        return Rz @ Ry @ Rx  # ZYX order
    selected_angles = []
    min_dist_angle = 10e7
    dist_angles = []
    for idx, smpl_param in enumerate(smplx_params):
        mask = masks[idx]
        h,w = mask.shape
        flag = np.any(mask[h-1,:])+np.any(mask[:,w-1])+np.any(mask[0,:]) + np.any(mask[:,0])
        #global_orients.append(nlf_param)
        global_angle = smpl_param['global_orient']
        # angle = rotation_matrix_to_angle(nlf_param['body_pose'], nlf_param['global_orient'])
        #angle = angle + np.pi #[0,2pi]
        default_rotation = torch.eye(3,device=global_angle.device)
        default_rotation[1:] = -default_rotation[1:]
        #default_rotation[2,2] = -default_rotation[2,2]
        #default_rotation[0,0] = -1
        #default_rotation[2,2] = -1

        dist_angle = rotation_matrix_distance(R1=euler_to_matrix(global_angle.squeeze()),R2=default_rotation)#torch.abs(rotation_distance_from_euler(torch.zeros_like(nlf_param['global_orient'])[0],nlf_param['global_orient'][0]))
        dist_angle = abs(dist_angle)#min(abs(2*np.pi-angle),abs(angle))
        #print('dist_angle'+str(idx), dist_angle)

        dist_angles.append(dist_angle)

        if dist_angle < math.pi/6 and flag==0:
            selected_angles.append(idx)
        if dist_angle < min_dist_angle:
            min_dist_angle=dist_angle
            min_dist_idx = idx
    if len(selected_angles) == 0:
        selected_angles.append(min_dist_idx)
    Da_pose = torch.zeros_like(smplx_params[0]['body_pose'])
    Da_pose[0,2] = 1.0
    Da_pose[0,5] = -1.0
    min_error = 10e7
    for selected_idx in selected_angles:
        #find a-posed image?
        pose = smplx_params[selected_idx]['body_pose']
        error = torch.sum(torch.abs(Da_pose - pose)) + 10*torch.sum(torch.abs(Da_pose[0,16*3:] - pose[0,16*3:])) \
        + 10*torch.sum(torch.abs(Da_pose[0,:6] - pose[0,:6])) 
        if error < min_error:
            min_error = error
            min_idx =selected_idx
    print(f'Run hunyuan3d v2 for selected image: {min_idx}-th image') #similar to da-pose + seeing 
    return min_idx

def train_wxb_model(x_train, y_train, lr=0.01, epochs=500):
    import torch.optim as optim

    """
    x_train, y_train: shape (N, 3)
    Model: ŷ = w * (x + b)
    """
    x_train = x_train.float().detach()
    y_train = y_train.float().detach()

    # Parameters to learn
    w = torch.nn.Parameter(torch.tensor(1.0, device=x_train.device))
    b = torch.nn.Parameter(torch.zeros(3, device=x_train.device))

    optimizer = optim.Adam([w, b], lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = w * (x_train + b)  # broadcasted addition
        loss = loss_fn(y_pred, y_train)
        loss.backward()  # only one backward per loss
        optimizer.step()

        #if (epoch + 1) % 100 == 0:
        #    print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

    return w.item(), b.detach()
                

def render_smpls(renderer, images, smpl_mesh, smpl_params, save_folder,camera ):
    os.makedirs(save_folder,exist_ok=True)
    for i, img in enumerate(images):
        smpl_output, mesh = smpl_mesh.predict(smpl_params[i], return_mesh=True)
        #nlf_list.append(nlf_params)
        #render mesh using camera
        mesh.xyz= torch.Tensor(mesh.vertices).to(device)
        mesh.rgb = 0.5*torch.ones_like(mesh.xyz)
        mesh.face = torch.Tensor(smpl_mesh.smpl_model.faces).to(device)
        mesh = Mesh(mesh.vertices,indices=smpl_mesh.smpl_model.faces,colors=0.5*torch.ones_like(torch.Tensor(mesh.vertices)),device=device)
        render_out = renderer.render_batch(mesh, [camera],bg=[0,0,0])
        cv2.imwrite(os.path.join(save_folder,str(i).zfill(5)+'.png'),img + 255*render_out['rgba'][0,...,:3].cpu().numpy())
def draw_points_on_image(image, points, save_path, point_color=(255, 0, 0), radius=5, thickness=-1):
    """
    Draw coordinate points on an image and save it.

    Parameters:
    - image_path (str): input image path
    - points (list of tuple): points to draw [(x1, y1), (x2, y2), ...]
    - save_path (str): output image path
    - point_color (tuple): point color (R, G, B)
    - radius (int): circle radius
    - thickness (int): line thickness (-1 = filled circle)

    Returns:
    - None
    """
    # Load the image

    # Draw a marker at each point
    for point in points:
        x = point[0]
        y = point[1]
        cv2.circle(image, (int(x), int(y)), radius=radius, color=point_color, thickness=thickness)

    # Save the image (in BGR format)
    cv2.imwrite(save_path, image)




try:
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
assert has_mmdet, 'Please install mmdet to run the demo.'

if __name__ == "__main__":
    device = torch.device('cuda')
    path2config =  os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','scarf.yaml')
    with open(path2config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    Path_params = params['Path']

    rndr = UnifiedRenderer(device=device)

    data_loader = DataLoader(model_type=params['SMPL']['model_type'])
    gpo_optimizer = GPOOptimizer(mesh_renderer=rndr.mesh_render, gs_renderer=None)
        # file names



    """load images/ masks"""
    data_folder = Path_params['training_folder']


    save_root = data_folder # Path_params['save_root']
    image_root = os.path.join(data_folder,'images') #Path_params['orig_image_folder']
    #os.makedirs(hunyuan_folder,exist_ok=True)
    os.makedirs(save_root,exist_ok=True)

    # read a list of images
    images, img_list = data_loader.load_images(image_root, return_mask=False)
    original_height,original_width,c = images[0].shape
    #_, masks, _ = data_loader.load_images(mask_root, return_mask=True)
    ext = '.' + img_list[0].split('.')[-1]
    ext = '.' + img_list[0].split('.')[-1]
    #h,w = masks[0].shape




    ######################################################################
    #   Below: DW pose test code:mmpose/demo/topdown_demo_with_mmdet.py  #
    ######################################################################
    #hyper parmaters
    det_config = 'libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
    det_checkpoint = 'libs/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    pose_config = 'libs/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    pose_checkpoint = 'libs/mmpose/dw-ll_ucoco_384.pth'
    bbox_thr = 0.3
    det_cat_id = 0
    nms_thr = 0.3
    kpt_thr =0.3
    show_interval = 0
    alpha = 0.8
    radius = 3
    thickness= 1
    skeleton_style='mmpose'
    mmpose_folder= os.path.join(save_root,'mmpose_result_'+skeleton_style)
    os.makedirs(mmpose_folder,exist_ok=True)
    keypoints_folder = os.path.join(mmpose_folder, 'keypoints_whole_body')
    bbox_folder = os.path.join(mmpose_folder, 'bboxes')
    crop_bbox_folder = os.path.join(mmpose_folder, 'cropped_bboxes')

    #output_path_list = glob(os.path.join(output_root, '*.json'))
    os.makedirs(bbox_folder,exist_ok=True)
    os.makedirs(crop_bbox_folder,exist_ok=True)

    os.makedirs(keypoints_folder,exist_ok=True)

    kpt_path_list = sorted(glob.glob(os.path.join(keypoints_folder, '*.json')))# dwpose results.
    kpts = {}
    dw_kpts = np.zeros((len(kpt_path_list),133,3))
    for frame_idx, kpt_path in enumerate(kpt_path_list):
        with open(kpt_path) as f:
            kpt = np.array(json.load(f), dtype=np.float32)
        dw_kpts[frame_idx] =kpt
        kpt = change_kpt_name(kpt, dw_pose_kpt['name'], tmp_kpt['name']) # dwpose -> tmp keypoint
        kpts[frame_idx] = torch.Tensor(kpt)



    #_, masks, _ = data_loader.load_masks(samurai_mask_root, return_mask=True)
    #for mask_idx in range(len(masks)):
    samurai_mask_root = os.path.join(save_root,'masks')
    if not os.path.exists(samurai_mask_root):
        print('run samurai')
        exit()
    mask_images = sorted(glob.glob(os.path.join(samurai_mask_root,'*.png')))
    masks = []
    bboxes = []

    crop_images_list = []
    for mask_image_path in mask_images:
        mask_name = os.path.basename(mask_image_path)
        mask = cv2.imread(mask_image_path,cv2.IMREAD_UNCHANGED)
        crop_bbox_path = os.path.join(crop_bbox_folder,mask_name.split('.')[-2]+'.txt')
        with open(crop_bbox_path, 'r') as file:
            content = file.read()
            values = [int(item.strip()) for item in content.split(',') if item.strip()]
        bboxes.append(values)

        mask = mask[values[1]:values[1]+values[3], values[0]:values[0] + values[2]]
        masks.append(mask)
        crop_image_path = os.path.join(save_root,'image_crop',mask_name)
        crop_image = cv2.imread(os.path.join(crop_image_path))
        crop_images_list.append(crop_image)
    masks = np.stack(masks,axis=0)
    crop_images = np.stack(crop_images_list,axis=0)
    #crop masks based on cropped_parts
    height,width = masks[0].shape
    try:
        cam_int = torch.Tensor(np.load(Path_params['calib_path']))
    except:
        cam_int = est_calib(images)
        cam_int = UnifiedCamera.intrinsic_from_param_each(fx=cam_int[0],fy=cam_int[1],cx=cam_int[2],cy=cam_int[3])

    #camera
    camera = UnifiedCamera.perspective_camera_from_K(K=torch.Tensor(cam_int),width=original_width,height=original_height,  znear=0.01, zfar=1000.0, device=torch.device('cuda:0') )
    crop_cameras = []
    for bbox in bboxes:
        crop_camera = copy.deepcopy(camera)
        crop_camera.K_update_bbox(bbox)
        crop_cameras.append(crop_camera)

    """
    if height>width:
        camera = UnifiedCamera.perspective_camera_from_arg(yaw=0, pitch=0, fovx=55*width/height, fovy=55, znear=00.1, zfar=1000.0,
                                                        cam_center=0.0, width=width, height=height,  device=torch.device('cuda:0'))
    else:
        camera = UnifiedCamera.perspective_camera_from_arg(yaw=0, pitch=0, fovx=55, fovy=55*height/width, znear=00.1, zfar=1000.0,
                                                        cam_center=0.0, width=width, height=height,  device=torch.device('cuda:0'))
    """
    debugging_folder = os.path.join(save_root,'nlf_debugging_folder')
    smpl_mesh = SMPLMesh(smpl_config=params['SMPL'])


    #outof mask -> confidence zero!, add init loss (not change too much... pose.)
    #smpl_params = gpo_optimizer.batch2list(smpl_params)

    #load smpl_params
    B = len(kpts)

    target_kpts = torch.zeros((len(kpts),135,3))
    smpl_params = []
    smpl_paths = sorted(glob.glob(os.path.join(save_root,'hmr_results','smplx_params','*.pt')))
    for idx, smpl_path in enumerate(smpl_paths):
        smpl_param = torch.load(smpl_path)
        smpl_params.append(smpl_param)
        target_kpts[idx] = kpts[idx]
    target_kpts = torch.Tensor(target_kpts).to(device)
    smpl_params = gpo_optimizer.list2batch(smpl_params)

    smpl_params = {k: v.detach().cuda() for k, v in smpl_params.items()}


    opt_options = {   'transl':1e-2, 'betas':1e-3} #align the smpl on the frames
    weight = {'body':1.0, 'hand':.0, 'face':1.0}
    loss_options = {
        'keypoint': 10.0,

    }

    #outof mask -> confidence zero!
    
    smpl_params = gpo_optimizer.fitting_smplx(target_kpts, smpl_mesh, smpl_params,crop_cameras,opt_options,loss_options=loss_options, max_iter=200,
                                                keys=opt_options.keys(), weight=weight,kpt_idx=tmp_kpt['idx'])

    output = smpl_mesh.predict(smpl_params)
    #proj_est_joint = projection(output.joints[:,:144], camera.K.expand(B, -1, -1),
    #                    camera.R.expand(B, -1, -1),
    #                    -camera.t.expand(B, -1).unsqueeze(-1))
    after_opt_folder = os.path.join(save_root,'global_smpl')
    os.makedirs(after_opt_folder,exist_ok=True)
    smpl_params = gpo_optimizer.batch2list(smpl_params)
    #smpl_images= []
    for idx, smpl_param in enumerate(smpl_params):
        smpl_output, mesh = smpl_mesh.predict(smpl_param, return_mesh=True)
        mesh.xyz= torch.Tensor(mesh.vertices).to(device)
        mesh.rgb = 0.5*torch.ones_like(mesh.xyz)
        mesh.face = torch.Tensor(smpl_mesh.smpl_model.faces).to(device)
        mesh = Mesh(mesh.vertices,indices=smpl_mesh.smpl_model.faces,colors=0.5*torch.ones_like(torch.Tensor(mesh.vertices)),device=device)
        render_out = rndr.mesh_render.render_batch(mesh, [camera],bg=[0,0,0])
        cv2.imwrite(os.path.join(after_opt_folder,os.path.basename(mask_images[idx])),images[idx]/2 + 255*render_out['normal'][0,...,:3].cpu().numpy()/2)
        # smpl_images.append(crop_images[idx]/(2*255) +render_out['normal'][0,...,:3].cpu().numpy()/2)

    smpl_params = gpo_optimizer.list2batch(smpl_params)
    
    #output = smpl_mesh.predict(smpl_params)

    #proj_est_joint = projection(output.joints[:,:144], camera.K.expand(B, -1, -1),
    #                    camera.R.expand(B, -1, -1),
    #                    -camera.t.expand(B, -1).unsqueeze(-1))
    #visualize_2d_points_on_numpy_images(target_kpts[...,:2],proj_est_joint[:,tmp_kpt['idx'],:],smpl_images,'opt_global_smpl')





    weight = {'body':1.0, 'hand':1.0, 'face':1.0}

    opt_options = { 'body_pose':1e-2,  'global_orient':1e-3, 'betas':1e-4, 'transl':1e-4} #, 'scale':1e-5}
                    
    loss_options = {
    'keypoint': 10.0,
    'init_body': 0.1,
    'init_jaw':1.0,
    'global_smooth': 1.0,
    'init_foot':1.0,
    'init_neck':1.0,
    'init_head':1.0,
    }
    smpl_params = gpo_optimizer.fitting_smplx(target_kpts, smpl_mesh, smpl_params,crop_cameras,opt_options,loss_options=loss_options, keys=opt_options.keys(),
                                                weight=weight,kpt_idx=tmp_kpt['idx'], max_iter=1000,)

    after_opt_folder = os.path.join(save_root,'global_after_shape_opt')
    os.makedirs(after_opt_folder,exist_ok=True)
    smpl_params = gpo_optimizer.batch2list(smpl_params)
    #smpl_images= []
    for idx, smpl_param in enumerate(smpl_params):
        smpl_output, mesh = smpl_mesh.predict(smpl_param, return_mesh=True)
        mesh.xyz= torch.Tensor(mesh.vertices).to(device)
        mesh.rgb = 0.5*torch.ones_like(mesh.xyz)
        mesh.face = torch.Tensor(smpl_mesh.smpl_model.faces).to(device)
        mesh = Mesh(mesh.vertices,indices=smpl_mesh.smpl_model.faces,colors=0.5*torch.ones_like(torch.Tensor(mesh.vertices)),device=device)
        render_out = rndr.mesh_render.render_batch(mesh, [camera],bg=[0,0,0])
        cv2.imwrite(os.path.join(after_opt_folder,os.path.basename(mask_images[idx])),images[idx]/2 + 255*render_out['normal'][0,...,:3].cpu().numpy()/2)
        #smpl_images.append(crop_images[idx]/(2*255) +render_out['normal'][0,...,:3].cpu().numpy()/2)
    smpl_params = gpo_optimizer.list2batch(smpl_params)
    #output = smpl_mesh.predict(smpl_params)

    #proj_est_joint = projection(output.joints[:,:144], camera.K.expand(B, -1, -1),
    #                    camera.R.expand(B, -1, -1),
    #                    -camera.t.expand(B, -1).unsqueeze(-1))
    #visualize_2d_points_on_numpy_images(target_kpts[...,:2],proj_est_joint[:,tmp_kpt['idx'],:],smpl_images,'joint_viz')


    #smpl_params = gpo_optimizer.list2batch(smpl_params)

    weight = {'body':1.0, 'hand':1.0, 'face':1.0}

    if params['use_hamer']:
        opt_options = { 'body_pose':1e-2,  'jaw_pose': 1e-4, 
                        'left_eye_pose': 1e-3, 'right_eye_pose': 1e-3, # 'left_hand_pose': 0.0, 'right_hand_pose':0.0, #1e-3,
                        'global_orient':1e-4, 'betas':1e-5, 'transl':1e-4} #, 'scale':1e-5}
    else:
        opt_options = { 'body_pose':1e-2,  'jaw_pose': 1e-4, 
                        'left_eye_pose': 1e-3, 'right_eye_pose': 1e-3, 'left_hand_pose': 1e-3, 'right_hand_pose':1e-3,  # 'left_hand_pose': 0.0, 'right_hand_pose':0.0, #1e-3,
                        'global_orient':1e-4, 'betas':1e-5, 'transl':1e-4} #, 'scale':1e-5}

    """
    loss_options = {
        'keypoint': 0.1,
        'init_body': 0.1,
        'init_jaw':10.0,

        'global_smooth': 1.0,
        'init_foot':100000,
        'init_neck':10.0,
        'init_head':10.0,

    }
    """

    loss_options = {
    'keypoint': 10.0,
    'init_body': 0.1,
    'init_jaw':1.0,
    'global_smooth': 1.0,
    'init_foot':1.0,
    'init_neck':1.0,
    'init_head':1.0,

    
    }

    #outof mask -> confidence zero!, add init loss (not change too much... pose.)
    #smpl_params = gpo_optimizer.batch2list(smpl_params)

    smpl_params = gpo_optimizer.fitting_smplx(target_kpts, smpl_mesh, smpl_params,crop_cameras,opt_options,loss_options=loss_options, keys=opt_options.keys(),
                                                weight=weight,kpt_idx=tmp_kpt['idx'], max_iter=1000,)
    after_opt_folder = os.path.join(save_root,'global_all_opt')
    os.makedirs(after_opt_folder,exist_ok=True)
    smpl_params = gpo_optimizer.batch2list(smpl_params)
    smpl_save_folder = os.path.join(save_root,'global_smpl_optimized')
    os.makedirs(smpl_save_folder,exist_ok=True)

    for idx, smpl_param in enumerate(smpl_params):
        smpl_output, mesh = smpl_mesh.predict(smpl_param, return_mesh=True)
        mesh.xyz= torch.Tensor(mesh.vertices).to(device)
        mesh.rgb = 0.5*torch.ones_like(mesh.xyz)
        mesh.face = torch.Tensor(smpl_mesh.smpl_model.faces).to(device)
        mesh = Mesh(mesh.vertices,indices=smpl_mesh.smpl_model.faces,colors=0.5*torch.ones_like(torch.Tensor(mesh.vertices)),device=device)
        render_out = rndr.mesh_render.render_batch(mesh, [camera],bg=[0,0,0])
        cv2.imwrite(os.path.join(after_opt_folder,os.path.basename(mask_images[idx])),images[idx]/2 + 255*render_out['normal'][0,...,:3].cpu().numpy()/2)
        #smpl_images.append(crop_images[idx]/(2*255) +render_out['normal'][0,...,:3].cpu().numpy()/2)
        smpl_param_cpu = {k: v.detach().cpu() for k, v in smpl_param.items()}
        torch.save(smpl_param_cpu, os.path.join(smpl_save_folder,os.path.basename(mask_images[idx]).split('.')[-2]+'.pt'))


    #additional smoothing from exavatar.
    from utils.geometry.pose_utils import smoothen_poses
    smpl_params = gpo_optimizer.list2batch(smpl_params)
    keys=['body_pose','jaw_pose','global_orient']
    for key in keys:

        pose = smpl_params[key].cpu().numpy()
        pose = smoothen_poses(pose, window_length=9)
        smpl_params[key] = torch.tensor(pose).cuda()

    after_opt_folder = os.path.join(save_root,'global_smoothed')
    os.makedirs(after_opt_folder,exist_ok=True)
    smpl_params = gpo_optimizer.batch2list(smpl_params)
    smpl_save_folder = os.path.join(save_root,'global_smpl_smoothed')
    os.makedirs(smpl_save_folder,exist_ok=True)

    for idx, smpl_param in enumerate(smpl_params):
        smpl_output, mesh = smpl_mesh.predict(smpl_param, return_mesh=True)
        mesh.xyz= torch.Tensor(mesh.vertices).to(device)
        mesh.rgb = 0.5*torch.ones_like(mesh.xyz)
        mesh.face = torch.Tensor(smpl_mesh.smpl_model.faces).to(device)
        mesh = Mesh(mesh.vertices,indices=smpl_mesh.smpl_model.faces,colors=0.5*torch.ones_like(torch.Tensor(mesh.vertices)),device=device)
        render_out = rndr.mesh_render.render_batch(mesh, [camera],bg=[0,0,0])
        cv2.imwrite(os.path.join(after_opt_folder,os.path.basename(mask_images[idx])),images[idx]/2 + 255*render_out['normal'][0,...,:3].cpu().numpy()/2)
        #smpl_images.append(crop_images[idx]/(2*255) +render_out['normal'][0,...,:3].cpu().numpy()/2)
        smpl_param_cpu = {k: v.detach().cpu() for k, v in smpl_param.items()}
        torch.save(smpl_param_cpu, os.path.join(smpl_save_folder,os.path.basename(mask_images[idx]).split('.')[-2]+'.pt'))


    #smpl_params = gpo_optimizer.list2batch(smpl_params)
    #change2ezmocap params and save them and render them...
    hunyuan_folder = os.path.join(save_root,'hunyuan_tmp')
    os.makedirs(hunyuan_folder,exist_ok=True)
    #hunyuan_path = os.path.join(hunyuan_folder,'hunyuan3dv2.obj')
    target_idx_path = os.path.join(hunyuan_folder,'target_idx.txt')
    #smpl_params = gpo_optimizer.batch2list(smpl_params)
    #if not os.path.exists(hunyuan_path):
    if 'target_idx'  in params.keys():
        target_idx = int(params['target_idx'])
    else:
        target_idx = check_best_hunyuan3d_candidate(smpl_params,masks)
    print('target_idx:',os.path.basename(mask_images[target_idx]).split('.')[-2])
    with open(target_idx_path,'w') as file:
        file.write(os.path.basename(mask_images[target_idx]).split('.')[-2])
    image_name = os.path.join( hunyuan_folder,'hunyuan.png')
    mask_name = os.path.join( hunyuan_folder,'mask.png')


    cv2.imwrite(mask_name,masks[target_idx])
    cv2.imwrite(image_name,crop_images[target_idx])

    #running hunyuan3d
    
    #depth_predictor = DepthAnythingV2(config=params['DEPTH'])
    # depth_anything = depth_predictor.to(device).eval()

    #depths = []
    #normals = []
    #depth_folder = os.path.join(save_root,'depths')
    #normal_folder = os.path.join(save_root,'normals')
    #if debug:
    #    debugging_depth_folder = os.path.join(save_root,'debug_depth')
    ##    debugging_normal_folder = os.path.join(save_root,'debug_normal')
     #   os.makedirs(debugging_depth_folder,exist_ok=True)
     #   os.makedirs(debugging_normal_folder,exist_ok=True)

    #os.makedirs(depth_folder,exist_ok=True)
    #os.makedirs(normal_folder,exist_ok=True)

    #for i, img in enumerate(crop_images):
    #    depth, normal = depth_anything.predict_depth_normal_cy(img, w=width, h=height, normalize=False)
    #    image_name = os.path.basename(mask_images[i]).split('.')[-2]
    #    #np.save(os.path.join(depth_folder,image_name+'.npy'),depth)
    #    np.save(os.path.join(normal_folder,image_name+'.npy'),normal)
    #    if debug:
     ##       cv2.imwrite(os.path.join(debugging_depth_folder,image_name+'.png'),depth)
      #      cv2.imwrite(os.path.join(debugging_normal_folder,image_name+'.png'),normal)

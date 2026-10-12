import torch
import glob
import os

import os
import cv2
import yaml
import torch
from scene.data_loader import DataLoader
#from scene.optimizer.avatar_optimizer import AvatarOptimizer as GPOOptimizer
#from libs.pose_detector import BodyPoseFacade
#from libs.depth_anything_v2.dpt import DepthAnythingV2
import glob
import torch
import glob
import os



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
        'num': 135, # 25 (body joints) + 40 (hand joints) + 70 (face keypoints): hand joints: match to openpose ( without first one ) - why remove L/R_Wrist_Hand..? #0~24,66,67: body(full body without hand details and ...)
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


try:
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
assert has_mmdet, 'Please install mmdet to run the demo.'

import argparse  # import argparse


if __name__ == "__main__":

    # --- argparse setup ---
    # 1. Create the ArgumentParser



    device = torch.device('cuda')

    #with open(path2config, 'r') as f:
    #    params = yaml.load(f, Loader=yaml.FullLoader)
    path2config =  os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','scarf.yaml')
    with open(path2config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    data_loader = DataLoader(model_type='smplx')
    save_root = os.path.join(params['Path']['training_folder'])
    os.makedirs(save_root,exist_ok=True)
    # read a list of images
    
    #img_list = [img_list[0]] #ONLY FIRST IMAGE!
    #masks,_ = data_loader.load_masks(mask_root, return_mask=False)
    #h,w = masks[0].shape
    mask_root = os.path.join(save_root,'masks')
    masks,mask_list = data_loader.load_masks(mask_root, return_mask=False)
    crop_mask_save = os.path.join(save_root,'mask_crop')
    os.makedirs(crop_mask_save,exist_ok=True)


   # bbox_folder = os.path.join(mmpose_folder, 'bboxes')
    crop_bbox_folder = os.path.join(save_root,'mmpose_result_mmpose', 'cropped_bboxes')

    crop_bbox_files = sorted(glob.glob(os.path.join(crop_bbox_folder,'*.txt')))

    for idx, crop_bbox_file in enumerate(crop_bbox_files):
        with open(crop_bbox_file, 'r') as f:
            # f.read().strip() yields a string like "87,244,410,535".
            data = f.read().strip()
        x1_crop, y1_crop, width, height = map(int, data.split(','))
        #crop image
        #image_crop = image_np[y1_crop:y1_crop+height,x1_crop:x1_crop+width]
        mask_np = masks[idx]

        mask_crop = mask_np[y1_crop:y1_crop+height,x1_crop:x1_crop+width]
        #import pdb;pdb.set_trace()
        #cv2.imwrite(os.path.join(crop_image_save,os.path.basename(img_paths[idx])),image_crop)
        cv2.imwrite(os.path.join(crop_mask_save,os.path.basename(mask_list[idx])),mask_crop)
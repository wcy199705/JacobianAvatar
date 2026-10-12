import torch
import numpy as np
import os
import json
import math

import tqdm
import os
import numpy as np
import cv2
import yaml
import json
import torch
from scene.data_loader import DataLoader
#from scene.optimizer.avatar_optimizer import AvatarOptimizer as GPOOptimizer
#from libs.pose_detector import BodyPoseFacade
#from libs.depth_anything_v2.dpt import DepthAnythingV2
import torch
import numpy as np
import os
import json


from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import mmcv

try:
    from mmdet.apis import inference_detector, init_detector
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
    from mmdet.apis import inference_detector, init_detector
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

    image_root = os.path.join(params['Path']['training_folder'], 'images') 
    data_loader = DataLoader(model_type='smplx')
    save_root = os.path.join(params['Path']['training_folder'])
    os.makedirs(save_root,exist_ok=True)

    # read a list of images
    
    images, img_list = data_loader.load_images(image_root, return_mask=False)
    #img_list = [img_list[0]] #ONLY FIRST IMAGE!
    h,w,c = images[0].shape
    #masks,_ = data_loader.load_masks(mask_root, return_mask=False)
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

    # build detector
    detector = init_detector(
        det_config, det_checkpoint, device='cpu')#'cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False, weights_only=False))))
    
    pose_estimator.cfg.visualizer.radius = radius
    pose_estimator.cfg.visualizer.alpha = alpha
    pose_estimator.cfg.visualizer.line_width = thickness

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=skeleton_style)

    pred_keypoints =[]
    img_paths = []
    bbox_list = []
    for img_path in tqdm.tqdm(img_list):
        #img = torch.Tensor(img/255).to(device).permute(2,0,1)
        det_result = inference_detector(detector, img_path)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id,
                                    pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]
        bbox_list.append(bboxes)
        pose_results = inference_topdown(pose_estimator, img_path, bboxes)
        data_samples = merge_data_samples(pose_results)
        # show the results
        if isinstance(img_path, str):
            img = mmcv.imread(img_path, channel_order='rgb')
        elif isinstance(img_path, np.ndarray):
            img = mmcv.bgr2rgb(img_path)

        visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=True,
                    show_kpt_idx=True,
                    skeleton_style=skeleton_style,
                    show=False,
                    wait_time=show_interval,
                    kpt_thr=kpt_thr)
        pred_instances = data_samples.get('pred_instances', None)
        pred_instances_list = split_instances(pred_instances)
        output_file = os.path.join(mmpose_folder, os.path.basename(img_path))
        img_vis = visualizer.get_image()
        mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
        pred_keypoints.append(pred_instances_list)
        img_paths.append(img_path)
        """
        pred_save_path = f'{mmpose_outputpath}/results_' \
            f'{os.path.splitext(os.path.basename(img_path))[0]}.json'
        with open(pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        """
    widths = []
    heights = []
    x1s = []
    y1s = []

    for idx, save_bbox in enumerate(bbox_list):
        image = cv2.imread(img_paths[idx])
        bbox = save_bbox.tolist()[0]
        x1=bbox[0]
        y1=bbox[1]
        w=bbox[2] - x1
        h=bbox[3] - y1
        widths.append(w)
        heights.append(h)

    #preprocessing crop human in the video.
    width = math.ceil(max(widths))
    height = math.ceil(max(heights))
    crop_image_save = os.path.join(save_root,'image_crop')
    #crop_mask_save = os.path.join(save_root,'mask_crop')

    os.makedirs(crop_image_save,exist_ok=True)
    #os.makedirs(crop_mask_save,exist_ok=True)

    for idx, save_bbox in enumerate(bbox_list):
        image_np = cv2.imread(img_paths[idx])
        #mask_np = masks[idx]

        img_height, img_width,_ = image_np.shape

        bbox = save_bbox.tolist()[0]
        x1=bbox[0]
        y1=bbox[1]
        w=bbox[2] - x1
        h=bbox[3] - y1
        if idx==0:
            with open(os.path.join(save_root, 'first_bbox.txt'), 'w') as f:
                f.write(str(x1) +','+str(y1) + ',' + str(w) +',' + str(h))
        body_center = [x1 + w/2, y1+h/2]
        x1_crop = x1 + w/2 - width/2
        if x1_crop<0:
            #import pdb;pdb.set_trace()
            x1_crop = 0
        x1_crop = round(x1_crop)
        if x1_crop + width > img_width:
            x1_crop = img_width- width
        y1_crop = y1 + h/2 - height/2
        if y1_crop<0:
                            #import pdb;pdb.set_trace()
            y1_crop = 0
        y1_crop = round(y1_crop)
        if y1_crop + height > img_height:
            y1_crop = img_height- height
                    
        #crop image
        image_crop = image_np[y1_crop:y1_crop+height,x1_crop:x1_crop+width]
        #mask_crop = mask_np[y1_crop:y1_crop+height,x1_crop:x1_crop+width]
        #import pdb;pdb.set_trace()
        cv2.imwrite(os.path.join(crop_image_save,os.path.basename(img_paths[idx])),image_crop)
        #v2.imwrite(os.path.join(crop_mask_save,os.path.basename(img_paths[idx])),mask_crop)

        kpt_save = None
        image_name = os.path.basename(img_paths[idx]).split('.')[-2]
        pred_instances_list = pred_keypoints[idx]
        for i in range(len(pred_instances_list)):
            xy = np.array(pred_instances_list[i]['keypoints'], dtype=np.float32).reshape(-1,2)
            xy[...,0] = xy[...,0] - x1_crop
            xy[...,1] = xy[...,1] - y1_crop
            score = np.array(pred_instances_list[i]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
            kpt = np.concatenate((xy, score),1) # x, y, score
            if (kpt_save is None) or (kpt_save[:,2].mean() < kpt[:,2].mean()):
                kpt_save = kpt

        with open(os.path.join(keypoints_folder, image_name + '.json'), 'w') as f:
            json.dump(kpt_save.tolist(), f)
        with open(os.path.join(bbox_folder, image_name + '.txt'), 'w') as f:
            f.write(str(x1) +','+str(y1) + ',' + str(w) +',' + str(h))
        with open(os.path.join(crop_bbox_folder, image_name + '.txt'), 'w') as f:
            f.write(str(x1_crop) +','+str(y1_crop) + ',' + str(width) +',' + str(height))

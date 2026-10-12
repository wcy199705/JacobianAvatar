import torch
import os

import os
import yaml
import torch
from scene.data_loader import DataLoader
#from scene.optimizer.avatar_optimizer import AvatarOptimizer as GPOOptimizer
#from libs.pose_detector import BodyPoseFacade
#from libs.depth_anything_v2.dpt import DepthAnythingV2
import torch
import os


from preprocessing.pipeline_a_custom.utils.run_samurai import run_samurai

if __name__ == "__main__":

    # --- argparse setup ---
    # 1. Create the ArgumentParser



    device = torch.device('cuda')

    path2config =  os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','scarf.yaml')
    with open(path2config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    image_root = os.path.join(params['Path']['training_folder'], 'images') 
    data_loader = DataLoader(model_type='smplx')
    save_root = os.path.join(params['Path']['training_folder'])
    os.makedirs(save_root,exist_ok=True)


    samurai_mask_root = os.path.join(save_root,'masks')

    images, img_list = data_loader.load_images(image_root, return_mask=False)
    crop_bboxes= []
    if not os.path.exists(samurai_mask_root):
        print('run samurai')
        first_image_name = os.path.basename(img_list[0])
        masks_out = run_samurai(save_folder=samurai_mask_root,bbox_path=os.path.join(save_root,'first_bbox.txt'),image_folder=image_root)

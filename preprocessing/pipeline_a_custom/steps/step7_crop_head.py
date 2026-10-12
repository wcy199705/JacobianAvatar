#keypoints extraction.
#load keypoints from dataset.
import numpy as np
import cv2
# load the models from the given paths
import torch
import os
import yaml
from utils.dataloader.CUSTOM_dataset_SMPLX import CUSTOM
from utils.geometry.sampling_utils import get_sampled_indices_from_length
from utils.dataloader.utils import get_sampled_data

#parser = argparse.ArgumentParser(description='Process SMPLX meshes with HaMeR.')
#parser.add_argument('--config', type=str, default=os.path.join('config', 'ZJU_dataset', '313.yaml'),
#                    help='Path to the configuration YAML file.')

device =torch.device('cuda')
#args = parser.parse_args()
#path2config = args.config


#if not os.path.exists(path2config):
#    print(f"[Error] Config file not found at: {path2config}")

#print(f"Loading configuration from: {path2config}")

# -----------------------------------------------------------
# 2. Load config and initialize
# -----------------------------------------------------------
#with open(path2config, 'r') as f:
#    params = yaml.load(f, Loader=yaml.FullLoader)
scenes=['scarf']
for scene in scenes:
    path2config = os.path.join('configs', 'legacy', 'SMPLX_dataloader', 'CUSTOM', f'{scene}.yaml')
    with open(path2config, 'r') as f:
        general_params = yaml.load(f, Loader=yaml.FullLoader)

    Path_params = general_params['Path']
    train_dataset = CUSTOM(
        root_folder=Path_params['training_folder'],
        smpl_config=general_params['SMPL'],
        device=device,
        img_scale_factor=1.0,
        split='train',
        use_head_data=False,
    )


    total_sampling_idxs = get_sampled_indices_from_length(train_dataset.num_frames, 1)  #[] #, sampling_boundaries = get_sampled_indices(sampling_file, step=L0_interval)
    sampling_imgs, sampling_masks, sampling_cams, sampling_normals, sampling_keypoints = get_sampled_data(train_dataset,total_sampling_idxs)
    #head_idxs = [i in range()]



    bboxes = []
    head_keypoints = sampling_keypoints[:,67:,:]
    head_img_folder = os.path.join(Path_params['training_folder'],'Head_img')
    head_mask_folder = os.path.join(Path_params['training_folder'],'Head_mask')
    head_bbox_file = os.path.join(Path_params['training_folder'],'Head_aabb.npy')
    os.makedirs(head_img_folder,exist_ok=True)
    os.makedirs(head_mask_folder,exist_ok=True)

    for idx, head_kpt in enumerate(head_keypoints):
        # 1. Strip the extension with [:-4]
        img_name = os.path.basename(train_dataset.file_names[idx])[:-4]
        img = sampling_imgs[idx]
        mask = sampling_masks[idx]

        H, W, c = img.shape
        resize_size_H =  H//4
        resize_size_W =  W//4

        x_pad = W * 0.05
        y_pad = H * 0.05
        conf = head_kpt[:, 2]
        
        # Keep valid keypoints with confidence >= 0.8
        valid_kpts = head_kpt[conf >= 0.80]
        
        if len(valid_kpts) > 0:
            xs = valid_kpts[:, 0]
            ys = valid_kpts[:, 1]
            
            # These are PyTorch tensors, so use torch.min / torch.max and .item() to get scalars.
            min_x = torch.min(xs).item()
            min_y = torch.min(ys).item()
            max_x = torch.max(xs).item()
            max_y = torch.max(ys).item()
            
            # Compute min/max, apply padding, and clamp to image bounds (0~W, 0~H) 
            x1 = np.clip(min_x - x_pad, 0, W)
            y1 = np.clip(min_y - y_pad, 0, H)
            x2 = np.clip(max_x + x_pad, 0, W)
            y2 = np.clip(max_y + y_pad, 0, H)
        else:
            # 2. x1, y1, x2, y2 must also be assigned in the else branch, otherwise cropping fails
            x1, y1, x2, y2 = 0, 0, W, H
            
        bboxes.append([x1, y1, x2, y2])
        
        # 3. Crop indexing is [y (rows), x (cols)] and must use int
        croped_img = img[int(y1):int(y2), int(x1):int(x2), :]
        cropped_mask = mask[int(y1):int(y2), int(x1):int(x2),]
        croped_img = cv2.resize(croped_img,(resize_size_W,resize_size_H),cv2.INTER_LANCZOS4)
        cropped_mask = cv2.resize(cropped_mask,(resize_size_W, resize_size_H),cv2.INTER_NEAREST)
        # --- [Note] check color channels and dtype ---
        # If sampling_imgs was loaded as RGB, convert to BGR for OpenCV saving
        # croped_img = cv2.cvtColor(croped_img, cv2.COLOR_RGB2BGR)
        # If values are floats in 0~1, convert to uint8 in 0~255
        # croped_img = (croped_img * 255).astype(np.uint8) 
        # ----------------------------------------
        
        # 4. cv2.imwrite must receive the image as its second argument
        save_path = os.path.join(head_img_folder, img_name + '.png')
        cv2.imwrite(save_path, croped_img)
        save_path = os.path.join(head_mask_folder, img_name + '.png')
        cv2.imwrite(save_path, cropped_mask)

    # 5. The old x_min, y_min, etc. do not fit the loop; just stack into an np.array
    bboxes_arr = np.array(bboxes) # final shape: (B, 4)
    np.save(head_bbox_file, bboxes_arr)

    print(f"[{len(bboxes_arr)}] head images cropped and saved.")
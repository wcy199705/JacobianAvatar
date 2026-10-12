import torch
from torch import nn
from typing import Union
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import axis_angle_to_matrix
import copy
import torch.nn.functional as F
import math


import lpips

    #if is_sparse:
    #    return sp_csc(diags(d_area))
    #return diags(d_area)

def vertex_disp_propagation_loss(mesh, disp, mask=None, cot=False):

    """Compute vertex color propagation loss using Laplacian matrix.
    
    This loss encourages smooth color distribution across the mesh surface
    by minimizing the Laplacian of the vertex colors. It helps propagate
    colors from regions with defined colors to neighboring vertices.
    
    Args:
        mesh: Mesh object containing vertices, colors, and Laplacian matrix
        mask: Optional boolean mask to apply loss only to specific vertices
    
    Returns:
        loss: Scalar loss value for color smoothness
    """
    if not cot:
        L = mesh.laplacian  # Sparse Laplacian matrix
    attr = disp  # Vertex colors [N, 3]
    
    if len(attr.shape) == 2:
        # Single mesh case
        # L @ colors computes the discrete Laplacian of colors
        laplacian_colors = L.mm(attr)
        # Squared norm encourages smoothness
        loss = laplacian_colors.norm(dim=1) ** 2
        
        if mask is not None:
            loss = loss[mask]
        
        return loss.mean()
    
    elif len(attr.shape) == 3:
        # Batch case
        loss = 0.0
        B = attr.shape[0]
        for k in range(B):
            if cot:
                from utils.geometry.mesh_utils import compute_laplacian_cot
                with torch.no_grad():
                    L = compute_laplacian_cot(mesh.xyz[k],F=mesh.face,device=mesh.device)

            laplacian_colors = L.mm(attr[k])
            loss_k = laplacian_colors.norm(dim=1) ** 2
            
            if mask is not None:
                loss_k = loss_k[mask[k]]
            
            loss += loss_k.mean()
        
        return loss / B
    

def compute_semantic_laplacian_loss(attribute, face_laplacian, label_indices, weight_config):
    """
    Computes Laplacian loss with per-part (semantic) weights.
    If 'hand' is set, it is applied to both lhand and rhand.

    Args:
        albedo: [B, F, C]
        face_laplacian: [F, F]
        label_indices: dict {'head': idxs, 'lhand': idxs, 'rhand': idxs}
        weight_config: dict {'default': 1.0, 'head': 5.0, 'hand': 10.0 ...}
    """
    if attribute.ndim==2:
        if attribute.shape[-1] > 100:
            attribute = attribute.unsqueeze(dim=-1)
        else:
            attribute = attribute.unsqueeze(dim=0)
    B, F, C = attribute.shape
    device = attribute.device

    # 1. Initialize with the default value
    # Shape [B, F] (single mesh weight case)
    weights = torch.ones(B, F, device=device) * weight_config.get('default', 1.0)

    # -----------------------------------------------------------
    # 2. Apply per-part weights
    # -----------------------------------------------------------
    
    # (1) Apply head
    if 'head' in weight_config:
        weights[:, label_indices.head] = weight_config['head']

    # (2) Apply unified hand setting
    # If config has 'hand' -> apply the value to both lhand and rhand indices
    if 'hand' in weight_config:
        hand_weight = weight_config['hand']
        
        weights[:, label_indices.lhand] = hand_weight
        weights[:, label_indices.rhand] = hand_weight

    # (3) Per-hand settings (optional)
    # If both 'hand' and 'lhand' exist, order so that the more specific 'lhand' overrides
    if 'lhand' in weight_config:
        weights[:, label_indices.lhand] = weight_config['lhand']
        
    if 'rhand' in weight_config :
        weights[:, label_indices.rhand] = weight_config['rhand']

    # -----------------------------------------------------------

    # 3. Compute loss (using the previously optimized function)
    loss_val = face_disp_propagation_loss(attribute, face_laplacian, face_weights=weights)
    
    return loss_val


def compute_semantic_base_loss(loss, label_indices, weight_config, vts=False):
    """
    Computes Laplacian loss with per-part (semantic) weights.
    If 'hand' is set, it is applied to both lhand and rhand.

    Args:
        albedo: [B, F, C]
        face_laplacian: [F, F]
        label_indices: dict {'head': idxs, 'lhand': idxs, 'rhand': idxs}
        weight_config: dict {'default': 1.0, 'head': 5.0, 'hand': 10.0 ...}
    """
    if loss.ndim==2:
        if loss.shape[-1] > 100:
            loss = loss.unsqueeze(dim=-1)
        else:
            loss = loss.unsqueeze(dim=0)
    B, F, C = loss.shape
    device = loss.device

    # 1. Initialize with the default value
    # Shape [B, F] (single mesh weight case)
    weights = torch.ones((B, F, 1), device=device) * weight_config['default'] #.get('default', 1.0)

    # -----------------------------------------------------------
    # 2. Apply per-part weights
    # -----------------------------------------------------------
    
    # (1) Apply head
    if 'head' in weight_config:
        if vts:
            head_labels = label_indices.head_vts
        else:
            head_labels = label_indices.head

        weights[:, head_labels] = weight_config['head']

    # (2) Apply unified hand setting
    # If config has 'hand' -> apply the value to both lhand and rhand indices
    if vts:
        lhand_labels = label_indices.lhand_vts
        rhand_labels = label_indices.rhand_vts

    else:
        lhand_labels = label_indices.lhand
        rhand_labels = label_indices.rhand

    if 'hand' in weight_config:
        hand_weight = weight_config['hand']


        weights[:, lhand_labels] = hand_weight
        weights[:, rhand_labels] = hand_weight

    # (3) Per-hand settings (optional)
    # If both 'hand' and 'lhand' exist, order so that the more specific 'lhand' overrides
    if 'lhand' in weight_config:
        weights[:, lhand_labels] = weight_config['lhand']
        
    if 'rhand' in weight_config:
        weights[:, rhand_labels] = weight_config['rhand']
    loss_val = (loss*weights).mean()
    # -----------------------------------------------------------

    # 3. Compute loss (using the previously optimized function)
    
    return loss_val


def face_disp_propagation_loss(disp, face_laplacian, face_weights=None, mask=None):
    """
    Vectorized version for [B, F, C] input avoiding for-loops.
    
    Args:
        disp: [B, F, C] (Batch, Faces, Channels) - e.g., Displacements
        face_laplacian: [F, F] (Sparse or Dense Matrix)
        face_weights: [B, F] (Per-face weights) or [F] (Shared weights) or None
        mask: [B, F] or [F] or None
    """
    
    # If input is [F, C], expand to [1, F, C] for unified handling
    if disp.ndim == 2:
        disp = disp.unsqueeze(0)
    
    B, F, C = disp.shape

    # ---------------------------------------------------------
    # 1. Reshape Trick for Batch Processing
    # [B, F, C] -> [F, B, C] -> [F, B * C]
    # This allows a single multiplication with the Laplacian [F, F].
    # ---------------------------------------------------------
    disp_reshaped = disp.permute(1, 0, 2).reshape(F, -1)  # [F, B*C]
    
    # 2. Laplacian Multiplication
    # [F, F] @ [F, B*C] -> [F, B*C]
    diff_reshaped = face_laplacian.mm(disp_reshaped)
    
    # 3. Reshape Back to [B, F, C]
    # [F, B*C] -> [F, B, C] -> [B, F, C]
    diff = diff_reshaped.view(F, B, C).permute(1, 0, 2)
    
    # ---------------------------------------------------------
    # 4. Loss Computation
    # ---------------------------------------------------------
    
    # L2 Norm per face (dim=2 is the channel axis: x,y,z)
    # Result shape: [B, F]
    loss_per_face = torch.norm(diff, dim=2) ** 2
    # loss_per_face = (diff ** 2).sum(dim=2)
    # 5. Apply Weights (if exists)
    if face_weights is not None:
        # Use [1, F] if face_weights is [F], or as-is if [B, F]
        # PyTorch broadcasting handles this automatically, but check dims explicitly
        if face_weights.ndim == 1:
            weights = face_weights.unsqueeze(0) # [1, F]
        else:
            weights = face_weights # [B, F]
            
        loss_per_face = loss_per_face * weights

    # 6. Apply Mask (if exists)
    if mask is not None:
        # Select only entries where mask is True into a 1D tensor
        loss_per_face = loss_per_face[mask]
    
    # 7. Final Mean Loss
    return loss_per_face.mean()
    


class LPIPS(nn.Module):
    def __init__(self, img_size, spatial=False, downsample_factor=2):
        super(LPIPS, self).__init__()

        self.lpips = lpips.LPIPS(net='vgg', spatial=spatial).eval().cuda()

        print("Disabling gradients for LPIPS network parameters.")
        for param in self.lpips.parameters():
            param.requires_grad = False

        H, W = img_size
        self.downsample_factor = downsample_factor

        # Compute padding based on the size after downsampling
        H_ds = H // downsample_factor
        W_ds = W // downsample_factor
        self.H_new = ((H_ds + 31) // 32) * 32
        self.W_new = ((W_ds + 31) // 32) * 32
        pad_h = self.H_new - H_ds
        pad_w = self.W_new - W_ds
        self.pad = (0, pad_w, 0, pad_h)

    def forward(self, img_out, img_target, render_mask):
        """
        img_out: B,H,W,C [0,1]
        img_target: B,H,W,C [0,1]
        render_mask: B,H,W,1 [0 or 1]
        """
        img_out = img_out * 2 - 1  # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1

        img_out = img_out * render_mask + (1 - render_mask) * img_target.detach()

        img_out = torch.permute(img_out, (0, 3, 1, 2))  # B,C,H,W
        img_target = torch.permute(img_target, (0, 3, 1, 2))

        # Half-size downsample — 4x VRAM saving, nearly identical perceptual quality
        if self.downsample_factor > 1:
            img_out = F.interpolate(img_out, scale_factor=1.0/self.downsample_factor, mode='bilinear', align_corners=False)
            img_target = F.interpolate(img_target, scale_factor=1.0/self.downsample_factor, mode='bilinear', align_corners=False)

        img_out_padded = F.pad(img_out, self.pad, mode='replicate')
        img_target_padded = F.pad(img_target, self.pad, mode='replicate')

        loss = self.lpips(img_out_padded, img_target_padded)
        return loss


def head_lpips(lpips_loss, rendered_img, render_mask, masked_head_gt, bbox):
    from torchvision.ops import roi_align
    """
    rendered_img: (B, H, W, 3) full rendered image
    render_mask: (B, H, W, 1) full rendered mask
    masked_head_gt: (B, H_crop, W_crop, 3) GT head image (already cropped and resized)
    bbox: (B, 4) bounding box coordinates as [x0, y0, x1, y1]
    """
    B, H, W, C = rendered_img.shape
    B_crop, H_crop, W_crop, C_crop = masked_head_gt.shape
    assert B == B_crop
    
    # 1. Concatenate image (3) and mask (1) along the channel dim (dim=3) -> (B, H, W, 4)
    combined_tensor = torch.cat([rendered_img, render_mask], dim=3)
    
    # 2. roi_align expects (B, C, H, W), so permute -> (B, 4, H, W)
    combined_bchw = combined_tensor.permute(0, 3, 1, 2).contiguous()
    
    # 3. Build bbox format: [batch index, x0, y0, x1, y1]
    # bbox is already (x0, y0, x1, y1), so just prepend the batch index.
    batch_indices = torch.arange(B, dtype=bbox.dtype, device=bbox.device).view(-1, 1)
    rois = torch.cat([batch_indices, bbox], dim=1) # shape: (B, 5)
    
    # 4. Crop & resize in one step with roi_align -> (B, 4, H_crop, W_crop)
    # spatial_scale=1.0 means bbox coordinates are in the input image pixel scale.
    cropped_combined_bchw = roi_align(
        combined_bchw, 
        rois, 
        output_size=(H_crop, W_crop), 
        spatial_scale=1.0, 
        aligned=True # option for more accurate pixel alignment
    )
    
    # 5. Restore to (B, H_crop, W_crop, 4)
    cropped_combined = cropped_combined_bchw.permute(0, 2, 3, 1).contiguous()
    
    # 6. Split the concatenated tensor back into image (3 channels) and mask (1 channel)
    # [..., :3] is indices 0,1,2 (RGB), [..., 3:4] is index 3 (mask).
    rendered_head = cropped_combined[..., :3]
    rendered_head_mask = cropped_combined[..., 3:4]
    #cv2.imwrite('head_test.png',rendered_head[0].detach().cpu().numpy()*255)
    #cv2.imwrite('rendered_head_mask.png',rendered_head_mask[0].detach().cpu().numpy()*255)

    #cv2.imwrite('masked_head_gt.png',masked_head_gt[0].detach().cpu().numpy()*255)

    # 7. Pass to the optimized lpips_loss
    loss = lpips_loss(rendered_head, masked_head_gt, rendered_head_mask)
    
    return loss




def head_l1(l1_loss, rendered_img, render_mask, masked_head_gt, bbox):
    from torchvision.ops import roi_align
    """
    rendered_img: (B, H, W, 3) full rendered image
    render_mask: (B, H, W, 1) full rendered mask
    masked_head_gt: (B, H_crop, W_crop, 3) GT head image (already cropped and resized)
    bbox: (B, 4) bounding box coordinates as [x0, y0, x1, y1]
    """
    B, H, W, C = rendered_img.shape
    B_crop, H_crop, W_crop, C_crop = masked_head_gt.shape
    assert B == B_crop
    
    # 1. Concatenate image (3) and mask (1) along the channel dim (dim=3) -> (B, H, W, 4)
    combined_tensor = torch.cat([rendered_img, render_mask], dim=3)
    
    # 2. roi_align expects (B, C, H, W), so permute -> (B, 4, H, W)
    combined_bchw = combined_tensor.permute(0, 3, 1, 2).contiguous()
    
    # 3. Build bbox format: [batch index, x0, y0, x1, y1]
    # bbox is already (x0, y0, x1, y1), so just prepend the batch index.
    batch_indices = torch.arange(B, dtype=bbox.dtype, device=bbox.device).view(-1, 1)
    rois = torch.cat([batch_indices, bbox], dim=1) # shape: (B, 5)
    
    # 4. Crop & resize in one step with roi_align -> (B, 4, H_crop, W_crop)
    # spatial_scale=1.0 means bbox coordinates are in the input image pixel scale.
    cropped_combined_bchw = roi_align(
        combined_bchw, 
        rois, 
        output_size=(H_crop, W_crop), 
        spatial_scale=1.0, 
        aligned=True # option for more accurate pixel alignment
    )
    
    # 5. Restore to (B, H_crop, W_crop, 4)
    cropped_combined = cropped_combined_bchw.permute(0, 2, 3, 1).contiguous()
    
    # 6. Split the concatenated tensor back into image (3 channels) and mask (1 channel)
    # [..., :3] is indices 0,1,2 (RGB), [..., 3:4] is index 3 (mask).
    rendered_head = cropped_combined[..., :3]
    rendered_head_mask = cropped_combined[..., 3:4]
    #cv2.imwrite('head_test.png',rendered_head[0].detach().cpu().numpy()*255)
    #cv2.imwrite('rendered_head_mask.png',rendered_head_mask[0].detach().cpu().numpy()*255)

    #cv2.imwrite('masked_head_gt.png',masked_head_gt[0].detach().cpu().numpy()*255)

    # 7. Pass to the optimized lpips_loss
    loss = l1_loss(rendered_head, masked_head_gt)
    
    return loss





def l1loss(tgt, gt):
    loss = torch.abs(tgt - gt)
    loss = loss.mean()
    return loss

class SSIM(nn.Module): #https://github.com/mks0601/ExAvatar_RELEASE/blob/main/avatar/common/nets/loss.py
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_target, bbox=None, mask=None, window_size=11):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if mask is not None:
            img_out = img_out * mask
            img_target = img_target * mask
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_target, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_target*img_target, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_target, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map
    

class DepthDistillationLoss(nn.Module):#https://github.com/astonishingwolf/NeuralGarmentRecon/blob/master/scripts/losses/utils.py
    def __init__(self, margin=0.05, reduction='mean'):
        super().__init__()

        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
    
    def sample_pairs(self, depth_map, num_boxes=50, box_size=64):
        """
        Vectorized sampling of pairs from small bounding boxes in the depth map
        Args:
            depth_map (torch.Tensor): Input depth map [B, 1, H, W]
            num_boxes (int): Number of boxes to sample per batch
            box_size (int): Size of each bounding box
        Returns:
            Tuple of tensors containing indices for pairs
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device
        
        x_boxes = torch.randint(0, W - box_size, (B, num_boxes), device=device)
        y_boxes = torch.randint(0, H - box_size, (B, num_boxes), device=device)
        
        x1_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        y1_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        x2_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        y2_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        x1 = x_boxes + x1_offset
        y1 = y_boxes + y1_offset
        x2 = x_boxes + x2_offset
        y2 = y_boxes + y2_offset
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, num_boxes)
        point1_indices = torch.stack([batch_indices, y1, x1], dim=-1).view(-1, 3)
        point2_indices = torch.stack([batch_indices, y2, x2], dim=-1).view(-1, 3)
        
        return point1_indices, point2_indices

    def forward(self, pred_depth, target_depth, gt_depth=None):
        """
        Compute depth distillation loss using vectorized operations
        Args:
            pred_depth (torch.Tensor): Predicted depth from DPT [B, 1, H, W]
            rendered_depth (torch.Tensor): Rendered depth map [B, 1, H, W]
            gt_depth (torch.Tensor, optional): Ground truth depth for additional supervision
        Returns:
            torch.Tensor: Computed loss
        """
        pred_depth = pred_depth.squeeze()
        target_depth = target_depth.squeeze()
        if len(pred_depth.shape)==3:
            B = pred_depth.shape[0]
            pred_depth = pred_depth[:,None,:,:]
            target_depth = target_depth[:,None,:,:]

        elif len(pred_depth.shape)==2:
            pred_depth = pred_depth[None,None,:,:]
            target_depth = target_depth[None,None,:,:]
        #predicted_depth = pred_depth.train_depth.mean(dim=3, keepdim=True).permute(0,3,1,2)
        #target_estimated_depth = target_depth.train_target_depth.unsqueeze(0)

        point1_indices, point2_indices = self.sample_pairs(target_depth)
        
        depth1_pred = target_depth[point1_indices[:, 0], 0, point1_indices[:, 1], point1_indices[:, 2]]
        depth2_pred = target_depth[point2_indices[:, 0], 0, point2_indices[:, 1], point2_indices[:, 2]]
        depth1_rendered = pred_depth[point1_indices[:, 0], 0, point1_indices[:, 1], point1_indices[:, 2]]
        depth2_rendered = pred_depth[point2_indices[:, 0], 0, point2_indices[:, 1], point2_indices[:, 2]]
        
        target = torch.sign(depth2_pred - depth1_pred)
        
        loss = self.ranking_loss(depth1_rendered, depth2_rendered, target)

        return loss
"""
def normal_l1_loss (norm_out, norm_tgt, overlap_mask): #check the output
    normal_out = norm_out#*2-1 # [0,1] -> [-1,1]
    normal_tgt = norm_tgt*2-1 # [0,1] -> [-1,1]
    loss = torch.abs(normal_out - normal_tgt)*overlap_mask
    loss = loss.mean()
    return loss
def normal_cos_loss(norm_out, norm_tgt, overlap_mask):
    normal_out = norm_out# *2-1 # [0,1] -> [-1,1]
    normal_tgt = norm_tgt*2-1 # [0,1] -> [-1,1]
    loss = (1 - torch.sum(normal_out*normal_tgt,1)[:,None])*overlap_mask
    loss = loss.mean()
    return loss
"""

#from persona

# https://github.com/facebookresearch/sapiens/blob/3e829ac27476e4a70b6a01f85e487492afe02df1/seg/mmseg/models/losses/metric_silog_loss.py
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_out, depth_tgt):
        batch_size = depth_out.shape[0]

        loss = 0
        for i in range(batch_size):
            # get foreground
            is_valid = (depth_out[i] > 0) * (depth_tgt[i] > 0)
            depth_out_i = depth_out[i][is_valid]
            depth_tgt_i = depth_tgt[i][is_valid]

            # normalize in [0,1]
            depth_out_normalized = (depth_out_i - torch.min(depth_out_i)) / (torch.max(depth_out_i) - torch.min(depth_out_i) + 1e-4)
            depth_tgt_normalized = (depth_tgt_i - torch.min(depth_tgt_i)) / (torch.max(depth_tgt_i) - torch.min(depth_tgt_i) + 1e-4)
            is_valid = (depth_out_normalized > 0) * (depth_tgt_normalized > 0)
            depth_out_normalized, depth_tgt_normalized = depth_out_normalized[is_valid], depth_tgt_normalized[is_valid]

            # compute loss
            diff_log = torch.log(depth_tgt_normalized) - torch.log(depth_out_normalized)
            diff_log_mean = torch.mean(diff_log)
            diff_log_sq_mean = torch.mean(diff_log**2)
            loss += torch.sqrt(diff_log_sq_mean - 0.5*diff_log_mean**2)
        loss = loss / batch_size
        return loss

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, normal_out, normal_tgt):
        is_valid_out = 1 - ((normal_out==0).sum(1)[:,None] == 3).float()
        is_valid_tgt = 1 - ((normal_tgt==0).sum(1)[:,None] == 3).float()
        normal_out = normal_out*2-1 # [0,1] -> [-1,1]
        normal_tgt = normal_tgt*2-1 # [0,1] -> [-1,1]
        loss = (torch.abs(normal_out - normal_tgt) + (1 - torch.sum(normal_out*normal_tgt,1)[:,None])) * is_valid_out * is_valid_tgt
        return loss

class WAFTLoss(nn.Module):
    def __init__(self, model, heat_threshold=0):
        super(WAFTLoss, self).__init__()
        self.model = model
        self.heat_threshold = heat_threshold

    def loss_w_images(self, tmp_masks, rendered_mask, of_cur_idxs,of_next_idxs, init_flow, tmp_images ):
        if tmp_images.shape[1] !=3:
            tmp_images = tmp_images.permute(0,3,1,2)
        
        curr_mask = tmp_masks[of_cur_idxs]
        next_mask = tmp_masks[of_next_idxs]
        if rendered_mask.shape[0] == tmp_masks.shape[0]:
            curr_rendered_mask = rendered_mask[of_cur_idxs].detach()
        else:
            curr_rendered_mask = rendered_mask.detach()
        curr_overlap_mask = curr_mask*curr_rendered_mask
        curr_overlap_mask = curr_overlap_mask.permute(0,3,1,2)
        init_flow_permute = init_flow.permute(0,3,1,2)

        curr_image = tmp_images[of_cur_idxs]
        next_images = tmp_images[of_next_idxs]
        of_loss = self.forward_w_images(init_flow_permute,curr_image,next_images,curr_overlap_mask)
        return of_loss


    def loss_w_init_of_w_waft_feature(self, tmp_masks, rendered_mask, of_cur_idxs,of_next_idxs, init_flow, tmp_waft_feature,no_grad=False ):
        curr_mask = tmp_masks[of_cur_idxs]
        next_mask = tmp_masks[of_next_idxs]
        if rendered_mask.shape[0] == tmp_masks.shape[0]:
            curr_rendered_mask = rendered_mask[of_cur_idxs].detach()
        else:
            curr_rendered_mask = rendered_mask.detach()
        curr_overlap_mask = curr_mask*curr_rendered_mask
        curr_overlap_mask = curr_overlap_mask.permute(0,3,1,2)
        init_flow_permute = init_flow.permute(0,3,1,2)

        curr_features = tmp_waft_feature[of_cur_idxs]
        next_features = tmp_waft_feature[of_next_idxs]
        if no_grad:
            of_loss = self.forward_w_features_wo_grad(init_flow_permute,curr_features,next_features,curr_overlap_mask,use_init_of=True)

        else:
            of_loss = self.forward_w_features(init_flow_permute,curr_features,next_features,curr_overlap_mask,use_init_of=True)
        return of_loss
    



    def loss_w_gt_of(self, tmp_masks, rendered_mask, of_cur_idxs,of_next_idxs, init_flow, tmp_waft_of ):
        curr_waft_of = tmp_waft_of[of_cur_idxs]

        curr_mask = tmp_masks[of_cur_idxs]
        #next_mask = tmp_masks[of_next_idxs]
        if rendered_mask.shape[0] == tmp_masks.shape[0]:
            curr_rendered_mask = rendered_mask[of_cur_idxs].detach()
        else:
            curr_rendered_mask = rendered_mask.detach()

        curr_overlap_mask = curr_mask*curr_rendered_mask
        curr_overlap_mask = curr_overlap_mask.permute(0,3,1,2)
        init_flow_permute = init_flow.permute(0,3,1,2)
        #cv2.imwrite('mask.png',curr_overlap_mask[0].detach().cpu().numpy()[0]*255)
        #cv2.imwrite('curr_waft_of.png',flow_to_image((curr_waft_of*curr_overlap_mask)[0].permute(1,2,0).detach().cpu().numpy()))
        # cv2.imwrite('init_flow_permute.png',flow_to_image((init_flow_permute*curr_overlap_mask)[0].permute(1,2,0).detach().cpu().numpy()))
        #warp_with_flow()
        of_loss = self.forward_init_w_of(init_flow_permute,curr_waft_of,curr_overlap_mask)
        return of_loss
                       # from libs.WAFT.utils.flow_viz import flow_to_image
                            #cv2.imwrite('init_flow.png',flow_to_image(mesh_init_flow[0].detach().cpu().numpy()))
                            #cv2.imwrite('tmp_waft_of.png',flow_to_image(tmp_waft_of[0].permute(1,2,0).detach().cpu().numpy()))

    def forward_init_w_of(self,init_flow, gt_of, mask):
        n, _, h, w = gt_of.shape

        diff = torch.abs(gt_of - init_flow)
        diff[:,0] = diff[:,0]/ w
        diff[:,1] = diff[:,1]/ h
        return torch.sum(mask*diff)/torch.sum(mask)


    def forward_w_images(self, init_flow, image1, image2, mask) : #, rendered_image1=None,rendered_image2=None ):

        output = None
        #with torch.no_grad():
        n, _, h, w = image1.shape
        #import pdb;pdb.set_trace()
        valid = torch.zeros((n, h, w), device=image1.device)
        if init_flow is None:
            init_flow = torch.zeros(n, 2, h, w).to(image1.device)
        weight = self.generate_gaussian((h, w), 1).to(image1.device).unsqueeze(0)
        output_ij = self.model.forward_w_init_flow(image1, image2, init_flow,iters=3)
        valid += weight

        if output is None:
            output = {}
            for key in output_ij.keys():
                if key!='init_flow':

                    if 'flow' in key:
                        output[key] = [torch.zeros((n, 2, h, w), device=image1.device) for _ in range(len(output_ij[key]))]
                    elif 'info' in key:
                        output[key] = [torch.zeros((n, 4, h, w), device=image1.device) for _ in range(len(output_ij[key]))]
                    else:
                        output[key] = [torch.zeros((n, 2, h, w), device=image1.device) for _ in range(len(output_ij[key]))]

        for i in range(len(output_ij['flow'])):
            for key in output.keys():
                if key!='init_flow':
                    output[key][i] += weight * output_ij[key][i]

        for i in range(len(output['flow'])):
            for key in output.keys():
                if key!='init_flow':
                    output[key][i] /= valid.unsqueeze(1)
    # output['init_flow']= output_ij['init_flow']
        flow = output['flow'][-1]
        """
        from utils.utils import load_ckpt, coords_grid, bilinear_sampler

        def warp_with_flow(image, flow):
            N, _, H, W = image.shape
            coords2 = coords_grid(N, H, W, device=image.device).permute(0, 2, 3, 1)
            coords2 = coords2 + flow.permute(0, 2, 3, 1)
            warped_image = bilinear_sampler(image, coords2)
            return warped_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()


        import cv2
        warped_image = warp_with_flow(image2.detach(),flow.detach())
        cv2.imwrite('w_init_warp.png',warped_image[0])
        cv2.imwrite('img2.png',image2[0].permute(1,2,0).cpu().numpy())
        cv2.imwrite('img1.png',image1[0].permute(1,2,0).cpu().numpy())

        rendered_image1_permute = rendered_image1[...,:3].permute(0,3,1,2).detach()
        rendered_image2_permute = rendered_image2[...,:3].permute(0,3,1,2).detach()

        warped_image = warp_with_flow(rendered_image2_permute.detach(),flow.detach())
        cv2.imwrite('w_init_warp_rndr.png',warped_image[0]*255)
        cv2.imwrite('img2_rndr.png',rendered_image2_permute[0].permute(1,2,0).cpu().numpy()*255)
        cv2.imwrite('img1_rndr.png',rendered_image1_permute[0].permute(1,2,0).cpu().numpy()*255)
        warped_image = warp_with_flow(rendered_image2_permute.detach(),init_flow.detach())

        cv2.imwrite('only_init_warp_rndr.png',warped_image[0]*255)
        """

        diff = torch.abs(flow - init_flow)
        diff[:,0] = diff[:,0]/ w
        diff[:,1] = diff[:,1]/ h
        #warp_with_flow()
        """


        #debugging
        from utils.flow_viz import flow_to_image
        import cv2
        index=2
        flow_init_out = flow_to_image((mask*init_flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('init_flow.png',flow_init_out)
        flow_init_out = flow_to_image((mask*flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('flow.png',flow_init_out)
        flow_init_out = flow_to_image((flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('flow_gt.png',flow_init_out)
        cv2.imwrite('mask.png',255*mask[index].permute(1, 2, 0).detach().cpu().numpy())
        cv2.imwrite('image1.png',image1[index].permute(1, 2, 0).detach().cpu().numpy())
        cv2.imwrite('image2.png',image2[index].permute(1, 2, 0).detach().cpu().numpy())

        with torch.no_grad():
            out_flow = self.model.forward(image1, image2)
            flow_init_out = flow_to_image((out_flow['flow'][-1])[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
            cv2.imwrite('flow_wo_init.png',flow_init_out)
        """
        return torch.sum(mask*diff)/torch.sum(mask)


    def forward_w_features(self, init_flow, feat1, feat2, mask, iters=2, use_init_of=True ):
        device = init_flow.device
        # Ensure float32 for WAFT model compatibility
        feat1 = feat1.float()
        feat2 = feat2.float()
        init_flow = init_flow.float()
        output = None
        #with torch.no_grad():
        n, _, h, w = init_flow.shape
        valid = torch.zeros((n, h, w), device=device)
        #    init_flow = torch.zeros(n, 2, h, w).to(device)
        weight = self.generate_gaussian((h, w), 1).to(device).unsqueeze(0)
        if use_init_of:
            input_flow = init_flow
        else:
            input_flow = torch.zeros_like(init_flow)
        output_ij = self.model.forward_w_init_flow_prefeat(feat1, feat2, input_flow, iters=iters)
        valid += weight

        if output is None:
            output = {}
            for key in output_ij.keys():
                if key!='init_flow':

                    if 'flow' in key:
                        output[key] = [torch.zeros((n, 2, h, w), device=device) for _ in range(len(output_ij[key]))]
                    elif 'info' in key:
                        output[key] = [torch.zeros((n, 4, h, w), device=device) for _ in range(len(output_ij[key]))]
                    else:
                        output[key] = [torch.zeros((n, 2, h, w), device=device) for _ in range(len(output_ij[key]))]

        for i in range(len(output_ij['flow'])):
            for key in output.keys():
                if key!='init_flow':
                    output[key][i] += weight * output_ij[key][i]

        for i in range(len(output['flow'])):
            for key in output.keys():
                if key!='init_flow':
                    output[key][i] /= valid.unsqueeze(1)
    # output['init_flow']= output_ij['init_flow']

        flow = output['flow'][-1]
        diff = torch.abs(flow - init_flow)
        diff[:,0] = diff[:,0]/ w
        diff[:,1] = diff[:,1]/ h
        """
        #debugging
        from utils.flow_viz import flow_to_image
        import cv2
        index=2
        flow_init_out = flow_to_image((mask*init_flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('init_flow.png',flow_init_out)
        flow_init_out = flow_to_image((mask*flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('flow.png',flow_init_out)
        flow_init_out = flow_to_image((flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('flow_gt.png',flow_init_out)
        cv2.imwrite('mask.png',255*mask[index].permute(1, 2, 0).detach().cpu().numpy())
        cv2.imwrite('image1.png',image1[index].permute(1, 2, 0).detach().cpu().numpy())
        cv2.imwrite('image2.png',image2[index].permute(1, 2, 0).detach().cpu().numpy())

        with torch.no_grad():
            out_flow = self.model.forward(image1, image2)
            flow_init_out = flow_to_image((out_flow['flow'][-1])[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
            cv2.imwrite('flow_wo_init.png',flow_init_out)
        """
        return torch.sum(mask*diff)/torch.sum(mask)
    def forward_w_features_wo_grad(self, init_flow, feat1, feat2, mask, iters=2, use_init_of=True ):
        device = init_flow.device
        # Ensure float32 for WAFT model compatibility
        feat1 = feat1.float()
        feat2 = feat2.float()
        init_flow = init_flow.float()
        with torch.no_grad():
            output = None
            #with torch.no_grad():
            n, _, h, w = init_flow.shape
            valid = torch.zeros((n, h, w), device=device)
            #    init_flow = torch.zeros(n, 2, h, w).to(device)
            weight = self.generate_gaussian((h, w), 1).to(device).unsqueeze(0)
            if use_init_of:
                input_flow = init_flow
            else:
                input_flow = torch.zeros_like(init_flow)
            output_ij = self.model.forward_w_init_flow_prefeat(feat1, feat2, input_flow, iters=iters)
            valid += weight

            if output is None:
                output = {}
                for key in output_ij.keys():
                    if key!='init_flow':

                        if 'flow' in key:
                            output[key] = [torch.zeros((n, 2, h, w), device=device) for _ in range(len(output_ij[key]))]
                        elif 'info' in key:
                            output[key] = [torch.zeros((n, 4, h, w), device=device) for _ in range(len(output_ij[key]))]
                        else:
                            output[key] = [torch.zeros((n, 2, h, w), device=device) for _ in range(len(output_ij[key]))]

            for i in range(len(output_ij['flow'])):
                for key in output.keys():
                    if key!='init_flow':
                        output[key][i] += weight * output_ij[key][i]

            for i in range(len(output['flow'])):
                for key in output.keys():
                    if key!='init_flow':
                        output[key][i] /= valid.unsqueeze(1)
        # output['init_flow']= output_ij['init_flow']

            flow = output['flow'][-1]
            flow = flow.detach()
        diff = torch.abs(flow - init_flow)
        diff[:,0] = diff[:,0]/ w
        diff[:,1] = diff[:,1]/ h
        """
        #debugging
        from utils.flow_viz import flow_to_image
        import cv2
        index=2
        flow_init_out = flow_to_image((mask*init_flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('init_flow.png',flow_init_out)
        flow_init_out = flow_to_image((mask*flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('flow.png',flow_init_out)
        flow_init_out = flow_to_image((flow)[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite('flow_gt.png',flow_init_out)
        cv2.imwrite('mask.png',255*mask[index].permute(1, 2, 0).detach().cpu().numpy())
        cv2.imwrite('image1.png',image1[index].permute(1, 2, 0).detach().cpu().numpy())
        cv2.imwrite('image2.png',image2[index].permute(1, 2, 0).detach().cpu().numpy())

        with torch.no_grad():
            out_flow = self.model.forward(image1, image2)
            flow_init_out = flow_to_image((out_flow['flow'][-1])[index].permute(1, 2, 0).detach().cpu().numpy(), convert_to_bgr=True)
            cv2.imwrite('flow_wo_init.png',flow_init_out)
        """
        return torch.sum(mask*diff)/torch.sum(mask)

    def generate_gaussian(self,size, sigma):
        """
        Generate a 2D Gaussian pattern based on the distance to the center.

        Args:
            size (tuple): (height, width) of the output pattern.
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            torch.Tensor: 2D Gaussian pattern.
        """
        height, width = size
        # Create a coordinate grid
        y = torch.arange(0, height).view(-1, 1) / height
        x = torch.arange(0, width).view(1, -1) / width
        
        # Compute the center coordinates
        center_y, center_x = 0.5, 0.5
        
        # Compute the squared distance from each point to the center
        distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
        
        # Apply the Gaussian function
        gaussian = torch.exp(-distance_squared / (2 * sigma ** 2))
        return gaussian


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, seg_out, seg_tgt):
        is_fg = 1 - ((seg_tgt==0).sum(1)[:,None] == 3).float()
        loss = torch.abs(seg_out - seg_tgt) * is_fg
        return loss

class GeoLoss(nn.Module):
    def __init__(self):
        super(GeoLoss, self).__init__()
        self.depth_loss = DepthLoss()
        self.normal_loss = NormalLoss()
        self.seg_loss = SegLoss()

    def forward(self, depth_out, normal_out, seg_out, depth_tgt, normal_tgt, seg_tgt):
        depth_loss = self.depth_loss(depth_out, depth_tgt) * 0.01
        normal_loss = self.normal_loss(normal_out, normal_tgt)
        seg_loss = self.seg_loss(seg_out, seg_tgt)
        loss = depth_loss + normal_loss + seg_loss
        return loss

def laplacian_loss(mesh: Union, mask=None):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """
    L = mesh.laplacian
    V = mesh.xyz

    if len(V.shape) == 2:
        loss = L.mm(V)
        loss = loss.norm(dim=1) ** 2
        return loss.mean()
    elif len(V.shape) == 3: # batch
        loss = 0.0
        for k in range(V.shape[0]):
            loss_ = L.mm(V[k])
            loss_ = loss_.norm(dim=1) ** 2
            loss += loss_.mean()
        loss = loss/V.shape[0] #normalize (cy.)
    if mask is not None:
        loss = loss[mask]
    return loss

def laplacian_offset_loss(mesh, base_verts):
    """Relative laplacian loss: penalizes change in laplacian coordinates from base mesh.

    loss = || L @ V_deformed - L @ V_base ||^2
         = || L @ (V_deformed - V_base) ||^2
         = || L @ offset ||^2

    This allows smooth deformation while penalizing high-frequency offset noise.
    Unlike absolute laplacian_loss which suppresses ALL deformation.

    Args:
        mesh: Mesh with deformed vertices (mesh.xyz) and mesh.laplacian
        base_verts: [V, 3] or [B, V, 3] base (undeformed) vertex positions, detached
    """
    L = mesh.laplacian
    V = mesh.xyz  # deformed vertices

    if len(V.shape) == 2:
        lap_deformed = L.mm(V)
        lap_base = L.mm(base_verts.detach())
        lap_diff = (lap_deformed - lap_base).norm(dim=1) ** 2
        return lap_diff.mean()
    elif len(V.shape) == 3:  # batch
        loss = 0.0
        for k in range(V.shape[0]):
            lap_deformed = L.mm(V[k])
            bv = base_verts[k] if base_verts.dim() == 3 else base_verts
            lap_base = L.mm(bv.detach())
            lap_diff = (lap_deformed - lap_base).norm(dim=1) ** 2
            loss += lap_diff.mean()
        return loss / V.shape[0]


def edge_length_preservation_loss(verts_deformed, verts_base, faces):
    """Penalize change in edge lengths after deformation.

    loss = mean(| ||e_deformed|| - ||e_base|| |^2)

    Args:
        verts_deformed: [B, V, 3] or [V, 3] deformed vertices
        verts_base: [V, 3] base vertices (detached)
        faces: [F, 3] face indices
    """
    # Get unique edges from faces
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)  # [3F, 2]
    edges = torch.unique(edges.sort(dim=1)[0], dim=0)  # unique undirected edges

    if verts_deformed.dim() == 2:
        verts_deformed = verts_deformed.unsqueeze(0)

    base_edge_len = (verts_base[edges[:, 0]] - verts_base[edges[:, 1]]).norm(dim=-1)  # [E]

    loss = 0.0
    B = verts_deformed.shape[0]
    for b in range(B):
        deformed_edge_len = (verts_deformed[b, edges[:, 0]] - verts_deformed[b, edges[:, 1]]).norm(dim=-1)
        loss += ((deformed_edge_len - base_edge_len.detach()) ** 2).mean()
    return loss / B


def pmesh_arap_loss(verts_deformed, verts_ref, faces, cos_weight=0.5):
    """Posed-mesh ARAP regularizer: pull posed mesh (after J+Poisson) toward the
    LBS-only reference posed mesh in edge length AND edge direction.

    Attacks the supervision null-space: J components that move invisible (back-side)
    vertices get no gradient from 2D losses; this loss provides a 3D geometric
    anchor on every face so arbitrary J directions in the null-space are penalized.

    loss = mean_e (||e_def|| - ||e_ref||)^2  + cos_weight * mean_e (1 - cos(e_def, e_ref))

    Args:
        verts_deformed: [B, V, 3] or [V, 3] (F_P_davts — posed mesh after J)
        verts_ref:      [B, V, 3] or [V, 3] (F_P_base_vts — LBS+pose_dirs, detached)
        faces:          [F, 3]
        cos_weight:     cosine term weight (default 0.5 per Avatar Expert sketch)
    """
    import torch.nn.functional as _F
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)
    edges = torch.unique(edges.sort(dim=1)[0], dim=0)
    if verts_deformed.dim() == 2:
        verts_deformed = verts_deformed.unsqueeze(0)
    if verts_ref.dim() == 2:
        verts_ref = verts_ref.unsqueeze(0)
    verts_ref = verts_ref.detach()
    vi, vj = edges[:, 0], edges[:, 1]
    e_def = verts_deformed[:, vi] - verts_deformed[:, vj]        # (B, E, 3)
    e_ref = verts_ref[:, vi] - verts_ref[:, vj]                   # (B, E, 3)
    len_err = (e_def.norm(dim=-1) - e_ref.norm(dim=-1)).pow(2).mean()
    cos_err = (1.0 - _F.cosine_similarity(e_def, e_ref, dim=-1)).mean()
    return len_err + cos_weight * cos_err


def face_normal_consistency_loss(verts, faces, edge_face_pairs):
    """Penalize difference between normals of adjacent faces (sharing an edge).

    Args:
        verts: [V, 3] deformed vertices
        faces: [F, 3] face indices
        edge_face_pairs: [E, 2] pairs of face indices that share an edge

    Returns:
        scalar: mean (1 - cos(n_a, n_b)) over adjacent face pairs
    """
    fv = verts[faces.long()]  # [F, 3, 3]
    fn = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=-1)
    fn_norm = fn.norm(dim=-1, keepdim=True)
    valid = (fn_norm > 1e-6).squeeze(-1)  # skip degenerate faces
    fn = fn / (fn_norm + 1e-8)
    n_a = fn[edge_face_pairs[:, 0]]
    n_b = fn[edge_face_pairs[:, 1]]
    valid_pairs = valid[edge_face_pairs[:, 0]] & valid[edge_face_pairs[:, 1]]
    cos = (n_a * n_b).sum(dim=-1)
    cos = cos.clamp(-1.0, 1.0)
    loss_per_pair = 1.0 - cos
    if valid_pairs.any():
        return loss_per_pair[valid_pairs].mean()
    return torch.zeros(1, device=verts.device, requires_grad=True).squeeze()


def build_edge_face_pairs(faces):
    """Build [E, 2] tensor of face index pairs that share an edge.
    Returns tensor on same device as faces."""
    import numpy as np
    F_np = faces.detach().cpu().numpy().astype(np.int64)
    edge_dict = {}  # (v_lo, v_hi) -> list of face idxs
    for fi, (a, b, c) in enumerate(F_np):
        for u, v in [(a, b), (b, c), (c, a)]:
            key = (min(u, v), max(u, v))
            edge_dict.setdefault(key, []).append(fi)
    pairs = []
    for face_list in edge_dict.values():
        if len(face_list) == 2:
            pairs.append(face_list)
    pairs = torch.tensor(pairs, dtype=torch.long, device=faces.device) if pairs else torch.zeros((0, 2), dtype=torch.long, device=faces.device)
    return pairs


def tv_loss(x):
    """Total Variation loss on image/feature map.
    Supports [B,H,W,C] (rendered normal) or [B,C,H,W]. Detects by last-dim=3.
    """
    if x.dim() == 4 and x.shape[-1] == 3:
        dh = (x[:, 1:, :, :] - x[:, :-1, :, :]).pow(2).mean()
        dw = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
    else:
        dh = (x[..., 1:, :] - x[..., :-1, :]).pow(2).mean()
        dw = (x[..., :, 1:] - x[..., :, :-1]).pow(2).mean()
    return dh + dw


def triplane_tv_loss(planes):
    """
    Total Variation loss on triplane planes for spatial smoothness.

    Args:
        planes: [1, C, H, W] triplane parameter (C = 3 * feature_dim)

    Returns:
        tv_loss: scalar
    """
    tv_h = ((planes[:, :, 1:, :] - planes[:, :, :-1, :]) ** 2).mean()
    tv_w = ((planes[:, :, :, 1:] - planes[:, :, :, :-1]) ** 2).mean()
    return tv_h + tv_w


def chamfer_distance_pytorch3d(v1, v2, n1=None, n2=None):
    # wrapper function for chamfer distance in pytorch3d
    loss_c, loss_n = chamfer_distance(v1, v2, x_normals=n1, y_normals=n2)

    return loss_c if loss_n is None else loss_c + loss_n




""""
def self_intersection_penalty(faces, vertices, adjacent_mask, threshold=0.01):
    # NOTE: oom occurs -> erase.
    #faces: ( num_faces, 3) - vertex indices forming each face - same in canonical mesh.
    #vertices: (batch, num_vertices, 3) - vertex coordinates
    #adjacent_mask: (num_faces, num_faces) - 1 for adjacent face pairs, 0 otherwise
    #threshold: distance threshold below which a penalty is applied
    # Compute face centroids
    centroids = (vertices[:,faces[..., 0]] + vertices[:,faces[..., 1]] + vertices[:,faces[:, 2]]) / 3
    
    # Compute distances between all face pairs
    num_faces = faces.shape[0]
    centroids_exp = centroids.unsqueeze(1).expand(-1, num_faces, -1, -1)
    distances = torch.norm(centroids_exp - centroids_exp.transpose(1, 2), dim=2)
    
    # Mask of non-adjacent face pairs (excluding self and adjacent faces)
    non_adjacent_mask = 1 - adjacent_mask - torch.eye(num_faces, device=faces.device)
    
    # Penalize non-adjacent face pairs closer than threshold
    close_pairs = (distances < threshold) & (non_adjacent_mask > 0)
    penalties = torch.clamp(threshold - distances, min=0)
    loss = torch.sum(penalties * close_pairs.float())
    return loss
"""
def project_and_sample_features(
    pcd_xyz: torch.Tensor,
    image_size,
    feature_map: torch.Tensor,
    camera_K: torch.Tensor,
    camera_R: torch.Tensor,
    camera_t: torch.Tensor,
    visibility_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects pcd_xyz onto the given feature_map/camera and
    returns the feature vectors at those locations and a validity mask.

    Args:
        pcd_xyz (torch.Tensor): 3D point cloud. Shape: (B, N, 3)
        feature_map (torch.Tensor): Feature map or image to sample from. Shape: (B, H, W, C)
        camera_K, camera_R, camera_t: camera parameters
        visibility_mask (torch.Tensor): Occlusion mask. Shape: (B, N)

    Returns:
        torch.Tensor: Sampled feature vectors. Shape: (B, N, C)
        torch.Tensor: Mask of points valid in this view. Shape: (B, N)
    """
    from utils.geometry.camera_utils import projection

    H, W = image_size
    B, _, _, C = feature_map.shape # take channel count (C) as a variable.
    num_points = pcd_xyz.shape[1]
    device = pcd_xyz.device

    if B == 0:
        # Handle empty batch (channel count generalized to C)
        return (
            torch.zeros(B, num_points, C, device=device),
            torch.zeros(B, num_points, dtype=torch.bool, device=device)
        )

    # 1. Project 3D points onto the 2D image plane (same logic)
    uv = projection(pcd_xyz, camera_K, camera_R, -camera_t.unsqueeze(-1))

    # 2. Normalize coordinates for F.grid_sample (same logic)
    u, v = uv[..., 0], uv[..., 1]
    normalized_u = (2.0 * u / (W - 1)) - 1.0
    normalized_v = (2.0 * v / (H - 1)) - 1.0
    grid = torch.stack((normalized_u, normalized_v), dim=-1).unsqueeze(1)

    # 3. Sample features from the feature map
    # Variable renamed from image -> feature_map
    feature_map_for_sampling = feature_map.permute(0, 3, 1, 2) # (B, C, H, W)
    sampled_features = F.grid_sample(
        feature_map_for_sampling, grid, mode='bilinear', padding_mode='border', align_corners=False
    ).squeeze(2).permute(0, 2, 1) # (B, N, C)
    #from scene.object.mesh import Mesh
    #Mesh(vertices=pcd_xyz[0].detach(),indices=torch.arange(pcd_xyz[0].shape[0])[:,None].repeat(1,3).detach(),colors=sampled_features[0].detach()).to_trimesh().export('debug.ply')
    # 4. Compute validity mask for this view (same logic)
    in_frame_mask = (grid >= -1) & (grid <= 1)
    in_frame_mask = in_frame_mask.all(dim=-1).squeeze(1)

    valid_mask_for_this_view = in_frame_mask & visibility_mask
    #s#ampled_vts =pcd_xyz[0][valid_mask_for_this_view[0]].detach()
    #Mesh(vertices=sampled_vts,indices=torch.arange(sampled_vts[0].shape[0],device=pcd_xyz.device)[:,None].repeat(1,3).detach(),colors=sampled_features[0].detach()[valid_mask_for_this_view[0]]).to_trimesh().export('debug.ply')

    return sampled_features, valid_mask_for_this_view

def project_points_to_2d(
    pcd_xyz: torch.Tensor,
    image_size: tuple[int, int],
    camera_K: torch.Tensor,
    camera_R: torch.Tensor,
    camera_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects pcd_xyz onto the 2D image plane with the given camera parameters,
    and returns the (u,v) coordinates along with a validity mask
    indicating whether each point lies inside the image frame.

    Args:
        pcd_xyz (torch.Tensor): 3D point cloud. Shape: (B, N, 3)
        image_size (tuple): Image (height, width).
        camera_K (torch.Tensor): Camera intrinsics. Shape: (B, 3, 3)
        camera_R (torch.Tensor): Camera rotation matrix. Shape: (B, 3, 3)
        camera_t (torch.Tensor): Camera translation vector. Shape: (B, 3)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - uv_coords (torch.Tensor): Projected (u,v) pixel coordinates. Shape: (B, N, 2)
        - in_frame_mask (torch.Tensor): Boolean mask indicating whether points lie
                                        inside the image frame. Shape: (B, N)
    """
    # Assumes this utility function is used the same as in the existing code.
    from utils.geometry.camera_utils import projection

    H, W = image_size
    B, num_points, _ = pcd_xyz.shape
    device = pcd_xyz.device

    if B == 0:
        # Handle empty batch
        return (
            torch.zeros(B, num_points, 2, device=device),
            torch.zeros(B, num_points, dtype=torch.bool, device=device)
        )

    # 1. Project 3D points onto the 2D image plane
    # This part is identical to the existing code.
    uv_coords = projection(pcd_xyz, camera_K, camera_R, -camera_t.unsqueeze(-1))

    # 2. Compute validity mask for this view
    # Check whether projected coordinates lie within image bounds (0 <= u < W, 0 <= v < H).
    u, v = uv_coords[..., 0], uv_coords[..., 1]

    mask_u = (u >= 0) & (u < W)
    mask_v = (v >= 0) & (v < H)
    in_frame_mask = mask_u & mask_v

    return uv_coords, in_frame_mask



def batched_patch_ncc_group_conv(templates: torch.Tensor,
                           search_regions: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """
    Computes batched NCC over small search regions very fast using grouped convolution.
    Much more efficient than F.unfold when H, W are close to P.

    Args:
        templates (torch.Tensor): Template patches. Format (B, N, P, P, C).
        search_regions (torch.Tensor): Search regions. Format (B, N, H, W, C).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: NCC map for each pair. Format (B, N, H-P+1, W-P+1).
    """
    B, N, P, _, C = templates.shape
    _, _, H, W, _ = search_regions.shape
    
    # 1. Permute and reshape: merge B, N, C into a single dim for computation
    # (B, N, C, P, P) -> (B*N*C, 1, P, P)
    templates_conv = templates.permute(0, 1, 4, 2, 3).reshape(B * N * C, 1, P, P)
    # (B, N, C, H, W) -> (1, B*N*C, H, W)
    search_conv = search_regions.permute(0, 1, 4, 2, 3).reshape(1, B * N * C, H, W)
    
    # Set groups to B*N*C so each channel runs an independent 2D convolution
    groups = B * N * C
    
    # --- Compute the NCC formula with grouped convolution ---
    
    # 2. Compute each term
    # Sum(t*f)
    sum_tf = F.conv2d(search_conv, templates_conv, groups=groups)
    
    # ones kernel for computing Sum(t) and Sum(f)
    ones_kernel = torch.ones_like(templates_conv)
    sum_t = F.conv2d(torch.ones_like(search_conv), templates_conv, groups=groups) # sum of template is constant
    sum_f = F.conv2d(search_conv, ones_kernel, groups=groups)
    
    # Sum(t^2) and Sum(f^2)
    sum_t2 = F.conv2d(torch.ones_like(search_conv), templates_conv**2, groups=groups) # sum of squared template
    sum_f2 = F.conv2d(search_conv**2, ones_kernel, groups=groups)
    
    # 3. Compute mean and variance
    num_pixels = P * P
    mean_t = sum_t / num_pixels
    mean_f = sum_f / num_pixels
    
    # Numerator: Sum(t*f) - N * mean_t * mean_f
    numerator = sum_tf - num_pixels * mean_t * mean_f
    
    # Denominator: Sqrt( (Sum(t^2) - N*mean_t^2) * (Sum(f^2) - N*mean_f^2) )
    var_t = sum_t2 - num_pixels * mean_t**2
    var_f = sum_f2 - num_pixels * mean_f**2


    #var mask
    var_t = torch.clamp(var_t,0)
    var_f = torch.clamp(var_f,0)
    denominator = torch.sqrt(var_t * var_f + eps)

    # 4. Compute NCC map and reshape to final form
    # Current ncc shape: (1, B*N*C, H-P+1, W-P+1)
    output_h, output_w = H - P + 1, W - P + 1
    ncc = numerator / denominator
    
    # Average over the channel (C) dim to get the final NCC score
    ncc_map = ncc.reshape(B, N, C, output_h, output_w).mean(dim=2)
    
    return torch.clamp(ncc_map, -1.0, 1.0)



def NCC_differentiable_patch_matching_loss_padded(
    source_projected_coords: torch.Tensor,
    target_projected_coords: torch.Tensor,
    source_feature_map: torch.Tensor,
    target_feature_map: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size,
    sample_patch_radius: int,
    patch_radius: int = 2,
    temperature: float = 0.001,
) -> torch.Tensor:
    """
    (Final OOM-fix version) Correctly restructures the grid_sample input dims
    to avoid creating huge intermediate tensors.
    """

    # target_feature_map: (B, H, W, C)
    B,max_n,_ = source_projected_coords.shape
    C_feat = source_feature_map.shape[-1]
    device = source_projected_coords.device

    # 1. Compute patch coordinates (same as before)
    target_radius = patch_radius+sample_patch_radius

    target_patch_size = 2 * target_radius + 1
    sample_patch_size = 2 * sample_patch_radius + 1

    target_P_sq = target_patch_size**2
    source_P_sq = sample_patch_size**2

    #P_sq = patch_size ** 2
    offset_range = torch.arange(-target_radius, target_radius + 1, device=device)
    grid_y, grid_x = torch.meshgrid(-offset_range, offset_range, indexing='ij')
    target_patch_offsets = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
    patch_offsets = torch.stack([grid_x[sample_patch_radius:-sample_patch_radius,sample_patch_radius:-sample_patch_radius],grid_y[sample_patch_radius:-sample_patch_radius,sample_patch_radius:-sample_patch_radius]]).view(1, 1, -1, 2)
    source_patch_offsets = torch.stack([grid_x[patch_radius:-patch_radius,patch_radius:-patch_radius],grid_y[patch_radius:-patch_radius,patch_radius:-patch_radius]]).view(1, 1, -1, 2)

    source_patch_coords = source_projected_coords.unsqueeze(2) + source_patch_offsets # (B, max_n, P*P, 2)
    target_patch_coords = target_projected_coords.unsqueeze(2) + target_patch_offsets # (B, max_n, P*P, 2)
    patch_coords = target_projected_coords.unsqueeze(2) + patch_offsets


    # 2. Normalize coordinates for grid_sample
    H, W = image_size
    image_dims = torch.tensor([W, H], device=device, dtype=torch.float32).view(1, 1, 1, 2)
    source_normalized_patch_coords = (source_patch_coords / (image_dims - 1)) * 2.0 - 1.0
    target_normalized_patch_coords = (target_patch_coords / (image_dims - 1)) * 2.0 - 1.0

    # 3. ✅ Key fix: reshape the grid tensor into a 1D list form
    # (B, max_n, P*P, 2) -> (B, 1, max_n * P*P, 2)
    source_grid = source_normalized_patch_coords.reshape(B, 1, max_n * source_P_sq, 2)
    target_grid = target_normalized_patch_coords.reshape(B, 1, max_n * target_P_sq, 2)

    # 4. Memory-safe grid_sample call
    # Output tensor shape: (B, C, 1, max_n * P*P)
    source_feature_map = source_feature_map.permute(0, 3, 1, 2)
    target_feature_map = target_feature_map.permute(0, 3, 1, 2)

    # Now call grid_sample safely
    source_features = F.grid_sample(
        source_feature_map,
        source_grid,
        mode='bilinear',
        align_corners=False,
        padding_mode='border'
    )
    target_features = F.grid_sample(
        target_feature_map,
        target_grid,
        mode='bilinear',
        align_corners=False,
        padding_mode='border'
    )


    # 5. Restore original shape for subsequent ops
    # (B, C, 1, max_n * P*P) -> (B, max_n * P*P, C) -> (B, max_n, P*P, C)
    source_patch_features = source_features.squeeze(2).permute(0, 2, 1).reshape(B, max_n, sample_patch_size,sample_patch_size, C_feat)
    target_patch_features = target_features.squeeze(2).permute(0, 2, 1).reshape(B, max_n, target_patch_size,target_patch_size, C_feat)

    
    NCC_Map = batched_patch_ncc_group_conv(source_patch_features,target_patch_features).view(B,max_n,-1)
    # 6. Remaining loss computation logic is the same as before
    NCC_Map_mask = (NCC_Map.max(dim=-1).values > 0.8)
    NCC_Map[NCC_Map<=0.8] = - 1e11
    attention_weights = torch.softmax(NCC_Map / temperature, dim=-1)

    expected_coords = torch.einsum('bnp,bnpc->bnc', attention_weights, patch_coords.float())
    regression_loss = torch.norm(target_projected_coords - expected_coords, p=2, dim=-1)
    regression_loss = regression_loss.masked_fill(~(attention_mask*NCC_Map_mask), 0.0)

    total_valid_points = (attention_mask*NCC_Map_mask).sum()
    if total_valid_points > 0:
        loss = regression_loss.sum() / total_valid_points
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def gather_and_pad_vectorized(
    covisible_mask: torch.Tensor,
    feature_curr: torch.Tensor,
    coords_curr: torch.Tensor,
    coords_prev: torch.Tensor,
    coords_next: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts data to padded tensors using covisible_mask without a for loop.
    """
    B, N = covisible_mask.shape
    device = covisible_mask.device

    # 1. Count covisible points per batch item and their maximum
    num_covisible_per_item = covisible_mask.sum(dim=1)
    max_n_covisible = num_covisible_per_item.max()

    # Initialize padded tensor and attention_mask
    padded_curr_features = torch.zeros(B, max_n_covisible, feature_curr.shape[-1], device=device)
    padded_coords_prev = torch.zeros(B, max_n_covisible, 2, device=device)
    padded_coords_next = torch.zeros(B, max_n_covisible, 2, device=device)
    padded_coords_curr = torch.zeros(B, max_n_covisible, 2, device=device)

    attention_mask = torch.arange(max_n_covisible, device=device).unsqueeze(0) < num_covisible_per_item.unsqueeze(1)

    # 2. Extract only indices where actual data exists
    # batch_indices: which batch each belongs to (e.g. [0, 0, 1, 2, 2, 2, ...])
    # point_indices: index among the original N points
    batch_indices, point_indices = covisible_mask.nonzero(as_tuple=True)

    # 3. ✨ Core logic: compute destination indices in the padded tensor ✨
    # Assign each covisible point its order within its batch item.
    # (e.g. if batch 0 has 2 and batch 1 has 1 covisible points -> [0, 1, 0])
    dest_indices = torch.arange(len(batch_indices), device=device) - \
                   torch.repeat_interleave(torch.cumsum(torch.cat((torch.tensor([0], device=device), num_covisible_per_item[:-1])), dim=0),
                                           num_covisible_per_item)

    # 4. Assign values all at once using the computed indices (scatter)
    padded_curr_features[batch_indices, dest_indices] = feature_curr[batch_indices, point_indices]
    padded_coords_prev[batch_indices, dest_indices] = coords_prev[batch_indices, point_indices]
    padded_coords_next[batch_indices, dest_indices] = coords_next[batch_indices, point_indices]
    padded_coords_curr[batch_indices, dest_indices] = coords_curr[batch_indices, point_indices]

    return padded_curr_features,padded_coords_curr, padded_coords_prev, padded_coords_next, attention_mask
"""

def calculate_covisible_geo_loss_patch_based_vectorized(
    pcd_xyz: torch.Tensor,
    features: torch.Tensor,
    image_size,
    camera_K: torch.Tensor,
    camera_R: torch.Tensor,
    camera_t: torch.Tensor,
    visibility_mask: torch.Tensor,
    previous_index,
    current_index,
    next_index,
    patch_radius: int = 13,
    temperature: float = 0.1
) -> torch.Tensor:
   # (Vectorized version) Computes the loss with padding and masking, without a for loop.
    B = pcd_xyz.shape[0]
    device = pcd_xyz.device
    
    # 1. Use the point cloud data (XYZ and RGB) at the current time (t) as the reference
    current_pcd_xyz = pcd_xyz[current_index]
    prev_pcd_xyz = pcd_xyz[previous_index]
    next_pcd_xyz = pcd_xyz[next_index]


    # (same as before) 3. Project and sample colors for each view
    #coords_curr, mask_curr = project_points_to_2d(current_pcd_xyz, image_size, camera_K[current_index], camera_R[current_index], camera_t[current_index])
    coords_prev, mask_prev = project_points_to_2d(prev_pcd_xyz, image_size, camera_K[previous_index], camera_R[previous_index], camera_t[previous_index])
    coords_next, mask_next = project_points_to_2d(next_pcd_xyz, image_size, camera_K[next_index], camera_R[next_index], camera_t[next_index])
    
    feature_curr, valid_mask_curr = project_and_sample_features(
        current_pcd_xyz, image_size, features[current_index],
        camera_K[current_index], camera_R[current_index], camera_t[current_index],
        visibility_mask[current_index]
    )

    
    # 2. Compute covisibility mask and count valid points
    # 2. Compute covisibility mask for points visible in all three views
    covisible_mask = valid_mask_curr & mask_prev & mask_next

    num_covisible_points = covisible_mask.sum()
    if num_covisible_points < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 3. Efficiently extract only info for covisible points
    # Use nonzero to get (batch, point) indices of valid points

    # 3.1. Sample features from the source (current view)
    # (B, N, 2) -> (N_total_covisible, 2)
    # 2. ✨ Replace the for loop with a vectorized function call ✨
    padded_curr_features, padded_coords_prev, padded_coords_next, attention_mask = \
        gather_and_pad_vectorized(covisible_mask, feature_curr, coords_prev, coords_next)
            
    # 3. Safe and efficient loss computation with padded tensors (remaining logic unchanged)
    loss_prev = differentiable_patch_matching_loss_padded(
        padded_coords_prev, padded_curr_features, features[previous_index],
        attention_mask, image_size,patch_radius, temperature,
    )
    loss_next = differentiable_patch_matching_loss_padded(
        padded_coords_next, padded_curr_features, features[next_index],
        attention_mask, image_size, patch_radius, temperature,
    )
    
    return loss_prev + loss_next
"""


# def mesh_edge_loss(meshes):
#     """
#     Computes mesh edge length regularization loss averaged across all meshes
#     in a batch. Each mesh contributes equally to the final loss, regardless of
#     the number of edges per mesh in the batch by weighting each mesh with the
#     inverse number of edges. For example, if mesh 3 (out of N) has only E=4
#     edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
#     contribute to the final loss.
#
#     Args:
#         meshes: Meshes object with a batch of meshes.
#         target_length: Resting value for the edge length.
#
#     Returns:
#         loss: Average loss across the batch. Returns 0 if meshes contains
#         no meshes or all empty meshes.
#     """
#     if meshes.isempty():
#         return torch.tensor(
#             [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
#         )
#     l1_loss = nn.L1Loss()
#     faces_packed = meshes.faces_packed()
#     verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
#     v0 = verts_packed[faces_packed[:, 0]]
#     v1 = verts_packed[faces_packed[:, 1]]
#     v2 = verts_packed[faces_packed[:, 2]]
#
#     e0 = (v0 - v1).norm(dim=1, p=2)
#     e1 = (v1 - v2).norm(dim=1, p=2)
#     e2 = (v2 - v0).norm(dim=1, p=2)
#     avg_len = torch.mean(e0 + e1 + e2)/3
#     loss = l1_loss(e0, e1) + l1_loss(e1, e2) + l1_loss(e2, e0) \
#            + l1_loss(e0, avg_len) + l1_loss(e1, avg_len)+ l1_loss(e2, avg_len)
#
#     return loss.sum(), avg_len

def compute_adjacent_pairs_vectorized(faces):
    """
    faces: (F, 3) LongTensor
    return: (Num_Pairs, 2) LongTensor - pairs of adjacent face indices
    """
    F = faces.shape[0]
    device = faces.device

    # (F, 3, 2) -> edges of each face (v0-v1, v1-v2, v2-v0)
    face_edges = torch.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=1)

    # Sort within each edge (to make it independent of v0, v1 order)
    face_edges, _ = torch.sort(face_edges, dim=-1)

    # Flatten to shape (3F, 2)
    edges_flat = face_edges.view(-1, 2)
    
    # Record which face each edge originally belonged to
    face_indices = torch.arange(F, device=device).repeat_interleave(3)

    # Edge hashing (generate unique IDs)
    # If the vertex index range is large, consider 64-bit integers, etc.
    max_v_idx = edges_flat.max().item() + 1
    edge_hashes = edges_flat[:, 0] * max_v_idx + edges_flat[:, 1]

    # Sort hashes -> entries with the same hash (same edge) become adjacent
    sorted_hashes, sort_idx = torch.sort(edge_hashes)
    sorted_faces = face_indices[sort_idx]

    # Compare hashes of neighboring entries (True means shared edge)
    mask = (sorted_hashes[:-1] == sorted_hashes[1:])

    # Extract indices of the two faces sharing an edge
    idx_a = sorted_faces[:-1][mask]
    idx_b = sorted_faces[1:][mask]

    pairs = torch.stack([idx_a, idx_b], dim=1)

    # Remove self-connections (degenerate case)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    return pairs

class MeshHuberLoss(nn.Module):
    def __init__(self, da_mesh, delta_scale=0.1):
        """
        Args:
            verts (Tensor): vertices of the initial mesh (V, 3) - for scale computation
            faces (Tensor): mesh faces (F, 3) - for edge length computation
            delta_scale (float): ratio of delta to mean edge length (default 0.1)
        """
        super().__init__()
        self.avg_edge_length= self.compute_avg_edge_length(da_mesh.xyz, da_mesh.face)
        self.delta = delta_scale*self.avg_edge_length #  self.compute_avg_edge_length(da_mesh.xyz, da_mesh.face)


        
        self.loss_fn = nn.HuberLoss(reduction='mean', delta=self.delta)
    
    def compute_avg_edge_length( self, verts, faces):
        """
        Internal function that computes all mesh edge lengths and returns their mean
        """
        # For face [v0, v1, v2], edges are (v0, v1), (v1, v2), (v2, v0)
        # Extract indices
        v0_idx = faces[:, 0]
        v1_idx = faces[:, 1]
        v2_idx = faces[:, 2]
        
        # Get start and end point coordinates of each edge
        # (F, 3)
        p0 = verts[:,v0_idx]
        p1 = verts[:,v1_idx]
        p2 = verts[:,v2_idx]
        
        # Compute each edge length (L2 norm)
        len_01 = torch.norm(p0 - p1, dim=-1)
        len_12 = torch.norm(p1 - p2, dim=-1)
        len_20 = torch.norm(p2 - p0, dim=-1)
        
        # Combine all edge lengths and compute the mean
        all_lengths = torch.cat([len_01, len_12, len_20])
        return all_lengths.mean().item()


    def forward(self, src_verts, tgt_verts):
        return self.loss_fn(src_verts, tgt_verts)/(self.avg_edge_length+1e-9)
    
class FaceConsistencyLoss(nn.Module):
    def __init__(self, faces, metric='l2', reduction='mean'):
        """
        Args:
            faces (torch.Tensor): (F, 3) Face topology
            metric (str): 'l1' (absolute) or 'l2' (squared error)
            reduction (str): 'mean' or 'sum'
        """
        super().__init__()
        self.metric = metric
        self.reduction = reduction
        
        # [Key] If the topology is fixed, precompute and store adjacent pairs.
        # With register_buffer, it is included in the model state_dict and moves with to(device).
        pairs = compute_adjacent_pairs_vectorized(faces)
        self.register_buffer('adjacency_pairs', pairs)

    def forward(self, face_values):
        """
        Args:
            face_values (torch.Tensor): tensor of shape (F, D) (e.g. face colors, normals)
                                        F is the number of faces, D is the data dim (RGB=3)
        """
        # Case with no adjacent pairs (e.g. only one face)
        
        if len(face_values.shape)==2:
            face_values = face_values[...,None]
        
        if self.adjacency_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=face_values.device, requires_grad=True)

        # 1. Indexing: fetch the values of adjacent faces.
        #    adjacency_pairs[:, 0] is the list of faces A
        #    adjacency_pairs[:, 1] is the list of faces B
        vals_a = face_values[:,self.adjacency_pairs[:, 0]]
        vals_b = face_values[:,self.adjacency_pairs[:, 1]]
        
        # 2. Compute difference
        diff = vals_a - vals_b
        
        # 3. Apply metric
        if self.metric == 'l2':
            # (vals_a - vals_b)^2
            loss_per_pair = torch.sum(diff ** 2, dim=-1) 
        else: # l1
            # |vals_a - vals_b|
            loss_per_pair = torch.sum(torch.abs(diff), dim=-1)
            
        # 4. Reduction (mean or sum)
        if self.reduction == 'mean':
            return loss_per_pair.mean()
        else:
            return loss_per_pair.sum()



def masked_normal_cosine_loss(normal,gt_nomral,mask, normalize=True):
    if normalize:
        normal = normal*2.0 -1.0
        normal = normal/(torch.norm(normal,dim=-1)[...,None]+1e-8)
    normal_loss = torch.sum(mask - mask*(torch.sum(normal*gt_nomral,dim=-1)[...,None]))/torch.sum(mask)
    return normal_loss

def masked_normal_z_cosine_loss(normal, gt_normal, mask, normalize=True):
    """Z-axis normal cosine loss: 1 - cos(nz, gz) where nz/gz are z-components treated as 1D vectors."""
    if normalize:
        normal = normal * 2.0 - 1.0
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
    nz = normal[..., 2:3]
    gz = gt_normal[..., 2:3]
    cos_sim = nz * gz  # both scalar per pixel, product = cosine of sign agreement + magnitude
    return torch.sum(mask * (1.0 - cos_sim)) / torch.sum(mask)

def masked_normal_z_l1_loss(normal, gt_normal, mask, normalize=True):
    """Z-axis normal L1 loss: |nz - gz|."""
    if normalize:
        normal = normal * 2.0 - 1.0
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
    z_diff = (normal[..., 2:3] - gt_normal[..., 2:3]).abs()
    return torch.sum(mask * z_diff) / torch.sum(mask)



def subdivide_differentiable(vertices, parents):
    """
    Args:
        vertices: (B, N, 3) original vertices
        parents: (M, 2) parent indices obtained from extract_midpoint_recipe
    Returns:
        new_vertices: (B, N+M, 3)
    """
    if len(vertices.shape) == 2:
        vertices =vertices[None,:,:]
    # Indices stored in parents are original vertex indices in range [0 ~ N-1]
    v0 = vertices[:, parents[:, 0]] # (B, M, 3)
    v1 = vertices[:, parents[:, 1]] # (B, M, 3)
    
    midpoints = (v0 + v1) / 2.0
    
    return torch.cat([vertices, midpoints], dim=1)

def extract_midpoint_recipe(new_faces, num_org_verts, num_new_verts_count):
    """
    Analyzes new_faces produced by Trimesh to find,
    for each newly created vertex, the two original (old) vertices it came from.
    
    Args:
        new_faces: (F_new, 3) faces from Trimesh output
        num_org_verts: int (N) number of original vertices
        num_new_verts_count: int (M) number of newly created vertices (total - N)
        
    Returns:
        midpoint_parents: (M, 2) 
            - row 0 is the parents [v_a, v_b] of vertex N
            - row 1 is the parents of vertex N+1 ...
            stored in order.
    """
    device = new_faces.device
    
    # 1. Extract all edges of the new mesh
    # (F, 3, 2) -> (3F, 2)
    edges = torch.cat([
        new_faces[:, [0, 1]],
        new_faces[:, [1, 2]],
        new_faces[:, [2, 0]]
    ], dim=0)
    
    # 2. Unify direction (sort)
    edges, _ = torch.sort(edges, dim=1)
    
    # 3. Remove duplicates
    edges = torch.unique(edges, dim=0) # (E_total, 2)
    
    # 4. Find "parent-child" edges
    # In Loop subdivision, a new vertex (idx >= N) is always connected to two old vertices (idx < N).
    # So we only need to filter edges of the form (Old, New).
    
    # edges[:, 0] is always <= edges[:, 1] (sorted above)
    # Condition: find edges where the smaller is Old (<N) and the larger is New (>=N)
    mask_mixed = (edges[:, 0] < num_org_verts) & (edges[:, 1] >= num_org_verts)
    mixed_edges = edges[mask_mixed] # form [Old_Idx, New_Idx]
    
    # 5. Sort by new vertex
    # Sort by column 1 (New Idx) of mixed_edges
    # This places the two old vertices sharing the same new vertex consecutively
    # e.g. [[10, 100], [25, 100], [5, 101], [50, 101] ...]
    sort_indices = torch.argsort(mixed_edges[:, 1])
    sorted_mixed = mixed_edges[sort_indices]
    
    # 6. Extract parents
    # sorted_mixed should have shape (2*M, 2) (each new vertex has 2 parents)
    # Take only column 0 (old vertices) and reshape to (M, 2)
    parents = sorted_mixed[:, 0].reshape(num_new_verts_count, 2)
    
    return parents


def mesh_mask_loss( pred_mask, gt_mask):

    # Compute difference (L1)
    mask_diff = torch.abs(pred_mask - gt_mask)

    denom = torch.sum(gt_mask)
    loss_val = torch.sum(mask_diff) / (denom + 1e-8)

    return loss_val


def boundary_weighted_mask_loss(pred_mask, gt_mask, sigma=3.0, alpha=3.0):
    """Distance-transform-weighted L1 on the rendered alpha.
    Weight = 1 + alpha * exp(-dist_to_silhouette_edge / sigma), capped to (1, 1+alpha].
    Reason: silhouette pixels are the most accurate monocular cue; amplifying the loss
    there sharpens boundaries without destabilizing interior. Weight normalized by GT area
    so magnitude is comparable with existing `mesh_mask_loss`.
    """
    import numpy as _np
    import scipy.ndimage as _ndi
    # Accept (B, 1, H, W) or (B, H, W)
    if pred_mask.dim() == 4:
        p = pred_mask.squeeze(1)
        g = gt_mask.squeeze(1)
    else:
        p, g = pred_mask, gt_mask
    g_bin = (g > 0.5).detach().cpu().numpy()
    dt_in  = _np.stack([_ndi.distance_transform_edt( m) for m in g_bin])
    dt_out = _np.stack([_ndi.distance_transform_edt(~m) for m in g_bin])
    dt = _np.minimum(dt_in, dt_out).astype(_np.float32)                # (B, H, W)
    w = 1.0 + alpha * _np.exp(-dt / max(sigma, 1e-6))
    w_t = torch.from_numpy(w).to(pred_mask.device)
    diff = torch.abs(p - g) * w_t
    denom = torch.sum(g)
    return torch.sum(diff) / (denom + 1e-8)


def compute_sdf_loss( mesh_normals, query_points, sdf_func):
    """
    Computes the consistency (cosine) loss between mesh normals and SDF gradient (normal)
    """
    # Query the normal (gradient) at the given positions from the SDF volume
    # query_points shape: (N, 3) or (B, N, 3) -> squeezeIfNeeded
    sdf_norm = sdf_func(query_points.squeeze())

    # Cosine Similarity Loss: | 1 - (Normal_mesh • Normal_sdf) |
    # 0 if both normals point the same way (1), 2 if opposite (-1), 1 if orthogonal (0)
    dot_prod = torch.sum(mesh_normals * sdf_norm, dim=-1)
    norm_loss = torch.mean(torch.abs(1 - dot_prod))

    # Logging and return
    return norm_loss

def compute_hinge_sdf_loss(mesh_normals, query_points, sdf_func, option='vts'):
    """
    Computes the consistency (cosine) loss between mesh normals and SDF gradient (normal).
    Applies a hinge loss that ignores errors smaller than the grid resolution limit (margin).
    """
    # 1. Query the normal (gradient) at the given positions from the SDF volume
    # sdf_func is a GridSDFGrad instance
    # If query_points is (B, N, 3), it is handled inside sdf_func
    sdf_norm = sdf_func(query_points)

    # 2. Compute cosine similarity error (point-wise)
    # Loss = | 1 - (Normal_mesh • Normal_sdf) |
    # (B, N) or (N,)
    dot_prod = torch.sum(mesh_normals * sdf_norm, dim=-1)
    per_point_error = torch.abs(1 - dot_prod)

    # 3. Get the hinge margin
    # Fetch 'vts_norm_low_bound' stored on the sdf_func (GridSDFGrad) object.
    # If the attribute is missing, use 0.0 (default) so all errors count.
    if option=='vts':
        margin = getattr(sdf_func, 'vts_norm_low_bound', 0.0)
    elif option=='face':
        margin = getattr(sdf_func, 'face_norm_low_bound', 0.0)


    # If margin is a tensor, convert to a scalar (detach from graph)
    if isinstance(margin, torch.Tensor):
        margin = margin.detach()

    # 4. Apply hinge loss (ReLU)
    # Formula: Loss = max(0, error - margin)
    # Errors below margin become 0; above it, only the excess counts as loss.
    hinge_loss = F.relu(per_point_error - margin)

    # 5. Return the mean
    return torch.mean(hinge_loss)

def acc_loss(prev_vts, curr_vts, next_vts, mass_weights=None):
    
    # 1. Compute acceleration (central difference)
    # a(t) = x(t+1) - 2x(t) + x(t-1)
    acc = next_vts - 2 * curr_vts + prev_vts
    
    # Squared magnitude of acceleration (squared L2 norm)
    acc_sq = (acc ** 2).sum(dim=-1) # (Batch, Time-2, Num_Vertices)
    
    # 2. Apply mass weights (FEM lumped mass matrix concept)
    if mass_weights is not None:
        # Reshape mass_weights for broadcasting
        if mass_weights.ndim == 1:
            mass_weights = mass_weights.view(1,  -1)
        #elif mass_weights.ndim == 2:
        #    mass_weights = mass_weights[None,:,:] #.view(1, 1, -1)

        # From F = ma, roughly minimizing the inertial force
        # From an energy perspective, close to minimizing the rate of change of kinetic energy
        weighted_loss = (acc_sq * mass_weights).mean()
        return weighted_loss
    
    else:
        # Plain mean if no mass information
        return acc_sq.mean()
def calculate_vertex_areas(verts, faces):
    """
    Computes per-vertex area (mass) for the FEM lumped mass matrix
    
    Args:
        verts (Tensor): vertex coordinates of shape (V, 3) (rest pose recommended)
        faces (Tensor): face indices of shape (F, 3) (LongTensor)
        
    Returns:
        vertex_areas (Tensor): shape (V, ). Area assigned to each vertex.
    """
    
    # 1. Gather vertex coordinates of each face (fancy indexing)
    # v0, v1, v2 shape: (F, 3)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # 2. Compute each triangle (face) area via cross product
    # Parallelogram area = ||(v1 - v0) x (v2 - v0)||
    # Triangle area = 0.5 * parallelogram area
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # (F, 3) -> vector
    cross_product = torch.cross(edge1, edge2, dim=1)
    
    # (F, ) -> scalar area per face
    face_areas = 0.5 * torch.norm(cross_product, dim=1)
    
    # 3. Distribute area to vertices (scatter / accumulate)
    # Each face's area is split 1/3 each among its 3 vertices
    vertex_areas = torch.zeros((verts.shape[0]), dtype=verts.dtype, device=verts.device)
    
    # Share for each vertex (face area / 3)
    val_to_add = face_areas / 3.0
    
    # Accumulate area at each vertex index using index_add_
    # Add val_to_add to the vertices in faces[:, 0]
    vertex_areas.index_add_(0, faces[:, 0], val_to_add)
    vertex_areas.index_add_(0, faces[:, 1], val_to_add)
    vertex_areas.index_add_(0, faces[:, 2], val_to_add)
    
    return vertex_areas




class vposer_loss:
    def __init__(self, vposer_sampling):
        self.vposer = vposer_sampling
        self.L2loss = torch.nn.MSELoss(reduce=False)

    def fullpose2boypose_loss(self, FineAvatar, pose1, pose2, offset1, offset2, Cparents=None, CoarseAvatar=None, alpha=None, return_new_pose=False, use_weight=True):
        B,V,_ = offset1.shape 
        if alpha is None:
            alpha = torch.rand(B).to(offset1.device)
        if pose1['full_pose'].dim() == 3: 
            pose1_bodypose = pose1['full_pose'][:, 1:22, :]
        else: 
            pose1_bodypose = pose1['full_pose'][:, 3:66].view(-1, 21, 3)

        if pose2['full_pose'].dim() == 3: 
            pose2_bodypose = pose2['full_pose'][:, 1:22, :]
        else: 
            pose2_bodypose = pose2['full_pose'][:, 3:66].view(-1, 21, 3)
        
        if use_weight:
            similar_weights = self.compute_geodesic_similarity_weights(pose1_bodypose, pose2_bodypose)

        new_pose = copy.deepcopy(pose1)
        new_pose['body_pose'] = self.vposer.interp_pose(body_pose1, body_pose2, alpha)

        total_offset = self.get_new_pose_offset(new_pose, FineAvatar, Cparents, CoarseAvatar)

        interp_offset= offset1*alpha + offset2*(1-alpha)
        if use_weight:
            loss = (self.L2loss(interp_offset - total_offset).view(B,-1) * similar_weights).mean()
        else:
            loss = self.L2loss(interp_offset - total_offset).mean()
        if return_new_pose:
            return loss, new_pose
        else:
            return loss

    def get_new_pose_offset(self, new_pose, FineAvatar, C_parents=None, CoarseAvatar=None):
        if CoarseAvatar is not None:
            C_outputs = CoarseAvatar(target_pose=new_pose ) #, time_index=tmp_time_idxs ) #MLP.get_posed_output(tmp_batch_params, use_noise=use_noise, training=True, return_GS_mesh=use_gaussian_mesh)
            C_offset =  subdivide_differentiable(C_outputs['P_offset'],C_parents.detach())#.detach()

            F_outputs = FineAvatar(target_pose=new_pose,init_offset=C_offset ) #, time_index=tmp_time_idxs )
            total_offset = F_outputs['P_offset'] + C_offset
        else:
            F_outputs = FineAvatar(target_pose=new_pose ) #, time_index=tmp_time_idxs )

            total_offset = F_outputs['P_offset'] 

        return total_offset

    def compute_geodesic_similarity_weights(self, pose1, pose2, tau=0.5, epsilon=1e-7):
        """
        Computes the geodesic distance (rotation angle difference) between two poses (axis-angle),
        returning a larger weight for higher similarity.
        
        Args:
            pose1 (Tensor): axis-angle pose of shape [B, 21, 3] or [B, 63] - bodypose
            pose2 (Tensor): axis-angle pose of shape [B, 21, 3] or [B, 63] - bodypose
            tau (float): temperature parameter. Smaller is more sensitive to similarity (default: 0.5)
            epsilon (float): small value for numerical stability
            
        Returns:
            weights (Tensor): weight tensor of shape [B] (values in 0 ~ 1)
        """
        
        # 1. Fix input shape: [B, -1] -> [B, N_Joints, 3]
        #    Typically N_Joints=21 for SMPL body pose
        B = pose1.shape[0]
        p1 = pose1.view(B, -1, 3)
        p2 = pose2.view(B, -1, 3)

        # 2. Axis-angle -> rotation matrix: [B, J, 3, 3]
        #    This resolves the 2pi periodicity (gimbal-lock-like issue).
        R1 = axis_angle_to_matrix(p1)
        R2 = axis_angle_to_matrix(p2)

        # 3. Compute relative rotation
        #    R_diff = R1 @ R2^T
        #    If the two rotation matrices are identical, this is the identity matrix.
        R_diff = torch.matmul(R1, R2.transpose(-1, -2))

        # 4. Compute trace
        #    Trace(R) = 1 + 2cos(theta)
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        
        # 5. Compute rotation angle (theta) (geodesic distance)
        #    theta = arccos((Trace - 1) / 2)
        cos_theta = (trace - 1) / 2
        
        #    [Important] Clamp cos to [-1, 1] for numerical stability
        #    Floating-point error like 1.000001 causes NaN in acos
        cos_theta = torch.clamp(cos_theta, -1.0 + epsilon, 1.0 - epsilon)
        
        theta = torch.acos(cos_theta) # [B, J] (per-joint angle difference, radians)

        # 6. Compute distance for the whole pose (mean over joints) -> [B]
        pose_distance = theta.mean(dim=1)

        # 7. Apply Gaussian weighting
        #    Distance 0 gives exp(0) = 1 (maximum weight)
        #    Converges to 0 as distance grows
        weights = torch.exp(-(pose_distance ** 2) / tau)

        return weights


# ==========================================================================
# Local-space Jacobian losses
#
# J_local shape (B, N_f, 3, 2):
#   J_local = [[a, b],
#              [c, d],
#              [tx, ty]]
# Identity: [[1,0],[0,1],[0,0]] → J_world = I_3, vertices unchanged.
# ==========================================================================

def local_jac_area_preserve_loss(J_local):
    """Penalize deviation from area-preserving in-plane deformation.

    area = det([[a, b], [c, d]]) = a*d - b*c
    loss = (area - 1)^2

    Args:
        J_local: (..., N_f, 3, 2)
    Returns:
        scalar
    """
    a = J_local[..., 0, 0]
    b = J_local[..., 0, 1]
    c = J_local[..., 1, 0]
    d = J_local[..., 1, 1]
    area = a * d - b * c
    return ((area - 1.0) ** 2).mean()


def local_jac_rigidity_loss(J_local):
    """Penalize stretch away from identity: (a-1)^2 + (d-1)^2.

    Args:
        J_local: (..., N_f, 3, 2)
    """
    a = J_local[..., 0, 0]
    d = J_local[..., 1, 1]
    return ((a - 1.0) ** 2 + (d - 1.0) ** 2).mean()


def local_jac_shear_loss(J_local):
    """Penalize in-plane shear: b^2 + c^2."""
    b = J_local[..., 0, 1]
    c = J_local[..., 1, 0]
    return (b ** 2 + c ** 2).mean()


def local_jac_tilt_loss(J_local):
    """Penalize out-of-plane tilt: tx^2 + ty^2."""
    tx = J_local[..., 2, 0]
    ty = J_local[..., 2, 1]
    return (tx ** 2 + ty ** 2).mean()


def local_jac_identity_loss(J_local):
    """Frobenius distance from identity J_local.

    identity = [[1,0],[0,1],[0,0]]
    """
    a = J_local[..., 0, 0]
    b = J_local[..., 0, 1]
    c = J_local[..., 1, 0]
    d = J_local[..., 1, 1]
    tx = J_local[..., 2, 0]
    ty = J_local[..., 2, 1]
    return ((a - 1.0) ** 2 + (d - 1.0) ** 2
            + b ** 2 + c ** 2 + tx ** 2 + ty ** 2).mean()


def local_jac_laplacian_loss(J_local, edge_face_pairs):
    """Adjacent-face smoothness on J_local (Frobenius distance).

    Args:
        J_local: (..., N_f, 3, 2)
        edge_face_pairs: (E_int, 2) long — face index pairs sharing an edge.
    """
    J_a = J_local[..., edge_face_pairs[:, 0], :, :]
    J_b = J_local[..., edge_face_pairs[:, 1], :, :]
    return ((J_a - J_b) ** 2).sum(dim=(-1, -2)).mean()


def local_jac_arap_loss(J_local):
    """ARAP (As-Rigid-As-Possible) energy on J_local.

    E_ARAP = ‖J_local^T @ J_local - I_2‖²_F

    Penalizes deviation of the induced tangent-plane metric from identity,
    which equals deviation from a rigid 3D isometric embedding. Pure rotations
    (in-plane θ and out-of-plane tilt that preserves metric) have E_ARAP = 0.

    Args:
        J_local: (..., N_f, 3, 2)
    Returns:
        scalar
    """
    # C = J^T @ J  (2x2 symmetric)
    C = torch.einsum("...fij,...fik->...fjk", J_local, J_local)
    # Subtract identity
    I2 = torch.eye(2, device=J_local.device, dtype=J_local.dtype)
    diff = C - I2
    # Frobenius norm squared (element-wise sum of squares)
    return (diff ** 2).sum(dim=(-1, -2)).mean()


def local_jac_arap_laplacian_loss(J_local, edge_face_pairs):
    """ARAP-style smoothness: adjacent faces should have the same in-plane
    metric tensor (C = J^T J).

    loss = ‖C_a - C_b‖²_F

    Args:
        J_local: (..., N_f, 3, 2)
        edge_face_pairs: (E_int, 2) long
    """
    C = torch.einsum("...fij,...fik->...fjk", J_local, J_local)  # (..., F, 2, 2)
    C_a = C[..., edge_face_pairs[:, 0], :, :]
    C_b = C[..., edge_face_pairs[:, 1], :, :]
    return ((C_a - C_b) ** 2).sum(dim=(-1, -2)).mean()


def local_jac_shared_edge_loss(J_restricted, edge_to_faces,
                               edge_tangent_a, edge_tangent_b):
    """Enforce that a shared edge deforms to the same 3D position in both
    adjacent faces.

    For each interior edge:
        e_from_A = J_A @ [u_a, v_a]^T
        e_from_B = J_B @ [u_b, v_b]^T
        loss    += ||e_from_A - e_from_B||^2

    Args:
        J_restricted: (B, N_f, 3, 2) or (N_f, 3, 2)
        edge_to_faces: (E_int, 2) long — face index pairs
        edge_tangent_a: (E_int, 2) — shared edge tangent coords in face_a
        edge_tangent_b: (E_int, 2) — shared edge tangent coords in face_b
    """
    face_a = edge_to_faces[:, 0]
    face_b = edge_to_faces[:, 1]
    if J_restricted.dim() == 4:  # (B, N_f, 3, 2)
        J_a = J_restricted[:, face_a]
        J_b = J_restricted[:, face_b]
        e_a = torch.einsum("beij,ej->bei", J_a, edge_tangent_a)
        e_b = torch.einsum("beij,ej->bei", J_b, edge_tangent_b)
    else:  # (N_f, 3, 2)
        J_a = J_restricted[face_a]
        J_b = J_restricted[face_b]
        e_a = torch.einsum("eij,ej->ei", J_a, edge_tangent_a)
        e_b = torch.einsum("eij,ej->ei", J_b, edge_tangent_b)
    return ((e_a - e_b) ** 2).sum(-1).mean()
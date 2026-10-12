
import torch
class sampling:
    def __init__(self, gt_masks:list, posed_meshes:list, cameras:list, mesh_renderer,device):
        #forward skinning
        self.device = device
        self.num_imgs = len(gt_masks)
        self.max_candidate = int(0.05* self.num_imgs)
        #self.body_idx = [
        #0, 2, 3, 4,  5, 6, 7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        #]
        self.minum_view_to_see = 3
        self.mesh_renderer = mesh_renderer
        #mesh_renderer.render_batch(posed_meshes,cameras)
        self.render_masks = [mesh_renderer.render_batch(posed_mesh,[cameras[idx]])['mask'] for idx, posed_mesh in enumerate(posed_meshes)]
        self.gt_masks = gt_masks
        self.num_vertices = posed_meshes[0].xyz.shape[0]

        self.visibility_tensor = torch.zeros((self.num_imgs, self.num_vertices), device=self.device)
        indexs = torch.arange(0,len(self.gt_masks))

        for i, idx in enumerate(indexs):
            vis = self.mesh_renderer.get_vert_visibility(posed_meshes[i], [cameras[i]])[0].float()
            self.visibility_tensor[i] = vis
        self.visiblity_max = (torch.sum(self.visibility_tensor,dim=0)>self.minum_view_to_see).sum()
        print("visiblity_max:",str(self.visiblity_max/self.num_vertices*100) + ' %')

        #self.ious = self.get_iou()


    
    def get_iou(self):
        if self.iou_type=='outside':
            iou = self.outside_iou(self.gt_masks, self.render_masks)
        return iou
    



    def outside_iou(self,gt_mask: torch.Tensor, mesh_mask: torch.Tensor) -> float:
        """
        Args:
            mask1: torch.Tensor of shape (H, W), binary (0 or 1)
            mask2: torch.Tensor of shape (H, W), binary (0 or 1)
        
        Returns:
            IoU score (float)
        """
        #intersection = torch.logical_and(gt_mask, smpl_mask).float()#.sum()
        #union = torch.logical_or(gt_mask, smpl_mask).float()#.sum()
        #smpl_mask_float = smpl_mask.float()
        gt_mask_float = gt_mask.float()
        non_smpl_mask_mask = torch.logical_not(mesh_mask)
        non_overlap = torch.logical_and(torch.logical_and(non_smpl_mask_mask,gt_mask), gt_mask).float().sum()
        
        union = gt_mask_float.sum()
        
        outside_body_iou = non_overlap/union
        #non_overlap_smpl = union - gt_mask  #intersection



        if non_overlap == 0:
            import pdb;pdb.set_trace()
            return 1.0 if union == 0 else 0.0  # avoid divide by zero
        
        return outside_body_iou.item()
    def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """
        Args:
            mask1: torch.Tensor of shape (H, W), binary (0 or 1)
            mask2: torch.Tensor of shape (H, W), binary (0 or 1)
        
        Returns:
            IoU score (float)
        """
        intersection = torch.logical_and(mask1, mask2).float().sum()
        union = torch.logical_or(mask1, mask2).float().sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0  # avoid divide by zero
        
        iou = intersection / union
        return iou.item()

    def inner_body_iou(gt_mask: torch.Tensor, smpl_mask: torch.Tensor) -> float:
        """
        Args:
            mask1: torch.Tensor of shape (H, W), binary (0 or 1)
            mask2: torch.Tensor of shape (H, W), binary (0 or 1)
        
        Returns:
            IoU score (float)
        """
        #intersection = torch.logical_and(gt_mask, smpl_mask).float()#.sum()
        #union = torch.logical_or(gt_mask, smpl_mask).float()#.sum()
        smpl_mask_float = smpl_mask.float()
        non_gt_mask = torch.logical_not(gt_mask)
        non_overlap_smpl = torch.logical_and(torch.logical_and(smpl_mask,non_gt_mask), smpl_mask).float().sum()
        
        union = smpl_mask_float.sum()
        
        inner_body_iou = 1.0 - non_overlap_smpl/union
        #non_overlap_smpl = union - gt_mask  #intersection



        if non_overlap_smpl == 0:
            return 1.0 if union == 0 else 0.0  # avoid divide by zero
        
        return inner_body_iou.item()






def get_sampled_indices_from_length(total_length, step=3):
    """
    Sample indices at `step` intervals over the given total length (total_length).
    
    Args:
        total_length (int): total length of the data (e.g., total number of frames)
        step (int): sampling interval (default 3)
        
    Returns:
        sampled_indices (list): list of sampled indices
        take_first_indices (list): list containing the start index [0] (single take)
    """
    
    sampled_indices = []

    # 1. Validity check (return empty lists if length <= 0)
    if total_length <= 0:
        return [], []

    # 2. Sample at step intervals
    # range(0, total_length, step) -> generates 0 to total_length-1 at step intervals
    # e.g., length=10, step=3 -> [0, 3, 6, 9]
    sampled_indices = list(range(0, total_length, step))
            
    return sampled_indices



def get_safe_triple_indices_from_current(total_len, 
                            curr_sampling_idxs, 
                            curr_except_frames=None, 
                            prev_except_frames=None, 
                            next_except_frames=None):
    """
    [Main Pipeline]
    Expands the user's sampling_idxs into (Prev, Curr, Next) triplets,
    then filters and returns only the safe index sets.
    
    Args:
        total_len (int): total dataset length
        sampling_idxs (list): current-frame indices the user wants to sample
        ...except_frames: lists of indices to exclude
        
    Returns:
        tuple: (final_prevs, final_currs, final_nexts)
    """
    
    # 1. Expand: first build all previous/next indices.
    raw_prevs, raw_currs, raw_nexts = expand_neighbor_indices(curr_sampling_idxs)
    
    # 2. Filter: drop invalid sets.
    final_prevs, final_currs, final_nexts = filter_triplet_indices(
        total_len=total_len,
        prev_idxs=raw_prevs,
        curr_idxs=raw_currs,
        next_idxs=raw_nexts,
        curr_except=curr_except_frames,
        prev_except=prev_except_frames,
        next_except=next_except_frames
    )
    
    return final_prevs, final_currs, final_nexts

def expand_neighbor_indices(curr_idxs):
    """
    [Helper 1] Builds Prev(-1) and Next(+1) lists from the current index list.
    Performs only simple arithmetic, without validity checks.
    
    Args:
        curr_idxs (list): reference list of current indices
        
    Returns:
        tuple: (prev_idxs, curr_idxs, next_idxs)
    """
    if not curr_idxs:
        return [], [], []
        
    prev_idxs = [i - 1 for i in curr_idxs]
    next_idxs = [i + 1 for i in curr_idxs]
    
    return prev_idxs, curr_idxs, next_idxs

def filter_triplet_indices(total_len, prev_idxs, curr_idxs, next_idxs,
                           curr_except=None, prev_except=None, next_except=None):
    """
    [Step 2] Checks the generated (Prev, Curr, Next) sets and
    keeps only the valid ones.
    
    Args:
        total_len (int): total data length (for range checks)
        prev_idxs, curr_idxs, next_idxs (list): index lists to check
        ...except (list): exclusion lists
        
    Returns:
        tuple: (valid_prevs, valid_currs, valid_nexts)
    """
    
    # 1. Build exclusion sets (faster lookup)
    curr_except_set = set(curr_except) if curr_except is not None else set()
    prev_except_set = set(prev_except) if prev_except is not None else set()
    next_except_set = set(next_except) if next_except is not None else set()
    
    valid_prevs, valid_currs, valid_nexts = [], [], []
    
    # 2. Iterate over the 3 lists together and check (zip)
    for p, c, n in zip(prev_idxs, curr_idxs, next_idxs):
        
        # [Check 1] Boundary check
        # Reject if Prev < 0 or Next >= total length
        if p < 0 or p >= total_len: continue
        if c < 0 or c >= total_len: continue
        if n < 0 or n >= total_len: continue
        
        # [Check 2] Exclusion list check (Invalid Set Check)
        if c in curr_except_set: continue
        if p in prev_except_set: continue  # e.g., scene start, cannot link to the previous frame
        if n in next_except_set: continue  # e.g., scene end, cannot link to the next frame
            
        # Keep only sets that pass all checks
        valid_prevs.append(p)
        valid_currs.append(c)
        valid_nexts.append(n)
        
    return valid_prevs, valid_currs, valid_nexts
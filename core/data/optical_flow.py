import os
import torch
import tqdm
from core.data.waft_avatar import build_waft_avatar

class OpticalFlowLoader:
    def __init__(self, waft_config, device='cuda'):
        self.config = waft_config
        self.device = device
        self.model = None  # load the model only when needed (lazy loading)


    def _load_model(self):
        """Load the model only when no flow file exists, to save memory."""
        if self.model is not None:
            return

        print("Loading WAFT Model...")
        self.model = build_waft_avatar(self.config)

        # Load checkpoint (adjust the path to your environment)
        ckpt_path = 'libs/WAFT/waftv2-ckpts/dinov3/spring.pth'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _reuse_sibling_cache(file_path: str, filename: str) -> bool:
        """If a sibling variant of this scene already holds an identical
        cache file, symlink it into `file_path` and return True. Prevents
        recomputing multi-GB flow/waft features per experiment variant.
        """
        if os.path.exists(file_path):
            return True  # already there
        save_root = os.path.dirname(file_path)
        scene_dir = os.path.dirname(save_root)   # parent of the variant folder
        if not os.path.isdir(scene_dir):
            return False
        for sibling in os.listdir(scene_dir):
            sib_path = os.path.join(scene_dir, sibling, filename)
            if sibling == os.path.basename(save_root):
                continue
            if os.path.exists(sib_path):
                target = os.path.realpath(sib_path)
                os.makedirs(save_root, exist_ok=True)
                os.symlink(target, file_path)
                print(f"[FlowLoader] Reused sibling cache "
                      f"({sibling}/{filename}) → symlinked into {os.path.basename(save_root)}")
                return True
        return False


    def get_flow_low_interval(self, save_root, images, idx_p, idx_c, idx_n, level_name, interval):
        self._load_model()
        assert interval < 3, "Interval must be small for this method."
        
        # File name (consider renaming, e.g. mapped -> full_tensor, since the structure changed)
        filename = f'waft_flow_{level_name}_{str(interval).zfill(2)}_full.pth'
        file_path = os.path.join(save_root, filename)

        # 1. Load cache — flow kept as float32 (WAFT model compatible, small at 2.8GB)
        if self._reuse_sibling_cache(file_path, filename):
            print(f"[FlowLoader] Loading cached flow from {filename}")
            flows = torch.load(file_path, map_location='cpu')
            return flows

        print(f"[FlowLoader] Calculating Flow...")

        # 2. Prepare a tensor holding all images
        num_triplets = len(idx_p)
        h, w, _ = images[0].shape
        num_imgs = len(images)
        
        # Shape (Num_Images, 2, H, W), initialized to Inf
        flows = torch.full((num_imgs, 2, h, w), float('inf'), dtype=torch.float32)

        with torch.no_grad():
            for i in tqdm.trange(num_triplets):
                p_id = int(idx_p[i])
                c_id = int(idx_c[i])
                n_id = int(idx_n[i])

                img_p = self._prepare_image(images[p_id])
                img_c = self._prepare_image(images[c_id])
                img_n = self._prepare_image(images[n_id])

                # [A] Compute Prev -> Curr (skip if already computed!)
                # Check whether the first value of flows[p_id] is inf
                if torch.isinf(flows[p_id, 0, 0, 0]):
                    flows[p_id] = self._forward_flow(img_p, img_c)

                # [B] Compute Curr -> Next (skip if already computed!)
                if torch.isinf(flows[c_id, 0, 0, 0]):
                    flows[c_id] = self._forward_flow(img_c, img_n)

        # 3. Save
        print(f"[FlowLoader] Saving flow to {filename}")
        torch.save(flows, file_path) # save the tensor itself

        return flows  # kept on CPU (moved to GPU per batch)


    def get_flow(self, save_root, images, sampling_idxs, level_name, interval):
        """
        Loads flow, or computes, saves and returns it if missing.
        
        Args:
            save_root (str): save path
            images (list): list of all images (AvatarDataset.images)
            sampling_idxs (list/array): sampled index list (assumed Prev, Curr, Next order)
            level_name (str): file name tag such as 'L0', 'L1'
            interval (int): sampling interval
        """
        self._load_model()  # load model

        # Set save file name
        filename = f'waft_init_{level_name}_{str(interval).zfill(2)}.pth'
        file_path = os.path.join(save_root, filename)

        # 1. Load cache file if present — flow kept as float32 (WAFT model compatible)
        if self._reuse_sibling_cache(file_path, filename):
            print(f"[FlowLoader] Loading cached flow from {filename}")
            return torch.load(file_path, map_location='cpu')

        # 2. Start computing if no file exists
        print(f"[FlowLoader] Calculating Flow for {level_name} (interval={interval})...")

        num_samples = len(sampling_idxs)
        num_triplets = num_samples // 3
        
        # Get image size
        h, w, _ = images[0].shape
        
        # Initialize everything to inf
        flows = torch.full((num_samples, 2, h, w), float('inf'), device=self.device)

        # Extract and process only the sampled images
        # Note: images is the full dataset, sampling_idxs are the selected indices
        
        with torch.no_grad():
            for idx in tqdm.trange(num_triplets, desc="Computing Flows"):
                # Index in the flattened sampling list
                p_out_idx = 3 * idx
                c_out_idx = 3 * idx + 1
                # n_out_idx = 3 * idx + 2  # no flow is stored for Next

                # Actual Image Indices
                p_img_idx = sampling_idxs[p_out_idx]
                c_img_idx = sampling_idxs[c_out_idx]
                n_img_idx = sampling_idxs[3 * idx + 2]

                # Prepare Images (H,W,C -> C,H,W)
                prev_img = self._prepare_image(images[p_img_idx])
                curr_img = self._prepare_image(images[c_img_idx])
                next_img = self._prepare_image(images[n_img_idx])

                # 1. Prev -> Curr
                flow_p2c = self._forward_flow(prev_img, curr_img)
                flows[p_out_idx] = flow_p2c

                # 2. Curr -> Next
                flow_c2n = self._forward_flow(curr_img, next_img)
                flows[c_out_idx] = flow_c2n

        # 3. Save results
        print(f"[FlowLoader] Saving flow to {filename}")
        flows = flows.cpu()
        torch.save(flows, file_path)

        return flows  # kept on CPU (moved to GPU per batch)


    def get_features(self, save_root, images, filename):
        """
        Loads WAFT features, or extracts, saves and returns them if missing.
        
        Args:
            save_root (str): save path
            images (list): list of all images
            filename (str): file name to save/load
        """
        file_path = os.path.join(save_root, filename)

        # 1. Load cache file if present — float16 to save RAM (~24GB → ~12GB)
        if self._reuse_sibling_cache(file_path, filename):
            print(f"[FlowLoader] Loading cached features from {filename}")
            features = torch.load(file_path, map_location='cpu')
            if features.dtype == torch.float32:
                features = features.half()
                print(f"[FlowLoader] Converted to float16 ({features.nelement() * 2 / 1024**3:.1f} GB)")
            return features

        # 2. If no file exists, load the model and start feature extraction
        self._load_model()
        print(f"[FlowLoader] Extracting Features...")
        
        features_list = []
        with torch.no_grad():
            for idx in tqdm.trange(len(images), desc="Extracting WAFT Features"):
                # Reuse _prepare_image to convert to (1, C, H, W) (keeping the 0~255 range)
                img_tensor = self._prepare_image(images[idx])
                
                # Extract features
                feature = self.model.est_prefeat(img_tensor)
                
                # Move to CPU immediately and keep in a list to save GPU memory
                features_list.append(feature.cpu())
                
        # 3. Merge the collected tensors at once (faster and more memory-efficient than repeated concat)
        waft_features = torch.cat(features_list, dim=0)
        del features_list

        # 4. Save results
        print(f"[FlowLoader] Saving features to {filename}")
        torch.save(waft_features, file_path)
        
        return waft_features
    def _prepare_image(self, img_numpy):
        """Convert a numpy image (H,W,C) to a tensor (1,C,H,W)"""
        tensor = torch.Tensor(img_numpy).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)


    def _forward_flow(self, img1, img2, iters=None):
        """Compute flow for a single pair (internal helper)"""
        # Embeds the user-provided forward_init_flow logic
        # (including Gaussian weighting logic)
        
        n, _, h, w = img1.shape
        init_flow = torch.zeros(n, 2, h, w).to(self.device)
        
        # Build Gaussian weight
        weight = self._generate_gaussian((h, w), 1).to(self.device).unsqueeze(0)
        
        output_ij = self.model.forward_w_init_flow(img1, img2, init_flow, iters=iters)
        
        # Weighting logic
        # output_ij['flow'] is assumed to be a list (based on the original user code)
        final_flow = torch.zeros((n, 2, h, w), device=self.device)
        valid_weight = weight # initial value if it is a plain accumulation
        
        # Simplification of the original code's complex logic:
        # The original code loops over the whole output list, multiplying and dividing by weight;
        # if only the last flow is needed, it could be simplified as below.
        # However, the original logic is preserved as much as possible:
        
        accumulated_flow = torch.zeros((n, 2, h, w), device=self.device)
        valid_sum = torch.zeros((n, h, w), device=self.device)
        
        # The last flow in the list is often the most accurate,
        # but the provided code takes a weighted sum over all flow outputs.
        for flow_est in output_ij['flow']:
            accumulated_flow += weight * flow_est
        
        # Normalize (since weight was added len(output_ij['flow']) times)
        # Original code: valid += weight (once, outside the loop) -> weight multiplied inside the loop
        # -> output[key][i] /= valid.unsqueeze(1)
        # Is this effectively (weight * sum(flows)) / weight = sum(flows)?
        # Since the original code does valid += weight only once outside the loop,
        # the result is the sum of flows over all stages / 1 (if the Gaussian max is 1).
        # Usually only the last (high-resolution) flow would be taken here,
        # following the original logic, only the "last flow" is extracted and moved to CPU.
        
        return output_ij['flow'][-1].squeeze(0).cpu()

    @staticmethod
    def _generate_gaussian(size, sigma):
        h, w = size
        y = torch.arange(0, h).view(-1, 1) / h
        x = torch.arange(0, w).view(1, -1) / w
        center_y, center_x = 0.5, 0.5
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))
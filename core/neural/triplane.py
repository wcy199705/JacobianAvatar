import torch
import torch.nn as nn
import torch.nn.functional as F

class TriplaneEncoder(nn.Module):
    def __init__(self, triplane_config, bounds,device=torch.device('cuda')):
        """
        Args:
            triplane_config (dict): 
                - plane_res, feature_dim, init_scale, agg_mode
                - lr: learning rate of this module (added)
            bounds (tuple): ((min_x, y, z), (max_x, y, z))
        """
        super().__init__()
        self.device =device
        # 1. Config Unpacking
        self.plane_res = triplane_config['plane_res']
        self.feature_dim = triplane_config['feature_dim']
        self.init_scale = triplane_config['init_scale']
        self.agg_mode = triplane_config['agg_mode']
        
        # [NEW] store learning rate
        self.lr = triplane_config.get('lr', 1e-3) # default 0.001 if key is missing

        # 2. Bounds Registration
        self.register_buffer('min_bound', torch.tensor(bounds['min'], dtype=torch.float32).view(1, 1, 3))
        self.register_buffer('max_bound', torch.tensor(bounds['max'], dtype=torch.float32).view(1, 1, 3))

        # 3. Output Dimension
        if self.agg_mode == 'sum':
            self.output_dim = self.feature_dim
        elif self.agg_mode == 'concat':
            self.output_dim = self.feature_dim * 3
        else:
            raise ValueError(f"Invalid agg_mode: {self.agg_mode}")

        # 4. Triplane Parameter Initialization
        # This variable is the trainable parameter.
        self.planes = nn.Parameter(
            (torch.randn(1, 3 * self.feature_dim, self.plane_res, self.plane_res) * self.init_scale).to(self.device)
)
    def normalize_coords(self, points):
        """ Coordinate normalization: uses internal buffers """
        normalized = (points - self.min_bound) / (self.max_bound - self.min_bound)
        normalized = normalized * 2 - 1
        return normalized #torch.clamp(normalized, -1, 1)

    def forward(self, points):
        # (same logic as before)
        points = self.normalize_coords(points)
        batch_size, n_points, _ = points.shape
        
        coord_xy = points[..., [0, 1]].view(batch_size, n_points, 1, 2)
        coord_xz = points[..., [0, 2]].view(batch_size, n_points, 1, 2)
        coord_yz = points[..., [1, 2]].view(batch_size, n_points, 1, 2)
        
        planes = self.planes.repeat(batch_size, 1, 1, 1)
        plane_xy, plane_xz, plane_yz = torch.split(planes, self.feature_dim, dim=1)
        
        feat_xy = F.grid_sample(plane_xy, coord_xy, align_corners=True, mode='bilinear')
        feat_xz = F.grid_sample(plane_xz, coord_xz, align_corners=True, mode='bilinear')
        feat_yz = F.grid_sample(plane_yz, coord_yz, align_corners=True, mode='bilinear')
        
        if self.agg_mode == 'sum':
            features = feat_xy + feat_xz + feat_yz
        else:
            features = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        
        features = features.view(batch_size, -1, n_points).permute(0, 2, 1)
        return features




    def get_optimizable_params(self):
        """
        Returns the list of parameter dicts used when initializing the optimizer.
        """
        optimizable_params = [
            {
                'params': [self.planes], # nn.Parameter must be wrapped in a list
                'name': 'triplane_embeddings',
                'lr': self.lr
            },
        ]
        return optimizable_params

# --- Usage example ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Add 'lr' to the config
    config = {
        'plane_res': 128,
        'feature_dim': 32,
        'init_scale': 0.1,
        'agg_mode': 'concat',
        'lr': 0.005 # Triplane learning rate
    }
    bounds = ((-1, -1, -1), (1, 1, 1))
    
    encoder = TriplaneEncoder(config, bounds).to(device)
    
    # 2. Use get_optimizable_params() when creating the optimizer
    # Fetch model parameters per group and pass them to the optimizer
    optimizer = torch.optim.Adam(
        encoder.get_optimizable_params(), 
        betas=(0.9, 0.999)
    )
    
    print("Optimizer Param Groups:")
    for group in optimizer.param_groups:
        print(f"- Name: {group['name']}, LR: {group['lr']}, Params: {len(group['params'])}")
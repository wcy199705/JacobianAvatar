import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneEncoderDino(nn.Module):
    """
    Triplane Encoder with DINOv2 feature embedding.

    Changes from original TriplaneEncoder:
    1. agg_mode forced to 'sum' (output_dim = feature_dim = 32)
    2. DINOv2 projection layer (768 → feature_dim)
    3. DINOv2 feature is added to triplane output
    4. Pre-fitting method to initialize triplane from DINOv2 features

    Projection modes (dino_proj_mode in config):
    - 'linear' (default): learnable nn.Linear(768 → feature_dim)
    - 'pca': fixed PCA projection, call fit_pca() with data before training
    """

    def __init__(self, triplane_config, bounds, device=torch.device('cuda'),
                 dino_feat_dim=768):
        super().__init__()
        self.device = device

        # 1. Config Unpacking
        self.plane_res = triplane_config['plane_res']
        self.feature_dim = triplane_config['feature_dim']
        self.init_scale = triplane_config['init_scale']
        self.agg_mode = triplane_config.get('agg_mode', 'sum')

        self.lr = triplane_config.get('lr', 1e-3)

        # 2. Bounds Registration
        self.register_buffer('min_bound', torch.tensor(bounds['min'], dtype=torch.float32).view(1, 1, 3))
        self.register_buffer('max_bound', torch.tensor(bounds['max'], dtype=torch.float32).view(1, 1, 3))

        # 3. Output Dimension
        if self.agg_mode == 'concat':
            self.output_dim = self.feature_dim * 3
        else:
            self.output_dim = self.feature_dim

        # 4. Triplane Parameter
        self.planes = nn.Parameter(
            (torch.randn(1, 3 * self.feature_dim, self.plane_res, self.plane_res) * self.init_scale).to(self.device)
        )

        # 5. DINOv2 Projection Layer (768 → feature_dim)
        self.dino_feat_dim = dino_feat_dim
        self.dino_proj_mode = triplane_config.get('dino_proj_mode', 'linear')

        if self.dino_proj_mode == 'pca':
            # PCA projection: fixed (non-learnable) linear transform
            self.register_buffer('pca_components', torch.zeros(self.output_dim, dino_feat_dim))
            self.register_buffer('pca_mean', torch.zeros(dino_feat_dim))
            self.register_buffer('pca_fitted', torch.tensor(False))
            self.dino_proj = None
        else:
            # Default: learnable linear projection
            self.dino_proj = nn.Linear(dino_feat_dim, self.feature_dim, bias=False).to(self.device)
            nn.init.xavier_uniform_(self.dino_proj.weight)
            self.register_buffer('pca_fitted', torch.tensor(False))

    @torch.no_grad()
    def fit_pca(self, dino_features):
        """
        Fit PCA on DINOv2 features and store components as fixed projection.
        pca_mean and pca_components are frozen after this call (register_buffer).

        Args:
            dino_features: [N, 768] single-view pixel features (NOT multi-view averaged).
                           Use a representative sample of pixels from one or a few frames.
        """
        assert self.dino_proj_mode == 'pca', "fit_pca() only available in 'pca' mode"
        feats = dino_features.float().to(self.device)
        if feats.dim() == 3:
            feats = feats.view(-1, feats.shape[-1])

        # Center
        mean = feats.mean(dim=0)
        centered = feats - mean

        # SVD (more numerically stable than covariance eigendecomposition)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Top-k components (use output_dim for concat mode)
        n_comp = self.output_dim
        components = Vh[:n_comp]  # [output_dim, 768]

        self.pca_components.copy_(components)
        self.pca_mean.copy_(mean)
        self.pca_fitted.fill_(True)

        # Log explained variance
        total_var = (S ** 2).sum()
        explained_var = (S[:n_comp] ** 2).sum() / total_var
        print(f"  [PCA] Fitted {n_comp} components from {feats.shape[0]} pixels")
        print(f"  [PCA] Explained variance: {explained_var.item():.4f} ({explained_var.item()*100:.1f}%)")

        return explained_var.item()

    def _project_dino(self, dino_features):
        """Apply projection (linear or PCA) to DINOv2 features."""
        if self.dino_proj_mode == 'pca':
            assert self.pca_fitted, "PCA not fitted yet. Call fit_pca() first."
            dev = dino_features.device
            projected = F.linear(dino_features - self.pca_mean.to(dev), self.pca_components.to(dev))
            return projected
        else:
            return self.dino_proj(dino_features)

    def normalize_coords(self, points):
        normalized = (points - self.min_bound) / (self.max_bound - self.min_bound)
        normalized = normalized * 2 - 1
        return normalized

    def forward(self, points):
        """
        Standard triplane query (sum mode).

        Args:
            points: [B, N, 3] 3D coordinates

        Returns:
            features: [B, N, feature_dim]
        """
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

        if self.agg_mode == 'concat':
            features = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        elif self.agg_mode == 'avg':
            features = (feat_xy + feat_xz + feat_yz) / 3.0
        else:
            features = feat_xy + feat_xz + feat_yz
        features = features.view(batch_size, -1, n_points).permute(0, 2, 1)
        return features

    def forward_with_dino(self, points, dino_features_per_point):
        """
        Triplane query + DINOv2 feature addition.

        Args:
            points: [B, N, 3] 3D coordinates
            dino_features_per_point: [B, N, 768] DINOv2 features for each point

        Returns:
            features: [B, N, feature_dim]
        """
        triplane_feat = self.forward(points)  # [B, N, feature_dim]
        dino_projected = self._project_dino(dino_features_per_point)  # [B, N, feature_dim]
        return triplane_feat + dino_projected

    def project_dino_to_triplane(self, dino_features):
        """
        Project DINOv2 features (768-dim) to triplane feature space (feature_dim).

        Args:
            dino_features: [N, 768] or [B, N, 768]

        Returns:
            projected: same shape but last dim = feature_dim
        """
        return self._project_dino(dino_features)

    @torch.no_grad()
    def pre_fit_from_dino(self, face_centers, dino_face_features,
                          num_iters=500, lr=1e-3, verbose=True):
        """
        Pre-fitting: initialize the triplane to match DINOv2 features.

        Optimizes the triplane planes so that querying face_centers on the triplane
        matches the projected DINOv2 features.

        Args:
            face_centers: [1, N_faces, 3] canonical face centers
            dino_face_features: [N_faces, feat_dim] already projected DINOv2 features
            num_iters: pre-fitting iterations
            lr: learning rate for pre-fitting
            verbose: print progress
        """
        # Enable gradient for this method
        with torch.enable_grad():
            planes_opt = torch.optim.Adam([self.planes], lr=lr)
            target = dino_face_features.detach()  # [N_faces, feat_dim]
            if target.dim() == 2:
                target = target.unsqueeze(0)  # [1, N_faces, feat_dim]

            for i in range(num_iters):
                planes_opt.zero_grad()
                pred = self.forward(face_centers)  # [1, N_faces, feat_dim]
                loss = F.mse_loss(pred, target)
                loss.backward()
                planes_opt.step()

                if verbose and (i == 0 or (i + 1) % 100 == 0):
                    print(f"  [Triplane Pre-fit] iter {i+1}/{num_iters}, loss: {loss.item():.6f}")

            if verbose:
                print(f"  [Triplane Pre-fit] Done. Final loss: {loss.item():.6f}")

    def get_optimizable_params(self):
        optimizable_params = [
            {
                'params': [self.planes],
                'name': 'triplane_embeddings',
                'lr': self.lr
            },
        ]
        if self.dino_proj_mode == 'linear' and self.dino_proj is not None:
            optimizable_params.append({
                'params': self.dino_proj.parameters(),
                'name': 'dino_projection',
                'lr': self.lr * 0.1  # projection learns more slowly
            })
        # PCA mode: no additional learnable params for projection
        return optimizable_params

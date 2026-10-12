import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HashGridEncoderDino(nn.Module):
    """
    Multi-resolution Hash Grid Encoder (Instant-NGP style) with DINOv2 feature embedding.
    Pure PyTorch implementation (no tiny-cuda-nn dependency).

    Drop-in replacement for TriplaneEncoderDino with identical public interface.

    Config keys (with defaults):
        n_levels: 16
        n_features_per_level: 2
        log2_hashmap_size: 19
        base_resolution: 16
        finest_resolution: 512
        feature_dim: 32  (should equal n_levels * n_features_per_level)
        init_scale: 1e-4
        lr: 1e-3
        dino_proj_mode: 'pca' or 'linear'
        agg_mode: 'sum' (unused, for config compatibility)
    """

    # Primes for spatial hashing (standard Instant-NGP primes)
    PRIMES = [1, 2654435761, 805459861]

    def __init__(self, config, bounds, device=torch.device('cuda'),
                 dino_feat_dim=768):
        super().__init__()
        self.device = device

        # 1. Hash Grid Config (cast to int — YAML may return numpy types)
        self.n_levels = int(config.get('n_levels', 16))
        self.n_features_per_level = int(config.get('n_features_per_level', 2))
        self.log2_hashmap_size = int(config.get('log2_hashmap_size', 19))
        self.base_resolution = int(config.get('base_resolution', 16))
        self.finest_resolution = int(config.get('finest_resolution', 512))
        self.init_scale = float(config.get('init_scale', 1e-4))
        self.lr = float(config.get('lr', 1e-3))

        self.hashmap_size = int(2 ** self.log2_hashmap_size)

        # Growth factor: b such that base_resolution * b^(n_levels-1) = finest_resolution
        if self.n_levels > 1:
            self.growth_factor = math.exp(
                math.log(self.finest_resolution / self.base_resolution) / (self.n_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        # 2. Output dimension
        self.feature_dim = config.get('feature_dim', self.n_levels * self.n_features_per_level)
        self.output_dim = self.n_levels * self.n_features_per_level
        # Compatibility: agg_mode is unused but kept for config compat
        self.agg_mode = config.get('agg_mode', 'sum')

        # 3. Bounds Registration (bounds values may already be tensors)
        min_b = bounds['min'] if isinstance(bounds['min'], torch.Tensor) else torch.tensor(bounds['min'])
        max_b = bounds['max'] if isinstance(bounds['max'], torch.Tensor) else torch.tensor(bounds['max'])
        self.register_buffer('min_bound', min_b.float().view(1, 1, 3))
        self.register_buffer('max_bound', max_b.float().view(1, 1, 3))

        # 4. Hash Tables — one per level, each [hashmap_size, n_features_per_level]
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.hashmap_size, self.n_features_per_level) * self.init_scale
            )
            for _ in range(self.n_levels)
        ])

        # 5. Pre-compute resolutions per level
        resolutions = []
        for l in range(self.n_levels):
            res = int(math.floor(self.base_resolution * (self.growth_factor ** l)))
            resolutions.append(res)
        self.register_buffer('resolutions', torch.tensor(resolutions, dtype=torch.long))

        # 6. DINOv2 Projection Layer (768 → output_dim)
        self.dino_feat_dim = dino_feat_dim
        self.dino_proj_mode = config.get('dino_proj_mode', 'linear')

        if self.dino_proj_mode == 'pca':
            self.register_buffer('pca_components', torch.zeros(self.output_dim, dino_feat_dim))
            self.register_buffer('pca_mean', torch.zeros(dino_feat_dim))
            self.register_buffer('pca_fitted', torch.tensor(False))
            self.dino_proj = None
        else:
            self.dino_proj = nn.Linear(dino_feat_dim, self.output_dim, bias=False)
            nn.init.xavier_uniform_(self.dino_proj.weight)
            self.register_buffer('pca_fitted', torch.tensor(False))

        # Move entire module to device
        self.to(device)

    def _hash_fn(self, coords, table_size):
        """
        Spatial hashing with XOR and primes.
        Uses bitmask instead of modulo to guarantee valid indices.

        Args:
            coords: [..., 3] integer grid coordinates
            table_size: int, hash table size (must be power of 2)

        Returns:
            indices: [...] hash indices in [0, table_size)
        """
        # Use smaller primes that won't overflow int32
        # and bitmask to guarantee [0, table_size)
        mask = table_size - 1  # works because table_size = 2^k
        x = coords[..., 0].int()
        y = coords[..., 1].int()
        z = coords[..., 2].int()
        # Spatial hash: XOR with shifted coordinates
        h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        return (h & mask).long()

    def _query_single_level(self, points_norm, level):
        """
        Query a single hash grid level with trilinear interpolation.

        Args:
            points_norm: [B, N, 3] coordinates in [0, 1]
            level: int, which level to query

        Returns:
            features: [B, N, n_features_per_level]
        """
        resolution = self.resolutions[level].item()
        table = self.hash_tables[level]  # [T, F]

        # Scale to grid coordinates
        # points_norm is in [0, 1], scale to [0, resolution]
        scaled = points_norm * resolution  # [B, N, 3]

        # Floor to get voxel corner (integer coords), clamp to valid range
        floor_coords = scaled.floor().long().clamp(min=0, max=resolution - 1)  # [B, N, 3]

        # Fractional part for interpolation weights
        frac = (scaled - floor_coords.float()).clamp(0.0, 1.0)  # [B, N, 3]

        B, N, _ = points_norm.shape

        # 8 corners of the enclosing voxel: offsets (0 or 1) for each axis
        # Shape: [8, 3]
        offsets = torch.tensor(
            [[0,0,0],[0,0,1],[0,1,0],[0,1,1],
             [1,0,0],[1,0,1],[1,1,0],[1,1,1]],
            dtype=torch.long, device=points_norm.device
        )

        # Corner coordinates: [B, N, 8, 3]
        corners = floor_coords.unsqueeze(2) + offsets.view(1, 1, 8, 3)

        # Hash each corner: [B, N, 8]
        hash_indices = self._hash_fn(corners, self.hashmap_size)
        hash_indices = hash_indices.clamp(0, self.hashmap_size - 1)  # safety clamp

        # Lookup features: [B, N, 8, F]
        # Flatten for indexing, then reshape
        flat_indices = hash_indices.reshape(-1)  # [B*N*8]
        corner_features = table[flat_indices].view(B, N, 8, self.n_features_per_level)

        # Trilinear interpolation weights
        # For each corner (dx, dy, dz) in {0,1}^3:
        # weight = prod of (frac if d==1 else 1-frac) for each axis
        fx, fy, fz = frac[..., 0], frac[..., 1], frac[..., 2]  # [B, N]

        # Weights for each of the 8 corners: [B, N, 8]
        w000 = (1 - fx) * (1 - fy) * (1 - fz)
        w001 = (1 - fx) * (1 - fy) * fz
        w010 = (1 - fx) * fy       * (1 - fz)
        w011 = (1 - fx) * fy       * fz
        w100 = fx       * (1 - fy) * (1 - fz)
        w101 = fx       * (1 - fy) * fz
        w110 = fx       * fy       * (1 - fz)
        w111 = fx       * fy       * fz

        weights = torch.stack([w000, w001, w010, w011, w100, w101, w110, w111], dim=-1)  # [B, N, 8]

        # Weighted sum: [B, N, F]
        features = (corner_features * weights.unsqueeze(-1)).sum(dim=2)

        return features

    def normalize_coords(self, points):
        """Normalize coordinates from world space to [-1, 1]."""
        normalized = (points - self.min_bound) / (self.max_bound - self.min_bound)
        normalized = normalized * 2 - 1
        return normalized

    def forward(self, points):
        """
        Multi-resolution hash grid query.

        Args:
            points: [B, N, 3] 3D coordinates in world space

        Returns:
            features: [B, N, output_dim]
        """
        # Normalize to [-1, 1] then shift to [0, 1] for hash grid
        normalized = self.normalize_coords(points)  # [-1, 1]
        points_01 = (normalized + 1.0) * 0.5  # [0, 1]

        # Clamp to avoid out-of-bounds
        points_01 = points_01.clamp(0.0, 1.0 - 1e-6)

        # Query each level and concatenate
        level_features = []
        for l in range(self.n_levels):
            feat = self._query_single_level(points_01, l)  # [B, N, F]
            level_features.append(feat)

        # Concatenate: [B, N, n_levels * n_features_per_level]
        features = torch.cat(level_features, dim=-1)
        return features

    def forward_with_dino(self, points, dino_features_per_point):
        """
        Hash grid query + DINOv2 feature addition.

        Args:
            points: [B, N, 3] 3D coordinates
            dino_features_per_point: [B, N, 768] DINOv2 features for each point

        Returns:
            features: [B, N, output_dim]
        """
        grid_feat = self.forward(points)  # [B, N, output_dim]
        dino_projected = self._project_dino(dino_features_per_point)  # [B, N, output_dim]
        return grid_feat + dino_projected

    # ---- DINOv2 Projection (identical to TriplaneEncoderDino) ----

    @torch.no_grad()
    def fit_pca(self, dino_features):
        """
        Fit PCA on DINOv2 features and store components as fixed projection.

        Args:
            dino_features: [N, 768] single-view pixel features
        """
        assert self.dino_proj_mode == 'pca', "fit_pca() only available in 'pca' mode"
        feats = dino_features.float().to(self.device)
        if feats.dim() == 3:
            feats = feats.view(-1, feats.shape[-1])

        # Center
        mean = feats.mean(dim=0)
        centered = feats - mean

        # SVD
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Top-k components
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

    def project_dino_to_triplane(self, dino_features):
        """
        Project DINOv2 features (768-dim) to encoder feature space (output_dim).
        Named 'project_dino_to_triplane' for interface compatibility.

        Args:
            dino_features: [N, 768] or [B, N, 768]

        Returns:
            projected: same shape but last dim = output_dim
        """
        return self._project_dino(dino_features)

    @torch.no_grad()
    def pre_fit_from_dino(self, face_centers, dino_face_features,
                          num_iters=500, lr=1e-3, verbose=True):
        """
        Pre-fitting: initialize the hash grid to match DINOv2 features.

        Optimizes the hash table entries so that querying face_centers on the hash grid
        matches the projected DINOv2 features.

        Args:
            face_centers: [1, N_faces, 3] canonical face centers
            dino_face_features: [N_faces, feat_dim] already projected DINOv2 features
            num_iters: pre-fitting iterations
            lr: learning rate for pre-fitting
            verbose: print progress
        """
        with torch.enable_grad():
            optimizer = torch.optim.Adam(self.hash_tables.parameters(), lr=lr)
            target = dino_face_features.detach()
            if target.dim() == 2:
                target = target.unsqueeze(0)  # [1, N_faces, feat_dim]

            for i in range(num_iters):
                optimizer.zero_grad()
                pred = self.forward(face_centers)  # [1, N_faces, output_dim]
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()

                if verbose and (i == 0 or (i + 1) % 100 == 0):
                    print(f"  [HashGrid Pre-fit] iter {i+1}/{num_iters}, loss: {loss.item():.6f}")

            if verbose:
                print(f"  [HashGrid Pre-fit] Done. Final loss: {loss.item():.6f}")

    def get_optimizable_params(self):
        """Return parameter groups for the optimizer."""
        optimizable_params = [
            {
                'params': list(self.hash_tables.parameters()),
                'name': 'hashgrid_embeddings',
                'lr': self.lr
            },
        ]
        if self.dino_proj_mode == 'linear' and self.dino_proj is not None:
            optimizable_params.append({
                'params': list(self.dino_proj.parameters()),
                'name': 'dino_projection',
                'lr': self.lr * 0.1
            })
        return optimizable_params


if __name__ == "__main__":
    """Quick unit tests for HashGridEncoderDino."""
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Default config
    config = {
        'n_levels': 16,
        'n_features_per_level': 2,
        'log2_hashmap_size': 19,
        'base_resolution': 16,
        'finest_resolution': 512,
        'feature_dim': 32,
        'init_scale': 1e-4,
        'lr': 1e-3,
        'dino_proj_mode': 'pca',
        'agg_mode': 'sum',
    }
    bounds = {
        'min': [-1.0, -1.5, -0.5],
        'max': [1.0, 1.5, 0.5],
    }

    # ---- Test 1: Create encoder, check output shape ----
    print("\n=== Test 1: Forward pass shape ===")
    enc = HashGridEncoderDino(config, bounds=bounds, device=device)
    B, N = 2, 1000
    pts = torch.randn(B, N, 3, device=device) * 0.5  # random points near origin
    out = enc.forward(pts)
    assert out.shape == (B, N, 32), f"Expected (2, 1000, 32), got {out.shape}"
    print(f"  Output shape: {out.shape} -- OK")
    print(f"  Output dim: {enc.output_dim} -- OK")

    # ---- Test 2: fit_pca ----
    print("\n=== Test 2: fit_pca ===")
    fake_dino = torch.randn(5000, 768, device=device)
    explained_var = enc.fit_pca(fake_dino)
    assert enc.pca_fitted.item() is True, "PCA should be fitted"
    print(f"  Explained variance: {explained_var:.4f} -- OK")

    # Test projection after PCA
    proj = enc.project_dino_to_triplane(fake_dino[:10])
    assert proj.shape == (10, 32), f"Expected (10, 32), got {proj.shape}"
    print(f"  Projection shape: {proj.shape} -- OK")

    # ---- Test 3: forward_with_dino ----
    print("\n=== Test 3: forward_with_dino ===")
    dino_per_point = torch.randn(B, N, 768, device=device)
    out_dino = enc.forward_with_dino(pts, dino_per_point)
    assert out_dino.shape == (B, N, 32), f"Expected (2, 1000, 32), got {out_dino.shape}"
    print(f"  Output shape: {out_dino.shape} -- OK")

    # ---- Test 4: pre_fit_from_dino (loss should decrease) ----
    print("\n=== Test 4: pre_fit_from_dino ===")
    n_faces = 500
    face_centers = torch.randn(1, n_faces, 3, device=device) * 0.3
    target_feats = torch.randn(n_faces, 32, device=device)

    # Record initial loss
    with torch.no_grad():
        pred_before = enc.forward(face_centers)
        loss_before = F.mse_loss(pred_before, target_feats.unsqueeze(0)).item()
    print(f"  Loss before pre-fit: {loss_before:.6f}")

    enc.pre_fit_from_dino(face_centers, target_feats, num_iters=300, lr=1e-2, verbose=True)

    with torch.no_grad():
        pred_after = enc.forward(face_centers)
        loss_after = F.mse_loss(pred_after, target_feats.unsqueeze(0)).item()
    print(f"  Loss after pre-fit: {loss_after:.6f}")
    assert loss_after < loss_before, f"Loss should decrease: {loss_before:.6f} -> {loss_after:.6f}"
    print(f"  Loss decreased: {loss_before:.6f} -> {loss_after:.6f} -- OK")

    # ---- Test 5: get_optimizable_params ----
    print("\n=== Test 5: get_optimizable_params ===")
    params = enc.get_optimizable_params()
    assert len(params) > 0, "Should have at least one param group"
    total_params = sum(p.numel() for group in params for p in group['params'])
    print(f"  Param groups: {len(params)}")
    print(f"  Total trainable params: {total_params:,}")
    assert total_params > 0, "Should have trainable params"
    print(f"  -- OK")

    # ---- Test 6: Linear mode ----
    print("\n=== Test 6: Linear projection mode ===")
    config_linear = config.copy()
    config_linear['dino_proj_mode'] = 'linear'
    enc_linear = HashGridEncoderDino(config_linear, bounds=bounds, device=device)
    params_linear = enc_linear.get_optimizable_params()
    assert len(params_linear) == 2, f"Linear mode should have 2 param groups, got {len(params_linear)}"
    param_names = [p['name'] for p in params_linear]
    assert 'dino_projection' in param_names, "Should have dino_projection param group"
    print(f"  Param groups: {param_names} -- OK")

    print("\n=== All tests passed! ===")

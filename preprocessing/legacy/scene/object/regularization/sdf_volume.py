import torch
import torch.nn.functional as F
import numpy as np

class GridSDF:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, resolution: int = 64, padding: float = 0.1):
        """
        Builds the grid directly with Trimesh, without mesh-to-sdf.
        
        Args:
            verts: (V, 3) vertices
            faces: (F, 3) faces
            resolution: grid resolution (e.g., 64 -> 64^3 voxels)
            padding: ratio of margin space around the mesh
        """
        self.device = verts.device
        self.resolution = resolution
        
        # 1. Create the Trimesh object (for baking, CPU computation)
        v_np = verts.detach().cpu().numpy()
        f_np = faces.detach().cpu().numpy()
        # Must use process=False to keep the original mesh structure (important)
        #mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
        
        # 2. Set up the bounding box and grid coordinate system
        bbox_min = v_np.min(axis=0)
        bbox_max = v_np.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        
        # Use a cube based on the longest axis
        max_dim = (bbox_max - bbox_min).max()
        span = max_dim * (1 + padding) # add padding
        
        self.center = torch.tensor(center, device=self.device, dtype=torch.float32)
        self.span = torch.tensor(span, device=self.device, dtype=torch.float32)
        
        # 3. Generate grid coordinates (CPU)
        # Generate coordinates in the range -span/2 ~ +span/2
        # Use linspace: evenly split into `resolution` samples
        half_span = span / 2
        xs = np.linspace(center[0] - half_span, center[0] + half_span, resolution)
        ys = np.linspace(center[1] - half_span, center[1] + half_span, resolution)
        zs = np.linspace(center[2] - half_span, center[2] + half_span, resolution)
        
        # Generate 3D grid points (meshgrid)
        # indexing='ij' uses matrix indexing order (helps keep x, y, z order)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
        
        # Convert to an (N, 3) coordinate list
        grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
        
        # 4. Compute SDF values (baking) - the key step
        # trimesh's signed_distance function is fairly fast.
        # Sign: positive (+) inside, negative (-) outside (Trimesh convention).
        # Flip the sign if needed (here the trimesh output is used as-is -> outside is usually -)
        #p#rint(f"Baking SDF Grid ({resolution}^3 voxels)... This might take a moment.")
       #sdf_values = trimesh.proximity.signed_distance(mesh, grid_points)
        from pysdf import SDF
        sdf_sampler = SDF(v_np, f_np)
        sdf_values = sdf_sampler(grid_points)
        # 5. Convert to tensor and upload to GPU
        # Restore to (Res, Res, Res) shape
        sdf_volume = sdf_values.reshape(resolution, resolution, resolution)
        
        # Convert to a 5D tensor for grid_sample: (Batch, Channel, D, H, W)
        # grid_sample needs D, H, W mapped to the (x, y, z) coordinate system rather than (z, y, x)
        # Since it was built with meshgrid('ij'), data is ordered as [x, y, z].
        # PyTorch grid_sample treats the [D, H, W] dims as the [z, y, x] axes.
        # So the data must be transposed -> (x, y, z) to (z, y, x) order
        sdf_volume = sdf_volume.transpose(2, 1, 0) 
        
        self.sdf_grid = torch.from_numpy(sdf_volume).float().to(self.device)
        self.sdf_grid = self.sdf_grid.unsqueeze(0).unsqueeze(0) # (1, 1, Res, Res, Res)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
            """
            Args:
                points: (B, N, 3) or (N, 3) 3D positions
            Returns:
                sdf: (B, N) or (N,) signed distance values
            """
            # Check input dims and parse B, N
            is_batched = points.dim() == 3
            if not is_batched:
                points = points.unsqueeze(0) # (N, 3) -> (1, N, 3)
                
            B, N, _ = points.shape
            
            # 1. Normalize coordinates (World -> NDC [-1, 1])
            # Broadcasting: (B, N, 3) - (3,) -> (B, N, 3)
            norm_points = 2.0 * (points - self.center) / self.span
            
            # 2. Reshape for grid_sample
            # Our SDF grid is (1, 1, D, H, W), i.e., Batch=1.
            # So when querying, instead of treating the B batches as separate grid batches,
            # treat them as one large image (space) and pass them as (1, 1, B, N, 3);
            # this allows a single lookup without copying grid memory (no expand).
            
            # grid shape: (Vol_Batch, D_out, H_out, W_out, 3)
            # Here we map D_out=1, H_out=B, W_out=N.
            grid_input = norm_points.view(1, 1, B, N, 3)
            
            # 3. Trilinear Interpolation
            # Output Shape: (Vol_Batch, Channel, D_out, H_out, W_out) -> (1, 1, 1, B, N)
            sampled_sdf = F.grid_sample(
                self.sdf_grid, 
                grid_input, 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=True
            )
            
            # 4. Restore the original shape
            if is_batched:
                return sampled_sdf.view(B, N)
            else:
                return sampled_sdf.view(N)
            

class GridSDFGrad:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, resolution: int = 64, padding: float = 0.1):
        """
        Precomputes the SDF gradient (similar to the normal vector) on a grid and returns interpolated values.
        """
        self.device = verts.device
        self.resolution = resolution
        
        # 1. Convert to Trimesh / Numpy
        v_np = verts.detach().cpu().numpy()
        f_np = faces.detach().cpu().numpy()
        #nomralize them


        # 2. Set up bounding box and grid
        bbox_min = v_np.min(axis=0)
        bbox_max = v_np.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        
        # Set up the cubic space
        max_dim = (bbox_max - bbox_min).max()
        span = max_dim * (1 + padding)
        
        self.center = torch.tensor(center, device=self.device, dtype=torch.float32)
        self.span = torch.tensor(span, device=self.device, dtype=torch.float32)
        #v_np_norm = 2.0 * (v_np - center) / span

        # 3. Generate grid coordinates (CPU)
        half_span = span / 2
        # np.linspace includes both start and stop, so be careful when computing the step size
        xs = np.linspace(center[0] - half_span, center[0] + half_span, resolution)
        ys = np.linspace(center[1] - half_span, center[1] + half_span, resolution)
        zs = np.linspace(center[2] - half_span, center[2] + half_span, resolution)
        
        # Physical size of a single voxel (required for gradient scaling)
        voxel_size = span / (resolution - 1)
        
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
        
        # 4. Compute SDF values (using pysdf)
        from pysdf import SDF
        sdf_sampler = SDF(v_np, f_np)
        # SDF result is 1D with shape (N,)
        sdf_values = sdf_sampler(grid_points)
        v_np
        # Restore to (Res, Res, Res) shape -> (X, Y, Z) order
        sdf_volume = sdf_values.reshape(resolution, resolution, resolution)
        
        # 5. Compute gradient (key fix)
        # np.gradient uses central differences to get accurate gradients without positional shift.
        # It must also be divided by voxel_size to obtain the correct physical gradient.
        # grad_x, grad_y, grad_z each have shape: (Res, Res, Res)
        grad_x, grad_y, grad_z = np.gradient(sdf_volume, voxel_size)
        
        # Stack into (3, Res, Res, Res) -> (Channel, X, Y, Z)
        grad_field_np = np.stack([grad_x, grad_y, grad_z], axis=0)
        
        # 6. Convert to tensor and permute dims (X,Y,Z -> Z,Y,X)
        grad_field = torch.from_numpy(grad_field_np).float().to(self.device)
        
        # grid_sample expects (Batch, Channel, Depth, Height, Width) input,
        # which corresponds to the (Batch, C, z, y, x) axes.
        # grad_field is currently (C, x, y, z), so the spatial dims must be reversed.
        # permute(0, 3, 2, 1) -> (Channel, z, y, x)
        grad_field = grad_field.permute(0, 3, 2, 1)
        
        # Add batch dimension: (1, 3, Res, Res, Res)
        self.sdf_grad_grid = grad_field.unsqueeze(0)

        #surface norm of mesh
        #debugging
        verts = torch.tensor(v_np).to(self.device)
        from scene.object.mesh import Mesh
        mesh_deubg =Mesh(verts,faces)
        mesh_norm = mesh_deubg.norm_xyz
        mesh_face_norm = mesh_deubg.norm_face

        #points = verts_norm
        is_batched = verts.dim() == 3
        if not is_batched:
            verts = verts.unsqueeze(0)
            
        B, N, _ = verts.shape
        
        # 1. Normalize coordinates
        norm_points = 2.0 * (verts - self.center) / self.span
        
        # 2. Prepare grid_sample input
        # Shape as (1, 1, B, N, 3), treating the batch as a spatial dimension
        grid_input = norm_points.view(1, 1, B, N, 3)
        
        # 3. Trilinear Interpolation
        # self.sdf_grad_grid shape: (1, 3, Res, Res, Res) -> Channel=3 (Gradient xyz)
        # Output shape: (1, 3, 1, B, N)
        sampled_grad = F.grid_sample(
            self.sdf_grad_grid, 
            grid_input, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        # 4. Restore dimensions
        # (1, 3, 1, B, N) -> (B, N, 3)
        # Use permute to reorder to (B, N, C).
        output = sampled_grad.view(3, B, N).permute(1, 2, 0)
        
        #if not is_batched:
        #    output = output.squeeze(0) # (N, 3)
        debugging_output = -output/(output.norm(dim=-1)[:,:,None]+1e-10)
        debugging_output = debugging_output.to(mesh_deubg.device)
        self.vts_norm_low_bound = torch.mean(torch.abs(1- torch.sum(debugging_output*mesh_norm,dim=-1)))

        print("error:", self.vts_norm_low_bound)
        

        face_centers= mesh_deubg.xyz[mesh_deubg.face].mean(dim=1)
        face_centers = torch.tensor(face_centers).to(self.device)
        norm_points = 2.0 * (face_centers - self.center) / self.span
        num_face = norm_points.shape[0]
        grid_input = norm_points.view(1, 1, B, num_face, 3)
        
        # 3. Trilinear Interpolation
        # self.sdf_grad_grid shape: (1, 3, Res, Res, Res) -> Channel=3 (Gradient xyz)
        # Output shape: (1, 3, 1, B, N)
        sampled_grad = F.grid_sample(
            self.sdf_grad_grid, 
            grid_input, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        # 4. Restore dimensions
        # (1, 3, 1, B, N) -> (B, N, 3)
        # Use permute to reorder to (B, N, C).
        output = sampled_grad.view(3, B, num_face).permute(1, 2, 0)
        
        #if not is_batched:
        #    output = output.squeeze(0) # (N, 3)
        debugging_output = -output/(output.norm(dim=-1)[:,:,None]+1e-10)
        debugging_output = debugging_output.to(mesh_deubg.device)
        self.face_norm_low_bound = torch.mean(torch.abs(1- torch.sum(debugging_output*mesh_face_norm,dim=-1)))

        print("error:", self.face_norm_low_bound)




    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3)
        Returns:
            grads: (B, N, 3) or (N, 3) - Normalized Gradient Vectors usually
        """
        is_batched = points.dim() == 3
        if not is_batched:
            points = points.unsqueeze(0)
            
        B, N, _ = points.shape
        
        # 1. Normalize coordinates
        norm_points = 2.0 * (points - self.center) / self.span
        
        # 2. Prepare grid_sample input
        # Shape as (1, 1, B, N, 3), treating the batch as a spatial dimension
        grid_input = norm_points.view(1, 1, B, N, 3)
        
        # 3. Trilinear Interpolation
        # self.sdf_grad_grid shape: (1, 3, Res, Res, Res) -> Channel=3 (Gradient xyz)
        # Output shape: (1, 3, 1, B, N)
        sampled_grad = F.grid_sample(
            self.sdf_grad_grid, 
            grid_input, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        # 4. Restore dimensions
        # (1, 3, 1, B, N) -> (B, N, 3)
        # Use permute to reorder to (B, N, C).
        output = - sampled_grad.view(3, B, N).permute(1, 2, 0)
        output = output/(output.norm(dim=-1)[:,:,None]+1e-10)
        if not is_batched:
            output = output.squeeze(0) # (N, 3)
            
        return output
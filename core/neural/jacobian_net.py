from core.neural.mlp_base import *
from pytorch3d.transforms import rotation_6d_to_matrix
class jacobian_Net(nn.Module):
    def __init__(self, config, device ):
        super().__init__() 
        self.device =device
        rotation_config= config['rotation_net']
        scale_config= config['scale_net']
        shear_config= config['shear_net']

        rotation_config['input_dim'] = config['input_dim']
        scale_config['input_dim'] = config['input_dim']
        shear_config['input_dim'] = config['input_dim']


        rotation_config['output_dim'] = 6
        scale_config['output_dim'] = 3
        shear_config['output_dim'] = 3
        self.identity_6d_tensor = torch.tensor([1,0,0,0,1,0],device=device)[None,None,:]

        self.rotation_net=  UniversalMLP(rotation_config,device=device)
        self.scale_net =  UniversalMLP(scale_config,device=device)
        self.shear_net =  UniversalMLP(shear_config,device=device)

        # Scale bias for identity Jacobian (J = R_identity @ diag(1,1,1) = I)
        # rotation: + identity_6d → identity, shear: 0 → correct
        # scale: if output_activation='exp', bias=0 → exp(0)=1; otherwise bias=1 → direct 1
        from core.neural.mlp_base import LinearScale, LinearNoClamp, SoftplusScale
        scale_has_identity_at_zero = isinstance(self.scale_net.output_act_fn, SafeExp) or \
                        isinstance(self.scale_net.output_act_fn, nn.Softplus) or \
                        isinstance(self.scale_net.output_act_fn, LinearScale) or \
                        isinstance(self.scale_net.output_act_fn, LinearNoClamp) or isinstance(self.scale_net.output_act_fn, SoftplusScale)
        scale_bias_val = 0.0 if scale_has_identity_at_zero else 1.0
        with torch.no_grad():
            last_layer = self.scale_net.core.net[-1]
            if isinstance(last_layer, nn.Linear) and last_layer.bias is not None:
                last_layer.bias.data.fill_(scale_bias_val)
                print(f"[jacobian_Net] scale_net last bias={scale_bias_val} "
                      f"(identity_at_zero={scale_has_identity_at_zero})")

    def forward(self, x, decompose=True):
        B, num_F, feat_dim = x.shape
        
        # 1. Rotation (6D -> 3x3 Matrix)
        # Add identity_6d_tensor to bias toward the identity matrix at the start of training
        rotation = self.rotation_net(x) + self.identity_6d_tensor
        rot_mat = rotation_6d_to_matrix(rotation) # (B, N, 3, 3)

        # 2. Scale & Shear Prediction
        # scale_net is assumed to already apply Exp internally (always positive)
        scale = self.scale_net(x)
        shear = self.shear_net(x) # (B, N, 3) -> [k_xy, k_xz, k_yz]

        # 3. Construct Upper Triangular Matrix U
        # Create a (B, N, 3, 3) zero matrix
        U = torch.zeros(B, num_F, 3, 3, device=x.device)

        # Fill diagonal entries (scale)
        U[..., 0, 0] = scale[..., 0]
        U[..., 1, 1] = scale[..., 1]
        U[..., 2, 2] = scale[..., 2]

        # Fill off-diagonal upper-triangular entries (shear)
        # Order: (0,1), (0,2), (1,2)
        U[..., 0, 1] = shear[..., 0] # k_xy
        U[..., 0, 2] = shear[..., 1] # k_xz
        U[..., 1, 2] = shear[..., 2] # k_yz

        # 4. Compute Jacobian
        # J = R @ U
        jacobian = torch.matmul(rot_mat, U).view(B,num_F,9)

        if decompose:
            return rotation, scale, shear, jacobian
        else:
            return jacobian
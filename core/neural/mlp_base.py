import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# ==============================================================================
# [Modules] Specialized Network Implementations (Modified forward)
# ==============================================================================
class SafeExp(nn.Module):
    def forward(self, x):
        # Clamp input to at most 20 (e^20 is already large enough)
        return torch.exp(torch.clamp(x, max=20.0))

class LinearScale(nn.Module):
    """1 + x, clamped to [min_val, max_val]. Identity at x=0, gradient always 1."""
    def __init__(self, min_val=0.3, max_val=3.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x):
        return (1.0 + x).clamp(self.min_val, self.max_val)

class LinearNoClamp(nn.Module):
    """1 + x, NO clamp. Identity at x=0, gradient always 1, unbounded both directions."""
    def forward(self, x):
        return 1.0 + x

class LinearShear(nn.Module):
    """c * tanh(x). Bidirectional, bounded [-c, c]. Identity (zero) at x=0."""
    def __init__(self, c=2.0):
        super().__init__()
        self.c = c
    def forward(self, x):
        return self.c * torch.tanh(x)

class SoftplusScale(nn.Module):
    """softplus(x) + 1 - log(2), then clamp(max=max_val). Identity at x=0.
    Range: [0.307, max_val]. Gradient at 0 = 0.5. Smooth bottom, hard top clamp."""
    def __init__(self, max_val=3.0):
        super().__init__()
        import math
        self.offset = 1.0 - math.log(2.0)
        self.max_val = max_val
    def forward(self, x):
        return (torch.nn.functional.softplus(x) + self.offset).clamp(max=self.max_val)
    
# --- 1. SIREN ---
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, bias=bias))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, return_feat=False):
        # Run up to (but not including) the last layer
        feat = coords
        for layer in self.net[:-1]:
            feat = layer(feat)
        
        # Run the last layer
        out = self.net[-1](feat)
        
        if return_feat:
            return out, feat
        return out

# --- 2. FINER ---
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale is not None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad: 
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        return torch.sin(self.omega_0 * scale * x)

class Finer(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))
        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))
        
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords, return_feat=False):
        feat = coords
        for layer in self.net[:-1]:
            feat = layer(feat)
        
        out = self.net[-1](feat)
        
        if return_feat:
            return out, feat
        return out

# --- 3. WIRE ---
class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        dtype = torch.float if is_first else torch.cfloat
        
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j*omega - scale.abs().square())
    
class Wire(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, first_omega_0=20, hidden_omega_0=20, scale=10.0):
        super().__init__()
        self.nonlin = ComplexGaborLayer
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, is_first=True, omega0=first_omega_0, sigma0=scale))
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale))
        
        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, return_feat=False):
        feat = coords
        for layer in self.net[:-1]:
            feat = layer(feat)
        
        out = self.net[-1](feat)
        
        # WIRE's final output is complex, so return only the real part
        if return_feat:
            return out.real, feat # feat stays complex
        return out.real

# --- 4. GAUSS ---
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.exp(-(self.scale * self.linear(input))**2)
    
class Gauss(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, scale=30.0):
        super().__init__()
        self.net = []
        self.net.append(GaussLayer(in_features, hidden_features, scale=scale))
        for i in range(hidden_layers):
            self.net.append(GaussLayer(hidden_features, hidden_features, scale=scale))
        
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, return_feat=False):
        feat = coords
        for layer in self.net[:-1]:
            feat = layer(feat)
        out = self.net[-1](feat)
        
        if return_feat:
            return out, feat
        return out

# --- 5. PEMLP ---
class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) 
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        if self.freq_bands.device != x.device:
            self.freq_bands = self.freq_bands.to(x.device)
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)

class PEMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, activation='relu', N_freqs=10):
        super().__init__()
        self.encoding = PositionalEncoding(in_channels=in_features, N_freqs=N_freqs)
        if activation=='relu':
            self.activation = nn.ReLU(True)
        elif activation=='tanh':
            self.activation = nn.Tanh()

        self.net = []
        self.net.append(nn.Linear(self.encoding.out_channels, hidden_features))
        self.net.append(self.activation)
        
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(self.activation)
            
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, return_feat=False):
        # 1. Positional Encoding
        x = self.encoding(coords)
        
        # 2. Loop until last layer
        feat = x
        for layer in self.net[:-1]:
            feat = layer(feat)
        
        # 3. Last Layer
        out = self.net[-1](feat)
        
        if return_feat:
            return out, feat
        return out
    
class LipLinear(nn.Module):
    """
    Lipschitz-constrained Linear Layer.
    Weights are normalized to ensure the Lipschitz constant of the layer is <= c.
    Uses softplus on the learned Lipschitz bound 'c' to keep it positive.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Learnable Lipschitz bound parameter (initialized to give softplus(c) approx 1)
        self.c = nn.Parameter(torch.tensor(1.0)) 
        
    def forward(self, x):
        # Calculate Lipschitz constant 'c' (ensure positive)
        c = F.softplus(self.c)
        
        # Calculate the L-infinity Lipschitz constant of the weight matrix (max row sum of absolute weights)
        # To handle division by zero, add a small epsilon
        weight_abs_sum = torch.sum(torch.abs(self.linear.weight), dim=1, keepdim=True)
        scale = c / (weight_abs_sum + 1e-6)
        
        # We only want to scale down if the norm exceeds c, but standard LipMLP usually 
        # enforces W_normalized = W * (c / norm).
        # Here we apply the normalization dynamically.
        weight_normalized = self.linear.weight * scale
        
        return F.linear(x, weight_normalized, self.linear.bias)

class LipMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        self.net = []
        
        # First layer
        self.net.append(LipLinear(in_features, hidden_features))
        self.net.append(nn.Softplus(beta=100)) # Softplus is commonly used in LipMLP for smooth gradients
        
        # Hidden layers
        for _ in range(hidden_layers):
            self.net.append(LipLinear(hidden_features, hidden_features))
            self.net.append(nn.Softplus(beta=100))
            
        # Final layer (Standard LipLinear)
        self.net.append(LipLinear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords, return_feat=False):
        feat = coords
        for layer in self.net[:-1]:
            feat = layer(feat)
        
        out = self.net[-1](feat)
        
        if return_feat:
            return out, feat
        
        return out
class TransposedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        # x: (Batch, Length, Channel) -> (Batch, Channel, Length)
        x = x.permute(0, 2, 1)
        x = self.gn(x)
        # Restore: (Batch, Channel, Length) -> (Batch, Length, Channel)
        x = x.permute(0, 2, 1)
        return x
    

# --- 6. Standard MLP Fallback ---
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, num_groups=1, relu_final=False, use_gn=False, hidden_activation='relu', norm_type=None):
        """
        norm_type (optional): 'gn' | 'ln' | 'none' | None
          - None (default): backward-compat. use_gn=True → GN, else no norm.
          - 'gn': Force GroupNorm (TransposedGroupNorm, needs permute).
          - 'ln': LayerNorm on last dim (no permute, memory-efficient).
          - 'none': no norm regardless of use_gn.
        use_gn is kept for backward compatibility but takes effect only when norm_type is None.
        """
        super().__init__()
        layers = []

        # Resolve final norm choice with backward compatibility.
        if norm_type is None:
            norm_type = 'gn' if use_gn else 'none'
        norm_type = norm_type.lower()
        assert norm_type in ('gn', 'ln', 'none'), f"norm_type must be 'gn'|'ln'|'none', got {norm_type!r}"

        def _make_norm(channels):
            if norm_type == 'gn':
                return TransposedGroupNorm(num_groups, channels)
            if norm_type == 'ln':
                return nn.LayerNorm(channels)
            return None  # 'none'

        def _get_activation(act_name):
            if act_name == 'leaky_relu':
                return nn.LeakyReLU(0.01, inplace=True)
            elif act_name == 'gelu':
                return nn.GELU()
            elif act_name == 'relu':
                return nn.ReLU(True)
            else:
                raise ValueError(
                    f"[StandardMLP._get_activation] Unknown hidden_activation '{act_name}'. "
                    f"Supported: 'relu', 'leaky_relu', 'gelu'.")

        # [Case 1] hidden_layers == 0
        if hidden_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim))

        # [Case 2] hidden_layers >= 1
        else:
            # 1. First hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(_get_activation(hidden_activation))

            # 2. Intermediate hidden layers
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                _norm = _make_norm(hidden_dim)
                if _norm is not None:
                    layers.append(_norm)
                layers.append(_get_activation(hidden_activation))

            # 3. Final output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        self.relu_final = relu_final
        self.last_relu = nn.ReLU(True)
        # Backward compatibility: expose use_gn with its original meaning (whether norm is applied)
        self.use_gn = (norm_type != 'none')
        self.norm_type = norm_type

        if self.use_gn and self.relu_final:
            # norm right after the last layer — branch on norm_type
            _last_norm = _make_norm(output_dim)
            if _last_norm is not None:
                self.last_gn = _last_norm
            else:
                self.last_gn = nn.Identity()

    def forward(self, x, return_feat=False):
        # Pass through layers (dimension permutation is now handled internally)
        feat = x
        for layer in self.net[:-1]:
            feat = layer(feat)
        
        # Pass through the last layer
        out = self.net[-1](feat)
        
        if self.relu_final:
            if self.use_gn:
                # TransposedGroupNorm, so it can be called directly without permute
                out = self.last_gn(out)
            out = self.last_relu(out)
            
        if return_feat:
            return out, feat
        return out
# ==============================================================================
# [Wrapper] Universal MLP (Modified)
# ==============================================================================
class UniversalMLP(nn.Module):
    def __init__(self, config, device):
        """
        Unified MLP Wrapper.
        """
        super().__init__()
        self.config = config
        self.device = device  # Fixed typo: self.device =device -> = device
        self.network_type = config.get('type', 'mlp').lower()
        self.lr = config.get('lr', 1e-4)

        # 1. Instantiate Core Network
        if self.network_type == 'siren':
            self.core = Siren(
                in_features=config['input_dim'],
                hidden_features=config['hidden_dim'],
                hidden_layers=config['num_layers'],
                out_features=config['output_dim'],
                first_omega_0=config.get('omega_0', 30),
                hidden_omega_0=config.get('omega_0', 30)
            )
        elif self.network_type == 'finer':
            self.core = Finer(
                in_features=config['input_dim'],
                hidden_features=config['hidden_dim'],
                hidden_layers=config['num_layers'],
                out_features=config['output_dim'],
                first_omega_0=config.get('omega_0', 30),
                hidden_omega_0=config.get('omega_0', 30),
                scale_req_grad=config.get('scale_req_grad', False)
            )
        elif self.network_type == 'wire':
            self.core = Wire(
                in_features=config['input_dim'],
                out_features=config['output_dim'],
                hidden_features=config['hidden_dim'],
                hidden_layers=config['num_layers'],
                first_omega_0=config.get('omega_0', 20),
                hidden_omega_0=config.get('omega_0', 20),
                scale=config.get('scale', 10.0)
            )
        elif self.network_type == 'gauss':
            self.core = Gauss(
                in_features=config['input_dim'],
                hidden_features=config['hidden_dim'],
                hidden_layers=config['num_layers'],
                out_features=config['output_dim'],
                scale=config.get('scale', 30.0)
            )
        elif self.network_type == 'pemlp':
            self.core = PEMLP(
                in_features=config['input_dim'],
                hidden_features=config['hidden_dim'],
                hidden_layers=config['num_layers'],
                out_features=config['output_dim'],
                activation=config['activation'],
                N_freqs=config.get('multires', 10)
            )
        elif self.network_type == 'lipmlp':
            self.core = LipMLP( #in_features, hidden_features, hidden_layers, out_features
                in_features=config['input_dim'],
                hidden_features=config['hidden_dim'],
                hidden_layers=config['num_layers'],
                out_features=config['output_dim'],
            )
        elif self.network_type == 'standard':
            self.core = StandardMLP(
                config['input_dim'],
                config['hidden_dim'],
                config['num_layers'],
                config['output_dim'],
                num_groups=config.get('num_groups',1),
                relu_final=config.get('relu_final', False),
                use_gn = config.get('use_gn',False),
                hidden_activation=config.get('hidden_activation', 'relu'),
                norm_type=config.get('norm_type', None),  # optional 'gn'|'ln'|'none'
            )

        # 2. Output Activation
        self.core = self.core.to(self.device)
        self.output_act_fn = self._get_activation(config.get('output_activation', None))

        # 3. [NEW] Zero Initialization for Last Layer
        if config.get('zero_last_init', False):
            self._init_last_layer_to_zero()

    def _get_activation(self, act_name):
        if act_name is None: return nn.Identity()
        act_name = act_name.lower()
        if act_name == 'sigmoid': return nn.Sigmoid()
        elif act_name == 'tanh': return nn.Tanh()
        elif act_name == 'relu': return nn.ReLU(True)
        elif act_name == 'leaky_relu': return nn.LeakyReLU(0.01, inplace=True)
        elif act_name == 'gelu': return nn.GELU()
        elif act_name == 'softplus': return nn.Softplus()
        elif act_name == 'exp': return SafeExp()
        elif act_name == 'linear_scale': return LinearScale()
        elif act_name == 'linear_noclamp': return LinearNoClamp()
        elif act_name == 'linear_shear': return LinearShear(c=2.0)
        elif act_name == 'softplus_scale': return SoftplusScale()
        else:
            raise ValueError(
                f"[UniversalMLP._get_activation] Unknown output_activation '{act_name}'. "
                f"Supported: None, 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu', 'softplus', "
                f"'exp', 'linear_scale', 'linear_noclamp', 'linear_shear', 'softplus_scale'.")
    # =========================================================================
    # [NEW] Zero Init Helper
    # =========================================================================
    def _init_last_layer_to_zero(self):
        """
        Zero-initializes the weight and bias of the network's last Linear layer.
        Makes the initial output close to 0 (or 0.5 etc. depending on the activation).
        """
        # Assumes every implementation (Siren, Finer, etc.) follows the self.net = nn.Sequential(...) structure
        if hasattr(self.core, 'net') and len(self.core.net) > 0:
            last_layer = self.core.net[-1]
            
            # Check whether the last layer is Linear
            if isinstance(last_layer, nn.Linear):
                with torch.no_grad():
                    last_layer.weight.zero_()
                    if last_layer.bias is not None:
                        last_layer.bias.zero_()
                        
                print(f"[{self.network_type.upper()}] Last layer initialized to ZERO.")
            elif  isinstance(last_layer, LipLinear):
                with torch.no_grad():
                    last_layer.linear.weight.zero_()
                    if last_layer.linear.bias is not None:
                        last_layer.linear.bias.zero_()
                        
                print(f"[{self.network_type.upper()}] Last layer initialized to ZERO.")
            elif  isinstance(last_layer, FinerLayer):
                with torch.no_grad():
                    last_layer.linear.weight.zero_()
                    if last_layer.linear.bias is not None:
                        last_layer.linear.bias.zero_()
                        
                print(f"[{self.network_type.upper()}] Last layer initialized to ZERO.")

            elif  isinstance(last_layer, SineLayer):
                with torch.no_grad():
                    last_layer.linear.weight.zero_()
                    if last_layer.linear.bias is not None:
                        last_layer.linear.bias.zero_()
                        
                print(f"[{self.network_type.upper()}] Last layer initialized to ZERO.")


            else:
                print(f"[Warning] The last layer of {self.network_type} is not nn.Linear. implementation skipped.")




    # =========================================================================
    # Modified Forward
    # =========================================================================
    def forward(self, x, return_feat=False):
        # 1. Core Network Forward
        if return_feat:
            out, feat = self.core(x, return_feat=True)
            out = self.output_act_fn(out)
            return out, feat
        else:
            out = self.core(x, return_feat=False)
            out = self.output_act_fn(out)
            return out

    # =========================================================================
    # Utility Functions
    # =========================================================================
    def toggle_trainable_params(self, mode=True):
        for param in self.parameters():
            param.requires_grad = mode

    def get_optimizable_params(self):
        return [{
            'params': list(self.parameters()),
            'name': f'UniversalMLP_{self.network_type}',
            'lr': self.lr
        }]
# ==============================================================================
# Usage Example
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'type': 'finer',
        'input_dim': 3,
        'hidden_dim': 32,
        'output_dim': 1,
        'num_layers': 2,
        'output_activation': 'none', # 'none' gives 0; 'sigmoid' gives 0.5.
        
        # [NEW] zero-initialize the last layer
        'zero_last_init': True 
    }

    model = UniversalMLP(config, device=device).to(device)
    dummy_input = torch.randn(2, 5, 3).to(device)

    # 1. Normal Forward
    out = model(dummy_input)
    print(f"Output Mean (Should be close to 0): {out.mean().item()}")
    print(f"Output Std  (Should be close to 0): {out.std().item()}")
    print(f"Normal Output Shape: {out.shape}")
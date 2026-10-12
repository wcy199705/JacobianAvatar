import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()  # initialize nn.Module
        self.kwargs = kwargs
        self.min_bound = kwargs['bound_min']
        self.max_bound = kwargs['bound_max']
        self.use_bound = kwargs['use_bound']
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # Using freq as a raw tensor can cause CPU/GPU device mismatch,
                # so the device should be aligned or the value captured.
                # Here we keep the original logic but wrap it in a lambda.
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.output_dim = out_dim  # renamed (out_dim -> output_dim)
        
    def forward(self, inputs):
        # nn.Module requires a forward method.
        # Iterate over embed_fns, compute each, then concat
        if self.use_bound:
            inputs = self.normalize_coords(inputs)
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def normalize_coords(self, points):
        """ Coordinate normalization: uses internal buffers """
        normalized = (points - self.min_bound) / (self.max_bound - self.min_bound)
        normalized = normalized * 2 - 1
        return normalized #torch.clamp(normalized, -1, 1)

def get_embedder(multires, bounds=None):
    if bounds is not None:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
                    'bound_min': bounds['min'],
                    'bound_max': bounds['max'],
                    'use_bound': True


        }
    else:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
                    'bound_min': None,
                    'bound_max': None,
                    'use_bound': False

        }

    # Create Embedder object
    embedder_obj = Embedder(**embed_kwargs)
    
    # Return the object itself so it can be placed in an nn.ModuleList.
    # Before: returned a lambda x ... -> Now: returns embedder_obj
    return embedder_obj, embedder_obj.output_dim

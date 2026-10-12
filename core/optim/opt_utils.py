import os
import torch
import numpy as np
def set_adam_optimizer(opt_params, lr=1e-3, decay=0.99):
    # decay 0.95 → 0.99 (less aggressive decay)
    # 7200 batches → 72 scheduler steps → LR end = 1e-3 * 0.99^72 ≈ 4.8e-4 (48% retained)
    #
    # env-var overrides for optimizer ablations (v5, v6):
    #   OPTIMIZER_TYPE=adamw          → torch.optim.AdamW (weight_decay from env)
    #   OPTIMIZER_WEIGHT_DECAY=1e-4   → AdamW weight decay (default 1e-4)
    #   SCHEDULER_DECAY_OVERRIDE=1.0  → override the caller-provided decay (v6: decay off)
    # Absence of env vars → identical to previous behavior (Adam, decay from caller).
    opt_type = os.environ.get('OPTIMIZER_TYPE', 'adam').lower()
    if opt_type == 'adamw':
        wd = float(os.environ.get('OPTIMIZER_WEIGHT_DECAY', '1e-4'))
        optimizer = torch.optim.AdamW(opt_params, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(opt_params, lr=lr)
    _decay_env = os.environ.get('SCHEDULER_DECAY_OVERRIDE')
    if _decay_env is not None:
        decay = float(_decay_env)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay ** epoch)
    return optimizer, scheduler


def set_sgd_optimizer(opt_params, lr=1e-3, decay=0.99, momentum=0.0 ): #0.9 momentum is good.
    optimizer = torch.optim.SGD(opt_params, lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay ** epoch)
    return optimizer, scheduler

def set_opt_group(params, lr=1e-3):
    params.requires_grad_()
    params.retain_grad()
    return {"params":params, "lr": lr}


def resize_normal_map(normal_map, new_shape):
  """
  Resize the normal map and normalize the result.
  
  :param normal_map: NumPy array of shape (H, W, 3)
  :param new_shape: tuple of (new_H, new_W)
  :return: resized and normalized normal map of shape (new_H, new_W, 3)
  """
  from scipy.ndimage import zoom

  # current shape and target shape
  H, W, _ = normal_map.shape
  new_H, new_W = new_shape
  
  # compute zoom factors (height, width, channel order)
  # channels (X, Y, Z) are not zoomed, so their factor is 1
  zoom_factors = (new_H / H, new_W / W, 1)
  
  # 1. Resize with linear interpolation (order=1) using SciPy
  #print(f"Resizing: ({H}, {W}) -> ({new_H}, {new_W})")
  resized_map = zoom(normal_map.astype(np.float32), zoom_factors, order=1)
  
  # 2. Normalize every pixel of the resulting map
  # compute the norm (length) of each pixel (vector)
  # keepdims=True keeps the dimension for broadcasting
  norms = np.linalg.norm(resized_map, axis=-1, keepdims=True)
  
  # add a tiny value (epsilon) to avoid division by zero
  epsilon = 1e-9
  
  # return the normalized map
  normalized_map = resized_map / (norms + epsilon)
  
  return normalized_map
def resize_depth_map(depth_map, new_shape):
  """
  Resize the normal map and normalize the result.
  
  :param depth_map: NumPy array of shape (H, W, 1)
  :param new_shape: tuple of (new_H, new_W)
  :return: resized and normalized normal map of shape (new_H, new_W, 3)
  """
  from scipy.ndimage import zoom

  # current shape and target shape
  H, W, _ = depth_map.shape
  new_H, new_W = new_shape
  
  # compute zoom factors (height, width, channel order)
  # channels (X, Y, Z) are not zoomed, so their factor is 1
  zoom_factors = (new_H / H, new_W / W, 1)
  
  # 1. Resize with linear interpolation (order=1) using SciPy
  #print(f"Resizing: ({H}, {W}) -> ({new_H}, {new_W})")
  resized_map = zoom(depth_map.astype(np.float32), zoom_factors, order=1)
  
  # 2. Normalize every pixel of the resulting map
  # compute the norm (length) of each pixel (vector)
  # keepdims=True keeps the dimension for broadcasting
  
  return resized_map



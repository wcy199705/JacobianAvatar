import torch
import numpy as np
def set_adam_optimizer(opt_params, lr=1e-3, decay=0.99):
    # decay 0.95 → 0.99 (less aggressive decay)
    # 7200 batches → 72 scheduler steps → LR end = 1e-3 * 0.99^72 ≈ 4.8e-4 (48% retained)
    optimizer = torch.optim.Adam(opt_params, lr=lr)
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


def axis_angle2sixrot(axis_angle):
    import pytorch3d
    six_rot = pytorch3d.transforms.matrix_to_rotation_6d(pytorch3d.transforms.axis_angle_to_matrix(axis_angle))
    return six_rot



def sixrot2axis_angle(sixrot):
    import pytorch3d
    axis_angle = pytorch3d.transforms.matrix_to_axis_angle(pytorch3d.transforms.rotation_6d_to_matrix(sixrot))
    return axis_angle


def resize_normal_map(normal_map, new_shape):
  """
  Resizes a normal map and normalizes the result.
  
  :param normal_map: NumPy array of shape (H, W, 3)
  :param new_shape: tuple of (new_H, new_W)
  :return: resized and normalized normal map of shape (new_H, new_W, 3)
  """
  from scipy.ndimage import zoom

  # Current shape and target shape
  H, W, _ = normal_map.shape
  new_H, new_W = new_shape
  
  # Compute zoom factors (height, width, channel order)
  # Channels (X, Y, Z) are not zoomed, so their factor is 1
  zoom_factors = (new_H / H, new_W / W, 1)
  
  # 1. Resize with SciPy using linear interpolation (order=1)
  #print(f"Resizing: ({H}, {W}) -> ({new_H}, {new_W})")
  resized_map = zoom(normal_map.astype(np.float32), zoom_factors, order=1)
  
  # 2. Normalize every pixel of the resulting map
  # Compute the norm (length) of each pixel (vector)
  # Use keepdims=True to keep dimensions for broadcasting
  norms = np.linalg.norm(resized_map, axis=-1, keepdims=True)
  
  # Add a tiny epsilon to avoid division by zero
  epsilon = 1e-9
  
  # Return the normalized map
  normalized_map = resized_map / (norms + epsilon)
  
  return normalized_map
def resize_depth_map(depth_map, new_shape):
  """
  Resizes a normal map and normalizes the result.
  
  :param depth_map: NumPy array of shape (H, W, 1)
  :param new_shape: tuple of (new_H, new_W)
  :return: resized and normalized normal map of shape (new_H, new_W, 3)
  """
  from scipy.ndimage import zoom

  # Current shape and target shape
  H, W, _ = depth_map.shape
  new_H, new_W = new_shape
  
  # Compute zoom factors (height, width, channel order)
  # Channels (X, Y, Z) are not zoomed, so their factor is 1
  zoom_factors = (new_H / H, new_W / W, 1)
  
  # 1. Resize with SciPy using linear interpolation (order=1)
  #print(f"Resizing: ({H}, {W}) -> ({new_H}, {new_W})")
  resized_map = zoom(depth_map.astype(np.float32), zoom_factors, order=1)
  
  # 2. Normalize every pixel of the resulting map
  # Compute the norm (length) of each pixel (vector)
  # Use keepdims=True to keep dimensions for broadcasting
  
  return resized_map






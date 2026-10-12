#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import math
import numpy as np
from core.math_ops.vector_ops import dot, safe_normalize


def getProjectionMatrixShift(znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY): #https://github.com/graphdeco-inria/gaussian-splatting/issues/144
    fovX = np.deg2rad(fovX) if fovX > np.pi else fovX
    fovY = np.deg2rad(fovY) if fovY > np.pi else fovY

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    offset_x = cx - (width/2)
    offset_x = (offset_x/focal_x)*znear
    offset_y = cy - (height/2)
    offset_y = (offset_y/focal_y)*znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from core.lpips.lpips import *

# class PerceptualLoss(torch.nn.Module):
#     def __init__(self, model='lpips', net='alex', spatial=False, use_gpu=False, gpu_ids=[0], version='0.1'): # VGG using our perceptually-learned weights (LPIPS metric)
#     # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
#         super(PerceptualLoss, self).__init__()
#         print('Setting up Perceptual loss...')
#         self.use_gpu = use_gpu
#         self.spatial = spatial
#         self.gpu_ids = gpu_ids
#         self.model = dist_model.DistModel()
#         self.model.initialize(model=model, net=net, use_gpu=use_gpu, spatial=self.spatial, gpu_ids=gpu_ids, version=version)
#         print('...[%s] initialized'%self.model.name())
#         print('...Done')

#     def forward(self, pred, target, normalize=False):
#         """
#         Pred and target are Variables.
#         If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
#         If normalize is False, assumes the images are already between [-1,+1]

#         Inputs pred and target are Nx3xHxW
#         Output pytorch Variable N long
#         """

#         if normalize:
#             target = 2 * target  - 1
#             pred = 2 * pred  - 1

#         return self.model.forward(target, pred)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True)+eps)
    return in_feat/(norm_factor+eps)

def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

def psnr(p0, p1, peak=255.):
    return 10*np.log10(peak**2/np.mean((1.*p0-1.*p1)**2))

def dssim(p0, p1, range=255.):
    from skimage.measure import compare_ssim
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.

def rgb2lab(in_img,mean_cent=False):
    from skimage import color
    img_lab = color.rgb2lab(in_img)
    if(mean_cent):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    return img_lab

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2tensorlab(image_tensor,to_norm=True,mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if(mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    if(to_norm and not mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
        img_lab = img_lab/100.

    return np2tensor(img_lab)


def rgb2lab(input):
    from skimage import color
    return color.rgb2lab(input / 255.)

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)



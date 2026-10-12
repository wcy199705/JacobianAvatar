# Adapted from WAFT (https://github.com/princeton-vl/WAFT, model/waft_a2.py @ b152ff1)
# Copyright (c) 2025, Princeton Vision & Learning Lab — BSD 3-Clause License (see libs/WAFT/LICENSE).
# JacobianAvatar changes: dict-style config, `forward_w_init_flow` (flow refinement from an initial flow),
# `forward_w_init_flow_prefeat` / `est_prefeat` (reuse precomputed encoder features).
# The WAFT source itself is fetched unmodified into libs/WAFT by scripts/usage/fetch_libs.sh.
import __future__
import contextlib
import importlib
import importlib.abc
import importlib.machinery
import os
import sys

_WAFT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'WAFT'))
_DINOV3_ROOT = os.path.join(_WAFT_ROOT, 'thirdparty', 'dinov3')
if _WAFT_ROOT not in sys.path:
    sys.path.append(_WAFT_ROOT)   # WAFT uses top-level `model`, `utils`, `thirdparty` imports

import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.twins import TwinsFeatureEncoder
from model.backbone.waftv2_dav2 import DepthAnythingFeature
from model.backbone.dinov3 import DinoV3Feature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS

from utils.utils import coords_grid, Padder, bilinear_sampler

import timm

class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = resconv(64, 64, k=3, s=1)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]

class WAFTv2_avatar(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.args = params
        if params['feature_encoder'] == 'twins':
            self.encoder = TwinsFeatureEncoder(frozen=True)
            self.factor = 32
        elif params['feature_encoder'] == 'dav2':
            self.encoder = DepthAnythingFeature(model_name="vits", pretrained=True, lvl=-3)
            self.factor = 112
        elif params['feature_encoder'] == 'dinov3':
            self.encoder = DinoV3Feature(model_name="vits", lvl=-3)
            self.factor = 16
        else:
            raise ValueError(f"Unknown feature encoder: {params['feature_encoder']}")

        self.pretrain_dim = self.encoder.output_dim
        self.fnet = ResNet18Deconv(3, self.pretrain_dim)
        self.iter_dim = MODEL_CONFIGS[params['iterative_module']]['features']
        self.refine_net = VisionTransformer(params['iterative_module'], self.iter_dim, patch_size=8)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.iter_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3*self.iter_dim+2, self.iter_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.iter_dim//2*3, self.iter_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 4*9, 1, padding=0, bias=True)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 6, 1, padding=0, bias=True)
        )

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1) #here OF -> 2*OF!
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 2*H, 2*W), up_info.reshape(N, C, 2*H, 2*W)

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def forward(self, image1, image2, iters=None, flow_gt=None):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.args['iters']
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = [] 
        N, _, H, W = image1.shape
        fmap1_pretrain = self.encoder(image1)
        fmap2_pretrain = self.encoder(image2)
        fmap1_img = self.fnet(image1)[0]
        fmap2_img = self.fnet(image2)[0]
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_pretrain, fmap2_img], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        
        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):                 
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args['var_max'])
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args['var_min'], max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}    
        
        return output
    


    def forward_w_init_flow(self, image1, image2, init_flow, iters=None, flow_gt=None):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.args['iters']
        #import time
        #start=time.time()
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = [] 
        N, _, H, W = image1.shape
        init_flow = padder.pad(init_flow)
        fmap1_pretrain = self.encoder(image1)
        fmap2_pretrain = self.encoder(image2)
        fmap1_img = self.fnet(image1)[0]
        fmap2_img = self.fnet(image2)[0]
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_pretrain, fmap2_img], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))

        init_flow = F.interpolate(init_flow, scale_factor=0.5, mode='bilinear', align_corners=True)
        flow_2x = init_flow/2
        #flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        #start=time.time()

        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
        #print(time.time()-start)
        #start=time.time()

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        #print(time.time()-start)
        #start=time.time()

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):                 
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args['var_max'])
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args['var_min'], max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions, 'init_flow':init_flow,}    
        #print(time.time()-start)

        return output
    



    def forward_w_init_flow_prefeat(self, feat1, feat2, init_flow, iters=None, flow_gt=None):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.args['iters']
        #import time
        #start=time.time()
        #image1 = self.normalize_image(image1)
        #image2 = self.normalize_image(image2)

        padder = Padder(init_flow.shape, factor=self.factor)
        #image1 = padder.pad(image1)
        #image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = [] 
        init_flow = padder.pad(init_flow)
        N, _, H, W = init_flow.shape

        #fmap1_pretrain = self.encoder(image1)
        #fmap2_pretrain = self.encoder(image2)
        #fmap1_img = self.fnet(image1)[0]
        #fmap2_img = self.fnet(image2)[0]
        #fmap1_2x = self.fmap_conv(torch.cat([feat1], dim=1))
        #fmap2_2x = self.fmap_conv(torch.cat([feat2], dim=1))
        net = self.hidden_conv(torch.cat([feat1, feat2], dim=1))

        init_flow = F.interpolate(init_flow, scale_factor=0.5, mode='bilinear', align_corners=True)
        flow_2x = init_flow/2 #half size.
        
        #flow_2x = torch.zeros(N, 2, H//2, W//2).to(image1.device)
        #start=time.time()

        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H//2, W//2, device=feat1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(feat2, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([feat1, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
        #print(time.time()-start)
        #start=time.time()

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        #print(time.time()-start)
        #start=time.time()

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):                 
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args['var_max'])
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args['var_min'], max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions, 'init_flow':init_flow,}    
        #print(time.time()-start)

        return output
    

    def est_prefeat(self,image):
        with torch.no_grad():
            image = self.normalize_image(image)
            padder = Padder(image.shape, factor=self.factor)
            image = padder.pad(image)
            N, _, H, W = image.shape
            #init_flow = padder.pad(init_flow)
            fmap1_pretrain = self.encoder(image)
            fmap1_img = self.fnet(image)[0]
            fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img], dim=1))

            return fmap1_2x


class _PostponedAnnotationsLoader(importlib.machinery.SourceFileLoader):
    """DINOv3 targets Python >= 3.10 (`X | None` annotations). Compile its modules with
    `from __future__ import annotations` semantics so they also import on Python 3.9."""
    def get_code(self, fullname):
        return compile(self.get_data(self.path), self.path, 'exec',
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)


class _Dinov3Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != 'dinov3' and not name.startswith('dinov3.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec and spec.origin and spec.origin.startswith(_DINOV3_ROOT) and spec.origin.endswith('.py'):
            spec.loader = _PostponedAnnotationsLoader(spec.name, spec.origin)
        return spec


def _dinov3_backbone(repo_or_dir, model, *args, **kwargs):
    """Stands in for WAFT's `torch.hub.load(thirdparty/dinov3, 'dinov3_vits16', ...)`: builds the
    backbone directly (no hubconf, no weight download — spring.pth holds every encoder weight)."""
    if not any(isinstance(f, _Dinov3Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Dinov3Finder())
    if _DINOV3_ROOT not in sys.path:
        sys.path.append(_DINOV3_ROOT)
    backbones = importlib.import_module('dinov3.hub.backbones')
    return getattr(backbones, model)(pretrained=False)


@contextlib.contextmanager
def _no_pretrained_download():
    """Build WAFT without fetching the DINOv3 / timm ViT ImageNet weights: every one of those
    parameters is overwritten by the WAFT checkpoint (spring.pth) right after construction."""
    import timm
    orig_hub_load, orig_create = torch.hub.load, timm.create_model
    torch.hub.load = _dinov3_backbone
    timm.create_model = lambda *a, **kw: orig_create(*a, **{**kw, 'pretrained': False})
    try:
        yield
    finally:
        torch.hub.load, timm.create_model = orig_hub_load, orig_create


def build_waft_avatar(params):
    with _no_pretrained_download():
        return WAFTv2_avatar(params)

import torch
import cv2
from typing import Union
#from render.gaussian.gs2d_renderer import GS2DRenderer

from render.gaussian.gs_renderer import GSRenderer
from render.nvdiffrast.nvdr_renderer import NVDiffRenderer
from scene.object.gaussian import GaussianModel
from scene.camera.cameraunified import UnifiedCamera


class UnifiedRenderer:
    def __init__(self, device=torch.device("cuda:0")):
        self.device = device
        self.mesh_render = NVDiffRenderer(lighting_type='constant', device=self.device)
        self.mesh_render_w_light = NVDiffRenderer(lighting_type='front', lighting_space='camera', device=self.device)

        self.gs_render = GSRenderer()
        #self.gs2d_render = GS2DRenderer()

    def render(self, object, camera: Union[UnifiedCamera]): # : Union[GaussianModel | TexturedMesh | SMPLMesh]
        # render image regardless of input data types and rendering methods
        if isinstance(object, GaussianModel):
            return self.gs_render.render(object, camera)
        else:
            return self.mesh_render.render(object, camera)

    def save_results(self, output, path2save, type='image'):
        if type == 'image' and ('rgba' in output or 'render' in output):
            if 'rgba' in output:
                data_np = output['rgba'].detach().cpu().numpy().squeeze(0) * 255
            elif 'render' in output:
                data_np = output['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255
        elif type == 'normal' and 'normal' in output:
            data_np = output['normal'].detach().cpu().numpy().squeeze(0) * 255
        elif type == 'mask':
            data_np = output['mask'].detach().cpu().numpy().squeeze(0) * 255
        else:
            data_np = output['normal'].detach().cpu().numpy().squeeze(0) * 255
        cv2.imwrite(path2save, data_np)

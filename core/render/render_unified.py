import torch
import cv2
from typing import Union
#from render.gaussian.gs2d_renderer import GS2DRenderer

from core.render.gaussian.gs_renderer import GSRenderer
from core.render.nvdiffrast.nvdr_renderer import NVDiffRenderer
from core.gaussian import GaussianModel
from core.camera import UnifiedCamera


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


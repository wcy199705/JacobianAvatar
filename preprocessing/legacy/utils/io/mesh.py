"""Stub for legacy `from utils.io.mesh import *` in preprocessing scripts.

`utils/io/mesh.py` was never committed to JacobianAvatar_v2 yet
`scene/data_loader.py` does `from utils.io.mesh import *` and then uses
`glob`, `cv2`, `os`, etc. — all without importing them. The legacy code
has always relied on this module to side-import those for it.

Replicating that behavior so preprocessing scripts can run.
"""
# All of these are referenced (unguardedly) by legacy scene/data_loader.py
import glob   # noqa: F401
import os     # noqa: F401
import cv2    # noqa: F401
import json   # noqa: F401
import numpy as np  # noqa: F401
import torch  # noqa: F401
import trimesh  # noqa: F401

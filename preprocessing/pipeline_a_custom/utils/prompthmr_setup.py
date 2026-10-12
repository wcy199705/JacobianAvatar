"""Use the unmodified PromptHMR repo (libs/PromptHMR, fetched by scripts/usage/fetch_libs.sh)
with this repository's paths. Everything PromptHMR would otherwise need patched is set here at runtime:

- import paths (PromptHMR expects to run from its own repo root),
- body-model locations (`data/body_models/...` → the scene yaml's `SMPL.smpl_root`),
- no MetaCLIP download (every CLIP weight is overwritten by the PromptHMR checkpoint),
- the optional MCS-viewer exporter (needs `gloss` / `smpl_rs`) and DROID-SLAM backends (static camera), not used here.
"""
import contextlib
import importlib.util
import os
import sys
import types

_JA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PHMR_ROOT = os.path.join(_JA_ROOT, 'libs', 'PromptHMR')

if PHMR_ROOT not in sys.path:
    sys.path.append(PHMR_ROOT)
for _p in ('pipeline/gvhmr', 'pipeline/droidcalib/droid_slam'):   # PromptHMR inserts these (cwd-relative)
    _p = os.path.join(PHMR_ROOT, _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

if importlib.util.find_spec('gloss') is None or importlib.util.find_spec('smpl_rs') is None:
    _stub = types.ModuleType('pipeline.mcs_export_cam')
    _stub.export_scene_with_camera = None
    sys.modules['pipeline.mcs_export_cam'] = _stub

# DROID-SLAM (camera tracking) is only used for moving cameras; this pipeline assumes a static camera.
# Without its compiled backends, give `pipeline.camera` an import-only placeholder.
if importlib.util.find_spec('lietorch') is None or importlib.util.find_spec('droid_backends_intr') is None:
    _droid = types.ModuleType('droid')
    _droid.Droid = None
    sys.modules.setdefault('droid', _droid)


@contextlib.contextmanager
def _no_clip_download():
    import open_clip
    orig = open_clip.create_model
    open_clip.create_model = lambda *a, **kw: orig(*a, **{**kw, 'pretrained': None})
    try:
        yield
    finally:
        open_clip.create_model = orig


def load_phmr(smpl_root, folder=os.path.join(PHMR_ROOT, 'data', 'pretrain', 'phmr')):
    """PromptHMR image model with SMPL-X / SMPL from `smpl_root` (e.g. 'resource')."""
    import prompt_hmr.models.phmr as phmr_module
    import prompt_hmr.smpl_family.smpl_wrapper as smpl_wrapper
    from prompt_hmr import load_model_from_folder

    phmr_module.SMPLX_MODEL_DIR = os.path.join(smpl_root, 'smplx')
    phmr_module.SMPL_MODEL_DIR = os.path.join(smpl_root, 'smpl')
    phmr_module.SMPLX2SMPL = os.path.join(smpl_root, 'model_transfer', 'smplx2smpl.pkl')
    smpl_wrapper.SMPLX2SMPL_JOINTS = os.path.join(smpl_root, 'smplx2smpl_joints.npy')
    with _no_clip_download():
        return load_model_from_folder(folder)

# Third-party licenses

Parts of this repository are derived from, or use libraries from, the projects below. Those parts are subject to their original licenses.
Files derived from 3D Gaussian Splatting and Neural Jacobian Fields may be used for noncommercial research purposes only.

## 3D Gaussian Splatting — [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

License: [gaussian-splatting/LICENSE.md](gaussian-splatting/LICENSE.md) (Inria / MPII, non-commercial research use)

- Files adapted from the original code (the "LICENSE.md" referenced in their headers is this file):
  - `core/gaussian_utils.py`, `core/graphics/graphics_utils.py`, `core/render/gaussian/gs_renderer.py`
  - `preprocessing/legacy/scene/object/gaussian_utils.py`, `preprocessing/legacy/utils/graphics/graphics_utils.py`,
    `preprocessing/legacy/render/gaussian/gs_renderer.py`
- The dependency `diff_gauss` ([slothfulxtx/diff-gaussian-rasterization](https://github.com/slothfulxtx/diff-gaussian-rasterization),
  installed via `requirements.txt`) is subject to the same license.

## nvdiffrast — [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast)

License: [nvdiffrast/LICENSE.txt](nvdiffrast/LICENSE.txt) (NVIDIA Source Code License)

- Used for mesh rasterization (commit `729261d` installed via `requirements.txt`):
  `core/render/nvdiffrast/nvdr_renderer.py`, `core/render/render_unified.py`, `core/geometry/camera_utils.py`,
  and the files of the same names under `preprocessing/legacy/render/`

## Neural Jacobian Fields — [ThibaultGROUEIX/NeuralJacobianFields](https://github.com/ThibaultGROUEIX/NeuralJacobianFields)

License: [NeuralJacobianFields/license.txt](NeuralJacobianFields/license.txt) (Adobe Research License, noncommercial research use only)

- Files adapted from the original code:
  - `core/optim/poisson.py` (from `source_njf/PoissonSystem.py`)
  - `core/neural/jacobian_system.py` (differential-operator setup from `source_njf/MeshProcessor.py`)

## Other external libraries

For the licenses of MHR, WAFT, SAMURAI, PromptHMR, Sapiens2, and mmpose (downloaded by `fetch_libs.sh`), see
the table in Section 0 of [scripts/usage/ASSETS.md](../scripts/usage/ASSETS.md) and each `libs/<name>/LICENSE*`.

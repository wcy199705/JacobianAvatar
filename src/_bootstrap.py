"""Set up `sys.path` and `cwd` for the jacobian_avatar standalone tree.

The project layout is::

    jacobian_avatar/
        main.py              ← entry point
        src/                 ← thin glue layer (configs, stages, features)
        core/                ← migrated training/model code (was scene/, utils/)
        configs/             ← scene + paths YAML
        resource/            ← symlink to SMPL-X / body model assets
        libs/                ← compiled CUDA libs (diff-gs, etc.)

`install_path()` puts the project root on `sys.path` so both `src.*` and
`core.*` resolve, and chdir's to the root so the relative paths in
``configs/legacy/SMPLX_dataloader/CUSTOM/*.yaml`` (e.g. ``smpl_root:
'resource'``) point at the symlinked resource tree.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def install_path() -> Path:
    """Make `PROJECT_ROOT` importable and the current working directory.

    Idempotent: safe to call more than once. Returns the resolved root.
    """
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root_str)
    return PROJECT_ROOT

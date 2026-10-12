"""Config loader for v5_release.

Loads `paths.yaml`, `default_v5.yaml`, and a dataset-specific yaml; merges them
into a single typed dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import re
import yaml


@dataclass
class PathConfig:
    """Resolved filesystem roots used by data loaders.

    `data` carries dataset roots (e.g. ``{'data_root': '/path/...'}``).
    """
    data: Dict[str, str] = field(default_factory=dict)


@dataclass
class LossConfig:
    mesh: Dict[str, Any] = field(default_factory=dict)
    gs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingHyperparams:
    body_triplane_lr: float = 0.01
    pose_noise_scale: float = 1.0
    inject_pose_lr: float = 1.0e-4
    inject_transl_lr: float = 1.0e-4
    gs_freeze_rgb_iters: int = 0


@dataclass
class TrainingConfig:
    paths: PathConfig
    dataset_name: str
    scene_name: str
    data_path: str
    model: Dict[str, Any]
    mesh2splat: Dict[str, Any]
    scale_clamp: Dict[str, Any]
    training: TrainingHyperparams
    features: Dict[str, Any]
    stages: Dict[str, Any]
    losses: LossConfig
    disabled_losses: List[str]
    # Optional override: writes outputs (ckpts, debug renders) here instead of
    # the save_root field in the legacy <scene>_data2.yaml. Set per-scene to
    # isolate fresh training runs from the cached legacy artifacts.
    save_root_override: str = ''
    # Optional per-scene filter: path to a text file listing the absolute or
    # basename paths of frames to KEEP. Empty / None ⇒ no filtering. Set in the
    # dataset yaml as `valid_frames_file: /path/to/list.txt`.
    valid_frames_file: str = ''
    # Optional anchor-frame head-loss: when anchor_head_only=true, head losses
    # (Mesh_head_rgb_lpips/l1, GS_head_rgb_lpips/l1) are computed only on the
    # forced-included anchor frame (batch[0]); other frames contribute body
    # losses as usual. anchor_frame_file: path to a text file with one frame
    # path (its index in the dataset is used as the forced batch[0]).
    anchor_head_only: bool = False
    anchor_frame_file: str = ''
    # Optional cfg-driven debug camera (fine_train_view / novel_posed_*_canon
    # renders during training). Supports: width_scale, height_scale (multiplied
    # on training W/H), OR explicit width/height, plus optional fx/fy/cx/cy.
    # Defaults produce training-camera-identical debug renders (backward compat).
    debug_cam: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VAR = re.compile(r'\$\{([^}]+)\}')


def _resolve_vars(value: Any, context: Dict[str, Any]) -> Any:
    """Recursively replace ${a.b.c} placeholders using `context`."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            cur: Any = context
            for part in key.split('.'):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return m.group(0)  # leave untouched
            return str(cur)
        return _VAR.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_vars(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_vars(v, context) for v in value]
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(dataset_yaml: str) -> TrainingConfig:
    """Load a dataset/scene yaml + the default + paths, return a `TrainingConfig`."""
    dataset_path = Path(dataset_yaml).resolve()
    # configs/datasets/<file>.yaml -> v5_release root is parents[2]
    repo_root = dataset_path.parents[2]

    paths_dict = _load_yaml(repo_root / 'configs' / 'paths.yaml')
    dataset_raw = _load_yaml(dataset_path)

    # support chained `extends` so dataset-flavor defaults (e.g.
    # default_v6_custom.yaml) can themselves extend default_v6.yaml.
    def _load_with_extends(path: Path) -> Dict[str, Any]:
        raw = _load_yaml(path)
        parent_ref = raw.pop('extends', None)
        if parent_ref:
            parent_path = (path.parent / parent_ref).resolve()
            parent_dict = _load_with_extends(parent_path)
            return _merge(parent_dict, raw)
        return raw

    extends = dataset_raw.pop('extends', None)
    if extends:
        default_path = (dataset_path.parent / extends).resolve()
        default_dict = _load_with_extends(default_path)
    else:
        default_path = repo_root / 'configs' / 'default_v5.yaml'
        default_dict = _load_yaml(default_path)

    merged = _merge(default_dict, dataset_raw)

    # variable substitution: ${paths.data.<key>} → actual path
    resolution_ctx = {'paths': paths_dict}
    merged = _resolve_vars(merged, resolution_ctx)

    losses_raw = merged.get('losses', {})
    losses = LossConfig(
        mesh=losses_raw.get('mesh', {}),
        gs=losses_raw.get('gs', {}),
    )

    training_raw = merged.get('training', {})
    training = TrainingHyperparams(
        body_triplane_lr=float(training_raw.get('body_triplane_lr', 0.01)),
        pose_noise_scale=float(training_raw.get('pose_noise_scale', 1.0)),
        inject_pose_lr=float(training_raw.get('inject_pose_lr', 1.0e-4)),
        inject_transl_lr=float(training_raw.get('inject_transl_lr', 1.0e-4)),
        gs_freeze_rgb_iters=int(training_raw.get('gs_freeze_rgb_iters', 0)),
    )

    paths = PathConfig(data=paths_dict.get('data', {}))

    return TrainingConfig(
        paths=paths,
        dataset_name=merged.get('dataset_name', 'unknown'),
        scene_name=merged.get('scene_name', 'unknown'),
        data_path=merged.get('data_path', ''),
        model=merged.get('model', {}),
        mesh2splat=merged.get('mesh2splat', {}),
        scale_clamp=merged.get('scale_clamp', {}),
        training=training,
        features=merged.get('features', {}),
        stages=merged.get('stages', {}),
        losses=losses,
        disabled_losses=list(merged.get('disabled_losses', [])),
        save_root_override=str(merged.get('save_root_override', '') or ''),
        valid_frames_file=str(merged.get('valid_frames_file', '') or ''),
        anchor_head_only=bool(merged.get('anchor_head_only', False)),
        anchor_frame_file=str(merged.get('anchor_frame_file', '') or ''),
        debug_cam=dict(merged.get('debug_cam', {}) or {}),
    )

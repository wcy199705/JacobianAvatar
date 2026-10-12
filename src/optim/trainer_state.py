"""Carrier for state that flows between training stages.

Every stage entry function accepts this object and may mutate it; subsequent
stages read from it. Introducing this struct replaces the implicit shared
state in the legacy training loop with an explicit, documented interface.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainerState:
    """Mutable training state shared across stages.

    Attributes set by Pre stage, consumed by L0/L1/All/GS:
    - avatar: Cascade_Avatar instance built from initial mesh + SMPL params
    - smpl_mesh: AvatarMesh wrapper (canonical mesh + skinning)
    - train_dataset: Custom dataset (image / mask / camera / SMPLX_param)
    - save_root: Result folder root (e.g., outputs/<scene>/default)

    Attributes set by L0/L1/All, read by GS:
    - avatar_optimizer: torch optimizer (built per stage; persists if scheduler reuses)
    - avatar_scheduler: lr scheduler
    - load_smpl_batch: smpl_params batch tensor returned by load_optimizer_outputs

    Attributes set by mesh stages, read by GS:
    - mesh_ckpt_path: Path to the latest mesh-stage Avatar.pth (e.g., All_iter_0_...)

    Attributes for WAFT optical flow:
    - flow_loader: OF_loader instance built during L0+

    Anything stage-specific can go into `extras` to keep the dataclass small.
    """
    # Core model / data
    avatar: Any = None
    smpl_mesh: Any = None
    train_dataset: Any = None
    save_root: str = ''
    curr_exp_name: str = ''

    # Optimizer / scheduler
    avatar_optimizer: Any = None
    avatar_scheduler: Any = None
    load_smpl_batch: Any = None

    # Mesh ckpt tracking
    mesh_ckpt_path: Optional[str] = None

    # Optical flow
    flow_loader: Any = None

    # Stage-specific scratch
    extras: Dict[str, Any] = field(default_factory=dict)

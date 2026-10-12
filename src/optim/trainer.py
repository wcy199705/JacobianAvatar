"""Top-level training orchestrator.

In-process dispatcher that calls each stage module in order. `main.py`
invokes `train(cfg)` to run the full v5 pipeline.
"""
from __future__ import annotations
import os

# Note: this module assumes `apps/Rebuttal/v5_release/` is already on sys.path
# (added by `main.py` or by the caller). Inserting `src/` here directly would
# shadow the legacy `utils.*` namespace package, since we also have
# `src/utils/`.
from src.config import TrainingConfig
from src.optim.setup import setup_trainer
from src.optim.mesh_setup import setup_mesh_stages
from src.stages.pre import run_pre_stage
from src.stages.pre_loose_smpl import run_pre_stage_loose_smpl
from src.stages.l0 import run_l0_stage
from src.stages.l1 import run_l1_stage
from src.stages.all_stage import run_all_stage
from src.stages.gs import run_gs_stage


def train(cfg: TrainingConfig, skip_gs: bool = False, skip_gs2: bool = False,
          only_pre: bool = False) -> int:
    """Run the v5 pipeline end to end.

    Args:
        cfg: parsed v5 config.
        skip_gs: if True, stop after All stage (do not enter GS#1). Useful for
                 mesh-only ablations.
        skip_gs2: accepted for CLI compatibility. The Custom release has no
                  GS#2 (test-time pose fit needs a ground-truth test video), so
                  training always ends after GS#1.

    Returns:
        0 on success; non-zero on failure (exceptions propagate by default —
        the return code is reserved for future graceful-shutdown wiring).
    """
    print(f'[trainer] scene={cfg.scene_name} dataset={cfg.dataset_name}')
    print(f'[trainer] subdivide_count={cfg.model.get("subdivide_count")}, '
          f'pose_noise_scale={cfg.training.pose_noise_scale}')

    state = setup_trainer(cfg)
    print('[trainer] setup → done')

    # yaml `stages.pre.loose_smpl: true` selects the
    # `pre_loose_smpl.py` variant (masks×5, laplacian×5, iso_sdf×0.1,
    # normal_cosine=1.0) for long-hair / loose-clothing scenes.
    _use_loose = bool(cfg.stages.get('pre', {}).get('loose_smpl', False))
    if _use_loose:
        print('[trainer] Pre variant: pre_loose_smpl (stages.pre.loose_smpl=true)')
        run_pre_stage_loose_smpl(cfg, state)
    else:
        run_pre_stage(cfg, state)
    print('[trainer] Pre → done')

    if only_pre:
        print('[trainer] only_pre=True → stopping before L0')
        return 0

    setup_mesh_stages(cfg, state)
    print('[trainer] mesh_setup → done')

    # env `MOMENTUM_RESET_PER_STAGE=1` clears avatar_optimizer.state
    # before each stage (fresh Adam m/v). Isolates stale-momentum effect at
    # stage boundaries. Scheduler state (last_epoch) NOT touched here — that
    # is a separate axis (SCHEDULER_DECAY_OVERRIDE / LR_SCHEDULER_ADAPTIVE).
    # env `SCHEDULER_RESET_PER_STAGE=1` resets scheduler.last_epoch = 0 at each
    # stage start (LR back to initial). Useful for per-stage mild decay.
    _mom_reset = os.environ.get('MOMENTUM_RESET_PER_STAGE', '0') == '1'
    _sch_reset = os.environ.get('SCHEDULER_RESET_PER_STAGE', '0') == '1'
    def _maybe_reset(stage_name):
        if _mom_reset and state.avatar_optimizer is not None:
            state.avatar_optimizer.state.clear()
            print(f'[trainer] MOMENTUM_RESET_PER_STAGE=1 → cleared avatar_optimizer.state before {stage_name}')
        if _sch_reset and state.avatar_scheduler is not None:
            # last_epoch = -1 → first .step() sets last_epoch=0 → LR = base * lambda(0) = base
            state.avatar_scheduler.last_epoch = -1
            state.avatar_scheduler._step_count = 0
            # Force LR back to base on the optimizer immediately so log/first iter sees base
            for pg, base_lr in zip(state.avatar_optimizer.param_groups, state.avatar_scheduler.base_lrs):
                pg['lr'] = base_lr
            print(f'[trainer] SCHEDULER_RESET_PER_STAGE=1 → scheduler.last_epoch=-1, LR reset to base before {stage_name}')

    _maybe_reset('L0')
    run_l0_stage(cfg, state)
    print('[trainer] L0 → done')

    _maybe_reset('L1')
    run_l1_stage(cfg, state)
    print('[trainer] L1 → done')

    _maybe_reset('All')
    run_all_stage(cfg, state)
    print('[trainer] All → done')

    if skip_gs:
        print('[trainer] skip_gs=True → stopping before GS#1')
        return 0

    _maybe_reset('GS#1')
    run_gs_stage(cfg, state)
    print('[trainer] GS#1 → done')

    print('[trainer] pipeline complete (Custom release: Pre → L0 → L1 → All → GS#1)')
    return 0

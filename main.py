"""jacobian_avatar entry point — standalone runner.

Thin CLI wrapper. Dispatches to per-stage modules via `src/optim/trainer.py`.
Path setup happens through `src/_bootstrap.py`, which adds the project root to
`sys.path` and chdir's there so relative paths in scene YAMLs resolve.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Capture the invocation cwd before the bootstrap chdir's, so a relative
# `--config` path still resolves against the directory the user was in.
INITIAL_CWD = os.getcwd()

from src._bootstrap import install_path
install_path()

from src.config import TrainingConfig, load_config  # noqa: E402
from src.optim.trainer import train                  # noqa: E402


def _abs_config(p: str) -> str:
    """Resolve `--config` against the user's original cwd."""
    pp = Path(p)
    return str(pp.resolve() if pp.is_absolute() else (Path(INITIAL_CWD) / pp).resolve())


def _install_global_cosine_annealing(cfg: TrainingConfig) -> None:
    """If yaml `stages.all_stages.cosine_annealing: true`, monkey-patch
    `core.optim.opt_utils.set_adam_optimizer` so ALL stages (Pre/L0/L1/All/GS#1/GS#2)
    use CosineAnnealingLR with T_max / eta_min_ratio from yaml.
    """
    _all_cfg = cfg.stages.get('all_stages', {})
    if not bool(_all_cfg.get('cosine_annealing', False)):
        return
    import torch as _torch
    import core.optim.opt_utils as _ou
    _t_max = int(_all_cfg.get('cosine_t_max', 10))
    _eta_min_ratio = float(_all_cfg.get('eta_min_ratio', 0.01))

    def _global_cosine(opt_params, lr=1e-3, decay=0.99):
        opt = _torch.optim.Adam(opt_params, lr=lr)
        sch = _torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=_t_max, eta_min=lr * _eta_min_ratio)
        return opt, sch

    _ou.set_adam_optimizer = _global_cosine
    print(f'[main] global cosine annealing active: T_max={_t_max}, eta_min_ratio={_eta_min_ratio}')


def _print_cfg_summary(cfg: TrainingConfig) -> None:
    """One-shot header line printout on startup — useful in log grep."""
    print(f'[jacobian_avatar] dataset={cfg.dataset_name} scene={cfg.scene_name}')
    print(f'[jacobian_avatar] data_path={cfg.data_path}')
    print(f'[jacobian_avatar] mesh losses: {len(cfg.losses.mesh)} keys')
    print(f'[jacobian_avatar] gs   losses: {len(cfg.losses.gs)} keys')


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to dataset/scene yaml')
    ap.add_argument('--skip_gs', action='store_true',
                    help='Stop after All stage (mesh-only).')
    ap.add_argument('--skip_gs2', action='store_true',
                    help='Kept for compatibility; the Custom release always stops after GS#1.')
    ap.add_argument('--only_pre', action='store_true',
                    help='Run Pre stage only (mesh + SMPL-X fit), stop before L0.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Load config and print effective settings without training.')
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(_abs_config(args.config))
    _print_cfg_summary(cfg)
    if args.dry_run:
        return 0
    _install_global_cosine_annealing(cfg)
    return train(cfg, skip_gs=args.skip_gs, skip_gs2=args.skip_gs2,
                 only_pre=args.only_pre)


if __name__ == '__main__':
    sys.exit(main())

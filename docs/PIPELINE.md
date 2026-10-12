# Training Pipeline

End-to-end training flow of the Jacobian avatar.

Entry for Custom: `scripts/main_hard_zero_headhand.py --config <yaml>` (wraps `main.py`; use `scripts/usage/01_train_custom.sh`).
The wrapper is required: it applies the GS head/hand hard-zero and per-region sigma patches that the Custom recipe expects.

## Flow (v5 pipeline)
```
main.py
  ├─ load_config(yaml)                     [src/config.py]
  ├─ (optional) global cosine annealing patch
  └─ src.optim.trainer.train(cfg)
       ├─ setup_trainer(cfg)                [src/optim/setup.py]
       │    ├─ load general/model/exp params (legacy yaml)
       │    ├─ build train_dataset (core/data/custom.py; frame filter = joint_valid_frames.txt)
       │    ├─ build novel-pose dataset (Custom: same folder as training)
       │    ├─ build optional novel-pose driver (exavatar_pose_folder; skipped if empty)
       │    ├─ build optical-flow loader
       │    ├─ build smpl_mesh (AvatarMesh)
       │    └─ init gpo_optimizer + save_folders
       │
       ├─ Pre stage                          [src/stages/pre.py or pre_loose_smpl.py]
       │    stages.pre.loose_smpl=true → pre_loose_smpl (masks×5, lap×5, iso_sdf×0.1)
       │    outputs to preprocessing/{Pre0, Pre_Step1..3}
       │
       ├─ setup_mesh_stages(cfg, state)      [src/optim/mesh_setup.py]
       │    builds Cascade_Avatar + debug_cams + loss confs
       │
       ├─ (optional per-stage reset via env)
       │    MOMENTUM_RESET_PER_STAGE=1 → clear avatar_optimizer.state
       │    SCHEDULER_RESET_PER_STAGE=1 → reset scheduler.last_epoch
       │
       ├─ L0 stage                           [src/stages/l0.py]
       ├─ L1 stage                           [src/stages/l1.py]
       ├─ All stage                          [src/stages/all_stage.py]  ← MESH ckpt (All_iter_*)
       └─ GS#1 stage                         [src/stages/gs.py]         ← GS ckpt (GS_iter_*)
```

## Skip Flags
- `--only_pre`: run Pre only.
- `--skip_gs`: run through All (mesh-only).
- `--skip_gs2`: accepted for compatibility only (this release always stops after GS#1).
- `--dry_run`: parse config, print, exit.

## Per-stage ckpt naming (in save_root/3DGS_SMPLX_Version_poisson/AFTER_.../)
- Pre: `preprocessing/{Pre_Step1..3}_...pth`
- L0: `l0_iter_0_sampling_5_more_smoothing_Avatar.pth`
- L1: `L1_iter_0_sampling_5_more_smoothing_Avatar.pth`
- All: `All_iter_0_sampling_5_Avatar.pth` ← **for viz Mesh + P_normal**
- GS#1: `GS_iter_0_sampling_1_w_both_training_levels_w_strong_color_lap_Avatar.pth` ← **for viz GS**

## Key Environment Variables (runtime)
- `LR_SCHEDULER_ADAPTIVE=1` — data-count-dependent scheduler step
- `SCHEDULER_RESET_PER_STAGE=1` — reset LR to base at each stage start
- `MOMENTUM_RESET_PER_STAGE=1` — clear Adam m/v at each stage boundary
- `GS_ZERO_HEAD_HAND=1` — hard-zero GS mean_offset for head/hand semantic faces
  (this and the next two are read ONLY by `scripts/main_hard_zero_headhand.py`; plain `main.py` ignores them)
- `MESH2GS_{HEAD,HAND,BODY}_SIGMA_MULT=<float>` — mesh2splat sigma scale per region
- `GS_BODY_DELAY_ITERS=N` — freeze body mean_offset for first N iterations of GS#1

## Dataset
Custom only — loader `core/data/custom.py`, scene yaml `configs/legacy/SMPLX_dataloader/CUSTOM/<scene>.yaml`.
See [DATASETS.md](DATASETS.md).

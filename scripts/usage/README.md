# Run Scripts

All scripts work from any working directory (they compute the repository path themselves). Shared settings live in `_env.sh`.

| # | Script | Purpose |
|---|---|---|
| 00 | `00_preprocess_custom.sh <scene> <gpu> [steps]` | Preprocess video frames into training data (7 steps + finalize) |
| 01 | `01_train_custom.sh <dataset_yaml> <gpu> [flags]` | Training: Pre → L0 → L1 → All → GS#1 |
| 02 | `02_only_pre_stage.sh <dataset_yaml> <gpu>` | Initial body estimation (Pre) only — e.g. to build the frame validation list in advance |
| 03 | `03_skip_gs_mesh_only.sh <dataset_yaml> <gpu>` | Train up to the mesh only (skip Gaussians) |
| 04 | `04_render_viz.sh <dataset_yaml> <out_dir> <gpu> [opts]` | 360-degree orbit rendering + video |
| 05 | `05_dry_run_config_check.sh <dataset_yaml>` | Check the config file without training |
| — | `fetch_assets.sh` | Fetch external libraries (`fetch_libs.sh`), weights, and body models (requires SMPL-X/SMPLify accounts) |
| — | `fetch_libs.sh` | Fetch only the external library sources from their original GitHub repositories |
| — | `check_assets.sh` | Check that the downloaded weights and body models are in place |

For details see [USAGE.md](USAGE.md); for installation see [INSTALL.md](INSTALL.md) · [ASSETS.md](ASSETS.md).

Defaults can be overridden with environment variables:
- `PY` — Python for training (default `$(conda info --base)/envs/Jacobian_avatar/bin/python`)
- `PREP_PY`, `SAPIENS_PY` — Python for preprocessing (default conda envs `jacobian_avatar_prep`, `sapiens2`)
- `LOG_DIR` — preprocessing log location (default `logs/preprocess`)

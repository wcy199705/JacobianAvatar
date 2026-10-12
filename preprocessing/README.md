# preprocessing/

Preprocessing of captured video (frame images) into training data. For usage, see Section 3 of [scripts/usage/USAGE.md](../scripts/usage/USAGE.md).

```
preprocessing/
├── pipeline_a_custom/
│   ├── run_scene.sh          run the full pipeline for one scene (switches conda env per step automatically)
│   ├── run_batch.sh          run multiple scenes in parallel, one per GPU
│   ├── run_step.py           step script runner (injects the scene yaml)
│   ├── run_smplx_only.sh     stop at SMPL-X (skip normals and face crop)
│   ├── steps/step{1..7}_*.py step implementations
│   ├── utils/                K.npy generation, mask check video, etc.
│   ├── export/               convert SMPL-X results to the novel-pose driving format
│   └── viz/                  visualizations for inspecting results
└── legacy/                   modules used by the step scripts (scene/, render/, utils/)
```

Run:

```bash
bash preprocessing/pipeline_a_custom/run_scene.sh <scene> <gpu_id> [steps]
# default steps: 1a,2,mv,1b,3,4,5,6,7,finalize
```

- Scene config: `Path.training_folder` in `configs/legacy/SMPLX_dataloader/CUSTOM/<scene>.yaml`
- conda env: step 4 uses `sapiens2`, all other steps use `jacobian_avatar_prep` (paths can be changed via `PREP_PY`, `SAPIENS_PY`)
- Log: `logs/preprocess/<scene>.log` (can be changed via `LOG_DIR`)

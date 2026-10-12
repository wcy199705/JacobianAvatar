# Shared settings for usage scripts (sourced). Override any variable before calling.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null)}
PY=${PY:-$CONDA_BASE/envs/Jacobian_avatar/bin/python}
# Required for all training runs (cosine LR schedule + per-stage reset).
TRAIN_ENV="LR_SCHEDULER_ADAPTIVE=1 SCHEDULER_RESET_PER_STAGE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
# Custom recipe: hard-zero head/hand GS offsets + per-part Gaussian sigma.
# (scripts/main_hard_zero_headhand.py reads and applies these values — ignored when running main.py alone)
CUSTOM_ENV="GS_ZERO_HEAD_HAND=1 MESH2GS_HEAD_SIGMA_MULT=0.3 MESH2GS_HAND_SIGMA_MULT=0.3 MESH2GS_BODY_SIGMA_MULT=0.6"
# If the recipe yaml (or a file it extends) defines runtime_env, use it instead of CUSTOM_ENV
# (e.g. configs/default2.yaml, default3.yaml).
apply_recipe_env() {
  local out
  out=$("$PY" - "$1" <<'PY_EOF'
import os, sys, yaml
p = os.path.abspath(sys.argv[1])
while p:
    d = yaml.safe_load(open(p)) or {}
    if 'runtime_env' in d:
        print(d['runtime_env'] or '')
        break
    ext = d.get('extends')
    p = os.path.normpath(os.path.join(os.path.dirname(p), ext)) if ext else None
else:
    print('__NO_RUNTIME_ENV__')
PY_EOF
) || return 1
  [ "$out" = "__NO_RUNTIME_ENV__" ] || CUSTOM_ENV="$out"
}

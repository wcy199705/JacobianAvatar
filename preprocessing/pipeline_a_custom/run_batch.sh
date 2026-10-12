#!/usr/bin/env bash
# Run several CUSTOM scenes in parallel, one GPU per scene (full preprocessing each).
#
# Usage:
#   bash run_batch.sh <gpu_list> <scene1> [scene2 ...]
#   e.g. bash run_batch.sh 2,3 sceneA sceneB
#   STEPS=3,4,5 bash run_batch.sh 2,3 sceneA sceneB      # selected steps only
set -u

GPU_LIST="${1:?usage: $0 <gpu_list> <scene1> [scene2 ...]}"
shift
SCENES=("$@")
[[ ${#SCENES[@]} -ge 1 ]] || { echo "[FATAL] no scene given" >&2; exit 1; }
STEPS="${STEPS:-1a,2,mv,1b,3,4,5,6,7,finalize}"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -lt ${#SCENES[@]} ]]; then
  echo "[FATAL] need at least ${#SCENES[@]} GPUs in --gpus, got ${#GPUS[@]}" >&2
  exit 1
fi

DIR=$(cd "$(dirname "$0")" && pwd)
RUNNER="$DIR/run_scene.sh"
chmod +x "$RUNNER" "$DIR/run_batch.sh" 2>/dev/null || true

LOG_DIR=${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs/preprocess}
mkdir -p "$LOG_DIR"
BATCH_LOG=$LOG_DIR/_batch.log

echo "[$(date +%H:%M:%S)] === BATCH START: ${#SCENES[@]} scenes, gpus=$GPU_LIST, steps=$STEPS ===" | tee -a "$BATCH_LOG"

pids=()
for i in "${!SCENES[@]}"; do
  scene="${SCENES[$i]}"
  gpu="${GPUS[$i]}"
  echo "[$(date +%H:%M:%S)] launch scene=$scene on gpu=$gpu  (log: $LOG_DIR/$scene.log)" | tee -a "$BATCH_LOG"
  nohup "$RUNNER" "$scene" "$gpu" "$STEPS" >/dev/null 2>&1 &
  pids+=($!)
  # small stagger to avoid simultaneous JIT compile / nvdiffrast cache thrash
  sleep 2
done

echo "[$(date +%H:%M:%S)] PIDs: ${pids[*]}" | tee -a "$BATCH_LOG"

# Wait for all and collect rc
fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  scene="${SCENES[$i]}"
  if wait "$pid"; then
    echo "[$(date +%H:%M:%S)] OK    scene=$scene (pid=$pid)" | tee -a "$BATCH_LOG"
  else
    rc=$?
    echo "[$(date +%H:%M:%S)] FAIL  scene=$scene (pid=$pid rc=$rc)" | tee -a "$BATCH_LOG"
    fail=$((fail + 1))
  fi
done

echo "[$(date +%H:%M:%S)] === BATCH DONE: $((${#SCENES[@]} - fail))/${#SCENES[@]} scenes OK ===" | tee -a "$BATCH_LOG"
exit $fail

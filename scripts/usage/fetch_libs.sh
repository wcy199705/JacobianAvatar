#!/bin/bash
# Fetches external library sources from their original GitHub repos at pinned commits into libs/ (unmodified).
# All glue code for this repository lives in our own code (core/, preprocessing/).
# Already-fetched libraries are skipped. Called automatically at the start of fetch_assets.sh.
#
# Usage: bash scripts/usage/fetch_libs.sh
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source scripts/usage/_env.sh
mkdir -p libs
FAILED=()

# <name> <GitHub repo> <commit> [submodules]
clone_at() {
  local name=$1 url=$2 commit=$3 sub=${4:-}
  local dst=libs/$name
  if [ -e "$dst/.fetched_$commit" ]; then echo "  already exists: $dst"; return 0; fi
  echo "=== $name ($url @ ${commit:0:7}) ==="
  mkdir -p "$dst"
  git -C "$dst" init -q 2>/dev/null
  git -C "$dst" remote add origin "$url" 2>/dev/null || git -C "$dst" remote set-url origin "$url"
  git -C "$dst" fetch -q --depth 1 origin "$commit" && git -C "$dst" checkout -q FETCH_HEAD || return 1
  if [ -n "$sub" ]; then git -C "$dst" submodule update -q --init --depth 1 || return 1; fi
  touch "$dst/.fetched_$commit"
}

clone_at MHR       https://github.com/facebookresearch/MHR.git      98d520f6e6ea5e4dbc9327ef89c204300a501ceb || FAILED+=(MHR)
clone_at WAFT      https://github.com/princeton-vl/WAFT.git         b152ff1cad1af8c185ee7b141997c48ff3334c87 submodules || FAILED+=(WAFT)
clone_at samurai   https://github.com/yangchris11/samurai.git       76ba195984892b0d1e3db5d9c9f90bb62175680a || FAILED+=(samurai)
clone_at PromptHMR https://github.com/yufu-wang/PromptHMR.git       2a1c013f54fe9588d1019ad9779961045cf76f65 || FAILED+=(PromptHMR)
clone_at sapiens2  https://github.com/facebookresearch/sapiens2.git 5c7391af3c395ad94213d6bfd391c7f9ae55ac29 || FAILED+=(sapiens2)
clone_at mmpose    https://github.com/open-mmlab/mmpose.git         71ec36ebd63c475ab589afc817868e749a61491f || FAILED+=(mmpose)

# SAM 2's optional CUDA extension (_C, mask hole-filling post-processing) is not built. This matches the paper experiments;
# without the extension, SAM 2 warns and skips only that post-processing step.

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed: ${FAILED[*]}"; exit 1
fi
echo "Library sources ready."

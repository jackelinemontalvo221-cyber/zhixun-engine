#!/usr/bin/env bash
# Build the contest image with the current commit recorded in it.
#
# Usage: bash scripts/build_image.sh [image]        (default zhixun-harness:0.1.0)
# Restricted network: PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/build_image.sh
# Trial image (forces one rollback round, see docs/平台试运行清单.md):
#   HARNESS_CONFIG=/app/configs/harness.trial.json bash scripts/build_image.sh zhixun-harness:trial
# Then:  bash scripts/verify_image.sh [image]
set -euo pipefail

IMAGE=${1:-zhixun-harness:0.1.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
[ -n "$COMMIT" ] || { echo "not a git checkout: cannot record the commit" >&2; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "warning: uncommitted changes; the image will be labelled $COMMIT-dirty" >&2
  COMMIT="$COMMIT-dirty"
fi

args=(--platform linux/amd64 --build-arg "GIT_COMMIT=$COMMIT" -t "$IMAGE")
[ -n "${PIP_INDEX_URL:-}" ] && args+=(--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL")
[ -n "${HARNESS_CONFIG:-}" ] && args+=(--build-arg "HARNESS_CONFIG=$HARNESS_CONFIG")
docker build "${args[@]}" .
echo "built $IMAGE (commit $COMMIT)"

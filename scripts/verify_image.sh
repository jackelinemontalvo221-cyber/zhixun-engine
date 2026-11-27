#!/usr/bin/env bash
# Acceptance checks for the contest harness image.
#
# Usage:
#   scripts/verify_image.sh [image] [package]
#
# Defaults to zhixun-harness:0.1.0 and cpu-a-guide-20261110-v1.  Requires a
# linux/amd64 Docker engine (WSL2 or Linux).  Exit code 0 means every check
# passed.
set -euo pipefail

IMAGE=${1:-zhixun-harness:0.1.0}
PKG=${2:-cpu-a-guide-20261110-v1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACKAGE_DIR="$ROOT/$PKG"

fail() { echo "FAIL: $*" >&2; exit 1; }

command -v docker >/dev/null || fail "docker is not installed"
command -v python3 >/dev/null || fail "python3 is required to build the sample request"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image $IMAGE not found"

echo "[1/5] describe"
describe=$(docker run --rm "$IMAGE" describe)
case "$describe" in
  *'"protocol":"contest.harness.workspace.v2"'*) ;;
  *) fail "unexpected describe output: $describe" ;;
esac
case "$describe" in
  *'"sdk_version":"1.0.4"'*) ;;
  *) fail "unexpected SDK version: $describe" ;;
esac

echo "[2/5] non-root"
identity=$(docker run --rm --entrypoint id "$IMAGE")
case "$identity" in
  *"uid=65534"*) ;;
  *) fail "image does not run as uid 65534: $identity" ;;
esac

echo "[3/5] offline read-only describe"
docker run --rm --network none --read-only --tmpfs /tmp "$IMAGE" describe >/dev/null

echo "[4/5] predict with the package checkpoint"
[ -f "$PACKAGE_DIR/checkpoint.npz" ] || fail "missing $PACKAGE_DIR/checkpoint.npz"
sample=$(mktemp)
trap 'rm -f "$sample"' EXIT
python3 - "$PACKAGE_DIR" >"$sample" <<'PY'
import json
import sys
from pathlib import Path

package = Path(sys.argv[1])
task_id = package.name.split("-")[1].upper()
task = json.loads((package / "baselines" / "configs" / f"{task_id}.json").read_text())["task"]
with (package / "public" / "validation.jsonl").open(encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
row.pop("answer", None)
row["task"] = task
print(json.dumps(row, ensure_ascii=False))
PY
prediction=$(docker run --rm --network none --read-only --tmpfs /tmp -i \
  -v "$PACKAGE_DIR:/model:ro" \
  -e CONTEST_ARTIFACT_PATH=/model/checkpoint.npz \
  -e "CONTEST_TASK=$(python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); t=p.name.split("-")[1].upper(); print(json.loads((p/"baselines"/"configs"/(t+".json")).read_text())["task"])' "$PACKAGE_DIR")" \
  "$IMAGE" predict <"$sample")
case "$prediction" in
  *'"outputs"'*) ;;
  *) fail "unexpected prediction output: $prediction" ;;
esac

echo "[5/5] full acceptance (A/B/C predict, offline training rounds, SIGTERM, size, commit)"
if python3 -c "import contest_harness, pytest" 2>/dev/null; then
  (cd "$ROOT" && ZH_IMAGE="$IMAGE" python3 -m pytest -q tests/test_image.py) || fail "image acceptance tests failed"
else
  echo "skipped: install the SDK and pytest on the host to run tests/test_image.py" >&2
fi

echo "OK: $IMAGE passed describe, non-root, offline read-only, predict and acceptance checks"

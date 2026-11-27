#!/usr/bin/env bash
# Export the committed code (git HEAD) as a clean snapshot for the public source repository
# that the platform submission form links to ("查看源码").
#
# Usage:
#   bash scripts/export_public_snapshot.sh <output-dir> [public-remote-url]
#
# Only committed files are exported (never .env, local configs or untracked files).
# Excluded on purpose: official contest packages and data (cpu-*), the problem statement
# PDF, the SDK tarball, local run artifacts.  With a remote URL the snapshot is committed
# as one commit "snapshot of <commit>" and pushed to that repository's main branch
# (force-push: the public repository only mirrors snapshots).
set -euo pipefail

OUT=${1:?usage: export_public_snapshot.sh <output-dir> [public-remote-url]}
REMOTE=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

COMMIT=$(git rev-parse HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  echo "commit or stash your changes first: the snapshot is taken from HEAD" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT already exists" >&2; exit 1; }
mkdir -p "$OUT"

git archive --format=tar HEAD | tar -x -C "$OUT"
rm -rf "$OUT"/cpu-*-20261110-v1 "$OUT"/artifacts "$OUT"/contest-harness-sdk-*.tar.gz "$OUT"/*.pdf
find "$OUT" -name "*.Zone.Identifier" -delete

# refuse to publish anything that looks like a credential
if grep -rIlE "(PYTRIO_API_KEY|api[_-]?key|password|密码)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}['\"]?\s*($|,|\))" "$OUT" >/dev/null 2>&1; then
  echo "possible credential found; inspect before publishing:" >&2
  grep -rIlE "(PYTRIO_API_KEY|api[_-]?key|password|密码)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}['\"]?\s*($|,|\))" "$OUT" >&2
  exit 1
fi
echo "$COMMIT" > "$OUT/GIT_COMMIT"
echo "snapshot of $COMMIT in $OUT ($(find "$OUT" -type f | wc -l) files)"

if [ -n "$REMOTE" ]; then
  cd "$OUT"
  git init -q -b main
  git add -A
  git commit -q -m "snapshot of $COMMIT"
  git remote add origin "$REMOTE"
  git push -f origin main
fi

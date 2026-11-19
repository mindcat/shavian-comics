#!/bin/bash
# Assemble the Hugging Face Space into a folder (default: ./space-build), ready for
# `docker build` or `hf upload <user>/<space> <folder> --type space`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO/deploy/space-build}"
rm -rf "$OUT"
mkdir -p "$OUT/src"
cp "$REPO"/deploy/space/{Dockerfile,README.md,requirements.txt} "$OUT/"
cp "$REPO"/src/{server.py,shaw.py,init.py} "$OUT/src/"
cp -r "$REPO"/src/{shavcomic,web} "$OUT/src/"
for d in models dict fonts; do mkdir -p "$OUT/src/$d"; cp "$REPO/src/$d/README.md" "$OUT/src/$d/" 2>/dev/null || true; done
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
echo "Space assembled in $OUT:"
(cd "$OUT" && find . -type f | sort)

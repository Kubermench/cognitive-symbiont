#!/usr/bin/env bash
set -euo pipefail

# Safety: consider stashing before applying changes:
#   git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git stash push -u -k -m pre-symbiont-$(date +%s) || true

pip install ruff && printf '[format]
line-length = 100
' > ruff.toml
pip install pre-commit && printf 'repos: []
' > .pre-commit-config.yaml && pre-commit install

# Sources (recent notes):
#   # Note from https://docs.astral.sh/ruff/ — ./data/artifacts/notes/note_1758689354.md


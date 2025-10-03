#!/usr/bin/env bash
set -euo pipefail

# Safety: consider stashing before applying changes:
#   git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git stash push -u -k -m pre-symbiont-$(date +%s) || true

printf 'MIT
' > LICENSE
printf '*
end_of_line = lf
insert_final_newline = true
indent_style = space
indent_size = 2
' > .editorconfig
mkdir -p tests && printf 'def test_placeholder(

# Sources (recent notes):
#   # Note from https://docs.astral.sh/ruff/ — ./data/artifacts/notes/note_1758689354.md


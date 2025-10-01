#!/usr/bin/env python3
"""Minimal offline LLM fallback that echoes guidance."""

from __future__ import annotations

import sys

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
prompt = prompt.strip()

if not prompt:
    print("(offline) No prompt provided.")
    sys.exit(0)

summary = prompt[:300].replace("\n", " ")
print(f"(offline-fallback) Unable to reach primary LLM. Prompt excerpt: {summary}")

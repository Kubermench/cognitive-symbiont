from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Optional


def transcribe_wav(path: str) -> Optional[str]:
    """Best-effort local transcription using available CLIs. Returns None if unavailable.

    Tries (in order):
    - whisper (OpenAI Whisper CLI)
    - whisper.cpp (via 'main' with default model path)
    """
    if not os.path.exists(path):
        return None
    # Try whisper CLI
    try:
        out = subprocess.run(["whisper", path, "--language", "en", "--model", "base", "--output_format", "txt", "--task", "transcribe"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout:
            return out.stdout
    except Exception:
        pass
    # Try whisper.cpp main
    try:
        # Expect a default model env var or fallback path
        model = os.environ.get("WHISPER_CPP_MODEL", "models/ggml-base.en.bin")
        out = subprocess.run(["main", "-m", model, "-f", path, "-otxt"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout:
            return out.stdout
    except Exception:
        pass
    # Try a lightweight Vosk-based CLI if available (community tool)
    if platform.system() != "Darwin":
        cli = shutil.which("vosk-transcriber")
        if cli:
            try:
                out = subprocess.run([cli, path], capture_output=True, text=True)
                if out.returncode == 0 and out.stdout:
                    return out.stdout
            except Exception:
                pass
    return None

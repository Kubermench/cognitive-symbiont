from __future__ import annotations
import subprocess


def speak(text: str) -> bool:
    """Best-effort TTS using platform tools when available (macOS 'say')."""
    try:
        out = subprocess.run(["say", text], capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False


from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from functools import lru_cache
from typing import Iterable, Optional

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _init_engine() -> Optional["pyttsx3.Engine"]:
    if pyttsx3 is None:
        return None
    try:
        system = platform.system()
        driver = "espeak" if system == "Linux" and shutil.which("espeak") else None
        return pyttsx3.init(driverName=driver)
    except Exception as exc:  # pragma: no cover
        logger.debug("pyttsx3 init failed: %s", exc)
        return None


def available_voices() -> Iterable[str]:
    engine = _init_engine()
    if not engine:
        return []
    try:
        return [v.id for v in engine.getProperty("voices")]
    except Exception:
        return []


def speak(text: str, *, voice: str | None = None, rate: int | None = None) -> bool:
    """Best-effort cross-platform TTS."""

    system = platform.system()

    if system == "Darwin" and shutil.which("say"):
        cmd = ["say"]
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            logger.debug("macOS say failed: %s", exc)

    engine = _init_engine()
    if not engine:
        logger.debug("pyttsx3 unavailable; skipping TTS")
        return False

    try:
        if voice:
            engine.setProperty("voice", voice)
        if rate is not None:
            engine.setProperty("rate", rate)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as exc:  # pragma: no cover
        logger.debug("pyttsx3 speak failed: %s", exc)
        return False

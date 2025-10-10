from __future__ import annotations

import importlib.metadata


def _load_version() -> str:
    try:
        return importlib.metadata.version("cognitive-symbiont")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = _load_version()

__all__ = ["__version__"]

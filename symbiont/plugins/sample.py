"""Reference plugin used by the Swarm beta manifest example.

The goal is to keep demo plugins lightweight so that the default install
remains lean. Users can copy this structure when building their own add-ons
without needing additional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SamplePlugin:
    """Minimal plugin that stores a greeting and optional metadata."""

    greeting: str
    meta: Dict[str, Any]

    def register(self, orchestrator) -> None:
        """Optionally called during orchestrator bootstrap."""
        bucket = orchestrator.config.setdefault("sample_plugins", [])
        bucket.append({"greeting": self.greeting, "meta": self.meta})


def build_plugin(greeting: str = "hello beta", **kwargs: Any) -> SamplePlugin:
    """Factory entry point referenced by the example manifest."""

    meta = dict(kwargs.pop("meta", {}))
    meta.update(kwargs)
    return SamplePlugin(greeting=greeting, meta=meta)

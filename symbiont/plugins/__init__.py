from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)


class PluginLoadError(RuntimeError):
    """Raised when a plugin cannot be loaded or instantiated."""


@dataclass
class PluginEntry:
    """Manifest definition for a single plugin."""

    name: str
    module: str
    attribute: Optional[str] = None
    description: str = ""
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> Any:
        """Return the module attribute referenced by this entry."""
        try:
            module_obj = importlib.import_module(self.module)
        except Exception as exc:  # pragma: no cover - surfaced via tests
            raise PluginLoadError(f"Failed to import module '{self.module}': {exc}") from exc
        if not self.attribute:
            return module_obj
        try:
            return getattr(module_obj, self.attribute)
        except AttributeError as exc:
            raise PluginLoadError(
                f"Module '{self.module}' has no attribute '{self.attribute}'"
            ) from exc

    def instantiate(self, **overrides: Any) -> Any:
        """Instantiate the target if callable, returning the raw attribute otherwise."""
        target = self.load()
        if not callable(target):
            return target
        params = {**self.config, **overrides}
        try:
            return target(**params)
        except TypeError as exc:
            raise PluginLoadError(
                f"Callable plugin '{self.name}' rejected parameters {params}: {exc}"
            ) from exc


class PluginRegistry:
    """Load and access plugins declared in a YAML manifest."""

    def __init__(self, manifest_path: Optional[str] = None):
        default_path = os.getenv("SYMBIONT_PLUGINS_FILE", "configs/plugins.yml")
        self.manifest_path = manifest_path or default_path
        self._entries = self._load_manifest(self.manifest_path)

    @property
    def entries(self) -> Dict[str, PluginEntry]:
        """Return all plugin entries, keyed by name."""
        return dict(self._entries)

    def enabled(self) -> Iterable[PluginEntry]:
        """Iterate over enabled plugin entries."""
        return (entry for entry in self._entries.values() if entry.enabled)

    def get(self, name: str) -> Optional[PluginEntry]:
        """Fetch an entry by name."""
        return self._entries.get(name)

    def instantiate_enabled(self) -> Dict[str, Any]:
        """Instantiate all enabled plugins and return a name->instance map."""
        instances: Dict[str, Any] = {}
        for entry in self.enabled():
            try:
                instances[entry.name] = entry.instantiate()
            except PluginLoadError as exc:
                logger.warning("Plugin '%s' failed to instantiate: %s", entry.name, exc)
        return instances

    def reload(self) -> None:
        """Reload the manifest from disk."""
        self._entries = self._load_manifest(self.manifest_path)

    # ------------------------------------------------------------------
    def _load_manifest(self, path: str) -> Dict[str, PluginEntry]:
        manifest_path = Path(path)
        if not manifest_path.exists():
            logger.debug("Plugin manifest %s not found; returning empty registry", path)
            return {}
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                raw_manifest = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning("Failed to read plugin manifest %s: %s", path, exc)
            return {}

        plugins = raw_manifest.get("plugins") if isinstance(raw_manifest, dict) else None
        if not plugins:
            return {}

        entries: Dict[str, PluginEntry] = {}
        for idx, data in enumerate(plugins):
            if not isinstance(data, dict):
                logger.debug("Skipping malformed plugin entry at index %s", idx)
                continue
            name = str(data.get("name") or "").strip()
            module = str(data.get("module") or "").strip()
            if not name or not module:
                logger.debug("Skipping plugin entry %s due to missing name or module", idx)
                continue
            entry = PluginEntry(
                name=name,
                module=module,
                attribute=(str(data.get("attribute")).strip() or None)
                if data.get("attribute") is not None
                else None,
                description=str(data.get("description") or ""),
                enabled=bool(data.get("enabled", True)),
                tags=list(data.get("tags") or []),
                config=dict(data.get("config") or {}),
            )
            entries[entry.name] = entry
        return entries


__all__ = ["PluginRegistry", "PluginEntry", "PluginLoadError"]

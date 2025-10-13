"""Template system for Symbiont configurations.

This module provides reusable YAML templates for common configurations
and patterns.
"""

from __future__ import annotations

from .template_manager import TemplateManager
from .builtin_templates import BUILTIN_TEMPLATES

__all__ = ["TemplateManager", "BUILTIN_TEMPLATES"]
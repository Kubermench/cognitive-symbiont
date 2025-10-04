"""Helpers for loading secrets from local-only stores.

This mirrors common agent stacks by sourcing credentials from
environment variables and project-local dotenv files, with optional
support for platform-specific stores when explicitly requested.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class SecretLoadError(RuntimeError):
    """Raised when a secret cannot be retrieved."""


def load_secret(spec: Optional[Any], *, fallback_env: Optional[str] = None) -> str:
    """Return the secret described by *spec*.

    *spec* can be a single mapping or a list/tuple of mappings. The first
    successful source wins, letting callers ship configs that work across
    environments (e.g., keychain locally, environment variables in prod).

    Supported methods for each mapping:
      - env      : read from environment variable (default/fallback)
      - file     : read the entire contents of a local file
      - env_file : parse a dotenv-style file and return the requested key
      - keychain : fetch from macOS Keychain via `security`
    """

    if isinstance(spec, (list, tuple)):
        errors: list[str] = []
        for entry in spec:
            try:
                return load_secret(entry, fallback_env=fallback_env)
            except SecretLoadError as exc:
                errors.append(str(exc))
                continue
        if errors:
            raise SecretLoadError("; ".join(errors))
        raise SecretLoadError("No usable secret sources provided")

    if spec is None:
        spec_dict: Dict[str, Any] = {}
    elif isinstance(spec, dict):
        spec_dict = spec
    else:
        raise SecretLoadError("Secret spec must be a mapping when not None")

    method = spec_dict.get("method", "env").lower()

    if method == "env":
        env_name = spec_dict.get("env") or fallback_env
        if not env_name:
            raise SecretLoadError("Environment variable name not provided for secret")
        value = os.getenv(env_name)
        if not value:
            raise SecretLoadError(f"Environment variable {env_name} is not set")
        return value

    if method == "file":
        path_value = spec_dict.get("path")
        if not path_value:
            raise SecretLoadError("File path not provided for secret")
        path = Path(path_value).expanduser().resolve()
        if not path.exists():
            raise SecretLoadError(f"Secret file not found: {path}")
        try:
            data = path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pragma: no cover - file errors are environment specific
            raise SecretLoadError(f"Failed reading secret file {path}: {exc}") from exc
        if not data:
            raise SecretLoadError(f"Secret file {path} is empty")
        return data

    if method == "env_file":
        path_value = spec_dict.get("path")
        key = spec_dict.get("key") or fallback_env
        if not path_value:
            raise SecretLoadError("env_file method requires 'path'")
        if not key:
            raise SecretLoadError("env_file method requires 'key'")
        path = Path(path_value).expanduser().resolve()
        if not path.exists():
            raise SecretLoadError(f"Env file not found: {path}")
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line or line.lstrip().startswith("#"):
                    continue
                if "=" not in line:
                    continue
                lhs, rhs = line.split("=", 1)
                if lhs.strip() == key:
                    value = rhs.strip().strip('"').strip("'")
                    if value:
                        return value
                    raise SecretLoadError(
                        f"Env file {path} contains empty value for {key}"
                    )
        except Exception as exc:  # pragma: no cover - parsing errors depend on host file
            raise SecretLoadError(f"Failed reading env file {path}: {exc}") from exc
        raise SecretLoadError(f"Key {key} not found in env file {path}")

    if method == "keychain":
        service = spec_dict.get("service")
        if not service:
            raise SecretLoadError("Keychain service name is required")
        account = spec_dict.get("account")
        cmd = ["security", "find-generic-password", "-s", service]
        if account:
            cmd.extend(["-a", account])
        cmd.append("-w")
        try:
            out = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on host config
            stderr = (exc.stderr or "").strip() or "unknown error"
            raise SecretLoadError(
                f"Keychain lookup failed for service '{service}': {stderr}"
            ) from exc
        except FileNotFoundError as exc:  # pragma: no cover - security missing
            raise SecretLoadError("macOS `security` command not available") from exc
        secret = (out.stdout or "").strip()
        if not secret:
            raise SecretLoadError(
                f"Keychain returned empty secret for service '{service}'"
            )
        return secret

    raise SecretLoadError(f"Unsupported secret method: {method}")

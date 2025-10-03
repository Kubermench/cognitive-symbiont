from __future__ import annotations

import logging
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, cfg: Dict):
        cfg = cfg or {}
        self.provider = cfg.get("provider", "none")
        self.model = cfg.get("model", "phi3:mini")
        self.cmd = cfg.get("cmd", "")
        self.timeout = int(cfg.get("timeout_seconds", 25))
        self.fallback_cfg: Dict = cfg.get("fallback", {}) or {}

    def generate(self, prompt: str) -> str:
        primary = self._dispatch(self.provider, self.model, self.cmd, prompt, timeout=self.timeout)
        if primary.strip():
            return primary

        fallback_provider = self.fallback_cfg.get("provider")
        if not fallback_provider:
            return ""

        logger.warning(
            "Primary LLM provider '%s' failed; falling back to '%s'",
            self.provider,
            fallback_provider,
        )
        fallback_model = self.fallback_cfg.get("model", self.model)
        fallback_cmd = self.fallback_cfg.get("cmd", self.cmd)
        fallback_timeout = int(self.fallback_cfg.get("timeout_seconds", self.timeout))

        return self._dispatch(fallback_provider, fallback_model, fallback_cmd, prompt, timeout=fallback_timeout)

    def _dispatch(self, provider: str, model: str, cmd: str, prompt: str, *, timeout: Optional[int]) -> str:
        timeout_val = timeout if timeout is not None else self.timeout
        if provider == "ollama":
            return self._generate_ollama(model, prompt, timeout_val)
        if provider == "cmd":
            return self._generate_cmd(cmd, prompt, timeout_val)
        return ""

    def _generate_ollama(self, model: str, prompt: str, timeout: int) -> str:
        try:
            out = subprocess.run(
                ["ollama", "generate", "-m", model, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout
        except Exception as exc:
            logger.debug("ollama generate failed: %s", exc)
        try:
            proc = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, _stderr = proc.communicate(prompt, timeout=timeout)
            if proc.returncode == 0 and stdout.strip():
                return stdout
        except Exception as exc:
            logger.debug("ollama run failed: %s", exc)
        return ""

    def _generate_cmd(self, cmd: str, prompt: str, timeout: int) -> str:
        if not cmd:
            return ""
        try:
            rendered = cmd.replace("{prompt}", prompt.replace('"', '\\"'))
            out = subprocess.run(
                rendered,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if out.stdout:
                return out.stdout
        except Exception as exc:
            logger.debug("cmd provider failed: %s", exc)
        return ""

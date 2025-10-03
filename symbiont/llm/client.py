from __future__ import annotations

import logging
import os
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
        self.mode = cfg.get("mode", "local").lower()  # local | cloud | hybrid
        self.hybrid_threshold = int(cfg.get("hybrid_threshold_tokens", 800))
        self.cloud_cfg = cfg.get("cloud", {}) or {}
        self._cloud_client = None
        if self.mode in {"cloud", "hybrid"} and self.cloud_cfg:
            self._init_cloud_client()

    def generate(self, prompt: str) -> str:
        # Hybrid mode chooses local vs cloud dynamically
        if self.mode == "cloud":
            primary = self._generate_cloud(prompt)
        elif self.mode == "hybrid":
            if self._should_use_cloud(prompt):
                primary = self._generate_cloud(prompt)
                if not primary.strip():
                    primary = self._dispatch(
                        self.provider, self.model, self.cmd, prompt, timeout=self.timeout
                    )
            else:
                primary = self._dispatch(
                    self.provider, self.model, self.cmd, prompt, timeout=self.timeout
                )
                if not primary.strip():
                    primary = self._generate_cloud(prompt)
        else:  # local only
            primary = self._dispatch(
                self.provider, self.model, self.cmd, prompt, timeout=self.timeout
            )

        if primary and primary.strip():
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

    def _dispatch(
        self, provider: str, model: str, cmd: str, prompt: str, *, timeout: Optional[int]
    ) -> str:
        timeout_val = timeout if timeout is not None else self.timeout
        if provider == "ollama":
            return self._generate_ollama(model, prompt, timeout_val)
        if provider == "cmd":
            return self._generate_cmd(cmd, prompt, timeout_val)
        return ""

    def _generate_ollama(self, model: str, prompt: str, timeout: int) -> str:
        env = os.environ.copy()
        env.setdefault("OLLAMA_NO_SPINNER", "1")
        # First try piping the prompt via stdin (supported on recent ollama versions)
        try:
            out = subprocess.run(
                ["ollama", "run", model],
                input=(prompt if prompt.endswith("\n") else prompt + "\n"),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout
            logger.debug("ollama run (stdin) returned code %s, stderr=%s", out.returncode, out.stderr)
        except Exception as exc:
            logger.debug("ollama run (stdin) failed: %s", exc)

        # Fallback to the older generate invocation for compatibility
        try:
            out = subprocess.run(
                ["ollama", "generate", "-m", model, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout
            logger.debug("ollama generate returned code %s, stderr=%s", out.returncode, out.stderr)
        except Exception as exc:
            logger.debug("ollama generate failed: %s", exc)
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

    # ------------------------------------------------------------------
    def _init_cloud_client(self) -> None:
        from .cloud import CloudLLMClient, CloudLLMConfig, CloudLLMError

        try:
            config = CloudLLMConfig(
                provider=self.cloud_cfg.get("provider", "openai"),
                model=self.cloud_cfg.get("model", "gpt-4o-mini"),
                api_key_env=self.cloud_cfg.get("api_key_env", "OPENAI_API_KEY"),
                api_key_source=self.cloud_cfg.get("api_key_source"),
                timeout_seconds=int(self.cloud_cfg.get("timeout_seconds", 30)),
            )
            self._cloud_client = CloudLLMClient(config)
        except CloudLLMError as exc:
            logger.warning("Cloud LLM unavailable: %s", exc)
            self._cloud_client = None

    def _should_use_cloud(self, prompt: str) -> bool:
        approx_tokens = max(1, len(prompt) // 4)
        if approx_tokens >= self.hybrid_threshold:
            return True
        if not self._cloud_client:
            return False
        # Optional keyword-based routing
        patterns = self.cloud_cfg.get("force_patterns") or []
        return any(pat.lower() in prompt.lower() for pat in patterns)

    def _generate_cloud(self, prompt: str) -> str:
        if not self._cloud_client:
            return ""
        try:
            return self._cloud_client.generate(prompt)
        except Exception as exc:  # pragma: no cover - surfaced via logs
            logger.warning("Cloud LLM error: %s", exc)
            return ""

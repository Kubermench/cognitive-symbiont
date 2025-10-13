from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Dict, Optional
import shlex

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..tools.retry_utils import (
    RetryConfig,
    LLM_RETRY_CONFIG,
    retry_call,
    get_circuit_breaker,
)

from .budget import TokenBudget

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
        self._cloud_last_refresh = 0.0
        self._cloud_refresh_seconds = int(self.cloud_cfg.get("refresh_seconds", 3600))
        if self.mode in {"cloud", "hybrid"} and self.cloud_cfg:
            self._init_cloud_client()

        retry_cfg = cfg.get("retry", {}) or {}
        self._retry_config = RetryConfig(
            attempts=max(1, int(retry_cfg.get("attempts", 3) or 3)),
            initial_wait=max(0.1, float(retry_cfg.get("initial_seconds", 0.5) or 0.5)),
            multiplier=max(1.0, float(retry_cfg.get("multiplier", 2.0) or 2.0)),
            max_wait=max(
                0.5,  # minimum max_wait
                float(retry_cfg.get("max_seconds", 20.0) or 20.0),
            ),
            jitter=bool(retry_cfg.get("jitter", True)),
            failure_threshold=max(1, int(retry_cfg.get("failure_threshold", 5) or 5)),
            recovery_timeout=max(30.0, float(retry_cfg.get("recovery_timeout", 60.0) or 60.0)),
        )
        self._retry_enabled = bool(retry_cfg.get("enabled", True))
        
        # Initialize circuit breakers for different providers
        self._circuit_breaker_name = f"llm_{self.provider}_{self.model}"

    def generate(
        self,
        prompt: str,
        *,
        budget: Optional[TokenBudget] = None,
        label: Optional[str] = None,
    ) -> str:
        label = label or "llm"
        prompt_tokens = TokenBudget.estimate(prompt) if budget else 0

        def attempt(
            source: str,
            provider: str,
            model: str,
            func,
        ) -> str:
            if budget and not budget.can_consume(prompt_tokens):
                budget.note_denied(
                    prompt_tokens=prompt_tokens,
                    provider=provider,
                    model=model,
                    label=label,
                    source=source,
                )
                return ""

            start = time.monotonic()
            try:
                if self._retry_enabled:
                    output = retry_call(
                        func,
                        config=self._retry_config,
                        circuit_breaker=self._circuit_breaker_name,
                    )
                else:
                    output = func()
            except Exception as exc:
                latency = time.monotonic() - start
                if budget:
                    budget.log_attempt(
                        prompt_tokens=prompt_tokens,
                        response_tokens=0,
                        provider=provider,
                        model=model,
                        label=label,
                        source=source,
                        outcome="error",
                        latency=latency,
                        error=str(exc),
                    )
                logger.warning(
                    "LLM call failed for %s/%s via %s: %s",
                    provider,
                    model,
                    source,
                    exc,
                )
                return ""

            if output is None:
                output = ""
            latency = time.monotonic() - start
            response_tokens = TokenBudget.estimate(output) if budget else 0
            if budget:
                budget.log_attempt(
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    provider=provider,
                    model=model,
                    label=label,
                    source=source,
                    outcome="ok",
                    latency=latency,
                )
            return output

        # Hybrid mode chooses local vs cloud dynamically
        if self.mode == "cloud":
            self._refresh_cloud_client_if_needed()
            cloud_provider = self.cloud_cfg.get("provider", "cloud")
            cloud_model = self.cloud_cfg.get("model", self.cloud_cfg.get("model_name", self.model))
            primary = attempt(
                "cloud",
                cloud_provider,
                cloud_model,
                lambda: self._generate_cloud(prompt),
            )
        elif self.mode == "hybrid":
            self._refresh_cloud_client_if_needed()
            if self._should_use_cloud(prompt):
                cloud_provider = self.cloud_cfg.get("provider", "cloud")
                cloud_model = self.cloud_cfg.get("model", self.cloud_cfg.get("model_name", self.model))
                primary = attempt(
                    "cloud",
                    cloud_provider,
                    cloud_model,
                    lambda: self._generate_cloud(prompt),
                )
                if not primary.strip():
                    primary = attempt(
                        "local",
                        self.provider,
                        self.model,
                        lambda: self._dispatch(
                            self.provider,
                            self.model,
                            self.cmd,
                            prompt,
                            timeout=self.timeout,
                        ),
                    )
            else:
                primary = attempt(
                    "local",
                    self.provider,
                    self.model,
                    lambda: self._dispatch(
                        self.provider,
                        self.model,
                        self.cmd,
                        prompt,
                        timeout=self.timeout,
                    ),
                )
                if not primary.strip():
                    cloud_provider = self.cloud_cfg.get("provider", "cloud")
                    cloud_model = self.cloud_cfg.get("model", self.cloud_cfg.get("model_name", self.model))
                    primary = attempt(
                        "cloud",
                        cloud_provider,
                        cloud_model,
                        lambda: self._generate_cloud(prompt),
                    )
        else:  # local only
            primary = attempt(
                "local",
                self.provider,
                self.model,
                lambda: self._dispatch(
                    self.provider,
                    self.model,
                    self.cmd,
                    prompt,
                    timeout=self.timeout,
                ),
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

        return attempt(
            "fallback",
            fallback_provider,
            fallback_model,
            lambda: self._dispatch(
                fallback_provider,
                fallback_model,
                fallback_cmd,
                prompt,
                timeout=fallback_timeout,
            ),
        )

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
            parts = shlex.split(cmd)
        except ValueError as exc:
            logger.debug("cmd provider parse failed: %s", exc)
            return ""
        substituted: list[str] = []
        for part in parts:
            substituted.append(part.replace("{prompt}", prompt))
        try:
            out = subprocess.run(
                substituted,
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
            self._cloud_last_refresh = time.monotonic()
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

    def _refresh_cloud_client_if_needed(self) -> None:
        if not self._cloud_client:
            if self.cloud_cfg:
                self._init_cloud_client()
            return
        if self._cloud_refresh_seconds <= 0:
            return
        now = time.monotonic()
        if now - self._cloud_last_refresh >= self._cloud_refresh_seconds:
            logger.info("Refreshing cloud LLM credentials after %s seconds", self._cloud_refresh_seconds)
            self._init_cloud_client()

    # ------------------------------------------------------------------
    def get_circuit_breaker_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for this LLM client."""
        cb = get_circuit_breaker(self._circuit_breaker_name, self._retry_config)
        return cb.get_metrics()

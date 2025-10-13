from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..tools.secrets import SecretLoadError, load_secret


class CloudLLMError(RuntimeError):
    """Raised when a cloud LLM invocation fails."""


@dataclass
class CloudLLMConfig:
    provider: str
    model: str
    api_key_env: Optional[str] = None
    api_key_source: Optional[Any] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_multiplier: float = 2.0


class CloudLLMClient:
    """Thin wrapper around supported cloud providers.

    Currently supports OpenAI-compatible chat completion endpoints.
    """

    def __init__(self, cfg: CloudLLMConfig):
        self.cfg = cfg
        provider = (cfg.provider or "").lower()
        if provider in {"openai", "azure-openai", "azure_openai"}:
            self._kind = "openai"
            self._init_openai()
        else:
            raise CloudLLMError(f"Unsupported cloud provider: {cfg.provider}")
        
        # Retry configuration
        self.retry_attempts = max(1, int(cfg.retry_attempts))
        self.retry_initial_delay = max(0.1, float(cfg.retry_initial_delay))
        self.retry_max_delay = max(self.retry_initial_delay, float(cfg.retry_max_delay))
        self.retry_multiplier = max(1.0, float(cfg.retry_multiplier))

    # ------------------------------------------------------------------
    def _init_openai(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CloudLLMError(
                "openai package not installed. Install with `pip install openai` or `pip install -r requirements-dev.txt`."
            ) from exc

        api_key_env = self.cfg.api_key_env or "OPENAI_API_KEY"
        api_key = None
        if self.cfg.api_key_source:
            try:
                api_key = load_secret(self.cfg.api_key_source, fallback_env=api_key_env)
            except SecretLoadError as exc:
                raise CloudLLMError(str(exc)) from exc
        if not api_key:
            api_key = os.getenv(api_key_env)
        if not api_key:
            raise CloudLLMError(
                f"OpenAI API key not available; set {api_key_env} or configure api_key_source"
            )
        self._openai_client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        if self._kind == "openai":
            return self._generate_with_retry(self._generate_openai, prompt)
        raise CloudLLMError("Unsupported cloud provider")

    # ------------------------------------------------------------------
    def _generate_openai(self, prompt: str) -> str:
        try:
            response = self._openai_client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=None,
                timeout=self.cfg.timeout_seconds,
            )
        except Exception as exc:
            raise CloudLLMError(f"OpenAI request failed: {exc}") from exc

        try:
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise CloudLLMError(f"Malformed OpenAI response: {exc}") from exc

    def _generate_with_retry(self, generate_func, prompt: str) -> str:
        """Generate with exponential backoff retry."""
        if self.retry_attempts <= 1:
            return generate_func(prompt)

        def _log_retry_warning(retry_state: RetryCallState) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Cloud LLM retry %s/%s failed: %s",
                    retry_state.attempt_number,
                    self.retry_attempts,
                    exc,
                )

        retryer = Retrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(
                multiplier=self.retry_initial_delay,
                exp_base=self.retry_multiplier,
                min=self.retry_initial_delay,
                max=self.retry_max_delay,
            ),
            reraise=True,
            before_sleep=_log_retry_warning,
        )
        
        return retryer(generate_func, prompt)

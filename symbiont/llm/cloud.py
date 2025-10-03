from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class CloudLLMError(RuntimeError):
    """Raised when a cloud LLM invocation fails."""


@dataclass
class CloudLLMConfig:
    provider: str
    model: str
    api_key_env: Optional[str] = None
    timeout_seconds: int = 30


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

    # ------------------------------------------------------------------
    def _init_openai(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CloudLLMError(
                "openai package not installed. Install with `pip install openai`."
            ) from exc

        api_key_env = self.cfg.api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise CloudLLMError(
                f"Environment variable {api_key_env} not set for OpenAI access."
            )
        self._openai_client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        if self._kind == "openai":
            return self._generate_openai(prompt)
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


"""Minimal chat clients over HTTP: Ollama, any OpenAI-compatible server, and Anthropic."""

from __future__ import annotations

import os
import re
from typing import Protocol

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class _Base:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.name = f"{cfg.provider}:{cfg.model}"

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.cfg.timeout)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _key(self, default_env: str) -> str:
        env = self.cfg.api_key_env or default_env
        key = os.environ.get(env)
        if not key:
            raise LLMError(f"{self.name}: set the {env} environment variable")
        return key


class Ollama(_Base):
    def complete(self, system: str, prompt: str) -> str:
        data = self._post(
            f"{self.cfg.host.rstrip('/')}/api/chat",
            {
                "model": self.cfg.model,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.cfg.context},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        return (data.get("message") or {}).get("content", "")


class OpenAICompatible(_Base):
    def complete(self, system: str, prompt: str) -> str:
        data = self._post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            {
                "model": self.cfg.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            headers={"Authorization": f"Bearer {self._key('OPENAI_API_KEY')}"},
        )
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content", "")


class Anthropic(_Base):
    def complete(self, system: str, prompt: str) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.cfg.model,
                "max_tokens": 8192,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "x-api-key": self._key("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            },
        )
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )


_PROVIDERS = {"ollama": Ollama, "openai": OpenAICompatible, "anthropic": Anthropic}


def from_config(cfg: LLMConfig) -> LLM | None:
    """Build the configured client, or None when the provider is 'none'."""
    if cfg.provider == "none":
        return None
    try:
        return _PROVIDERS[cfg.provider](cfg)
    except KeyError:
        raise LLMError(
            f"unknown llm provider {cfg.provider!r}; use one of: none, {', '.join(_PROVIDERS)}"
        ) from None


_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str | None:
    """Return the largest fenced code block in a reply, which is where we ask for the file."""
    blocks = _FENCE.findall(reply)
    if not blocks:
        return None
    return max(blocks, key=len)

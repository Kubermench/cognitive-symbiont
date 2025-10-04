import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.llm.client import LLMClient
from symbiont.llm.cloud import CloudLLMClient, CloudLLMConfig, CloudLLMError
from symbiont.tools.secrets import SecretLoadError, load_secret


def test_load_secret_prefers_env_over_env_file(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    value = load_secret([
        {"method": "env", "env": "OPENAI_API_KEY"},
        {"method": "env_file", "path": str(dotenv), "key": "OPENAI_API_KEY"},
    ])
    assert value == "from-env"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    value = load_secret([
        {"method": "env", "env": "OPENAI_API_KEY"},
        {"method": "env_file", "path": str(dotenv), "key": "OPENAI_API_KEY"},
    ])
    assert value == "from-dotenv"


def test_load_secret_raises_when_sources_fail(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OTHER_KEY=value\n", encoding="utf-8")

    with pytest.raises(SecretLoadError):
        load_secret([
            {"method": "env", "env": "OPENAI_API_KEY"},
            {"method": "env_file", "path": str(dotenv), "key": "OPENAI_API_KEY"},
        ])


def test_llmclient_hybrid_escalates_to_cloud(monkeypatch):
    class DummyCloud:
        def __init__(self):
            self.calls = []

        def generate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return "cloud-response"

    dummy_cloud = DummyCloud()

    def fake_init_cloud(self):
        self._cloud_client = dummy_cloud

    monkeypatch.setattr(LLMClient, "_init_cloud_client", fake_init_cloud)
    monkeypatch.setattr(LLMClient, "_dispatch", lambda self, provider, model, cmd, prompt, timeout: "")
    monkeypatch.setattr(LLMClient, "_should_use_cloud", lambda self, prompt: False)

    cfg = {
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "hybrid",
        "cloud": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = LLMClient(cfg)
    result = client.generate("any prompt")

    assert result == "cloud-response"
    assert dummy_cloud.calls == ["any prompt"]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_cloud_client_requires_secret(monkeypatch):
    dummy_module = types.SimpleNamespace(OpenAI=lambda api_key: api_key)
    monkeypatch.setitem(sys.modules, "openai", dummy_module)

    cfg = CloudLLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key_env="MISSING_KEY",
    )

    with pytest.raises(CloudLLMError) as exc:
        CloudLLMClient(cfg)
    assert "MISSING_KEY" in str(exc.value)

    monkeypatch.setenv("MISSING_KEY", "value")
    client = CloudLLMClient(cfg)
    assert client._openai_client == "value"

    monkeypatch.delenv("MISSING_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "openai", raising=False)

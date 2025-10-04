import os
import sys
import time
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


def test_llmclient_refreshes_cloud_credentials(monkeypatch):
    class DummyCloud:
        def __init__(self):
            self.calls = []

        def generate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return "cloud"

    dummy_cloud = DummyCloud()
    init_calls = []
    real_monotonic = time.monotonic

    def fake_init(self):
        init_calls.append("init")
        self._cloud_client = dummy_cloud
        self._cloud_last_refresh = real_monotonic()

    monkeypatch.setenv("OPENAI_API_KEY", "value")
    monkeypatch.setattr(LLMClient, "_init_cloud_client", fake_init)
    monkeypatch.setattr(LLMClient, "_should_use_cloud", lambda self, prompt: True)

    cfg = {
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "hybrid",
        "cloud": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "refresh_seconds": 1,
        },
    }

    client = LLMClient(cfg)
    assert init_calls == ["init"]

    client.generate("first")
    assert dummy_cloud.calls == ["first"]

    client._cloud_last_refresh -= 5
    client.generate("second")

    assert dummy_cloud.calls == ["first", "second"]
    assert init_calls.count("init") == 2

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_llmclient_fallback_provider(monkeypatch):
    calls = {"dispatch": [], "fallback": []}

    def fake_dispatch(self, provider, model, cmd, prompt, timeout):
        calls["dispatch"].append(provider)
        return "" if provider == "ollama" else "fallback-response"

    def fake_cloud(self, prompt):
        calls["fallback"].append(prompt)
        return ""

    monkeypatch.setattr(LLMClient, "_dispatch", fake_dispatch, raising=True)
    monkeypatch.setattr(LLMClient, "_generate_cloud", fake_cloud, raising=True)
    monkeypatch.setattr(LLMClient, "_should_use_cloud", lambda self, prompt: False)

    cfg = {
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "hybrid",
        "fallback": {"provider": "cmd", "cmd": "fake"},
    }

    client = LLMClient(cfg)
    result = client.generate("hello")
    assert result == "fallback-response"
    assert calls["dispatch"] == ["ollama", "cmd"]


def test_llmclient_generate_cmd(monkeypatch):
    class Dummy:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: Dummy("ok"),
    )

    client = LLMClient({"provider": "cmd", "cmd": "echo"})
    assert client._dispatch("cmd", "model", "echo", "prompt", timeout=5) == "ok"


def test_llmclient_generate_ollama(monkeypatch):
    outputs = [
        types.SimpleNamespace(returncode=1, stdout="", stderr="fail"),
        types.SimpleNamespace(returncode=0, stdout="success"),
    ]

    def fake_run(cmd, **kwargs):
        return outputs.pop(0)

    monkeypatch.setattr("subprocess.run", fake_run)
    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    text = client._generate_ollama("phi3:mini", "prompt", timeout=5)
    assert text == "success"


def test_llmclient_refresh_cloud(monkeypatch):
    calls = []

    def fake_init(self):
        calls.append("init")
        self._cloud_client = object()
        self._cloud_last_refresh = time.monotonic()

    monkeypatch.setenv("OPENAI_API_KEY", "value")
    monkeypatch.setattr(LLMClient, "_init_cloud_client", fake_init)
    monkeypatch.setattr(LLMClient, "_should_use_cloud", lambda self, prompt: True)

    cfg = {
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "hybrid",
        "cloud": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "refresh_seconds": 1,
        },
    }

    client = LLMClient(cfg)
    assert calls == ["init"]
    client._cloud_last_refresh -= 5
    client._refresh_cloud_client_if_needed()
    assert calls == ["init", "init"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_llmclient_hybrid_cloud_then_local(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "value")
    def fake_cloud(self, prompt):
        return ""
    monkeypatch.setattr(LLMClient, "_generate_cloud", fake_cloud, raising=True)
    monkeypatch.setattr(LLMClient, "_dispatch", lambda self, provider, model, cmd, prompt, timeout: "local")
    monkeypatch.setattr(LLMClient, "_should_use_cloud", lambda self, prompt: True)
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
    client = LLMClient(cfg)
    assert client.generate("prompt") == "local"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_llmclient_generate_ollama_exception(monkeypatch):
    def first_call(*args, **kwargs):
        raise RuntimeError("boom")

    calls = iter([
        first_call,
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=""),
    ])

    def fake_run(*args, **kwargs):
        func = next(calls)
        return func(*args, **kwargs)

    monkeypatch.setattr("subprocess.run", fake_run)
    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    assert client._generate_ollama("phi3:mini", "prompt", timeout=5) == ""


def test_llmclient_generate_cmd_exception(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("cmd boom")

    monkeypatch.setattr("subprocess.run", fail)
    client = LLMClient({"provider": "cmd", "cmd": "echo"})
    assert client._generate_cmd("echo", "prompt", timeout=5) == ""


def test_llmclient_generate_cmd_missing_command():
    client = LLMClient({"provider": "cmd", "cmd": ""})
    assert client._generate_cmd("", "prompt", timeout=5) == ""


def test_llmclient_dispatch_routing(monkeypatch):
    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    monkeypatch.setattr(client, "_generate_ollama", lambda model, prompt, timeout: "ollama")
    assert client._dispatch("ollama", "phi3:mini", "", "prompt", timeout=5) == "ollama"
    monkeypatch.setattr(client, "_generate_cmd", lambda cmd, prompt, timeout: "cmd")
    assert client._dispatch("cmd", "model", "echo", "prompt", timeout=None) == "cmd"


def test_llmclient_generate_ollama_direct_success(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="direct", stderr=""),
    )
    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    assert client._generate_ollama("phi3:mini", "prompt", timeout=1) == "direct"


def test_llmclient_should_use_cloud_patterns(monkeypatch):
    client = LLMClient({"provider": "ollama", "model": "phi3:mini", "mode": "hybrid", "cloud": {}})
    client._cloud_client = object()
    client.cloud_cfg["force_patterns"] = ["urgent"]
    assert client._should_use_cloud("This is urgent task") is True
    assert client._should_use_cloud("short") is False


def test_llmclient_generate_cloud_without_client():
    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    assert client._generate_cloud("prompt") == ""


def test_llmclient_init_cloud_handles_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "value")

    class DummyError(Exception):
        pass

    class RaisingClient:
        def __init__(self, cfg):
            raise CloudLLMError("nope")

    monkeypatch.setattr("symbiont.llm.cloud.CloudLLMClient", RaisingClient)
    client = LLMClient({
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "cloud",
        "cloud": {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    })
    assert client._cloud_client is None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_llmclient_cloud_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "value")

    class DummyCloud:
        def __init__(self, config):
            self.calls = []

        def generate(self, prompt):
            self.calls.append(prompt)
            return "cloud-output"

    monkeypatch.setattr("symbiont.llm.cloud.CloudLLMClient", lambda cfg: DummyCloud(cfg))

    cfg = {
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "cloud",
        "cloud": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    }

    client = LLMClient(cfg)
    result = client.generate("prompt")
    assert result == "cloud-output"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_llmclient_generate_cloud_handles_exception(monkeypatch):
    class DummyCloud:
        def generate(self, prompt):
            raise RuntimeError("boom")

    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    client._cloud_client = DummyCloud()
    output = client._generate_cloud("prompt")
    assert output == ""


def test_llmclient_refresh_cloud_without_config():
    client = LLMClient({"provider": "ollama", "model": "phi3:mini"})
    client._cloud_client = None
    client._refresh_cloud_client_if_needed()


def test_llmclient_refresh_cloud_zero_interval(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "value")
    cfg = {
        "provider": "ollama",
        "model": "phi3:mini",
        "mode": "cloud",
        "cloud": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "refresh_seconds": 0,
        },
    }
    init_calls = []

    def fake_init(self):
        init_calls.append("init")
        self._cloud_client = object()

    monkeypatch.setattr(LLMClient, "_init_cloud_client", fake_init)
    client = LLMClient(cfg)
    client._refresh_cloud_client_if_needed()
    assert init_calls == ["init"]
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

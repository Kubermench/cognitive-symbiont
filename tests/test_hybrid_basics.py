import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.llm.client import LLMClient
from symbiont.orchestration.graph import GraphSpec
from symbiont.ports.ai_peer import AIPeerBridge


def test_graphspec_resolves_relative_crew_config(tmp_path):
    crews_path = tmp_path / "crews.yaml"
    crews_path.write_text("agents: {}\n", encoding="utf-8")

    graph_yaml = tmp_path / "graph.yaml"
    graph_yaml.write_text(
        """
crew_config: crews.yaml

graph:
  start: scout
  nodes:
    scout:
      agent: scout
        
""".strip()
        + "\n",
        encoding="utf-8",
    )

    spec = GraphSpec.from_yaml(graph_yaml)
    assert spec.crew_config == crews_path.resolve()


def test_llmclient_hybrid_prefers_local(monkeypatch):
    # Avoid importing third-party cloud SDKs by stubbing cloud init.
    class DummyCloud:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            return "cloud-response"

    dummy_cloud = DummyCloud()

    def fake_init_cloud(self):
        self._cloud_client = dummy_cloud

    monkeypatch.setattr(LLMClient, "_init_cloud_client", fake_init_cloud)

    dispatch_calls = []

    def fake_dispatch(self, provider, model, cmd, prompt, timeout):
        dispatch_calls.append((provider, model))
        return "local-response"

    monkeypatch.setattr(LLMClient, "_dispatch", fake_dispatch, raising=True)
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

    os.environ["OPENAI_API_KEY"] = "test-key"
    client = LLMClient(cfg)
    result = client.generate("short prompt")

    assert result == "local-response"
    assert dispatch_calls == [("ollama", "phi3:mini")]
    assert dummy_cloud.calls == 0

    os.environ.pop("OPENAI_API_KEY", None)


def test_ai_peer_chat_persists_agent_id(tmp_path):
    cfg = {
        "data_root": str(tmp_path),
        "ports": {"ai_peer": {"stub_mode": True}},
        "llm": {},
    }
    bridge = AIPeerBridge(cfg)
    transcript = bridge.chat("Hello", simulate_only=False, agent_id="peer-123")
    payload = json.loads(Path(transcript.path).read_text())
    assert payload["agent_id"] == "peer-123"
    assert payload["prompt"] == "Hello"


def test_ai_peer_chat_relays_when_not_stub(tmp_path, monkeypatch):
    cfg = {
        "data_root": str(tmp_path),
        "ports": {"ai_peer": {"stub_mode": False}},
        "llm": {},
    }
    bridge = AIPeerBridge(cfg)
    monkeypatch.setattr(bridge, "_relay", lambda prompt: "relayed")
    transcript = bridge.chat("Ping", simulate_only=False, agent_id=None)
    assert transcript.response == "relayed"

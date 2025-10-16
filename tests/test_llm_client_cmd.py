import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.llm.client import LLMClient


def test_cmd_provider_substitutes_prompt_safely():
    py = sys.executable
    cfg = {
        "provider": "cmd",
        "cmd": f"{py} -c \"import sys; print(sys.argv[1])\" \"{{prompt}}\"",
        "mode": "local",
    }
    client = LLMClient(cfg)
    payload = "\"; import os; print('boom') #"
    out = client.generate(payload, label="cmd-safe")
    assert out.strip() == payload


def test_cmd_provider_handles_whitespace_prompts():
    py = sys.executable
    cfg = {
        "provider": "cmd",
        "cmd": f"{py} -c \"import sys; print(len(sys.argv[1]))\" \"{{prompt}}\"",
        "mode": "local",
    }
    client = LLMClient(cfg)
    payload = "with\nmultiple lines\nand spaces"
    out = client.generate(payload, label="cmd-length")
    assert out.strip() == str(len(payload))

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from ..llm.client import LLMClient
from ..tools.files import ensure_dirs


@dataclass
class PeerTranscript:
    prompt: str
    response: str
    simulated: bool
    path: str
    agent_id: Optional[str]


class AIPeerBridge:
    """Mediates guarded conversations with external AI peers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.peer_cfg = (((self.config.get("ports", {}) or {}).get("ai_peer") or {}))
        self.stub_mode = bool(self.peer_cfg.get("stub_mode", True))
        self.transcripts_dir = Path(self.config.get("data_root", "data")) / "artifacts" / "ai_peer"
        ensure_dirs([self.transcripts_dir])
        self.llm = LLMClient(self.config.get("llm", {}))

    def chat(self, prompt: str, simulate_only: bool = False, agent_id: Optional[str] = None) -> PeerTranscript:
        simulated = self.stub_mode or simulate_only
        if simulated:
            reply = self._simulate(prompt)
        else:
            reply = self._relay(prompt)
        path = self._store(prompt, reply, simulated, agent_id)
        return PeerTranscript(
            prompt=prompt,
            response=reply,
            simulated=simulated,
            path=str(path),
            agent_id=agent_id,
        )

    def _simulate(self, prompt: str) -> str:
        guidance = (
            "You are an AI pair-programmer offering concise advice."
            " Respond in under 120 words with numbered steps if actionable."
        )
        return self.llm.generate(f"{guidance}\nPrompt: {prompt}")

    def _relay(self, prompt: str) -> str:
        # Placeholder for future external API integration.
        # For now we reuse simulation to guarantee deterministic behaviour.
        return self._simulate(prompt)

    def _store(self, prompt: str, reply: str, simulated: bool, agent_id: Optional[str]) -> Path:
        ts = int(time.time())
        path = self.transcripts_dir / f"peer_{ts}.json"
        payload = {
            "prompt": prompt,
            "response": reply,
            "simulated": simulated,
            "timestamp": ts,
            "agent_id": agent_id,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

"""Deterministic zero-knowledge style attestations for foresight diffs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

PROOF_DIR = Path("data/artifacts/foresight/proofs")


def prove_diff(diff_text: str) -> Dict[str, Any]:
    payload = diff_text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    proof = {
        "hash": digest,
        "length": len(payload),
        "timestamp": int(time.time()),
    }
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    proof_path = PROOF_DIR / f"diff_{digest[:12]}.json"
    proof_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    proof["path"] = str(proof_path)
    return proof


def verify_diff(proof: Dict[str, Any], diff_text: str) -> bool:
    if not proof:
        return False
    expected = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    path = proof.get("path")
    if path and Path(path).exists():
        try:
            recorded = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            recorded = {}
        if recorded.get("hash") != proof.get("hash"):
            return False
    return proof.get("hash") == expected

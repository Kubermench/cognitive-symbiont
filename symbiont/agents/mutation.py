from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..tools.files import ensure_dirs


@dataclass
class MutationIntent:
    kind: str
    rationale: str
    details: Dict[str, Any]


class MutationEngine:
    """Proposes guarded mutations and validates them in a sandbox."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.repo_root = Path(self.config.get("initiative", {}).get("repo_path", ".")).resolve()
        self.artifacts_dir = Path(self.repo_root, "data", "artifacts", "mutations")
        ensure_dirs([self.artifacts_dir])

    # ------------------------------------------------------------------
    def schedule(self, intent: MutationIntent) -> None:
        if intent.kind != "planner_prompt":
            return
        proposal = self._mutate_planner_prompt(intent)
        if not proposal:
            return
        if self._sandbox_validate(proposal["preview_path"], proposal["modified_text"]):
            self._persist_proposal(intent, proposal)

    # ------------------------------------------------------------------
    def _mutate_planner_prompt(self, intent: MutationIntent) -> Optional[Dict[str, Any]]:
        target = self.repo_root / "symbiont" / "agents" / "subself.py"
        if not target.exists():
            return None
        original = target.read_text(encoding="utf-8")

        if "Always diversify" in original and intent.details.get("strategy") == "promote_diversity":
            # Already patched
            return None

        replacement = original
        if intent.details.get("strategy") == "promote_diversity":
            needle = "Only bullets. No extra text."
            addition = (
                "Always diversify suggestions across cycles; reference recent history to avoid repeats."
            )
        else:
            needle = "constraint: ≤60s, one-step action"
            addition = "signal: run repo scan if heuristics are empty"

        if needle not in original:
            return None

        replacement = original.replace(
            needle,
            f"{needle}\n        # evolution tweak\n        # {addition}",
            1,
        )

        # Ensure modification stays within 5% of file size
        if abs(len(replacement) - len(original)) > max(1, int(len(original) * 0.05)):
            return None

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            replacement.splitlines(keepends=True),
            fromfile="a/symbiont/agents/subself.py",
            tofile="b/symbiont/agents/subself.py",
        )
        diff_text = "".join(diff)
        if not diff_text.strip():
            return None

        timestamp = int(time.time())
        preview_path = self.artifacts_dir / f"mutation_{timestamp}.diff"
        preview_path.write_text(diff_text, encoding="utf-8")
        return {
            "preview_path": preview_path,
            "diff_text": diff_text,
            "modified_text": replacement,
            "target": target,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    def _sandbox_validate(self, preview_path: Path, modified_text: str) -> bool:
        """Clone repo into temp dir and ensure mutation compiles three times."""

        try:
            with tempfile.TemporaryDirectory(prefix="symbiont_mutate_") as tmpdir:
                tmp_root = Path(tmpdir, "workspace")
                shutil.copytree(self.repo_root, tmp_root, dirs_exist_ok=True)
                target = tmp_root / "symbiont" / "agents" / "subself.py"
                target.write_text(modified_text, encoding="utf-8")
                success = True
                for _ in range(3):
                    proc = subprocess.run(
                        ["python", "-m", "compileall", "symbiont/agents/subself.py"],
                        cwd=tmp_root,
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0:
                        success = False
                        preview_path.write_text(
                            preview_path.read_text() + "\n# sandbox failure\n" + proc.stderr,
                            encoding="utf-8",
                        )
                        break
                return success
        except Exception as exc:
            preview_path.write_text(
                preview_path.read_text() + f"\n# sandbox exception: {exc}\n",
                encoding="utf-8",
            )
            return False

    def _persist_proposal(self, intent: MutationIntent, proposal: Dict[str, Any]) -> None:
        manifest_path = self.artifacts_dir / "manifest.json"
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        manifest[str(proposal["timestamp"])] = {
            "kind": intent.kind,
            "rationale": intent.rationale,
            "diff": str(proposal["preview_path"]),
            "details": intent.details,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


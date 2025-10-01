from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests

from ..llm.client import LLMClient
from ..memory.db import MemoryDB
from ..memory import graphrag
from ..tools.files import ensure_dirs


@dataclass
class OracleResult:
    query: str
    url: str
    note_path: str
    triples: List[int]


class QueryOracle:
    """Plans whitelisted web queries and ingests summaries into the belief store."""

    def __init__(self, config: Dict[str, any]):
        self.config = config or {}
        self.llm = LLMClient(self.config.get("llm", {}))
        self.db = MemoryDB(db_path=self.config.get("db_path", "./data/symbiont.db"))
        self.oracle_cfg = (((self.config.get("ports", {}) or {}).get("oracle") or {}))
        self.allowlist = {self._normalize_domain(d) for d in self.oracle_cfg.get("allowlist", [])}
        self.max_queries = int(self.oracle_cfg.get("max_queries", 5))
        self.notes_dir = Path(self.config.get("data_root", "data")) / "artifacts" / "notes"
        ensure_dirs([self.notes_dir])

    # ------------------------------------------------------------------
    def run(self, prompt: str, limit: int | None = None) -> List[OracleResult]:
        limit = min(limit or self.max_queries, self.max_queries)
        self.db.ensure_schema()
        queries = self._plan_queries(prompt)
        results: List[OracleResult] = []
        for query in queries[:limit]:
            url = self._choose_url(query)
            if not url:
                continue
            if not self._is_allowed(url):
                continue
            content = self._fetch(url)
            if not content:
                continue
            note_path = self._write_note(query, url, content)
            triples = self._extract_triples(prompt, content)
            ids = []
            for triple in triples:
                try:
                    cid, _ = graphrag.add_claim(self.db, triple[0], triple[1], triple[2], importance=0.4, source_url=url)
                    ids.append(cid)
                except Exception:
                    continue
            results.append(OracleResult(query=query, url=url, note_path=str(note_path), triples=ids))
        return results

    # ------------------------------------------------------------------
    def _plan_queries(self, prompt: str) -> List[str]:
        guidance = (
            "List up to 5 focused search queries that help a developer research: "
            + prompt
            + ". Output JSON array of strings."
        )
        raw = self.llm.generate(guidance) or "[]"
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(item) for item in data if isinstance(item, str)]
        except Exception:
            pass
        # fallback simple heuristics
        return [prompt]

    def _choose_url(self, query: str) -> str | None:
        suggestion = self.llm.generate(
            f"Provide a single documentation URL (only allowlisted domains) answering: {query}."
        )
        if suggestion:
            suggestion = suggestion.strip().split()[0]
        return suggestion if suggestion else None

    def _is_allowed(self, url: str) -> bool:
        if not self.allowlist:
            return True
        domain = self._normalize_domain(url)
        return domain in self.allowlist

    def _fetch(self, url: str) -> str | None:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code >= 400:
                return None
            text = resp.text
            return text[:5000]
        except Exception:
            return None

    def _write_note(self, query: str, url: str, content: str) -> Path:
        ts = int(time.time())
        fname = self.notes_dir / f"oracle_{ts}.md"
        summary = content[:1000]
        fname.write_text(
            f"# Oracle Result\nQuery: {query}\nURL: {url}\n\n```\n{summary}\n```\n",
            encoding="utf-8",
        )
        return fname

    def _extract_triples(self, prompt: str, content: str) -> List[List[str]]:
        raw = self.llm.generate(
            """Convert the following documentation into at most 3 belief triples in format
[{"subject": str, "relation": str, "object": str}].
Focus on actionable dev guidance.
"""
            + content[:1500]
        )
        try:
            data = json.loads(raw)
            triples: List[List[str]] = []
            for item in data[:3]:
                subject = str(item.get("subject", prompt[:40]) or prompt[:40])
                relation = str(item.get("relation", "related_to"))
                obj = str(item.get("object", ""))
                if obj:
                    triples.append([subject, relation, obj])
            return triples
        except Exception:
            return [[prompt[:40], "related_to", prompt[:80]]]

    def _normalize_domain(self, url: str) -> str:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        return parsed.netloc.lower()

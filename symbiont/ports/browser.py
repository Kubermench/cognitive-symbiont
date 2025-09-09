from __future__ import annotations
from typing import Dict, Any
import os, time
import httpx


def _allowed(url: str, cfg: Dict[str, Any]) -> bool:
    browser = cfg.get("browser", {})
    allow = browser.get("allowlist", []) or []
    return any(url.startswith(pfx) for pfx in allow)


def fetch_to_artifact(url: str, cfg: Dict[str, Any]) -> str:
    if not cfg.get("browser", {}).get("enabled", False):
        raise RuntimeError("browser.enabled=false; enable in configs/config.yaml")
    if not _allowed(url, cfg):
        raise RuntimeError("URL not in allowlist; add to configs/config.yaml -> browser.allowlist")
    if not cfg.get("tools", {}).get("network_access", False):
        raise RuntimeError("tools.network_access=false; enable to allow fetching")
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        txt = r.text
    base_dir = os.path.join(os.path.dirname(cfg["db_path"]), "artifacts", "notes")
    os.makedirs(base_dir, exist_ok=True)
    ts = int(time.time())
    fname = f"note_{ts}.md"
    path = os.path.join(base_dir, fname)
    content = f"""# Note from {url}

Fetched at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}

---
{txt}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


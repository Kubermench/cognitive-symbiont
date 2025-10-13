"""Lightweight security helpers for credential rotation and PII scrubbing."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?1?[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{4}\b")
SECRET_RE = re.compile(r"(api[_-]?key|token|secret)\s*[:=]\s*([A-Za-z0-9-_]{8,})", re.IGNORECASE)
SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")


def detect_pii_patterns(text: str) -> List[Dict[str, Any]]:
    """Detect PII patterns in text and return details about what was found."""
    if not text:
        return []
    
    detections = []
    
    # Email detection
    for match in EMAIL_RE.finditer(text):
        detections.append({
            "type": "email",
            "pattern": match.group(),
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.9,
        })
    
    # Phone detection
    for match in PHONE_RE.finditer(text):
        detections.append({
            "type": "phone",
            "pattern": match.group(),
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.8,
        })
    
    # Secret detection
    for match in SECRET_RE.finditer(text):
        detections.append({
            "type": "secret",
            "pattern": match.group(1),  # Just the key name, not the value
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.95,
        })
    
    # SSN detection
    for match in SSN_RE.finditer(text):
        detections.append({
            "type": "ssn",
            "pattern": "***-**-****",  # Redacted pattern
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.85,
        })
    
    # Credit card detection
    for match in CREDIT_CARD_RE.finditer(text):
        detections.append({
            "type": "credit_card",
            "pattern": "****-****-****-****",  # Redacted pattern
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.75,
        })
    
    return detections


def scrub_text(text: str) -> str:
    """Redact common forms of PII and obvious secrets from *text*."""

    if not text:
        return text
    # Order matters - do SSN and credit cards before phone to avoid conflicts
    redacted = EMAIL_RE.sub("[redacted-email]", text)
    redacted = SSN_RE.sub("[redacted-ssn]", redacted)
    redacted = CREDIT_CARD_RE.sub("[redacted-card]", redacted)
    redacted = PHONE_RE.sub("[redacted-phone]", redacted)
    redacted = SECRET_RE.sub(lambda m: f"{m.group(1)}=[redacted]", redacted)
    return redacted


def scrub_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *payload* with string fields scrubbed for PII."""

    sanitized: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            sanitized[key] = scrub_text(value)
        elif isinstance(value, dict):
            sanitized[key] = scrub_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [scrub_text(item) if isinstance(item, str) else item for item in value]
        else:
            sanitized[key] = value
    return sanitized


def write_audit_log(db_path: str, *, actor: str, action: str, context: str | None = None) -> None:
    """Append a credential/security action to the audit log."""

    db_file = Path(db_path).expanduser()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO audit_logs (actor, action, context) VALUES (?,?,?)",
            (actor, action, context),
        )


def rotate_env_secret(
    env_key: str,
    new_value: str,
    *,
    db_path: str,
    actor: str = "system",
    persist_path: str | None = None,
) -> None:
    """Rotate an environment credential and log the action.

    If *persist_path* is provided it will be treated as a dotenv-style file and
    updated (or created) with the new value so restarts pick it up.
    """

    os.environ[env_key] = new_value
    if persist_path:
        dotenv = Path(persist_path).expanduser()
        lines: list[str] = []
        if dotenv.exists():
            try:
                lines = dotenv.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines = []
        key_prefix = f"{env_key}="
        replaced = False
        for idx, line in enumerate(lines):
            if line.startswith(key_prefix):
                lines[idx] = f"{env_key}={new_value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{env_key}={new_value}")
        dotenv.parent.mkdir(parents=True, exist_ok=True)
        dotenv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_audit_log(db_path, actor=actor, action=f"rotate:{env_key}", context=persist_path or "env")

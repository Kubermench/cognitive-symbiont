from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ShadowClip:
    kind: str
    ts: int
    tags: Sequence[str]
    meta: Dict[str, Any]
    payload: Dict[str, Any]

    @property
    def rogue_score(self) -> Optional[float]:
        if self.kind != "guard":
            return None
        analysis = self.payload.get("analysis") or {}
        score = analysis.get("rogue_score")
        if isinstance(score, (int, float)):
            return float(score)
        return None

    @property
    def reward(self) -> Optional[float]:
        if self.kind != "cycle":
            return None
        reward = self.payload.get("reward")
        if isinstance(reward, (int, float)):
            return float(reward)
        meta_reward = self.meta.get("reward")
        if isinstance(meta_reward, (int, float)):
            return float(meta_reward)
        decision = self.payload.get("decision") or {}
        reward_dec = decision.get("reward")
        if isinstance(reward_dec, (int, float)):
            return float(reward_dec)
        return None


def load_clips(path: Path) -> List[ShadowClip]:
    clips: List[ShadowClip] = []
    if not path.exists():
        return clips
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            clip = _parse_clip(data)
            if clip:
                clips.append(clip)
    return clips


def _parse_clip(data: Dict[str, Any]) -> Optional[ShadowClip]:
    kind = data.get("kind")
    ts = data.get("ts")
    payload = data.get("payload") or {}
    if not isinstance(kind, str) or not isinstance(payload, dict):
        return None
    if not isinstance(ts, int):
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            ts = 0
    tags = tuple(str(tag) for tag in (data.get("tags") or []))
    meta = data.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    return ShadowClip(kind=kind, ts=ts, tags=tags, meta=meta, payload=payload)


def summarize(clips: Iterable[ShadowClip]) -> Dict[str, Any]:
    total = 0
    by_kind: Dict[str, int] = {}
    for clip in clips:
        total += 1
        by_kind[clip.kind] = by_kind.get(clip.kind, 0) + 1
    return {"total": total, "by_kind": by_kind}


class ShadowCurator:
    """Rank shadow clips to surface high-signal interventions for the flywheel."""

    def __init__(self, clip_path: Path):
        self.clip_path = clip_path
        self._clips: Optional[List[ShadowClip]] = None

    @property
    def clips(self) -> List[ShadowClip]:
        if self._clips is None:
            self._clips = load_clips(self.clip_path)
        return self._clips

    def curate(
        self,
        *,
        guard_threshold: float = 0.5,
        guard_limit: Optional[int] = None,
        reward_threshold: float = 0.5,
        cycle_limit: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        if limit is not None:
            guard_limit = guard_limit or limit
            cycle_limit = cycle_limit or limit
        guard_limit = guard_limit or 10
        cycle_limit = cycle_limit or 10
        guards_high, guards_medium = self._rank_guard_clips(
            guard_threshold=guard_threshold, limit=guard_limit
        )
        cycles_low = self._rank_cycle_clips(
            reward_threshold=reward_threshold, limit=cycle_limit
        )
        return {
            "meta": {
                "path": str(self.clip_path),
                "guard_threshold": guard_threshold,
                "reward_threshold": reward_threshold,
                "counts": summarize(self.clips),
            },
            "guards": {
                "high": [clip_to_dict(c) for c in guards_high],
                "medium": [clip_to_dict(c) for c in guards_medium],
            },
            "cycles": {
                "low_reward": [clip_to_dict(c) for c in cycles_low],
            },
        }

    def _rank_guard_clips(
        self,
        *,
        guard_threshold: float,
        limit: int,
    ) -> Tuple[List[ShadowClip], List[ShadowClip]]:
        high: List[ShadowClip] = []
        medium: List[ShadowClip] = []
        guards = [c for c in self.clips if c.kind == "guard" and c.rogue_score is not None]
        guards.sort(key=lambda c: c.rogue_score or 0.0, reverse=True)
        for clip in guards:
            score = clip.rogue_score or 0.0
            if score >= guard_threshold:
                if len(high) < limit:
                    high.append(clip)
            elif score >= guard_threshold * 0.5:
                if len(medium) < limit:
                    medium.append(clip)
        return high, medium

    def _rank_cycle_clips(
        self,
        *,
        reward_threshold: float,
        limit: int,
    ) -> List[ShadowClip]:
        cycles = [
            c for c in self.clips if c.kind == "cycle" and c.reward is not None
        ]
        cycles = [c for c in cycles if c.reward is not None and c.reward <= reward_threshold]
        cycles.sort(
            key=lambda c: (c.reward if c.reward is not None else 0.0, -c.ts)
        )
        return cycles[:limit]


def clip_to_dict(clip: ShadowClip) -> Dict[str, Any]:
    base = {
        "kind": clip.kind,
        "ts": clip.ts,
        "tags": list(clip.tags),
        "meta": clip.meta,
        "payload": clip.payload,
    }
    score = clip.rogue_score
    if score is not None:
        base["rogue_score"] = score
    reward = clip.reward
    if reward is not None:
        base["reward"] = reward
    return base

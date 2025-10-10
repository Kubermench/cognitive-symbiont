from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml
from typer.testing import CliRunner

from symbiont.cli import app as cli_app
from symbiont.observability.shadow import ShadowClipCollector
from symbiont.observability.shadow_curator import ShadowCurator, load_clips
from symbiont.observability.shadow_labeler import annotate_summary


def _write_config(tmp_path: Path) -> Path:
    config = {
        "db_path": str(tmp_path / "data" / "symbiont.db"),
        "data_root": str(tmp_path / "data"),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return cfg_path


def test_shadow_collector_and_curator(tmp_path: Path) -> None:
    clip_dir = tmp_path / "data" / "artifacts" / "shadow"
    collector = ShadowClipCollector(clip_dir)

    collector.record_guard(
        script_path=clip_dir / "apply_test.sh",
        analysis={
            "rogue_score": 0.72,
            "path": "apply_test.sh",
            "issues": [{"kind": "dangerous_command"}],
        },
        meta={"note": "simulated"},
        tags=["guard", "rogue:high"],
    )
    collector.record_guard(
        script_path=clip_dir / "apply_warn.sh",
        analysis={"rogue_score": 0.35, "path": "apply_warn.sh"},
        meta={"note": "medium"},
        tags=["guard", "rogue:medium"],
    )
    collector.record_cycle(
        goal="Test cycle",
        decision={"action": "noop"},
        trace=[{"role": "architect", "output": {"bullets": []}}],
        reward=0.42,
        tags=["orchestrator"],
        meta={"episode_id": 1},
    )

    clip_path = clip_dir / "shadow_clips.jsonl"
    clips = load_clips(clip_path)
    assert len(clips) == 3
    curator = ShadowCurator(clip_path)
    summary = curator.curate(guard_threshold=0.5, reward_threshold=0.5, limit=5)

    guards_high = summary["guards"]["high"]
    guards_medium = summary["guards"]["medium"]
    cycles_low = summary["cycles"]["low_reward"]
    assert len(guards_high) == 1
    assert len(guards_medium) == 1
    assert len(cycles_low) == 1
    assert guards_high[0]["rogue_score"] >= 0.5
    assert cycles_low[0]["reward"] <= 0.5

    labeled = annotate_summary(summary)
    assert labeled["guards"]["high"][0]["labels"]
    assert "issue.dangerous_command" in labeled["guards"]["high"][0]["labels"]
    assert any(label.startswith("reward.") for label in labeled["cycles"]["low_reward"][0]["labels"])


def test_shadow_cli_commands(tmp_path: Path) -> None:
    clip_dir = tmp_path / "data" / "artifacts" / "shadow"
    collector = ShadowClipCollector(clip_dir)
    collector.record_guard(
        script_path=clip_dir / "apply_fail.sh",
        analysis={"rogue_score": 0.8, "path": "apply_fail.sh"},
        tags=["guard"],
    )
    collector.record_cycle(
        goal="Guarded goal",
        decision={"action": "Do risky thing"},
        trace=[{"role": "architect", "output": {"bullets": ["bullet"]}}],
        reward=0.3,
        tags=["orchestrator"],
        meta={"episode_id": 7},
    )

    cfg_path = _write_config(tmp_path)
    runner = CliRunner()

    report = runner.invoke(
        cli_app,
        ["shadow_report", "--config-path", str(cfg_path)],
    )
    assert report.exit_code == 0
    assert "Shadow clips" in report.stdout

    summary_path = tmp_path / "summary.json"
    curated = runner.invoke(
        cli_app,
        [
            "shadow_curate",
            "--config-path",
            str(cfg_path),
            "--guard-threshold",
            "0.6",
            "--reward-threshold",
            "0.4",
            "--output",
            str(summary_path),
        ],
    )
    assert curated.exit_code == 0
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["guards"]["high"]
    assert data["cycles"]["low_reward"]

    labeled_path = tmp_path / "labeled.json"
    labeled_res = runner.invoke(
        cli_app,
        [
            "shadow_label",
            "--config-path",
            str(cfg_path),
            "--guard-threshold",
            "0.6",
            "--reward-threshold",
            "0.4",
            "--output",
            str(labeled_path),
        ],
    )
    assert labeled_res.exit_code == 0
    assert labeled_path.exists()
    labeled_data = json.loads(labeled_path.read_text(encoding="utf-8"))
    assert labeled_data["labels"]["counts"]
    with sqlite3.connect(str(tmp_path / "data" / "symbiont.db")) as conn:
        row = conn.execute(
            "SELECT type, path, summary FROM artifacts WHERE type='shadow_labels' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert Path(row[1]).exists()
        assert "Shadow labels" in (row[2] or "")

    ingest_res = runner.invoke(
        cli_app,
        [
            "shadow_ingest",
            "--config-path",
            str(cfg_path),
            "--path",
            str(labeled_path),
            "--top",
            "3",
        ],
    )
    assert ingest_res.exit_code == 0

    default_res = runner.invoke(
        cli_app,
        [
            "shadow_label",
            "--config-path",
            str(cfg_path),
            "--ingest",
            "--ingest-top",
            "4",
        ],
    )
    assert default_res.exit_code == 0
    label_dir = tmp_path / "data" / "artifacts" / "shadow" / "labels"
    assert any(label_dir.glob("shadow_labels_*.json"))
    with sqlite3.connect(str(tmp_path / "data" / "symbiont.db")) as conn:
        msg = conn.execute(
            "SELECT content FROM messages WHERE role='shadow' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert msg and "shadow_labels" in msg[0]
        beliefs = conn.execute(
            "SELECT statement FROM beliefs WHERE statement LIKE 'Shadow label:%'"
        ).fetchall()
        assert beliefs

    dashboard_path = tmp_path / "systems" / "ShadowDashboard.md"
    dashboard_res = runner.invoke(
        cli_app,
        [
            "shadow_dashboard",
            "--config-path",
            str(cfg_path),
            "--top",
            "5",
            "--output",
            str(dashboard_path),
        ],
    )
    assert dashboard_res.exit_code == 0
    assert dashboard_path.exists()
    text = dashboard_path.read_text(encoding="utf-8")
    assert "Shadow Dashboard" in text

    batch_dashboard = tmp_path / "systems" / "ShadowBatch.md"
    batch_label_path = tmp_path / "batch_labels.json"
    batch_res = runner.invoke(
        cli_app,
        [
            "shadow_batch",
            "--config-path",
            str(cfg_path),
            "--label-output",
            str(batch_label_path),
            "--dashboard-output",
            str(batch_dashboard),
            "--dashboard-top",
            "3",
            "--ingest-top",
            "2",
        ],
    )
    assert batch_res.exit_code == 0
    assert batch_label_path.exists()
    assert batch_dashboard.exists()
    assert "Shadow Dashboard" in batch_dashboard.read_text(encoding="utf-8")

    history_path = tmp_path / "data" / "artifacts" / "shadow" / "history.jsonl"
    assert history_path.exists()

    history_export = tmp_path / "systems" / "ShadowHistory.md"
    history_res = runner.invoke(
        cli_app,
        [
            "shadow_history",
            "--config-path",
            str(cfg_path),
            "--limit",
            "3",
            "--output",
            str(history_export),
        ],
    )
    assert history_res.exit_code == 0
    assert "Shadow history entries" in history_res.stdout
    assert history_export.exists()
    contents = history_export.read_text(encoding="utf-8")
    assert "Shadow History" in contents

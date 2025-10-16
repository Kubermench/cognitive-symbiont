from typer.testing import CliRunner

from symbiont.cli import app


def test_quickstart_runs_with_skip_external(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("- First bullet\n- Second bullet\n## Sources\n", encoding="utf-8")
    captured = {}

    class DummyOrchestrator:
        def __init__(self, cfg):
            captured["config"] = cfg

        def cycle(self, goal: str) -> str:
            captured["goal"] = goal
            return str(plan_path)

    monkeypatch.setattr("symbiont.cli.Orchestrator", DummyOrchestrator)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            f"db_path: {tmp_path / 'sym.db'}\n"
            "retrieval:\n"
            "  external:\n"
            "    enabled: true\n"
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "quickstart",
            "--config-path",
            str(config_path),
            "--goal",
            "demo goal",
        ],
    )

    assert result.exit_code == 0
    assert captured["goal"] == "demo goal"
    retrieval_cfg = captured["config"].get("retrieval", {}).get("external", {})
    assert retrieval_cfg.get("enabled") is False

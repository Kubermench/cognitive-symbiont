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


def test_bias_check_cli_invokes_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"db_path: {tmp_path / 'sym.db'}\n", encoding="utf-8")
    crew_path = tmp_path / "crew.yaml"
    crew_path.write_text("agents: {}\ncrew: {}\n", encoding="utf-8")

    dummy_registry = object()
    monkeypatch.setattr("symbiont.cli.AgentRegistry.from_yaml", lambda path: dummy_registry)

    captured: dict[str, object] = {}

    class DummyCrewRunner:
        def __init__(self, registry, cfg, db):
            captured["registry"] = registry
            captured["cfg"] = cfg
            captured["db_path"] = getattr(db, "db_path", None)

        def run(self, crew: str, goal: str) -> str:
            captured["crew"] = crew
            captured["goal"] = goal
            artifact = tmp_path / "bias_report.md"
            artifact.write_text("Bias score: 0.5", encoding="utf-8")
            return str(artifact)

    monkeypatch.setattr("symbiont.cli.CrewRunner", DummyCrewRunner)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bias-check",
            "AGI safety",
            "--config-path",
            str(config_path),
            "--crew-path",
            str(crew_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["registry"] is dummy_registry
    assert captured["crew"] == "bias_hunter"
    assert captured["goal"] == "AGI safety"

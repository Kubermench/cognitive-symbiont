from __future__ import annotations

from pathlib import Path

from symbiont.initiative.watchers import build_repo_watch_configs


def test_build_repo_watch_configs_inline_overrides(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = {
        "initiative": {
            "repo_path": str(repo),
            "watchers": ["file", "git", "timer"],
            "watch_targets": [
                {
                    "path": str(repo),
                    "watchers": ["timer", "git", "timer"],  # duplicates should collapse
                    "idle_minutes": 45,
                    "git_idle_minutes": 30,
                    "timer_minutes": 20,
                    "trigger_mode": "any",
                    "verify_rollback": True,
                }
            ],
        }
    }

    configs = build_repo_watch_configs(cfg)
    assert len(configs) == 1
    conf = configs[0]
    assert conf.watchers == ("timer", "git")
    assert conf.idle_minutes == 45
    assert conf.git_idle_minutes == 30
    assert conf.timer_minutes == 20
    assert conf.trigger_mode == "any"
    assert conf.verify_rollback is True


def test_build_repo_watch_configs_yaml_fallback(tmp_path: Path) -> None:
    repo = tmp_path / "other"
    repo.mkdir()
    yaml_path = tmp_path / "watch.yaml"
    yaml_path.write_text(
        """
repositories:
  - path: "{path}"
    watchers: ["file", "unknown", "git"]
    idle_minutes: 90
    trigger_mode: "idle_and_timer"
""".format(
            path=str(repo)
        )
    )
    cfg = {"initiative": {"watch_config_path": str(yaml_path)}}

    configs = build_repo_watch_configs(cfg)
    assert len(configs) == 1
    conf = configs[0]
    assert conf.path == repo.resolve()
    assert conf.watchers == ("file", "git")
    assert conf.idle_minutes == 90
    assert conf.trigger_mode == "idle_and_timer"

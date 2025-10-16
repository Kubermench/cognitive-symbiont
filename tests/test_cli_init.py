from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "symbiont.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sym_init_lite_generates_config_and_schema(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    base_config = ROOT / "configs" / "config-lite.yaml"
    target_config = configs_dir / "config.generated.yaml"
    target_config.parent.mkdir(parents=True, exist_ok=True)
    base_config_contents = base_config.read_text(encoding="utf-8")
    (configs_dir / "config-lite.yaml").write_text(base_config_contents, encoding="utf-8")

    result = _run_cli(
        [
            "init",
            "--lite",
            "--non-interactive",
            "--path",
            str(target_config),
            "--force",
        ],
        cwd=tmp_path,
    )

    assert (
        result.returncode == 0
    ), f"sym init failed: stdout={result.stdout}\nstderr={result.stderr}"

    assert target_config.exists(), "Configuration file was not created."
    generated = yaml.safe_load(target_config.read_text(encoding="utf-8"))
    assert generated.get("env") == "dev"
    assert "llm" in generated

    db_file = tmp_path / "data" / "symbiont.db"
    assert db_file.exists(), "Database schema was not initialised."

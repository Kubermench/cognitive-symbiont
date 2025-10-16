import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.tools import scriptify


def test_write_script_includes_commands_and_guard(tmp_path):
    bullets = ["Tidy temp files (cmd: rm -rf build)"]
    base_dir = tmp_path / "scripts"
    path = Path(scriptify.write_script(bullets, str(base_dir)))
    contents = path.read_text(encoding="utf-8")

    assert "rm -rf build" in contents
    assert "SYM_ROLLBACK_FORCE=1" in contents
    assert contents.startswith("#!/usr/bin/env bash")


def test_write_rollback_script_requires_force_flag(tmp_path):
    apply_path = tmp_path / "scripts" / "apply_0.sh"
    apply_path.parent.mkdir(parents=True, exist_ok=True)
    apply_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    rollback_path = Path(scriptify.write_rollback_script(str(apply_path)))
    contents = rollback_path.read_text(encoding="utf-8")

    assert "SYM_ROLLBACK_FORCE" in contents
    assert "exit 1" in contents
    assert rollback_path.stat().st_mode & 0o111, "rollback script should be executable"

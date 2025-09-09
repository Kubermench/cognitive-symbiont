from __future__ import annotations
import os, time, json, subprocess
from typing import Dict, Any, Tuple, List

from ..orchestrator import Orchestrator
from ..tools.files import ensure_dirs


STATE_DIR = os.path.join("./data", "initiative")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
STOP_PATH = os.path.join(STATE_DIR, "STOP")


def _now() -> int:
    return int(time.time())


def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_proposal_ts": 0, "last_check_ts": 0, "daemon_running": False, "daemon_started_ts": 0, "daemon_pid": 0}


def _save_state(st: Dict[str, Any]):
    ensure_dirs([STATE_DIR])
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f)


def request_stop():
    ensure_dirs([STATE_DIR])
    try:
        with open(STOP_PATH, "w") as f:
            f.write("stop")
    except Exception:
        pass


def clear_stop():
    try:
        if os.path.exists(STOP_PATH):
            os.remove(STOP_PATH)
    except Exception:
        pass


def is_stop_requested() -> bool:
    return os.path.exists(STOP_PATH)


def _is_git_repo(path: str) -> bool:
    try:
        out = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, text=True)
        return out.returncode == 0 and out.stdout.strip() == "true"
    except Exception:
        return False


def _git_last_commit_ts(path: str) -> int:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=path, capture_output=True, text=True)
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0
    except Exception:
        return 0


def _git_is_dirty(path: str) -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def _latest_file_mtime(path: str) -> int:
    latest = 0
    skip_dirs = {".git", "data", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
    for root, dirs, files in os.walk(path):
        # prune dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            try:
                full = os.path.join(root, fn)
                m = int(os.path.getmtime(full))
                if m > latest:
                    latest = m
            except Exception:
                pass
    return latest


def _should_trigger(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ini = cfg.get("initiative", {})
    if not ini.get("enabled", False):
        return False, ["initiative.disabled"]
    repo_path = os.path.abspath(ini.get("repo_path", "."))
    watchers: List[str] = ini.get("watchers", ["file", "git", "timer"]) or []
    idle_minutes = int(ini.get("idle_minutes", 120))
    timer_minutes = int(ini.get("timer_minutes", 120))
    trigger_mode = ini.get("trigger_mode", "idle_and_timer")  # any | idle_and_timer

    state = _load_state()
    now = _now()
    reasons: List[str] = []

    idle_ok = False
    timer_ok = False

    if "file" in watchers:
        mtime = _latest_file_mtime(repo_path)
        if mtime > 0 and (now - mtime) >= idle_minutes * 60:
            idle_ok = True
            reasons.append(f"file.idle>={idle_minutes}m")

    if "git" in watchers:
        if _is_git_repo(repo_path):
            last = _git_last_commit_ts(repo_path)
            dirty = _git_is_dirty(repo_path)
            if last > 0 and (now - last) >= idle_minutes * 60 and not dirty:
                idle_ok = True
                reasons.append(f"git.idle>={idle_minutes}m")

    if "timer" in watchers:
        last_prop = int(state.get("last_proposal_ts", 0))
        if (now - last_prop) >= timer_minutes * 60:
            timer_ok = True
            reasons.append(f"timer>={timer_minutes}m")

    if trigger_mode == "idle_and_timer":
        ok = idle_ok and timer_ok
    else:  # any
        ok = idle_ok or timer_ok

    return ok, reasons


def propose_once(cfg: Dict[str, Any], reason: str = "watchers") -> Dict[str, Any]:
    goal = cfg.get("initiative", {}).get(
        "goal_template",
        "Repo idle; propose one 10-minute refactor and draft a script.",
    )
    goal = f"{goal} [trigger: {reason}]"
    orch = Orchestrator(cfg)
    res = orch.cycle(goal=goal)
    st = _load_state()
    st["last_proposal_ts"] = _now()
    st["last_check_ts"] = _now()
    _save_state(st)
    return res


def run_once_if_triggered(cfg: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any] | None]:
    ok, reasons = _should_trigger(cfg)
    if not ok:
        return False, reasons, None
    res = propose_once(cfg, reason=",".join(reasons))
    st = _load_state()
    st["last_proposal_ts"] = _now()
    st["last_check_ts"] = _now()
    _save_state(st)
    return True, reasons, res


def daemon_loop(cfg: Dict[str, Any], poll_seconds: int = 60):
    ensure_dirs([STATE_DIR])
    clear_stop()
    print("[initiative] daemon started; polling", poll_seconds, "s")
    st = _load_state()
    st.update({
        "daemon_running": True,
        "daemon_started_ts": _now(),
        "daemon_pid": os.getpid(),
    })
    _save_state(st)
    while True:
        try:
            ok, reasons, _ = run_once_if_triggered(cfg)
            now = _now()
            st = _load_state()
            st["last_check_ts"] = now
            _save_state(st)
            if ok:
                print("[initiative] proposed (reasons:", ",".join(reasons), ")")
            if is_stop_requested():
                print("[initiative] stop requested; exiting daemon")
                break
        except KeyboardInterrupt:
            print("[initiative] daemon stopped by user")
            break
        except Exception as e:
            print("[initiative] error:", e)
        time.sleep(poll_seconds)
    st = _load_state()
    st.update({"daemon_running": False})
    _save_state(st)


def get_status() -> Dict[str, Any]:
    st = _load_state()
    return {
        "last_check_ts": int(st.get("last_check_ts", 0)),
        "last_proposal_ts": int(st.get("last_proposal_ts", 0)),
        "daemon_running": bool(st.get("daemon_running", False)),
        "daemon_started_ts": int(st.get("daemon_started_ts", 0)),
        "daemon_pid": int(st.get("daemon_pid", 0)),
        "state_path": os.path.abspath(STATE_PATH),
    }

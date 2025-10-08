from __future__ import annotations
import os, time, json, subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import logging
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from ..orchestrator import Orchestrator
from ..agents.registry import AgentRegistry, CrewRunner
from ..memory.db import MemoryDB
from ..tools.files import ensure_dirs
from .watchers import RepoWatchConfig, build_repo_watch_configs
from .pubsub import get_client
from .state import get_state_store, resolve_node_id


STATE_DIR = os.path.join("./data", "initiative")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
STOP_PATH = os.path.join(STATE_DIR, "STOP")


def _now() -> int:
    return int(time.time())


def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data.setdefault("targets", {})
    data.setdefault("last_proposal_ts", 0)
    data.setdefault("last_check_ts", 0)
    data.setdefault("daemon_running", False)
    data.setdefault("daemon_started_ts", 0)
    data.setdefault("daemon_pid", 0)
    data.setdefault("foresight_last_run_ts", 0)
    data.setdefault("foresight_last_goal", "")
    data.setdefault("foresight_last_artifact", "")
    return data


def _save_state(st: Dict[str, Any]):
    ensure_dirs([STATE_DIR])
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f)


def _publish_event(cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    client = get_client(cfg)
    if not getattr(client, "enabled", False):
        return
    event = dict(payload)
    event.setdefault("timestamp", _now())
    try:
        client.publish(event)
    except Exception as exc:  # pragma: no cover - should be rare
        logger.warning("Failed publishing initiative event: %s", exc)


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


def _should_trigger(cfg: Dict[str, Any]) -> Tuple[bool, List[str], Optional[RepoWatchConfig]]:
    ini = cfg.get("initiative", {})
    if not ini.get("enabled", False):
        return False, ["initiative.disabled"], None

    configs = build_repo_watch_configs(cfg)
    if not configs:
        return False, ["initiative.no_targets"], None

    state = _load_state()
    now = _now()

    for target in configs:
        path = str(target.path)
        target_state = state.setdefault("targets", {}).setdefault(path, {"last_proposal": 0, "last_check": 0})
        reasons: List[str] = []

        idle_ok = False
        timer_ok = False

        target_state["last_check"] = now

        if "file" in target.watchers:
            mtime = _latest_file_mtime(path)
            if mtime > 0 and (now - mtime) >= target.idle_minutes * 60:
                idle_ok = True
                reasons.append(f"file.idle>={target.idle_minutes}m")

        if "git" in target.watchers and _is_git_repo(path):
            last = _git_last_commit_ts(path)
            dirty = _git_is_dirty(path)
            if last > 0 and (now - last) >= target.git_idle_minutes * 60 and not dirty:
                idle_ok = True
                reasons.append(f"git.idle>={target.git_idle_minutes}m")

        if "timer" in target.watchers:
            last_prop = int(target_state.get("last_proposal", 0))
            if (now - last_prop) >= target.timer_minutes * 60:
                timer_ok = True
                reasons.append(f"timer>={target.timer_minutes}m")

        if target.trigger_mode == "idle_and_timer":
            ok = idle_ok and timer_ok
        else:
            ok = idle_ok or timer_ok

        if ok:
            _save_state(state)
            return True, reasons, target

    _save_state(state)
    return False, ["no_target_ready"], None


def _foresight_should_trigger(
    cfg: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, Any], Dict[str, Any], int]:
    foresight_cfg = cfg.get("foresight")
    state = _load_state()
    now = _now()
    if not isinstance(foresight_cfg, dict) or not foresight_cfg.get("enabled", False):
        return False, ["foresight.disabled"], {}, state, now

    interval = int(foresight_cfg.get("timer_minutes", 1440) or 1440)
    last_run = int(state.get("foresight_last_run_ts", 0))
    elapsed = now - last_run
    pending: List[str] = []
    reasons: List[str] = []

    timer_ok = True
    if interval > 0:
        if elapsed >= interval * 60:
            reasons.append(f"foresight.timer>={interval}m")
        else:
            timer_ok = False
            remaining = max(0, (interval * 60) - elapsed)
            minutes = max(1, remaining // 60) if remaining else 1
            pending.append(f"foresight.wait<{minutes}m")

    idle_minutes = int(foresight_cfg.get("idle_minutes", 0) or 0)
    idle_ok = True
    if idle_minutes > 0:
        repo_root = foresight_cfg.get("repo_path") or cfg.get("initiative", {}).get("repo_path", ".")
        latest = _latest_file_mtime(str(Path(repo_root).expanduser().resolve()))
        if latest and (now - latest) < idle_minutes * 60:
            idle_ok = False
            pending.append(f"foresight.idle<{idle_minutes}m")
        else:
            reasons.append(f"foresight.idle>={idle_minutes}m")

    trigger_mode = str(foresight_cfg.get("trigger_mode", "any") or "any").lower()
    if trigger_mode == "idle_and_timer":
        due = timer_ok and idle_ok
    else:
        due = timer_ok or idle_ok

    if due:
        if not reasons:
            reasons.append("foresight.manual")
        return True, reasons, foresight_cfg, state, now

    return False, (pending or reasons or ["foresight.not_due"]), foresight_cfg, state, now


def _run_foresight(
    cfg: Dict[str, Any],
    foresight_cfg: Dict[str, Any],
    state: Dict[str, Any],
    now: int,
    reasons: List[str],
) -> Dict[str, Any]:
    crew_name = foresight_cfg.get("crew", "foresight_weaver")
    goal = foresight_cfg.get("goal", "Emerging agentic AI trends")
    crew_config_path = foresight_cfg.get("crew_config") or foresight_cfg.get("crews_path") or "./configs/crews/foresight_weaver.yaml"
    crews_file = Path(crew_config_path).expanduser().resolve()

    try:
        registry = AgentRegistry.from_yaml(crews_file)
        db = MemoryDB(cfg["db_path"])
        runner = CrewRunner(registry, cfg, db)
        artifact_path = runner.run(crew_name, goal)
    except Exception as exc:
        _publish_event(
            cfg,
            {
                "type": "foresight.error",
                "goal": goal,
                "crew": crew_name,
                "reasons": reasons,
                "error": str(exc),
            },
        )
        raise

    state["foresight_last_run_ts"] = now
    state["foresight_last_goal"] = goal
    state["foresight_last_artifact"] = str(artifact_path)
    state["last_proposal_ts"] = now
    _save_state(state)

    _publish_event(
        cfg,
        {
            "type": "foresight.run",
            "goal": goal,
            "crew": crew_name,
            "artifact": str(artifact_path),
            "reasons": reasons,
        },
    )
    return {
        "crew": crew_name,
        "goal": goal,
        "artifact": str(artifact_path),
        "reasons": reasons,
    }


def _retry(
    operation,
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    sleep_fn = time.sleep,
):
    """Retry *operation* with exponential backoff via tenacity."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")

    multiplier = max(0.0, float(base_delay))
    exp_base = max(1.0, float(backoff))
    max_wait = max(multiplier, multiplier * (exp_base ** max(0, attempts - 1)))

    def _log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        if exc:
            logger.warning(
                "Retry %s/%s failed: %s",
                state.attempt_number,
                attempts,
                exc,
            )

    if multiplier <= 0:
        wait_strategy = wait_fixed(0)
    else:
        wait_strategy = wait_exponential(
            multiplier=multiplier,
            exp_base=exp_base,
            min=multiplier,
            max=max_wait,
        )

    retryer = Retrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(attempts),
        wait=wait_strategy,
        sleep=sleep_fn,
        reraise=True,
        before_sleep=_log_retry,
    )
    return retryer(operation)


def propose_once(cfg: Dict[str, Any], reason: str = "watchers", *, target: RepoWatchConfig | None = None) -> Dict[str, Any]:
    goal = cfg.get("initiative", {}).get(
        "goal_template",
        "Repo idle; propose one 10-minute refactor and draft a script.",
    )
    if target is not None:
        goal = f"{goal} [repo:{target.path}] [trigger:{reason}]"
    else:
        goal = f"{goal} [trigger:{reason}]"
    orch = Orchestrator(cfg)

    retry_cfg = (cfg.get("initiative") or {}).get("retry", {})
    attempts = int(retry_cfg.get("attempts", 3))
    base_delay = float(retry_cfg.get("base_delay", 1.0))
    backoff = float(retry_cfg.get("backoff", 2.0))

    res = _retry(
        lambda: orch.cycle(goal=goal),
        attempts=max(1, attempts),
        base_delay=base_delay,
        backoff=max(1.0, backoff),
    )
    st = _load_state()
    now = _now()
    st["last_proposal_ts"] = now
    st["last_check_ts"] = now
    if target is not None:
        path = str(target.path)
        st.setdefault("targets", {}).setdefault(path, {"last_proposal": 0, "last_check": 0})
        st["targets"][path]["last_proposal"] = now
        st["targets"][path]["last_check"] = now
    _save_state(st)
    _publish_event(
        cfg,
        {
            "type": "initiative.proposal",
            "reason": reason,
            "goal": goal,
            "target": str(target.path) if target else None,
            "decision": res.get("decision", {}).get("action"),
        },
    )
    return res


def run_once_if_triggered(cfg: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any] | None, Optional[RepoWatchConfig]]:
    ok, reasons, target = _should_trigger(cfg)
    if ok:
        res = propose_once(cfg, reason=",".join(reasons), target=target)
        st = _load_state()
        now = _now()
        st["last_proposal_ts"] = now
        st["last_check_ts"] = now
        if target is not None:
            path = str(target.path)
            st.setdefault("targets", {}).setdefault(path, {"last_proposal": 0, "last_check": 0})
            st["targets"][path]["last_proposal"] = now
            st["targets"][path]["last_check"] = now
        _save_state(st)
        _publish_event(
            cfg,
            {
                "type": "initiative.proposal",
                "reason": ",".join(reasons),
                "goal": res.get("goal") if isinstance(res, dict) else None,
                "target": str(target.path) if target else None,
            },
        )
        return True, reasons, res, target

    foresight_ok, foresight_reasons, foresight_cfg, foresight_state, now = _foresight_should_trigger(cfg)
    if foresight_ok:
        result = _run_foresight(cfg, foresight_cfg, foresight_state, now, foresight_reasons)
        return True, foresight_reasons, result, None

    combined_reasons = reasons + foresight_reasons
    _publish_event(
        cfg,
        {
            "type": "initiative.idle",
            "reasons": combined_reasons,
        },
    )
    return False, combined_reasons, None, target


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

    store = get_state_store(cfg)
    node_id = resolve_node_id(cfg)
    store.record_daemon(
        node_id=node_id,
        pid=os.getpid(),
        status="running",
        poll_seconds=poll_seconds,
        last_check_ts=int(st.get("last_check_ts", 0)),
        last_proposal_ts=int(st.get("last_proposal_ts", 0)),
        details={
            "startup": True,
            "started_ts": int(st.get("daemon_started_ts", _now())),
        },
    )

    retry_cfg = (cfg.get("initiative") or {}).get("retry", {})
    daemon_attempts = max(1, int(retry_cfg.get("attempts", 3) or 3))
    daemon_base = max(0.1, float(retry_cfg.get("base_delay", 1.0) or 1.0))
    daemon_backoff = max(1.0, float(retry_cfg.get("backoff", 2.0) or 2.0))

    while True:
        try:
            ok, reasons, _, target = _retry(
                lambda: run_once_if_triggered(cfg),
                attempts=daemon_attempts,
                base_delay=daemon_base,
                backoff=daemon_backoff,
            )
            now = _now()
            st = _load_state()
            st["last_check_ts"] = now
            _save_state(st)
            store.record_daemon(
                node_id=node_id,
                pid=os.getpid(),
                status="running",
                poll_seconds=poll_seconds,
                last_check_ts=now,
                last_proposal_ts=int(st.get("last_proposal_ts", 0)),
                details={
                    "ok": bool(ok),
                    "reasons": reasons,
                    "target": str(getattr(target, "path", "")) if target else None,
                    "started_ts": int(st.get("daemon_started_ts", now)),
                },
            )
            if ok:
                repo_msg = f" repo={getattr(target, 'path', '?')}" if target else ""
                print("[initiative] proposed (reasons:", ",".join(reasons), ")" + repo_msg)
            if is_stop_requested():
                print("[initiative] stop requested; exiting daemon")
                break
        except KeyboardInterrupt:
            print("[initiative] daemon stopped by user")
            break
        except Exception as e:
            print("[initiative] error:", e)
            store.record_daemon(
                node_id=node_id,
                pid=os.getpid(),
                status="error",
                poll_seconds=poll_seconds,
                last_check_ts=_now(),
                last_proposal_ts=int(st.get("last_proposal_ts", 0)),
                details={
                    "error": str(e),
                    "started_ts": int(st.get("daemon_started_ts", _now())),
                },
            )
            time.sleep(daemon_base)
            continue
        time.sleep(poll_seconds)
    st = _load_state()
    st.update({"daemon_running": False})
    _save_state(st)
    store.mark_daemon_stopped(node_id)


def get_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        store = get_state_store(cfg)
        node_id = resolve_node_id(cfg)
        current = store.load_daemon(node_id)
    except Exception:
        current = None

    if current:
        return {
            "last_check_ts": int(current.get("last_check_ts") or 0),
            "last_proposal_ts": int(current.get("last_proposal_ts") or 0),
            "daemon_running": current.get("status") in {"running", "error"},
            "daemon_started_ts": int(current.get("details", {}).get("started_ts", 0) or 0),
            "daemon_pid": int(current.get("pid") or 0),
            "state_path": os.path.abspath(STATE_PATH),
            "status": current.get("status"),
            "node_id": current.get("node_id"),
        }

    st = _load_state()
    return {
        "last_check_ts": int(st.get("last_check_ts", 0)),
        "last_proposal_ts": int(st.get("last_proposal_ts", 0)),
        "daemon_running": bool(st.get("daemon_running", False)),
        "daemon_started_ts": int(st.get("daemon_started_ts", 0)),
        "daemon_pid": int(st.get("daemon_pid", 0)),
        "state_path": os.path.abspath(STATE_PATH),
    }
logger = logging.getLogger(__name__)

import os, json, typer, yaml, sqlite3, time, stat
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich import print as rprint
from .orchestrator import Orchestrator
from .memory.db import MemoryDB
from .memory import retrieval, coerce_backend_name, available_backends
from .memory.external_sources import list_cache_entries, clear_cache
from .tools import repo_scan, scriptify
from .llm.client import LLMClient
from .llm.budget import TokenBudget
from .initiative import daemon as initiative
from .agents.mutation import MutationEngine, MutationIntent
from .agents.registry import AgentRegistry, CrewRunner
from .orchestration.graph import GraphSpec, GraphRunner
from .orchestration.dynamics_weaver import run_dynamics_weaver
from .agents.swarm import SwarmCoordinator
from . import guards as guard_mod
from .ports.oracle import QueryOracle
from .ports.ai_peer import AIPeerBridge
from .ports.github import GitHubGuard
from .observability.metrics import serve_metrics as start_metrics_server
from .tools.security import rotate_env_secret
from .tools.systems_os import update_flow_metrics
from .initiative.watchers import WatchEvent, build_repo_watch_configs  # noqa: F401 (placeholder for future use)
from .runtime.guard import Guard, Action, Capability  # noqa: F401 (placeholder for future use)
from .ports import browser as browser_port
from .memory import graphrag
from .observability.shadow_curator import ShadowCurator
from .observability.shadow_labeler import annotate_summary
from .observability.shadow_ingest import (
    latest_labeled_path,
    load_labeled_summary,
    ingest_labels,
    summarize_labels,
)
from .observability.shadow_dashboard import render_dashboard
from .observability.shadow_history import record_history, load_history
from .plugins import PluginRegistry

app = typer.Typer(help="Cognitive Symbiont — MVP CLI v2.3")

def load_config(path: str = "./configs/config.yaml"):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)


MEMORY_LAYER_HELP = f"Memory backend to use ({', '.join(available_backends())}). Defaults to config/env."


def _memory_layer(cfg: Dict[str, Any], override: Optional[str] = None) -> Optional[str]:
    return coerce_backend_name(cfg, override=override)


def _memory_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    section = cfg.get("memory")
    return section if isinstance(section, dict) else {}

def _label_shadow(
    cfg: Dict[str, Any],
    *,
    guard_threshold: float,
    reward_threshold: float,
    limit: int,
    output: Optional[str],
) -> tuple[Dict[str, Any], Dict[str, Any], Path, int, int, int, Path]:
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    clip_dir = data_root / "artifacts" / "shadow"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "shadow_clips.jsonl"
    curator = ShadowCurator(clip_path)
    summary = curator.curate(
        guard_threshold=guard_threshold,
        guard_limit=limit,
        reward_threshold=reward_threshold,
        cycle_limit=limit,
        limit=limit,
    )
    labeled = annotate_summary(summary)
    guard_total = len(summary.get("guards", {}).get("high", [])) + len(summary.get("guards", {}).get("medium", []))
    cycle_total = len(summary.get("cycles", {}).get("low_reward", []))
    meta = summary.get("meta") or {}
    counts = meta.get("counts") or {}
    total = counts.get("total", 0)

    if output:
        out_path = Path(output).expanduser().resolve()
    else:
        label_dir = clip_dir / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = label_dir / f"shadow_labels_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta["path"] = str(out_path)
    labeled.setdefault("meta", {})
    labeled["meta"]["path"] = str(out_path)

    out_path.write_text(json.dumps(labeled, indent=2), encoding="utf-8")
    history_cfg = cfg.get("shadow_history", {}) if isinstance(cfg.get("shadow_history"), dict) else {}
    history_max = int(history_cfg.get("max_entries", 200)) if history_cfg else 200
    top_k = max(1, min(limit, int(history_cfg.get("top_k", limit or 10))))
    history_path = record_history(
        data_root=data_root,
        labeled=labeled,
        summary=summary,
        source_path=out_path,
        top_k=top_k,
        max_entries=history_max,
    )
    return labeled, summary, out_path, guard_total, cycle_total, total, history_path


def _prepare_script_sandbox(script_path: Path, cfg: dict, sandbox: Optional[str]):
    import shutil, tempfile

    script_path = script_path.expanduser().resolve()
    repo_root = Path(cfg.get("initiative", {}).get("repo_path", ".")).expanduser().resolve()
    if not repo_root.exists():
        try:
            repo_root = script_path.parents[3]
        except IndexError:
            repo_root = script_path.parent

    if sandbox:
        sandbox_root = Path(sandbox).expanduser().resolve()
        sandbox_root.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        sandbox_root = Path(tempfile.mkdtemp(prefix="symbiont_rollback_"))
        cleanup = True

    dest_repo = sandbox_root / "workspace"
    if dest_repo.exists():
        shutil.rmtree(dest_repo)
    shutil.copytree(
        repo_root,
        dest_repo,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", "data/artifacts/logs"),
    )

    rollback_name = script_path.name.replace("apply_", "rollback_", 1)

    try:
        rel_script = script_path.relative_to(repo_root)
    except ValueError:
        try:
            rel_script = script_path.relative_to(Path.cwd())
        except ValueError:
            rel_script = Path("data/artifacts/scripts") / script_path.name

    dest_script = dest_repo / Path(rel_script)
    dest_rollback = dest_script.parent / rollback_name

    return {
        "repo_root": repo_root,
        "sandbox_root": sandbox_root,
        "dest_repo": dest_repo,
        "dest_script": dest_script,
        "dest_rollback": dest_rollback,
        "cleanup": cleanup,
    }

@app.command()
def init(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path); db=MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()
    rprint(f"[green]DB initialized at[/green] {cfg['db_path']}")

@app.command()
def rag_reindex(
    config_path: str = "./configs/config.yaml",
    memory_layer: Optional[str] = typer.Option(None, "--memory-layer", help=MEMORY_LAYER_HELP),
):
    cfg = load_config(config_path)
    db = MemoryDB(db_path=cfg["db_path"])
    db.ensure_schema()
    backend = _memory_layer(cfg, override=memory_layer)
    mem_cfg = _memory_config(cfg)
    n = retrieval.build_indices(db, backend=backend, config=mem_cfg)
    rprint(f"[green]Indexed[/green] {n} items.")


@app.command("plugins-list")
def plugins_list(
    manifest: Optional[str] = typer.Option(
        None,
        "--manifest",
        help="Optional path to a plugin manifest (defaults to configs/plugins.yml or $SYMBIONT_PLUGINS_FILE).",
    ),
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Only show enabled plugins."),
    show_config: bool = typer.Option(False, "--show-config", help="Include plugin config payloads."),
):
    """List plugins declared in the Swarm beta manifest."""

    registry = PluginRegistry(manifest_path=manifest) if manifest else PluginRegistry()
    entries = list(registry.entries.values())
    if enabled_only:
        entries = [entry for entry in entries if entry.enabled]

    if not entries:
        rprint("[yellow]No plugins registered.[/yellow]")
        return

    for entry in entries:
        status = "[green]enabled[/green]" if entry.enabled else "[red]disabled[/red]"
        target = entry.module + (f":{entry.attribute}" if entry.attribute else "")
        rprint(f"[bold]{entry.name}[/bold] ({status}) -> {target}")
        if entry.description:
            rprint(f"  {entry.description}")
        if entry.tags:
            rprint(f"  tags: {', '.join(entry.tags)}")
        if show_config and entry.config:
            rprint(f"  config: {entry.config}")


@app.command()
def install_hooks(config_path: str = "./configs/config.yaml", repo: list[str] = typer.Option(None, "--repo", help="One or more repo roots to install Symbiont hooks")):
    """Install a lightweight git pre-push hook that rebuilds RAG indices."""
    cfg=load_config(config_path)
    default_repo = cfg.get("initiative", {}).get("repo_path", ".")
    targets = repo or [default_repo]
    sym_home = Path(__file__).resolve().parents[1]
    cfg_path_abs = os.path.abspath(config_path)
    for root in targets:
        root_path = Path(root).expanduser().resolve()
        hooks_dir = root_path / ".git" / "hooks"
        if not hooks_dir.exists():
            rprint(f"[yellow]Skip {root_path}: .git/hooks not found.")
            continue
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "pre-push"
        script = """#!/usr/bin/env bash
set -euo pipefail
SYMBIONT_HOME="{sym_home}"
CONFIG_PATH="{cfg}"
PYTHON_BIN="${{PYTHON_BIN:-python}}"
cd "$SYMBIONT_HOME"
$PYTHON_BIN -m symbiont.cli rag_reindex --config-path "$CONFIG_PATH"
exit 0
""".format(sym_home=sym_home, cfg=cfg_path_abs)
        hook_path.write_text(script, encoding="utf-8")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)
        rprint(f"[green]Installed pre-push hook at[/green] {hook_path}")

@app.command()
def rag_search(
    query: str,
    k: int = 5,
    config_path: str = "./configs/config.yaml",
    memory_layer: Optional[str] = typer.Option(None, "--memory-layer", help=MEMORY_LAYER_HELP),
):
    cfg = load_config(config_path)
    db = MemoryDB(db_path=cfg["db_path"])
    backend = _memory_layer(cfg, override=memory_layer)
    mem_cfg = _memory_config(cfg)
    res = retrieval.search(db, query, k=k, backend=backend, config=mem_cfg)
    for r in res:
        rprint(f"[cyan]{r['kind']}[/cyan] {r['ref_table']}#{r['ref_id']} score={r['score']:.3f}\n{r['preview']}")


@app.command()
def rag_fetch_external(
    query: str,
    max_items: int = typer.Option(6, "--max-items", "-m", help="Maximum external items to consider"),
    min_relevance: float = typer.Option(0.7, "--min-relevance", "-r", help="Minimum relevance (0-1) required"),
    config_path: str = "./configs/config.yaml",
    memory_layer: Optional[str] = typer.Option(None, "--memory-layer", help=MEMORY_LAYER_HELP),
):
    """
    Pull external research context (arXiv + Semantic Scholar) and merge high-confidence triples into GraphRAG.
    """
    cfg = load_config(config_path)
    db = MemoryDB(db_path=cfg["db_path"])
    db.ensure_schema()
    backend = _memory_layer(cfg, override=memory_layer)
    mem_cfg = _memory_config(cfg)
    result = retrieval.fetch_external_context(
        db,
        query,
        max_items=max_items,
        min_relevance=min_relevance,
        backend=backend,
        config=mem_cfg,
    )
    accepted = result.get("accepted") or []
    claims = result.get("claims") or []
    if not accepted:
        rprint("[yellow]No external items cleared the relevance threshold.[/yellow]")
        return
    rprint(f"[green]Merged {len(claims)} claims from {len(accepted)} external items.[/green]")
    for item in accepted:
        summary = item.get("summary") or ""
        snippet = (summary[:160] + "…") if len(summary) > 160 else summary
        rprint(
            f"[cyan]{item['source']}[/cyan] score={item.get('relevance', 0):.2f} "
            f"[link={item.get('url', '')}]{item.get('title', '')}[/link]\n"
            f"{snippet}\n"
        )


@app.command()
def rag_cache(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Filter cache entries by stored query"),
    max_items: Optional[int] = typer.Option(None, "--max-items", "-m", help="Cache max-items key (required with --clear)"),
    clear: bool = typer.Option(False, "--clear", help="Delete the cache entry for the given query/max-items"),
    clear_all: bool = typer.Option(False, "--clear-all", help="Delete all cached external responses"),
    show_items: bool = typer.Option(False, "--show-items", help="Print cached item titles (up to 5 per entry)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum cache entries to display"),
):
    """Inspect or clear cached external fetch results (stored under data/external/)."""

    cache_dir = Path("data/external")

    if clear_all:
        removed = clear_cache(cache_dir=cache_dir)
        rprint(f"[green]Cleared[/green] {removed} cache file(s).")

    if clear:
        if not query:
            rprint("[red]--clear requires --query to select a cache entry.[/red]")
            raise typer.Exit(1)
        if max_items is None:
            rprint("[red]--clear requires --max-items to compute the cache fingerprint.[/red]")
            raise typer.Exit(1)
        removed = clear_cache(query=query, max_items=max_items, cache_dir=cache_dir)
        if removed:
            rprint(f"[green]Removed cache entry[/green] query='{query}' max_items={max_items}.")
        else:
            rprint(f"[yellow]No cache entry found for[/yellow] query='{query}' max_items={max_items}.")

    entries = list_cache_entries(cache_dir=cache_dir)
    if query:
        entries = [entry for entry in entries if entry.get("query") == query]

    if not entries:
        rprint("[yellow]No cached external responses found.[/yellow]")
        return

    now = time.time()
    for entry in entries[: max(1, limit)]:
        ts = entry.get("updated_at")
        age = ""
        if ts:
            minutes = max(0, int((now - ts) // 60))
            age = f"{minutes}m ago"
        rprint(
            f"[cyan]{entry.get('query')}[/cyan] items={entry.get('item_count', 0)} "
            f"max_items={entry.get('max_items')} path={entry.get('path')} {age}"
        )
        if show_items:
            items = entry.get("items") or []
            for item in items[:5]:
                title = str(item.get("title") or "(no title)")
                source = str(item.get("source") or "-")
                rprint(f"    - [{source}] {title}")
            if len(items) > 5:
                rprint("    …")


@app.command()
def watchers_config(config_path: str = "./configs/config.yaml"):
    """Print normalized initiative watcher configuration."""

    cfg = load_config(config_path)
    configs = build_repo_watch_configs(cfg)
    if not configs:
        rprint("[yellow]No watcher configuration detected.[/yellow]")
        return

    for conf in configs:
        watchers = ", ".join(conf.watchers) or "(none)"
        rprint(
            f"[cyan]{conf.path}[/cyan] watchers=[{watchers}] "
            f"idle={conf.idle_minutes}m git_idle={conf.git_idle_minutes}m "
            f"timer={conf.timer_minutes}m mode={conf.trigger_mode}"
        )
        if conf.verify_rollback:
            rprint("    verify_rollback: true")


@app.command()
def evolution_status(
    state_path: Path = typer.Option(Path("data/evolution/state.json"), "--state-path", "-s", help="Path to evolution state file"),
    recent: int = typer.Option(5, "--recent", "-n", help="Number of recent cycles to display"),
):
    """Inspect evolution history and current meta-learning adjustments."""

    state_path = state_path.expanduser().resolve()
    if not state_path.exists():
        rprint(f"[yellow]No evolution state found at[/yellow] {state_path}")
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rprint(f"[red]Failed to read state:[/red] {exc}")
        return

    adjustments = state.get("meta_adjustments") or {}
    meta_stats = (state.get("meta_learner") or {}).get("stats") or {}
    history = state.get("history") or []
    empty_streak = state.get("empty_streak", 0)

    rprint(f"[bold]State path:[/bold] {state_path}")
    if adjustments:
        rprint("[green]Active adjustments:[/green]", ", ".join(f"{k}={v}" for k, v in adjustments.items()))
    else:
        rprint("[green]Active adjustments:[/green] none (defaults in effect)")

    if meta_stats:
        rprint(
            "[blue]Meta stats:[/blue]",
            ", ".join(f"{k}={meta_stats[k]}" for k in sorted(meta_stats)),
        )
    rprint(f"[cyan]Empty bullet streak:[/cyan] {empty_streak}")

    if history:
        recent_entries = history[-max(1, recent):]
        rprint(f"[bold]Last {len(recent_entries)} cycle(s):[/bold]")
        for entry in recent_entries:
            episode = entry.get("episode_id")
            action = entry.get("action") or "(none)"
            bullets = len(entry.get("bullets") or [])
            reward = entry.get("reward", "n/a")
            rprint(f"- episode={episode} action={action} bullets={bullets} reward={reward}")
@app.command("shadow_report")
def shadow_report(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent clips to display"),
    kind: Optional[str] = typer.Option(None, "--kind", help="Filter by clip kind (e.g. cycle, guard)"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Summarize shadow clips captured during orchestration and guard runs."""
    cfg = load_config(config_path)
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    clip_dir = data_root / "artifacts" / "shadow"
    clip_path = clip_dir / "shadow_clips.jsonl"
    if not clip_path.exists():
        rprint(f"[yellow]No shadow clips found at[/yellow] {clip_path}")
        return

    total = 0
    kind_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    filtered = deque(maxlen=max(limit, 1))

    with clip_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                clip = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            clip_kind = clip.get("kind", "unknown")
            kind_counts[clip_kind] += 1
            for tag in clip.get("tags") or []:
                tag_counts[str(tag)] += 1
            if kind is None or clip_kind == kind:
                filtered.append(clip)

    if not total:
        rprint("[yellow]Shadow log exists but contains no entries.[/yellow]")
        return

    rprint(f"[green]Shadow clips:[/green] total={total} path={clip_path}")
    rprint("[bold]By kind:[/bold]", ", ".join(f"{k}={v}" for k, v in kind_counts.items()))
    if tag_counts:
        top_tags = ", ".join(f"{tag}={count}" for tag, count in tag_counts.most_common(6))
        rprint("[bold]Top tags:[/bold]", top_tags)

    if not filtered:
        rprint("[yellow]No clips match the requested filters.[/yellow]")
        return

    rprint(f"[bold]Last {len(filtered)} clip(s):[/bold]")
    for clip in filtered:
        ts = clip.get("ts")
        dt = datetime.fromtimestamp(ts).isoformat() if isinstance(ts, int) else "n/a"
        clip_kind = clip.get("kind", "unknown")
        tags = ", ".join(clip.get("tags") or []) or "-"
        meta = clip.get("meta") or {}
        payload = clip.get("payload") or {}
        preview = ""
        if clip_kind == "cycle":
            decision = payload.get("decision", {})
            preview = decision.get("action") or ""
        elif clip_kind == "guard":
            analysis = payload.get("analysis", {})
            preview = f"rogue={analysis.get('rogue_score')}"
        preview = (preview or json.dumps(payload)[:80]).strip()
        rprint(f"- [{dt}] {clip_kind} tags={tags} meta={meta} :: {preview}")

@app.command("shadow_curate")
def shadow_curate(
    guard_threshold: float = typer.Option(0.5, help="Rogue score threshold for high-risk guard clips"),
    reward_threshold: float = typer.Option(0.5, help="Reward threshold for low-performing cycles"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum entries per bucket"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write summary JSON to this path"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Rank shadow clips and surface high-signal guard/cycle examples."""
    cfg = load_config(config_path)
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    clip_dir = data_root / "artifacts" / "shadow"
    clip_path = clip_dir / "shadow_clips.jsonl"
    curator = ShadowCurator(clip_path)
    summary = curator.curate(
        guard_threshold=guard_threshold,
        guard_limit=limit,
        reward_threshold=reward_threshold,
        cycle_limit=limit,
        limit=limit,
    )
    meta = summary["meta"]
    counts = meta["counts"]
    total = counts.get("total", 0)
    by_kind = counts.get("by_kind") or {}
    rprint(f"[green]Curated shadow summary[/green] — path={meta['path']} total={total}")
    if by_kind:
        breakdown = ", ".join(f"{k}={v}" for k, v in by_kind.items())
        rprint("[bold]Kind breakdown:[/bold]", breakdown)
    if not total:
        return
    guards_high = summary["guards"]["high"]
    guards_medium = summary["guards"]["medium"]
    cycles_low = summary["cycles"]["low_reward"]
    if guards_high:
        rprint(f"[bold]High-risk guards (>= {guard_threshold}):[/bold]")
        for clip in guards_high:
            script = clip["payload"].get("script_path") or clip["payload"].get("analysis", {}).get("path")
            score = clip.get("rogue_score")
            rprint(f"- rogue={score} script={script} tags={clip.get('tags')}")
    if guards_medium:
        rprint(f"[bold]Medium-risk guards (>= {guard_threshold*0.5:.2f}):[/bold]")
        for clip in guards_medium:
            script = clip["payload"].get("script_path") or clip["payload"].get("analysis", {}).get("path")
            score = clip.get("rogue_score")
            rprint(f"- rogue={score} script={script} tags={clip.get('tags')}")
    if cycles_low:
        rprint(f"[bold]Low-reward cycles (<= {reward_threshold}):[/bold]")
        for clip in cycles_low:
            action = (clip["payload"].get("decision") or {}).get("action")
            reward = clip.get("reward")
            rprint(f"- reward={reward} action={action}")
    if output:
        out_path = Path(output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rprint(f"[green]Summary written to[/green] {out_path}")

@app.command("shadow_label")
def shadow_label(
    guard_threshold: float = typer.Option(0.5, help="Rogue score threshold for high-risk guard clips"),
    reward_threshold: float = typer.Option(0.5, help="Reward threshold for low-performing cycles"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum entries per bucket"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write labeled JSON to this path"),
    ingest: bool = typer.Option(False, "--ingest/--no-ingest", help="Immediately ingest top labels into beliefs"),
    ingest_top: int = typer.Option(5, "--ingest-top", "-k", help="How many labels to ingest when --ingest is set"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Curate and label shadow clips for downstream training."""
    cfg = load_config(config_path)
    labeled, summary, out_path, guard_total, cycle_total, total, history_path = _label_shadow(
        cfg,
        guard_threshold=guard_threshold,
        reward_threshold=reward_threshold,
        limit=limit,
        output=output,
    )
    meta = summary.get("meta", {})
    label_counts = labeled["labels"]["counts"]
    rprint(f"[green]Labeled shadow clips[/green] — path={meta.get('path')} total={total}")
    if label_counts:
        top_labels = ", ".join(f"{label}={count}" for label, count in Counter(label_counts).most_common(10))
        rprint("[bold]Top labels:[/bold]", top_labels)
    else:
        rprint("[yellow]No labels generated (shadow log may be empty).[/yellow]")
    rprint(f"[green]Labeled summary written to[/green] {out_path}")
    rprint(f"[dim]History appended at[/dim] {history_path}")

    summary_line = f"Shadow labels guards={guard_total} cycles={cycle_total} total={total}"
    try:
        db = MemoryDB(db_path=cfg["db_path"])
        db.ensure_schema()
        db.add_artifact(task_id=None, kind="shadow_labels", path=str(out_path), summary=summary_line)
        backend = _memory_layer(cfg)
        mem_cfg = _memory_config(cfg)
        retrieval.build_indices(db, limit_if_new=64, backend=backend, config=mem_cfg)
        if ingest:
            digest = ingest_labels(db, summary=labeled, source_path=out_path, top=ingest_top)
            retrieval.build_indices(db, limit_if_new=64, backend=backend, config=mem_cfg)
            top_labels = ", ".join(f"{label}={count}" for label, count in digest["top"]) or "(none)"
            rprint(f"[green]Ingested labels:[/green] {top_labels}")
    except Exception as exc:  # pragma: no cover - best effort
        rprint(f"[yellow]Failed to record shadow labels artifact:[/yellow] {exc}")

@app.command("shadow_ingest")
def shadow_ingest(
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Path to labeled shadow JSON"),
    top: int = typer.Option(5, "--top", "-k", help="Number of top labels to ingest"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Convert labeled shadow summaries into beliefs and memory messages."""
    cfg = load_config(config_path)
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    if path:
        label_path = Path(path).expanduser().resolve()
        if not label_path.exists():
            rprint(f"[red]Labeled summary not found:[/red] {label_path}")
            raise typer.Exit(1)
    else:
        label_path = latest_labeled_path(data_root)
        if not label_path:
            rprint("[yellow]No labeled shadow summaries found. Run `shadow_label` first.[/yellow]")
            raise typer.Exit(1)

    try:
        labeled = load_labeled_summary(label_path)
    except Exception as exc:
        rprint(f"[red]Failed to read labeled summary:[/red] {exc}")
        raise typer.Exit(1)

    db = MemoryDB(db_path=cfg["db_path"])
    db.ensure_schema()
    digest = ingest_labels(db, summary=labeled, source_path=label_path, top=top)
    backend = _memory_layer(cfg)
    mem_cfg = _memory_config(cfg)
    retrieval.build_indices(db, limit_if_new=64, backend=backend, config=mem_cfg)

    top_labels = ", ".join(f"{label}={count}" for label, count in digest["top"]) or "(none)"
    rprint(f"[green]Ingested shadow labels[/green] source={label_path}")
    rprint(f"[bold]Top labels:[/bold] {top_labels}")
    rprint(
        f"[dim]guards={digest['guard_total']} cycles={digest['cycle_total']} total_labels={len(digest['counts'])}[/dim]"
    )

@app.command("shadow_dashboard")
def shadow_dashboard(
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Path to labeled shadow JSON"),
    output: Optional[str] = typer.Option("systems/ShadowDashboard.md", "--output", "-o", help="Location to write markdown"),
    top: int = typer.Option(10, "--top", "-k", help="Number of top labels to display"),
    include_ingest: bool = typer.Option(False, "--include-ingest", help="Display latest ingest summary in dashboard"),
    to_stdout: bool = typer.Option(False, "--stdout/--no-stdout", help="Print dashboard markdown to stdout"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Render a markdown dashboard for labeled shadow data."""
    cfg = load_config(config_path)
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    if path:
        label_path = Path(path).expanduser().resolve()
        if not label_path.exists():
            rprint(f"[red]Labeled summary not found:[/red] {label_path}")
            raise typer.Exit(1)
    else:
        label_path = latest_labeled_path(data_root)
        if not label_path:
            rprint("[yellow]No labeled shadow summaries found. Run `shadow_label` first.[/yellow]")
            raise typer.Exit(1)

    try:
        labeled = load_labeled_summary(label_path)
    except Exception as exc:
        rprint(f"[red]Failed to read labeled summary:[/red] {exc}")
        raise typer.Exit(1)

    digest = summarize_labels(labeled, top=top) if include_ingest else None
    markdown = render_dashboard(labeled=labeled, ingest_digest=digest, top_limit=top)

    if output:
        out_path = Path(output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        rprint(f"[green]Dashboard written to[/green] {out_path}")

    if to_stdout or not output:
        rprint(markdown)

@app.command("shadow_history")
def shadow_history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of history entries to display"),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON instead of human summary"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write markdown summary to this path"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Show recent shadow label history entries."""
    cfg = load_config(config_path)
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    entries = load_history(data_root, limit=limit)
    if not entries:
        rprint("[yellow]No shadow history entries yet. Run `shadow_label` to generate one.[/yellow]")
        return
    if as_json:
        rprint(json.dumps(entries, indent=2))
    else:
        rprint(f"[green]Shadow history entries[/green] total={len(entries)}")
        for entry in entries:
            ts = entry.get("ts")
            ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts or 0))
            top = ", ".join(f"{label}={count}" for label, count in entry.get("top") or []) or "(none)"
            rprint(
                f"- [{ts_text}] guards={entry.get('guard_total')} cycles={entry.get('cycle_total')} labels={entry.get('total_labels')}\n  top: {top}"
            )

    if output:
        out_path = Path(output).expanduser().resolve()
        lines: List[str] = ["# Shadow History", ""]
        aggregate: Counter[str] = Counter()
        occurrences: Counter[str] = Counter()
        for entry in entries:
            for label, count in entry.get("top") or []:
                aggregate[label] += int(count)
                occurrences[label] += 1
        if aggregate:
            lines.extend([
                "## Aggregated Labels",
                "",
                "| Label | Total Count | Occurrences |",
                "| --- | --- | --- |",
            ])
            for label, total_count in aggregate.most_common(10):
                lines.append(f"| `{label}` | {total_count} | {occurrences[label]} |")
        else:
            lines.extend(["## Aggregated Labels", "", "(none)"])

        lines.extend(["", "## Recent Runs", ""])
        for entry in entries:
            ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("ts", 0)))
            top = ", ".join(f"{label}={count}" for label, count in entry.get("top") or []) or "(none)"
            lines.append(
                f"- **{ts_text}** — guards={entry.get('guard_total')} cycles={entry.get('cycle_total')} labels={entry.get('total_labels')}; top: {top}"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rprint(f"[green]History summary written to[/green] {out_path}")

@app.command("shadow_batch")
def shadow_batch(
    guard_threshold: float = typer.Option(0.5, help="Rogue score threshold for high-risk guard clips"),
    reward_threshold: float = typer.Option(0.5, help="Reward threshold for low-performing cycles"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum entries per bucket"),
    ingest_top: int = typer.Option(5, "--ingest-top", "-k", help="How many labels to ingest"),
    label_output: Optional[str] = typer.Option(None, "--label-output", help="Write labeled JSON to this path"),
    dashboard_output: Optional[str] = typer.Option("systems/ShadowDashboard.md", "--dashboard-output", help="Location to write dashboard markdown"),
    dashboard_top: int = typer.Option(10, "--dashboard-top", help="Number of top labels to display on dashboard"),
    to_stdout: bool = typer.Option(False, "--stdout/--no-stdout", help="Print dashboard markdown to stdout"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to Symbiont config file"),
):
    """Run labeling, ingestion, and dashboard generation in one step."""
    cfg = load_config(config_path)
    labeled, summary, label_path, guard_total, cycle_total, total, history_path = _label_shadow(
        cfg,
        guard_threshold=guard_threshold,
        reward_threshold=reward_threshold,
        limit=limit,
        output=label_output,
    )
    counts = labeled["labels"]["counts"]
    meta = summary.get("meta", {})
    rprint(f"[green]Labeled shadow clips[/green] — path={meta.get('path')} total={total}")
    if counts:
        top_labels = ", ".join(f"{label}={count}" for label, count in Counter(counts).most_common(10))
        rprint("[bold]Top labels:[/bold]", top_labels)
    else:
        rprint("[yellow]No labels generated (shadow log may be empty).[/yellow]")
    rprint(f"[green]Labeled summary written to[/green] {label_path}")
    rprint(f"[dim]History appended at[/dim] {history_path}")

    summary_line = f"Shadow labels guards={guard_total} cycles={cycle_total} total={total}"
    digest = None
    try:
        db = MemoryDB(db_path=cfg["db_path"])
        db.ensure_schema()
        db.add_artifact(task_id=None, kind="shadow_labels", path=str(label_path), summary=summary_line)
        backend = _memory_layer(cfg)
        mem_cfg = _memory_config(cfg)
        retrieval.build_indices(db, limit_if_new=64, backend=backend, config=mem_cfg)
        digest = ingest_labels(db, summary=labeled, source_path=label_path, top=ingest_top)
        retrieval.build_indices(db, limit_if_new=64, backend=backend, config=mem_cfg)
        top_ingested = ", ".join(f"{label}={count}" for label, count in digest["top"]) or "(none)"
        rprint(f"[green]Ingested labels:[/green] {top_ingested}")
    except Exception as exc:  # pragma: no cover - best effort
        rprint(f"[yellow]Failed to ingest labels:[/yellow] {exc}")

    markdown = render_dashboard(labeled=labeled, ingest_digest=digest, top_limit=dashboard_top)
    if dashboard_output:
        dashboard_path = Path(dashboard_output).expanduser().resolve()
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(markdown, encoding="utf-8")
        rprint(f"[green]Dashboard written to[/green] {dashboard_path}")
    if to_stdout or not dashboard_output:
        rprint(markdown)

@app.command()
def scan(path: str = "."):
    info=repo_scan.inspect_repo(path); rprint("[bold]Detected:[/bold]", info["detected"]); rprint("[bold]Quick fixes:[/bold]")
    for s in info["suggestions"][:10]:
        rprint(f"- {s['title']}  [dim](~{s['effort_min']} min)[/dim]\n  {s['details']}")
        for c in s.get("commands",[]): rprint(f"    $ {c}")

@app.command()
def llmtest(
    prompt: str = "List 3 tiny refactors you can do in a repo in 10 minutes.",
    config_path: str = "./configs/config.yaml",
):
    cfg = load_config(config_path)
    limit = 0
    try:
        limit = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit,
        label="cli:llmtest",
        sink_path=data_root / "token_budget" / "cli_llmtest.json",
    )
    out = LLMClient(cfg.get("llm", {})).generate(prompt, budget=budget, label="cli:llmtest")
    rprint(out if out.strip() else "[dim]No output (provider may be 'none').[/dim]")

@app.command()
def cycle(goal: str, config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path); res=Orchestrator(cfg).cycle(goal=goal)
    rprint("[bold]Trace:[/bold]", json.dumps(res["trace"], indent=2))

@app.command()
def tasks(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path)
    with sqlite3.connect(cfg["db_path"]) as c:
        rows=c.execute("SELECT id,episode_id,description,status,assignee_role,created_at FROM tasks ORDER BY id DESC LIMIT 20").fetchall()
    for (i,ep,desc,st,who,ts) in rows: rprint(f"[cyan]#{i}[/cyan] epi={ep} [{st}] @{who} — {desc}")

@app.command()
def episodes(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path)
    with sqlite3.connect(cfg["db_path"]) as c:
        rows=c.execute("SELECT id,title,started_at,status FROM episodes ORDER BY id DESC LIMIT 20").fetchall()
    for (i,title,st,stat) in rows: rprint(f"[magenta]Episode {i}[/magenta] [{stat}] {title} — started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st or 0))}")

@app.command()
def scriptify_last(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path)
    with sqlite3.connect(cfg["db_path"]) as c:
        row=c.execute("SELECT content FROM messages WHERE role='architect' ORDER BY id DESC LIMIT 1").fetchone()
    if not row: rprint("[red]No architect output found.[/red]"); raise typer.Exit(1)
    import json as _j, os as _o
    bullets=_j.loads(row[0]).get("bullets",[])
    if not bullets: rprint("[yellow]No bullets to scriptify.[/yellow]"); raise typer.Exit(1)
    path=scriptify.write_script(bullets, base_dir=os.path.join(os.path.dirname(cfg['db_path']), 'artifacts','scripts'))
    rprint(f"[green]Script saved:[/green] {path}")

@app.command()
def initiative_once(config_path: str = "./configs/config.yaml", force: bool = False):
    """Run the initiative proposer's single pass. If --force, skip watcher checks."""
    cfg=load_config(config_path)
    if force:
        res = initiative.propose_once(cfg, reason="forced")
        rprint("[green]Proposed (forced). Episode:[/green]", res.get("episode_id"))
        return
    ok, reasons, res, target = initiative.run_once_if_triggered(cfg)
    if ok:
        repo_hint = f" repo={getattr(target, 'path', '-')}" if target else ""
        rprint("[green]Proposed.[/green] reasons:", ",".join(reasons), repo_hint, "episode:", res.get("episode_id"))
    else:
        rprint("[yellow]No proposal. Reasons:", ",".join(reasons))

@app.command()
def initiative_daemon(config_path: str = "./configs/config.yaml", poll_seconds: int = 60):
    """Start a simple initiative daemon that polls watchers."""
    cfg=load_config(config_path)
    if not cfg.get("initiative",{}).get("enabled", False):
        rprint("[yellow]initiative.enabled=false[/yellow] — enable it in configs/config.yaml or via UI.")
    initiative.daemon_loop(cfg, poll_seconds=poll_seconds)

@app.command()
def voice_simulate(utterance: str = "symbiont, ship the quick cleanup script", config_path: str = "./configs/config.yaml"):
    """Simulate a voice command without STT. Parses intent and proposes once."""
    cfg=load_config(config_path)
    utt = (utterance or "").lower()
    if any(k in utt for k in ["ship", "propose", "cleanup", "suggest"]):
        res = initiative.propose_once(cfg, reason="voice-sim")
        rprint(f"[green]Proposed.[/green] episode={res.get('episode_id')}")
    else:
        rprint("[yellow]No recognized intent. Try including 'propose' or 'ship'.")

@app.command()
def propose_here(config_path: str = "./configs/config.yaml"):
    """Propose a tiny refactor for the current repo and draft a script."""
    cfg=load_config(config_path)
    res=Orchestrator(cfg).cycle(goal="Propose one 10-minute refactor for my repo")
    rprint("[green]Decision:[/green]", res["decision"]["action"]) 
    rprint("[bold]Trace:[/bold]", json.dumps(res["trace"], indent=2))

@app.command()
def run_script(path: str, yes: bool = False, config_path: str = "./configs/config.yaml"):
    """Run a generated script with a confirmation gate. Saves stdout/stderr as an artifact log."""
    import subprocess, time, os
    cfg = load_config(config_path)
    if not os.path.exists(path):
        rprint(f"[red]Script not found:[/red] {path}")
        raise typer.Exit(1)
    preview = "\n".join(open(path, 'r', encoding='utf-8').read().splitlines()[:40])
    rprint("[bold]About to run script:[/bold]", path)
    rprint("[dim]\n--- preview (first 40 lines) ---\n"+preview+"\n--- end preview ---\n[/dim]")
    if not yes:
        ans = input("Proceed? Type 'yes' to continue: ").strip().lower()
        if ans != 'yes':
            rprint("[yellow]Aborted by user.[/yellow]")
            raise typer.Exit(1)
    ts = int(time.time())
    proc = subprocess.run(["bash", path], capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    logs_dir = os.path.join(os.path.dirname(cfg['db_path']), 'artifacts', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"exec_{ts}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(out)
    rprint(f"[green]Executed with code {proc.returncode}. Log:[/green] {log_path}")
    from .memory.db import MemoryDB
    db=MemoryDB(db_path=cfg['db_path'])
    with db._conn() as c:
        c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'log',log_path,f'Execution log for {os.path.basename(path)}'))
        # audit record
        preview = "\n".join(preview.splitlines()[:20])
        c.execute("INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)", ("proc_run", f"Run script {os.path.basename(path)}", preview, 1))


@app.command()
def rollback_test(script: str, sandbox: str = typer.Option(None, "--sandbox", help="Optional existing sandbox directory"), config_path: str = "./configs/config.yaml"):
    """Validate apply/rollback scripts by running them in a throwaway workspace."""
    import shutil, subprocess

    cfg = load_config(config_path)
    script_path = Path(script).expanduser().resolve()
    if not script_path.exists():
        rprint(f"[red]Script not found:[/red] {script_path}")
        raise typer.Exit(1)
    if "apply_" not in script_path.name:
        rprint("[yellow]Expected an apply_*.sh script for validation.")
    rollback_name = script_path.name.replace("apply_", "rollback_", 1)
    rollback_path = script_path.parent / rollback_name
    if not rollback_path.exists():
        rprint(f"[red]Matching rollback script not found:[/red] {rollback_path}")
        raise typer.Exit(1)

    ctx = _prepare_script_sandbox(script_path, cfg, sandbox)
    dest_repo = ctx["dest_repo"]
    dest_script = ctx["dest_script"]
    dest_rollback = ctx["dest_rollback"]
    sandbox_root = ctx["sandbox_root"]
    cleanup = ctx["cleanup"]

    if not dest_script.exists() or not dest_rollback.exists():
        rprint("[red]Copied sandbox missing scripts. Ensure repo root is correct.")
        if cleanup:
            shutil.rmtree(sandbox_root, ignore_errors=True)
        raise typer.Exit(1)

    def _run(label: str, path_obj: Path):
        proc = subprocess.run(["bash", str(path_obj.resolve())], cwd=dest_repo, capture_output=True, text=True)
        if proc.returncode != 0:
            rprint(f"[red]{label} failed (exit {proc.returncode}).[/red]\n" + (proc.stdout or "") + (proc.stderr or ""))
            if cleanup:
                shutil.rmtree(sandbox_root, ignore_errors=True)
            raise typer.Exit(proc.returncode)
        return proc.stdout, proc.stderr

    rprint(f"[cyan]Sandboxed rollback test under[/cyan] {dest_repo}")
    _run("apply", dest_script)
    _run("rollback", dest_rollback)
    _run("apply(second)", dest_script)
    rprint("[green]Rollback verified idempotent.[/green]")

    if cleanup:
        shutil.rmtree(sandbox_root, ignore_errors=True)


@app.command()
def script_diff(script: str, sandbox: str = typer.Option(None, "--sandbox", help="Optional existing sandbox directory"), config_path: str = "./configs/config.yaml"):
    """Show a git diff preview after applying an apply_*.sh script in a sandbox."""
    import shutil, subprocess

    cfg = load_config(config_path)
    script_path = Path(script).expanduser().resolve()
    if not script_path.exists():
        rprint(f"[red]Script not found:[/red] {script_path}")
        raise typer.Exit(1)

    ctx = _prepare_script_sandbox(script_path, cfg, sandbox)
    dest_repo = ctx["dest_repo"]
    dest_script = ctx["dest_script"]
    sandbox_root = ctx["sandbox_root"]
    cleanup = ctx["cleanup"]

    if not dest_script.exists():
        rprint("[red]Copied sandbox missing script. Ensure repo root is correct.")
        if cleanup:
            shutil.rmtree(sandbox_root, ignore_errors=True)
        raise typer.Exit(1)

    proc = subprocess.run(["bash", str(dest_script.resolve())], cwd=dest_repo, capture_output=True, text=True)
    if proc.returncode != 0:
        rprint(f"[red]apply failed (exit {proc.returncode}).[/red]\n" + (proc.stdout or "") + (proc.stderr or ""))
        if cleanup:
            shutil.rmtree(sandbox_root, ignore_errors=True)
        raise typer.Exit(proc.returncode)

    diff = subprocess.run(["git", "diff", "--no-color"], cwd=dest_repo, capture_output=True, text=True)
    if diff.returncode != 0:
        rprint("[yellow]git diff unavailable in sandbox.")
    else:
        print(diff.stdout)

    if cleanup:
        shutil.rmtree(sandbox_root, ignore_errors=True)


@app.command()
def guard(
    script: str = typer.Option(None, "--script", help="Path to script to analyse"),
    plan: str = typer.Option(None, "--plan", help="Path to plan markdown"),
    json_out: bool = typer.Option(False, "--json", help="Return machine readable output"),
    config_path: str = typer.Option("./configs/config.yaml", "--config-path", help="Path to config"),
):
    """Run guard heuristics against scripts or plans."""

    cfg = load_config(config_path)
    limit = 0
    try:
        limit = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit,
        label="cli:guard",
        sink_path=data_root / "token_budget" / "cli_guard.json",
    )

    report: Dict[str, Any] = {}
    if script:
        report = guard_mod.analyze_script(Path(script))
    if plan:
        plan_path = Path(plan)
        ptext = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""
        plan_report = guard_mod.analyze_plan(ptext, cfg, budget=budget)
        report.setdefault("plan", plan_report)
    if not report:
        rprint("[yellow]Nothing to analyse. Provide --script or --plan.")
        raise typer.Exit(1)
    if json_out:
        print(guard_mod.serialize_report(report))
    else:
        rprint(f"[bold]Rogue score:[/bold] {report.get('rogue_score', 0.0)}")
        if report.get("issues"):
            for issue in report["issues"]:
                rprint(f" - [red]{issue['reason']}[/red] (weight {issue['weight']})")
        plan_flags = (report.get("plan") or {}).get("flags", [])
        if plan_flags:
            for flag in plan_flags:
                rprint(f" - [yellow]{flag}[/yellow]")


@app.command()
def evolve_self(scope: str = typer.Option("planner", "--scope", case_sensitive=False, help="Mutation scope"), strategy: str = typer.Option("promote_diversity", "--strategy", help="Mutation strategy"), config_path: str = "./configs/config.yaml"):
    """Manually trigger a self-evolution proposal."""

    cfg = load_config(config_path)
    engine = MutationEngine(cfg)
    scope = scope.lower()
    if scope != "planner":
        rprint("[yellow]Currently only planner scope is supported.")
        raise typer.Exit(1)
    intent = MutationIntent(
        kind="planner_prompt",
        rationale=f"Manual evolution request with strategy {strategy}",
        details={"strategy": strategy},
    )
    engine.schedule(intent)
    rprint("[green]Evolution intent queued. Inspect data/artifacts/mutations for proposals.[/green]")


@app.command()
def swarm_evolve(
    belief: Optional[str] = typer.Argument(None, help="Belief triple or hint (e.g., 'subject->relation->object')."),
    variants: int = typer.Option(3, "--variants", help="Number of swarm forks"),
    auto: bool = typer.Option(False, "--auto", help="Use latest belief when available"),
    config_path: str = "./configs/config.yaml",
):
    """Run swarm evolution across planner/belief space."""

    cfg = load_config(config_path)
    swarm = SwarmCoordinator(cfg)
    if not swarm.enabled:
        rprint("[yellow]Swarm evolution disabled (set evolution.swarm_enabled=true).")
        raise typer.Exit(1)
    limit = 0
    try:
        limit = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit,
        label="cli:swarm_evolve",
        sink_path=data_root / "token_budget" / "cli_swarm_evolve.json",
    )
    winners = swarm.run(belief or "", variants=variants, auto=auto, apply=True, budget=budget)
    if not winners:
        rprint("[yellow]No swarm winners produced.")
        return
    for variant in winners:
        triple = variant.triple
        rprint(
            f"[green]Winner:[/green] {triple['subject']} -> {triple['relation']} -> {triple['object']} (score {variant.score:.2f})"
        )
        if variant.justification:
            rprint(f"  justification: {variant.justification}")


@app.command()
def swarm_propose(
    belief: Optional[str] = typer.Argument(None, help="Belief triple or hint (e.g., 'subject->relation->object')."),
    variants: int = typer.Option(3, "--variants", help="Number of swarm forks"),
    config_path: str = "./configs/config.yaml",
):
    """Propose swarm variants without applying them."""

    cfg = load_config(config_path)
    swarm = SwarmCoordinator(cfg)
    if not swarm.enabled:
        rprint("[yellow]Swarm evolution disabled (set evolution.swarm_enabled=true).")
        raise typer.Exit(1)
    limit = 0
    try:
        limit = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit,
        label="cli:swarm_propose",
        sink_path=data_root / "token_budget" / "cli_swarm_propose.json",
    )
    winners = swarm.run(belief or "", variants=variants, auto=False, apply=False, budget=budget)
    if not winners:
        rprint("[yellow]No swarm variants produced.")
        return
    for variant in winners:
        triple = variant.triple
        rprint(
            f"[cyan]Proposal:[/cyan] {triple['subject']} -> {triple['relation']} -> {triple['object']} (score {variant.score:.2f})"
        )
        if variant.justification:
            rprint(f"  justification: {variant.justification}")


@app.command()
def swarm_merge_transcripts(config_path: str = "./configs/config.yaml"):
    """Merge outstanding AI peer transcripts into beliefs."""

    cfg = load_config(config_path)
    swarm = SwarmCoordinator(cfg)
    if not swarm.enabled:
        rprint("[yellow]Swarm evolution disabled (set evolution.swarm_enabled=true).")
        raise typer.Exit(1)
    limit = 0
    try:
        limit = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit,
        label="cli:swarm_merge",
        sink_path=data_root / "token_budget" / "cli_swarm_merge.json",
    )
    winners = swarm.merge_from_transcripts(budget=budget)
    if not winners:
        rprint("[yellow]No new transcripts to merge.")
        return
    for variant in winners:
        triple = variant.triple
        rprint(
            f"[green]Merged from transcripts:[/green] {triple['subject']} -> {triple['relation']} -> {triple['object']} (score {variant.score:.2f})"
        )
        if variant.justification:
            rprint(f"  justification: {variant.justification}")


@app.command()
def crew_run(
    crew: str,
    goal: str = typer.Argument(..., help="Goal for the crew to address"),
    config_path: str = "./configs/config.yaml",
    crews_path: str = "./configs/crews.yaml",
    dynamic: bool = typer.Option(False, "--dynamic", help="Generate a crew on the fly for this goal"),
):
    """Run a YAML-configured crew (researcher/planner/critic/etc.)."""

    cfg = load_config(config_path)
    dynamic_path: Optional[Path] = None
    if dynamic:
        from symbiont.orchestration.dynamic import generate_dynamic_crew_yaml

        new_name, dynamic_path = generate_dynamic_crew_yaml(goal, cfg)
        crews_file = dynamic_path
        target_crew = new_name
        rprint(f"[cyan]Dynamic crew generated:[/cyan] {dynamic_path}")
    else:
        crews_file = Path(crews_path)
        if not crews_file.exists():
            rprint(f"[red]Crew config not found:[/red] {crews_file}")
            raise typer.Exit(1)
        target_crew = crew

    registry = AgentRegistry.from_yaml(crews_file)
    db = MemoryDB(db_path=cfg["db_path"])
    runner = CrewRunner(registry, cfg, db)
    try:
        artifact_path = runner.run(target_crew, goal)
    except KeyError as exc:
        rprint(f"[red]{exc}" )
        raise typer.Exit(1)
    rprint(f"[green]Crew {target_crew} finished.[/green] Transcript: {artifact_path}")
    if dynamic_path:
        rprint(f"[blue]Dynamic spec saved at:[/blue] {dynamic_path}")


@app.command()
def dynamics_weaver(
    goal: str,
    config_path: str = "./configs/config.yaml",
):
    """Run the hybrid System Dynamics + ABM "Dynamics Weaver" foresight crew."""

    cfg = load_config(config_path)
    result = run_dynamics_weaver(goal, cfg)
    rprint("[green]Dynamics Weaver completed.[/green]")
    rprint(f"  goal: {result.goal}")
    rprint(f"  risk_score: {result.risk_score:.3f}")
    rprint(f"  artifact: {result.artifact_path}")
    rprint(f"  sd_plot: {result.sd_results.get('plot_path')}")
    rprint(f"  abm_plot: {result.abm_results.get('plot_path')}")


@app.command()
def run_graph(
    graph: str,
    goal: str = typer.Argument(..., help="Goal for the workflow"),
    config_path: str = "./configs/config.yaml",
):
    """Execute a graph-based workflow defined in YAML."""

    cfg = load_config(config_path)
    graph_path = Path(graph)
    if not graph_path.exists():
        rprint(f"[red]Graph spec not found:[/red] {graph_path}")
        raise typer.Exit(1)

    try:
        spec = GraphSpec.from_yaml(graph_path)
    except Exception as exc:
        rprint(f"[red]Failed to load graph:[/red] {exc}")
        raise typer.Exit(1)

    try:
        registry = AgentRegistry.from_yaml(spec.crew_config)
    except Exception as exc:
        rprint(f"[red]Failed to load crew config:[/red] {exc}")
        raise typer.Exit(1)

    db = MemoryDB(db_path=cfg["db_path"])
    runner = GraphRunner(spec, registry, cfg, db, graph_path=graph_path)
    result = runner.run(goal)
    if isinstance(result, dict) and result.get("status") == "paused":
        rprint(
            f"[yellow]Graph paused[/yellow] at node {result.get('last_node', '?')} — awaiting human input."
            f" State saved to: {result.get('state')}"
        )
        return
    rprint(f"[green]Graph completed.[/green] Transcript: {result}")


@app.command()
def graph_resume(
    state: str,
    config_path: str = "./configs/config.yaml",
    graph: Optional[str] = typer.Option(None, "--graph", help="Optional graph spec path override"),
):
    """Resume a previously saved graph run from a state file."""

    state_path = Path(state)
    if not state_path.exists():
        rprint(f"[red]State file not found:[/red] {state_path}")
        raise typer.Exit(1)

    try:
        state_data = json.loads(state_path.read_text())
    except Exception as exc:
        rprint(f"[red]Invalid state file:[/red] {exc}")
        raise typer.Exit(1)

    goal = state_data.get("goal")
    if not goal:
        rprint("[red]State file missing goal")
        raise typer.Exit(1)

    graph_path_str = graph or state_data.get("graph_path")
    if not graph_path_str:
        rprint("[red]State file missing graph path. Pass --graph to resume.")
        raise typer.Exit(1)
    graph_path = Path(graph_path_str)

    try:
        spec = GraphSpec.from_yaml(graph_path)
    except Exception as exc:
        rprint(f"[red]Failed to load graph:[/red] {exc}")
        raise typer.Exit(1)

    crew_config = state_data.get("crew_config")
    if crew_config:
        crew_path = Path(crew_config)
    else:
        crew_path = spec.crew_config

    try:
        registry = AgentRegistry.from_yaml(crew_path)
    except Exception as exc:
        rprint(f"[red]Failed to load crew config:[/red] {exc}")
        raise typer.Exit(1)

    cfg = load_config(config_path)
    db = MemoryDB(db_path=cfg["db_path"])
    runner = GraphRunner(spec, registry, cfg, db, graph_path=graph_path)
    result = runner.run(goal, resume_state=state_path)
    if isinstance(result, dict) and result.get("status") == "paused":
        rprint(
            f"[yellow]Graph paused[/yellow] at node {result.get('last_node', '?')} — awaiting human input."
            f" State saved to: {result.get('state')}"
        )
        return
    rprint(f"[green]Graph resumed.[/green] Transcript: {result}")


@app.command()
def graph_handoff_complete(
    state: str,
    outcome: str = typer.Option(
        "success",
        "--outcome",
        help="Outcome of the handoff (success, failure, block).",
    ),
    result: Optional[str] = typer.Option(
        None,
        "--result",
        help="JSON string describing the resolved result payload",
    ),
    result_file: Optional[str] = typer.Option(
        None,
        "--result-file",
        help="Path to a JSON file describing the resolved result payload",
    ),
    note: Optional[str] = typer.Option(None, "--note", help="Optional note to log with the resolution"),
    config_path: str = "./configs/config.yaml",
):
    """Mark a pending graph handoff as resolved."""

    state_path = Path(state)
    if not state_path.exists():
        rprint(f"[red]State file not found:[/red] {state_path}")
        raise typer.Exit(1)

    if result and result_file:
        rprint("[red]Pass either --result or --result-file, not both.")
        raise typer.Exit(1)

    try:
        state_data = json.loads(state_path.read_text())
    except Exception as exc:
        rprint(f"[red]Invalid state file:[/red] {exc}")
        raise typer.Exit(1)

    handoff = state_data.get("handoff")
    if not handoff:
        rprint("[red]No pending handoff recorded in this state file.")
        raise typer.Exit(1)
    if handoff.get("status") == "resolved":
        rprint("[yellow]Handoff already resolved; nothing to do.")
        raise typer.Exit()

    if result_file:
        payload_path = Path(result_file)
        if not payload_path.exists():
            rprint(f"[red]Result file not found:[/red] {payload_path}")
            raise typer.Exit(1)
        try:
            result_payload = json.loads(payload_path.read_text())
        except Exception as exc:
            rprint(f"[red]Failed to parse result file:[/red] {exc}")
            raise typer.Exit(1)
    elif result:
        try:
            result_payload = json.loads(result)
        except json.JSONDecodeError as exc:
            rprint(f"[red]Invalid JSON in --result:[/red] {exc}")
            raise typer.Exit(1)
    else:
        result_payload = {}

    normalized_outcome = outcome.strip().lower()
    if normalized_outcome not in {"success", "failure", "block"}:
        rprint("[red]Outcome must be one of: success, failure, block")
        raise typer.Exit(1)

    handoff.update(
        {
            "status": "resolved",
            "outcome": normalized_outcome,
            "result": result_payload,
            "resolved_at": int(time.time()),
        }
    )
    if note:
        handoff["note"] = note

    state_data["handoff"] = handoff
    state_data["awaiting_human"] = False
    state_path.write_text(json.dumps(state_data, indent=2), encoding="utf-8")

    cfg = load_config(config_path)
    db = MemoryDB(db_path=cfg["db_path"])
    task_id = handoff.get("task_id")
    if task_id:
        try:
            db.update_task_status(
                int(task_id),
                status=normalized_outcome,
                result=json.dumps(result_payload),
            )
        except Exception as exc:
            rprint(f"[yellow]Warning:[/yellow] failed to update task #{task_id}: {exc}")

    rprint(
        f"[green]Handoff resolved[/green] with outcome '{normalized_outcome}'."
        f" State updated at {state_path}"
    )


@app.command()
def sd_runs(
    limit: int = typer.Option(5, "--limit", help="Number of recent SD runs to display"),
    goal: Optional[str] = typer.Option(None, "--goal", help="Filter results by goal substring"),
    config_path: str = "./configs/config.yaml",
):
    """Show recent system-dynamics simulations recorded in telemetry."""

    cfg = load_config(config_path)
    db_path = cfg["db_path"]
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if goal:
            rows = conn.execute(
                "SELECT goal, label, horizon, timestep, stats_json, plot_path, created_at "
                "FROM sd_runs WHERE goal LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{goal}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT goal, label, horizon, timestep, stats_json, plot_path, created_at "
                "FROM sd_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

    if not rows:
        rprint("[yellow]No system-dynamics runs recorded yet.[/yellow]")
        raise typer.Exit()

    for row in rows:
        stats = {}
        try:
            stats = json.loads(row["stats_json"] or "{}")
        except json.JSONDecodeError:
            stats = {}
        headline = []
        for key in ("rogue", "autonomy", "latency"):
            if key in stats and isinstance(stats[key], dict):
                last_value = stats[key].get("last")
                if isinstance(last_value, (int, float)):
                    headline.append(f"{key}={last_value:.2f}")
        created_at = row["created_at"] or 0
        if created_at:
            created_dt = datetime.fromtimestamp(created_at, timezone.utc)
            created_label = created_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_label = "—"
        rprint(
            f"[bold]{row['goal']}[/bold]\n"
            f"  label={row['label']} horizon={row['horizon']} timestep={row['timestep']:.2f} at {created_label} UTC\n"
            f"  plot={row['plot_path'] or '(none)'}\n"
            f"  stats: {', '.join(headline) if headline else '(none)'}"
        )


@app.command()
def query_web(prompt: str, limit: int = typer.Option(3, "--limit", help="Maximum number of queries"), config_path: str = "./configs/config.yaml"):
    """Guarded search assistant that ingests findings into the belief store."""

    cfg = load_config(config_path)
    oracle = QueryOracle(cfg)
    limit_value = 0
    try:
        limit_value = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit_value = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit_value,
        label="cli:oracle",
        sink_path=data_root / "token_budget" / "cli_oracle.json",
    )
    results = oracle.run(prompt, limit=limit, budget=budget)
    if not results:
        rprint("[yellow]No oracle results (check allowlist or connectivity).")
        return
    for res in results:
        rprint(f"[green]Query:[/green] {res.query}\n  URL: {res.url}\n  Note: {res.note_path}\n  Triples: {res.triples}")


@app.command()
def peer_chat(prompt: str, simulate: bool = typer.Option(False, "--simulate", help="Force simulation only"), config_path: str = "./configs/config.yaml"):
    """Talk to a guarded AI peer (stubbed unless configured)."""

    cfg = load_config(config_path)
    bridge = AIPeerBridge(cfg)
    limit = 0
    try:
        limit = int(cfg.get("max_tokens", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent)
    budget = TokenBudget(
        limit=limit,
        label="cli:peer_chat",
        sink_path=data_root / "token_budget" / "cli_peer_chat.json",
    )
    transcript = bridge.chat(prompt, simulate_only=simulate, budget=budget)
    mode = "simulation" if transcript.simulated else "live"
    rprint(f"[green]{mode} peer response saved:[/green] {transcript.path}")
    print(transcript.response)


@app.command()
def metrics(
    config_path: str = "./configs/config.yaml",
    port: int = typer.Option(8001, "--port", help="Port for Prometheus metrics"),
    interval: int = typer.Option(5, "--interval", help="Polling interval in seconds"),
):
    """Expose Prometheus /metrics for observability dashboards."""

    cfg_path = Path(config_path).expanduser()
    if not cfg_path.exists():
        raise typer.BadParameter(f"Config not found: {cfg_path}")
    rprint(f"[green]Serving metrics on port {port} (interval {interval}s). Press Ctrl+C to stop.")
    start_metrics_server(str(cfg_path), port=port, interval=interval)


@app.command()
def rotate_credential(
    env_key: str,
    new_value: str,
    config_path: str = "./configs/config.yaml",
    env_file: str = typer.Option(None, "--env-file", help="Optional dotenv file to update"),
    actor: str = typer.Option("operator", "--actor", help="Who initiated the rotation"),
):
    """Rotate an environment credential and log the event."""

    cfg = load_config(config_path)
    db_path = cfg.get("db_path", "./data/symbiont.db")
    rotate_env_secret(env_key, new_value, db_path=db_path, actor=actor, persist_path=env_file)
    rprint(f"[green]Rotated credential {env_key}. Audit trail recorded.")


@app.command()
def flow_metrics(
    lead_time: float = typer.Option(0.0, help="Lead time in hours"),
    deploy_freq: float = typer.Option(0.0, help="Deployments per day"),
    mttr: float = typer.Option(0.0, help="Mean time to recovery in minutes"),
    fail_rate: float = typer.Option(0.0, help="Change fail percentage (0-1)"),
    path: str = typer.Option("systems/FlowMetrics.json", "--path", help="Metrics output path"),
):
    """Update flow metrics snapshot for dashboards."""

    payload = {
        "lead_time_hours": round(lead_time, 3),
        "deploy_frequency_per_day": round(deploy_freq, 3),
        "mttr_minutes": round(mttr, 3),
        "change_fail_rate": round(fail_rate, 3),
    }
    update_flow_metrics(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rprint(f"[green]Flow metrics written:[/green] {out_path}")


@app.command()
def github_pr(title: str, body: str = typer.Option("Autopilot proposal", "--body"), head: str = typer.Option("symbiont/autopilot", "--head"), base: str = typer.Option(None, "--base"), dry_run: bool = typer.Option(True, "--dry-run"), config_path: str = "./configs/config.yaml"):
    """Create a guarded pull request for the current repository."""

    cfg = load_config(config_path)
    guard = GitHubGuard(cfg)
    result = guard.create_pull_request(title=title, body=body, head=head, base=base, dry_run=dry_run)
    status = "dry-run" if result.dry_run else "submitted"
    rprint(f"[green]GitHub PR {status}:[/green] {result.message}")
    if result.url:
        rprint(f"URL: {result.url}")


@app.command()
def latest_artifact(type: str = "script", config_path: str = "./configs/config.yaml"):
    """Print path to latest artifact of a given type (plan/script/log)."""
    import sqlite3
    cfg=load_config(config_path)
    with sqlite3.connect(cfg['db_path']) as c:
        row = c.execute("SELECT path FROM artifacts WHERE type=? ORDER BY id DESC LIMIT 1", (type,)).fetchone()
    if not row:
        rprint(f"[yellow]No artifact of type '{type}' found.")
        raise typer.Exit(1)
    print(row[0])

@app.command()
def stt_transcribe(path: str):
    """Transcribe a WAV file using local STT backends if available."""
    from .ports import voice_stt
    txt = voice_stt.transcribe_wav(path)
    if not txt:
        rprint("[yellow]No local STT available or failed to transcribe.")
        raise typer.Exit(1)
    print(txt)

@app.command()
def browse_fetch(url: str, yes: bool = False, config_path: str = "./configs/config.yaml"):
    """Fetch a URL into a note artifact (read-only, allowlist only)."""
    cfg = load_config(config_path)
    import sqlite3, time, os
    preview = f"GET {url}"
    if not yes:
        ans = input(f"About to fetch (net_read): {url}. Type 'yes' to continue: ").strip().lower()
        if ans != 'yes':
            # audit deny
            with sqlite3.connect(cfg['db_path']) as c:
                c.execute("INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)", ("net_read", f"Fetch {url}", preview, 0))
            raise typer.Exit(1)
    try:
        path = browser_port.fetch_to_artifact(url, cfg)
        with sqlite3.connect(cfg['db_path']) as c:
            c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'note',path,f'Note from {url}'))
            c.execute("INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)", ("net_read", f"Fetch {url}", preview, 1))
        rprint(f"[green]Saved note:[/green] {path}")
    except Exception as e:
        rprint(f"[red]Fetch failed:[/red] {e}")
        raise typer.Exit(1)

@app.command()
def graph_add_claim(subject: str, relation: str, obj: str, importance: float = 0.5, source_url: str = "", config_path: str = "./configs/config.yaml"):
    """Add a GraphRAG-lite claim: <subject> <relation> <object> (with optional importance, source)."""
    cfg=load_config(config_path)
    db=MemoryDB(db_path=cfg['db_path']); db.ensure_schema()
    cid, status = graphrag.add_claim(db, subject, relation, obj, importance=importance, source_url=(source_url or None))
    if status == "inserted":
        rprint(f"[green]Added claim #[/green]{cid}: {subject} {relation} {obj}")
    elif status == "merged":
        rprint(f"[green]Merged claim #[/green]{cid}: {subject} {relation} {obj} (confidence vote)")
    else:
        rprint(f"[yellow]Vote recorded.[/yellow] Winner claim #{cid} retained for {subject} {relation} *")

@app.command()
def graph_query(term: str, k: int = 5, config_path: str = "./configs/config.yaml"):
    """Query GraphRAG-lite claims by keyword."""
    cfg=load_config(config_path)
    db=MemoryDB(db_path=cfg['db_path'])
    res = graphrag.query_claims(db, term, limit=k)
    if not res:
        rprint("[yellow]No matching claims.")
        return
    for c in res:
        rprint(f"- {c['subject']} {c['relation']} {c['object']} (imp {c['importance']:.2f}) src={c['source_url'] or '-'}")

if __name__ == "__main__":
    app()

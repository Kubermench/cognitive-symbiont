import os, json, typer, yaml, sqlite3, time, stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from rich import print as rprint
from .orchestrator import Orchestrator
from .memory.db import MemoryDB
from .memory import retrieval
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
from .initiative.watchers import WatchEvent  # noqa: F401 (placeholder for future use)
from .runtime.guard import Guard, Action, Capability  # noqa: F401 (placeholder for future use)
from .ports import browser as browser_port
from .memory import graphrag

app = typer.Typer(help="Cognitive Symbiont — MVP CLI v2.3")

def load_config(path: str = "./configs/config.yaml"):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)


def _prepare_script_sandbox(script_path: Path, cfg: dict, sandbox: str | None):
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
def rag_reindex(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path); db=MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()
    n=retrieval.build_indices(db); rprint(f"[green]Indexed[/green] {n} items.")


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
def rag_search(query: str, k: int = 5, config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path); db=MemoryDB(db_path=cfg["db_path"])
    res=retrieval.search(db, query, k=k)
    for r in res: rprint(f"[cyan]{r['kind']}[/cyan] {r['ref_table']}#{r['ref_id']} score={r['score']:.3f}\n{r['preview']}")

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
    winners = swarm.run(belief or "", variants=variants, auto=auto, apply=True)
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
    winners = swarm.run(belief or "", variants=variants, auto=False, apply=False)
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
    winners = swarm.merge_from_transcripts()
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
):
    """Run a YAML-configured crew (researcher/planner/critic/etc.)."""

    cfg = load_config(config_path)
    crews_file = Path(crews_path)
    if not crews_file.exists():
        rprint(f"[red]Crew config not found:[/red] {crews_file}")
        raise typer.Exit(1)

    registry = AgentRegistry.from_yaml(crews_file)
    db = MemoryDB(db_path=cfg["db_path"])
    runner = CrewRunner(registry, cfg, db)
    try:
        artifact_path = runner.run(crew, goal)
    except KeyError as exc:
        rprint(f"[red]{exc}" )
        raise typer.Exit(1)
    rprint(f"[green]Crew {crew} finished.[/green] Transcript: {artifact_path}")


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

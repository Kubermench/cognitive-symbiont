import os, json, typer, yaml, sqlite3, time
from rich import print as rprint
from .orchestrator import Orchestrator
from .memory.db import MemoryDB
from .memory import retrieval
from .tools import repo_scan, scriptify
from .llm.client import LLMClient
from .initiative import daemon as initiative
from .initiative.watchers import WatchEvent  # noqa: F401 (placeholder for future use)
from .runtime.guard import Guard, Action, Capability  # noqa: F401 (placeholder for future use)
from .ports import browser as browser_port
from .memory import graphrag

app = typer.Typer(help="Cognitive Symbiont — MVP CLI v2.3")

def load_config(path: str = "./configs/config.yaml"):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

@app.command()
def init(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path); db=MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()
    rprint(f"[green]DB initialized at[/green] {cfg['db_path']}")

@app.command()
def rag_reindex(config_path: str = "./configs/config.yaml"):
    cfg=load_config(config_path); db=MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()
    n=retrieval.build_indices(db); rprint(f"[green]Indexed[/green] {n} items.")

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
def llmtest(prompt: str = "List 3 tiny refactors you can do in a repo in 10 minutes."):
    cfg=load_config(); out=LLMClient(cfg.get("llm",{})).generate(prompt)
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
    ok, reasons, res = initiative.run_once_if_triggered(cfg)
    if ok:
        rprint("[green]Proposed.[/green] reasons:", ",".join(reasons), "episode:", res.get("episode_id"))
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
    cid = graphrag.add_claim(db, subject, relation, obj, importance=importance, source_url=(source_url or None))
    rprint(f"[green]Added claim #[/green]{cid}: {subject} {relation} {obj}")

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

import os, json, typer, yaml, sqlite3, time
from rich import print as rprint
from .orchestrator import Orchestrator
from .memory.db import MemoryDB
from .memory import retrieval
from .tools import repo_scan, scriptify
from .llm.client import LLMClient

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

if __name__ == "__main__":
    app()

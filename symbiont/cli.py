"""`sym` command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from . import __version__, checks, config, engine, lint
from .git import GitError, repo_root
from .llm import LLMError, from_config
from .store import Store

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Find small problems in a repo, fix them in an isolated worktree, verify, then hand you a patch.",
)
console = Console()
err = Console(stderr=True)  # warnings and errors, so stdout stays clean for `sym summary`


def _root() -> Path:
    try:
        root = repo_root(Path.cwd())
    except GitError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    problem = lint.config_problem(root)
    if problem:
        err.print(
            f"[yellow]Ignoring the repo's ruff config ({problem}); using Symbiont's lint rules.[/yellow]"
        )
    return root


def _config(root: Path) -> config.Config:
    try:
        return config.load(root)
    except config.ConfigError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


def _plan(root: Path, cfg: config.Config) -> list[engine.Task]:
    try:
        return engine.plan(root, cfg)
    except RuntimeError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


def _engine(root: Path, no_llm: bool) -> engine.Engine:
    cfg = _config(root)
    if no_llm:
        cfg.llm.provider = "none"
    try:
        llm = from_config(cfg.llm)
    except LLMError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    return engine.Engine(
        root,
        cfg,
        llm,
        log=lambda msg: err.print(msg, style="dim", markup=False, highlight=False),
    )


def _report(outcomes: list[engine.Outcome]) -> None:
    made = [o for o in outcomes if o.proposal]
    for o in outcomes:
        if o.proposal:
            console.print(f"[green]#{o.proposal.id}[/green] {o.task.title}")
        else:
            console.print(f"[yellow]skipped[/yellow] {o.task.title}\n  [dim]{o.reason}[/dim]")
    if made:
        console.print("\nReview with [bold]sym show <id>[/bold], then [bold]sym apply <id>[/bold].")
    elif not outcomes:
        console.print("Nothing to do: no lint findings that aren't already proposed.")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


@app.command()
def init() -> None:
    """Write a starter .symbiont.yml at the repo root."""
    path = _root() / config.CONFIG_NAME
    if path.exists():
        console.print(f"{path} already exists")
        raise typer.Exit(1)
    path.write_text(config.TEMPLATE, encoding="utf-8")
    console.print(f"Wrote {path}")


@app.command()
def doctor() -> None:
    """Show what Symbiont detected and whether the LLM is reachable."""
    root = _root()
    cfg = _config(root)
    console.print(f"repo:   {root}")
    console.print(
        f"ruff:   {'ok' if lint.available() else '[red]missing (pip install ruff)[/red]'}"
    )
    found = checks.detect(root, cfg)
    console.print(
        "checks: "
        + (
            ", ".join(f"{c.name} ({c.run})" for c in found)
            or "[yellow]none; fixes are only lint-verified[/yellow]"
        )
    )
    try:
        llm = from_config(cfg.llm)
        if llm is None:
            console.print("llm:    none (only ruff auto-fixes will be proposed)")
        else:
            reply = llm.complete("Reply with the single word: ok", "ping")
            console.print(f"llm:    {llm.name} [green]reachable[/green] ({reply.strip()[:20]!r})")
    except LLMError as exc:
        console.print(f"llm:    [red]{exc}[/red]")


@app.command()
def scan() -> None:
    """List the tasks Symbiont would attempt, without changing anything."""
    root = _root()
    tasks = _plan(root, _config(root))
    if not tasks:
        console.print("No lint findings.")
        return
    table = Table("kind", "task")
    for t in tasks:
        table.add_row(t.kind, t.title)
    console.print(table)


@app.command()
def propose(
    limit: int = typer.Option(3, "--limit", "-n", help="Stop after this many verified proposals."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Only use ruff's deterministic fixes."),
    apply_: bool = typer.Option(
        False, "--apply", help="Apply each verified fix right away so the next one builds on it."
    ),
) -> None:
    """Attempt fixes for scanned tasks; keep only those that verify."""
    root = _root()
    eng = _engine(root, no_llm)
    _report(eng.propose(_plan(root, eng.cfg), limit, apply=apply_))


@app.command()
def ask(
    instruction: str = typer.Argument(..., help="What to change, in plain words."),
    file: Path = typer.Argument(..., help="The file to change."),
) -> None:
    """Ask the LLM for a specific change to one file; it is kept only if it verifies."""
    root = _root()
    target = file.resolve()
    if not target.is_file():
        err.print(f"[red]{file} is not a file[/red]")
        raise typer.Exit(2)
    rel = str(target.relative_to(root))
    eng = _engine(root, no_llm=False)
    if eng.llm is None:
        err.print("[red]ask needs an LLM; set llm.provider in .symbiont.yml[/red]")
        raise typer.Exit(2)
    _report(eng.propose([engine.ask_task(rel, instruction)], limit=1))


@app.command()
def summary(
    status: str = typer.Option(
        "applied", help="Which proposals to include: applied, ready or all."
    ),
) -> None:
    """Print proposals and their evidence as Markdown (used as the automated PR description)."""
    props = [p for p in Store(_root()).all() if status == "all" or p.status == status]
    if not props:
        print("No verified fixes this run.")
        return
    print(
        f"Symbiont made {len(props)} verified fix(es). Each one was made in an isolated worktree;"
    )
    print("it parses, removes the targeted lint findings without adding new ones, and every check")
    print("that passed before still passes.\n")
    for p in props:
        print(f"### {p.title}\n")
        print(f"Files: {', '.join(f'`{f}`' for f in p.files)}\n")
        for line in p.evidence:
            print(f"- {line}")
        print()


@app.command("list")
def list_() -> None:
    """List proposals."""
    props = Store(_root()).all()
    if not props:
        console.print("No proposals yet. Run [bold]sym propose[/bold].")
        return
    table = Table("id", "status", "kind", "title")
    colors = {"ready": "green", "applied": "cyan", "undone": "yellow", "dropped": "dim"}
    for p in props:
        table.add_row(str(p.id), f"[{colors.get(p.status, '')}]{p.status}[/]", p.kind, p.title)
    console.print(table)


@app.command()
def show(pid: int) -> None:
    """Show a proposal's evidence and diff."""
    store = Store(_root())
    try:
        p = store.get(pid)
    except KeyError as exc:
        err.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[bold]#{p.id} {p.title}[/bold]  ({p.status})")
    for line in p.evidence:
        console.print(f"  [green]✓[/green] {line}")
    console.print(Syntax(store.patch(pid), "diff", theme="ansi_dark", background_color="default"))


def _transition(action: str, pid: int) -> None:
    store = Store(_root())
    try:
        p = getattr(store, action)(pid)
    except (KeyError, ValueError, GitError) as exc:
        err.print(f"[red]{exc.args[0] if exc.args else exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"#{p.id} {p.status}: {p.title}")


@app.command()
def apply(pid: int) -> None:
    """Apply a proposal to your working tree (nothing is committed)."""
    _transition("apply", pid)


@app.command()
def undo(pid: int) -> None:
    """Reverse an applied proposal."""
    _transition("undo", pid)


@app.command()
def drop(pid: int) -> None:
    """Discard a proposal you don't want."""
    _transition("drop", pid)


if __name__ == "__main__":
    app()

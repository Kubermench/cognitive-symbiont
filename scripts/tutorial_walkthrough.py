#!/usr/bin/env python3
"""Interactive CLI walkthrough for Symbiont newcomers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import List

import typer

app = typer.Typer(add_completion=False, help="Symbiont tutorial helper")


@dataclass
class Step:
    title: str
    command: str


STEPS: List[Step] = [
    Step("Install dependencies", "pip install -r requirements.txt"),
    Step("Start Homebase", "streamlit run app.py"),
    Step("Force initiative cycle", "python -m symbiont.cli initiative_once --force"),
    Step("Inspect artifacts", "ls data/artifacts/scripts"),
    Step("Run sandbox rollback test", "python -m symbiont.cli rollback-test data/artifacts/scripts/apply_*.sh"),
]


def _maybe_open_docs() -> None:
    if shutil.which("open"):
        subprocess.run(["open", "README.md"], check=False)


@app.command()
def walkthrough() -> None:
    typer.secho("Cognitive Symbiont guided walkthrough", fg=typer.colors.CYAN)
    _maybe_open_docs()
    for idx, step in enumerate(STEPS, start=1):
        typer.echo(f"\n[{idx}/{len(STEPS)}] {step.title}")
        typer.echo(f"    → try: {step.command}")
        if not typer.confirm("Mark complete?", default=True):
            typer.echo("Stopping tutorial.")
            return
    typer.secho("\nTutorial finished. Explore the UI, CLI, and VS Code extension next!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

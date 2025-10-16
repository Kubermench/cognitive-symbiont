from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
from typing import Any

import typer
import yaml
from rich import print as rprint

from ..config import SymbiontConfig


@dataclass
class InitOptions:
    lite: bool = False
    target_path: Path = Path("./configs/config.local.yaml")
    force: bool = False
    non_interactive: bool = False


def detect_llm() -> dict[str, Any]:
    ollama_available = shutil.which("ollama") is not None
    if ollama_available:
        return {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "local",
            "hint": "Detected `ollama` on PATH; defaulting to local provider.",
        }
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "mode": "cloud",
        "hint": "No local provider detected; defaulting to cloud `openai`.",
    }


def detect_gpu() -> bool:
    if shutil.which("nvidia-smi"):
        return True
    if Path("/proc/driver/nvidia/version").exists():
        return True
    if shutil.which("rocminfo"):
        return True
    return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


def load_base_config(lite: bool) -> SymbiontConfig:
    base_file = Path("configs/config-lite.yaml" if lite else "configs/config.yaml")
    if not base_file.exists():
        raise FileNotFoundError(f"Base configuration file missing: {base_file}")
    with base_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return SymbiontConfig.model_validate(data)


def ensure_dirs(cfg: SymbiontConfig) -> None:
    db_parent = Path(cfg.db_path).expanduser().resolve().parent
    db_parent.mkdir(parents=True, exist_ok=True)
    data_root = cfg.data_root or str(db_parent)
    Path(data_root).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def interactive_adjustments(cfg: SymbiontConfig) -> SymbiontConfig:
    rprint("[bold cyan]Symbiont setup wizard[/bold cyan]")
    cfg.env = typer.prompt("Environment tag", default=cfg.env)
    cfg.db_path = typer.prompt("SQLite database path", default=cfg.db_path)

    llm_provider = typer.prompt("Primary LLM provider", default=cfg.llm.provider)
    cfg.llm.provider = llm_provider
    cfg.llm.model = typer.prompt("Primary model name", default=cfg.llm.model)
    cfg.llm.mode = typer.prompt("LLM mode (local/cloud/hybrid)", default=cfg.llm.mode)
    cfg.llm.timeout_seconds = typer.prompt(
        "LLM timeout seconds", default=cfg.llm.timeout_seconds, type=int
    )

    if typer.confirm("Enable initiative daemon?", default=cfg.initiative.enabled):
        cfg.initiative.enabled = True
        cfg.initiative.repo_path = typer.prompt(
            "Repository path for initiative actions", default=cfg.initiative.repo_path
        )
    else:
        cfg.initiative.enabled = False

    cfg.foresight.enabled = typer.confirm(
        "Enable foresight hunts (requires extras)?", default=cfg.foresight.enabled
    )

    cfg.tools.network_access = typer.confirm(
        "Allow network access for tools?", default=cfg.tools.network_access
    )

    cfg.guard.auto_approve_safe = typer.confirm(
        "Auto-approve low-risk actions?", default=cfg.guard.auto_approve_safe
    )

    return cfg


def run_init_wizard(options: InitOptions) -> Path:
    cfg = load_base_config(options.lite)
    hints: list[str] = []

    llm_defaults = detect_llm()
    gpu_available = detect_gpu()
    cfg.llm.provider = llm_defaults["provider"]
    cfg.llm.model = llm_defaults["model"]
    cfg.llm.mode = llm_defaults["mode"]
    hints.append(llm_defaults["hint"])
    hints.append(
        "GPU detected; higher token budgets are available."
        if gpu_available
        else "No GPU detected; keeping conservative defaults."
    )

    if options.lite:
        cfg.foresight.enabled = False
        cfg.evolution.enabled = False
        cfg.initiative.enabled = False
        cfg.tools.network_access = False

    if not options.non_interactive:
        cfg = interactive_adjustments(cfg)

    try:
        cfg = SymbiontConfig.model_validate(cfg.dump())
    except Exception as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(f"Configuration validation failed: {exc}") from exc

    ensure_dirs(cfg)

    target = options.target_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not options.force:
        raise typer.BadParameter(
            f"Target config already exists at {target}. Use --force to overwrite."
        )

    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.dump(), handle, sort_keys=False)

    summary = {
        "env": cfg.env,
        "db_path": cfg.db_path,
        "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model, "mode": cfg.llm.mode},
        "initiative_enabled": cfg.initiative.enabled,
        "foresight_enabled": cfg.foresight.enabled,
        "network_access": cfg.tools.network_access,
        "gpu_detected": gpu_available,
    }

    rprint("[green]Configuration written to[/green]", target)
    rprint("[bold]Summary:[/bold]", summary)
    for hint in hints:
        rprint(f"- {hint}")

    return target

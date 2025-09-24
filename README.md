# Cognitive Symbiont — MVP v2.4
This bundle includes RAG memory, a local-first LLM adapter, a Streamlit homebase, script generation, initiative daemon (file/git/timer), Beliefs v1, and stubs for Voice/IDE.

## Initiative & Watchers (v2.4)
- Daemon: `python -m symbiont.cli initiative_daemon` (off by default).
- Watchers: file idle, git idle, and timer. Default trigger mode requires both idle and timer.
- Proposals: generates a one-step plan and a non-executing apply script plus a rollback script.

### Quick Start
- In the UI sidebar, toggle Initiative → Enabled and set minutes, or run once via button “Propose Now (ignore watchers)”.
- CLI: `python -m symbiont.cli initiative_once --force` to draft a proposal immediately.
- Rollback: a `rollback_*.sh` script is saved next to `apply_*.sh`.

## New Utilities (v2.4+)
- Guarded actions: UI and CLI confirmations for script writes and runs; actions recorded in `audits`.
- Intent checkpoint: save an "intent summary" (Settings) to align future proposals.
- Browser (read-only): allowlisted fetches into `data/artifacts/notes/` with source URLs; `tools.network_access` must be true.
- GraphRAG-lite: add/query simple claims to inform plans.
- MCP server (minimal): newline-delimited JSON-RPC over TCP for ports.

### Commands
- Graph claims:
  - Add: `python -m symbiont.cli graph_add_claim "ruff" "preferred_over" "black" --importance 0.8 --source_url https://docs.astral.sh/ruff/`
  - Query: `python -m symbiont.cli graph_query "ruff" -k 5`
- Browser fetch (guarded): `python -m symbiont.cli browse_fetch "https://docs.astral.sh/ruff/"`
- MCP server: `python -m symbiont.ports.mcp_server --host 127.0.0.1 --port 8765`

## Autopilot (optional, local)

- Local script: `scripts/autopilot.sh` proposes → applies latest script (guarded `--yes`) → runs Sandbox CI if present → commits to `symbiont/autopilot`.
- Cron example (Mon–Fri, every 30 min):
  - `crontab -e`
  - `*/30 9-18 * * 1-5 cd /path/to/repo && . .venv/bin/activate && ./scripts/autopilot.sh >> data/artifacts/logs/autopilot.log 2>&1`
- GitHub Actions (auto‑PR): `.github/workflows/symbiont_autopilot.yml` creates a nightly PR with changes. Production still manual.

## How to See It Working (Detailed)

1) Install
- Python 3.10+ and git installed.
- `pip install -r requirements.txt`

2) Homebase UI
- Start: `streamlit run app.py`
- In sidebar:
  - LLM: choose `ollama` if available (optional).
  - RAG: rebuild index.
  - Initiative: Enable, set Idle/Timer, click Start Daemon (optional), or “Propose Now”.
  - Beliefs: add a statement + confidence; shows as “Assumptions” in new plans.
- Main view:
  - Click “Run Cycle” with a goal → plan + bullets; click “Generate Script from Bullets” if shown.
  - Artifacts: pick a `plan_*.md` or `apply_*.sh`. For scripts, use “Run Safely (guarded)” to confirm and execute; logs save under `data/artifacts/logs/`.

3) CLI (shell alias optional)
- Set alias (optional): `alias sym="python -m symbiont.cli"`
- Propose here: `sym propose_here`
- Initiative once: `sym initiative_once --force`
- Daemon: `sym initiative_daemon`
- Run a script with confirm: `sym run_script ./data/artifacts/scripts/apply_*.sh`

4) BigKit v3.0-alpha (multi-tab)
- Start: `streamlit run bigkit_v3alpha/app.py`
- Tabs: Cycles (run), Memory (browse), Beliefs (add/list), Agency (one-click proposal), Ports (stubs).

5) VS Code Port (stub)
- Open `vscode/symbiont-vscode` in VS Code.
- `npm install` then `npm run build` or press F5 (Extension Dev Host).
- Run command: “Symbiont: Propose Tiny Refactor”.

6) Voice-mini (stubs)
- TTS: macOS `say` utility via `symbiont/ports/voice.py` (best effort).
- STT: `symbiont/ports/voice_stt.py` tries `whisper` or `whisper.cpp` if installed; otherwise no-op.

## Safety & Guardrails
- Scripts never auto-execute. UI and CLI require explicit confirmation for `run_script`.
- Rollback scripts are generated for quick recovery.
- Recommend: keep a clean git working tree; consider `git stash` before applying changes.

# Cognitive Symbiont — MVP v3.5 Autonomous Edition
This bundle includes RAG memory, a local-first LLM adapter, a Streamlit homebase, script generation, initiative daemon (file/git/timer), Beliefs v1, and stubs for Voice/IDE.

## Initiative & Watchers (v3.5)
- Daemon: `python -m symbiont.cli initiative_daemon` (off by default).
- Watchers: file idle, git idle, and timer. Default trigger mode requires both idle and timer.
- Multi-repo: configure `initiative.watch_targets` inline or point `initiative.watch_config_path` to `configs/watch_targets.yaml` for per-repo triggers, idle timers, and `verify_rollback` flags.
- Auto hooks: `python -m symbiont.cli install-hooks --repo <path>` installs a git `pre-push` hook that runs `sym rag_reindex` to keep RAG fresh after each push.
- Autonomy mode: set `initiative.autonomy_mode` (`manual|guarded|full`) to control how Eternal cycles behave.
- Proposals: generates a one-step plan and a non-executing apply script plus a rollback script (one per repo configured).

### Quick Start (non-technical friendly)
1. Open a terminal in the project folder and run `streamlit run app.py`.
2. In the web app, click **Run a cycle**. Symbiont will write a short plan on the left and a matching script on the right.
3. Read the plan. If it sounds good, press **Run safely (guarded)** → **Confirm**. Symbiont runs the script, saves the log, and shows a success message.
4. Need a web reference? Use **Settings → Browser → Fetch to notes** and confirm; the note appears in “Sources” for the next plan.

Symbiont never changes files until you press **Confirm**, and every action (plans, scripts, logs) is written under `data/artifacts/`.

### Power Quick Start
- In the UI sidebar, toggle Initiative → Enabled and set minutes, or run once via button “Propose Now (ignore watchers)”.
- CLI: `python -m symbiont.cli initiative_once --force` to draft a proposal immediately.
- Rollback: a `rollback_*.sh` script is saved next to `apply_*.sh`.

### Hybrid LLM setup (optional)
- Default mode is **local**: Symbiont calls Ollama (e.g., `phi3:mini`).
- To enable cloud fallback, edit `configs/config.yaml`:
  ```yaml
  llm:
    mode: hybrid          # local | cloud | hybrid
    hybrid_threshold_tokens: 800
    cloud:
      provider: openai
      model: gpt-4o-mini
      api_key_env: OPENAI_API_KEY
  ```
- Export your API key (`export OPENAI_API_KEY=sk-…`). Prompts below the threshold run locally; longer ones automatically escalate to the cloud model and fall back to local if the cloud call fails.
- **Crew orchestration (experimental)** – Define agents and crews in `configs/crews.yaml`. Run a crew with `python -m symbiont.cli crew_run quick_fix "Tidy the repo"`. Results land under `data/artifacts/crews/<crew>/crew_<timestamp>.json`; scripts still require approval via the guard dialogs.
- **Graph workflows (experimental)** – Describe branching workflows in YAML (e.g., `configs/graphs/quick_fix.yaml`) and run them with `python -m symbiont.cli run_graph configs/graphs/quick_fix.yaml "Fix lint"`. If a run pauses, resume via `python -m symbiont.cli graph_resume data/evolution/graph_state_<ts>.json`.

## New Utilities (v3.5)
- Guarded actions: UI and CLI confirmations for script writes and runs; actions recorded in `audits`.
- Reflection + mutation: each cycle feeds `data/evolution/state.json`; `sym evolve_self --scope planner` queues guarded prompt tweaks (≤5% diff) saved under `data/artifacts/mutations/` after triple sandbox validation.
- Swarm evolution: enable `evolution.swarm_enabled=true` to spawn parallel belief variants, score via peer chats, and merge consensus claims (`sym swarm_evolve "belief: UI->prefers->dark_mode"`).
- Rollback sandbox: `python -m symbiont.cli rollback-test data/artifacts/scripts/apply_*.sh` runs apply→rollback→apply in a temp checkout to guarantee idempotence before human approval.
- Diff preview: `python -m symbiont.cli script_diff data/artifacts/scripts/apply_*.sh` renders a git diff without touching the working tree; the VS Code command “Symbiont: Show Proposal Diff” mirrors the output in a webview.
- Intent checkpoint: save an "intent summary" (Settings) to align future proposals.
- Browser (read-only): allowlisted fetches into `data/artifacts/notes/` with source URLs; `tools.network_access` must be true.
- GraphRAG-lite: add/query simple claims, resolve conflicts with confidence voting, and visualise triples in BigKit.
- Voice: cross-platform TTS via `pyttsx3`/`espeak` on Linux/Windows and `say` on macOS; STT falls back to `vosk-transcriber` if Whisper is absent.
- Query Oracle: `sym query_web "React hooks" --limit 3` plans allowlisted searches, stores notes in `data/artifacts/notes/`, and ingests belief triples.
- AI peer bridge: `sym peer_chat --prompt "Summarise migration risks"` simulates external conversations (stubbed unless configured).
- GitHub guard: `sym github_pr --title "Symbiont autopilot" --dry-run false` opens PRs under allowlisted owners using PyGitHub and stored tokens.
- MCP server (minimal) + CLI `install-hooks` for RAG automation.
- **System dynamics foresight (v4.1)**: run `python -m symbiont.cli run_graph configs/graphs/foresight_sd.yaml "Stress-test autonomy guard"` to trigger the new `dynamics_scout → sd_modeler → strategist` crew. A baseline simulation is recorded under `data/artifacts/graphs/simulations/`, plotted, and summarised in the BigKit governance dashboard (look for **Recent SD runs**) so you can inspect autonomy/rogue trajectories before executing scripts. Telemetry for each projection also lands in the SQLite `sd_runs` table; inspect it via `python -m symbiont.cli sd-runs --limit 3`.
- **Dynamics Weaver (hybrid SD+ABM)**: `python -m symbiont.cli dynamics-weaver "Model rogue drift in swarm"` runs the new hybrid crew that couples the macro SD engine with an agent-based noise layer. Results (plots + JSON) live under `data/artifacts/crews/dynamics_weaver/` and the summary appears both in the CLI output and the SD telemetry tables.
- **Rotating credentials & peer identities**: cloud LLM keys reload automatically (configurable `refresh_seconds`, default 1h) and swarm peers now carry UUIDs so governance logs can trace each agent’s input without long-lived secrets.
- **Pub/Sub hooks for initiative**: enable `initiative.pubsub` in `configs/config.yaml` to broadcast daemon events (memory JSONL or Redis) so distributed watchers can react without polling.
- **Human-in-loop graph controls**: set `ui.pause_between_nodes=true` to pause crews between nodes; edit the pending state in BigKit’s dashboard and resume the graph directly from the UI.

### Commands
- Graph claims:
  - Add: `python -m symbiont.cli graph_add_claim "ruff" "preferred_over" "black" --importance 0.8 --source_url https://docs.astral.sh/ruff/`
  - Query: `python -m symbiont.cli graph_query "ruff" -k 5`
- Browser fetch (guarded): `python -m symbiont.cli browse_fetch "https://docs.astral.sh/ruff/"`
- MCP server: `python -m symbiont.ports.mcp_server --host 127.0.0.1 --port 8765`

## Autopilot (optional, local)

- Local script: `scripts/autopilot.sh` proposes → applies latest script (guarded `--yes`) → runs Sandbox CI if present → commits to `symbiont/autopilot`.
- Rollback validation runs automatically before applying; failures abort the autopilot loop.
- Eternal mode: `./scripts/autopilot.sh --mode=eternal --cycles=5 --autonomy=guarded` loops with rogue-score checks (`sym guard --script ...`) and kill-switch thresholds.
- Swarm mode: append `--swarm=true` to autopilot to fork mini-agents, evaluate with peers, and merge consensus updates.
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
- Install git hooks: `sym install_hooks --repo .`
- Sandbox rollback: `sym rollback_test ./data/artifacts/scripts/apply_*.sh`
- Diff preview: `sym script_diff ./data/artifacts/scripts/apply_*.sh`
- Self evolution: `sym evolve_self --scope planner`
- Query Oracle: `sym query_web "Async retries"`
- Peer chat: `sym peer_chat --prompt "Which tests to add?"`
- Guard scan: `sym guard --script ./data/artifacts/scripts/apply_latest.sh`
- GitHub PR: `sym github_pr --title "Symbiont autopilot" --dry-run false`

4) BigKit v3.0-alpha (multi-tab)
- Start: `streamlit run bigkit_v3alpha/app.py`
- Tabs: Cycles (run), Memory (browse), Beliefs (statement + GraphRAG triples with graph + edit actions), Agency (one-click proposal), Ports (stubs).
- Governance tab surfaces rogue-score metrics, reflection history, and exports compliance reports.

5) VS Code Port (stub)
- Open `vscode/symbiont-vscode` in VS Code.
- `npm install` then `npm run build` or press F5 (Extension Dev Host).
- Run command: “Symbiont: Propose Tiny Refactor”.

6) Voice-mini (stubs)
- TTS: macOS `say`, Linux `espeak`/Windows SAPI via `pyttsx3` fallback in `symbiont/ports/voice.py`.
- STT: `symbiont/ports/voice_stt.py` tries `whisper`, `whisper.cpp`, then `vosk-transcriber` if available; otherwise no-op.

## Safety & Guardrails
- Scripts never auto-execute. UI and CLI require explicit confirmation for `run_script`.
- Rollback scripts are generated for quick recovery.
- Recommend: keep a clean git working tree; consider `git stash` before applying changes.
- Use `sym rollback_test` to validate scripts and `Symbiont: Show Proposal Diff` in VS Code for a visual check before approval.

## Onboarding & Tutorial
- Quick orientation: `python scripts/tutorial_walkthrough.py walkthrough` walks through setup, initiative, artifact inspection, and rollback validation.
- LLM fallback: if the primary provider fails, `scripts/offline_llm.py` echoes a safe placeholder so initiative cycles stay responsive; configure via `llm.fallback` in `configs/config.yaml`.
- Autonomy guardrails: use `sym guard --script` before approving scripts and monitor Eternal cycles with BigKit’s governance dashboard.

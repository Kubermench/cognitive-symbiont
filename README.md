# Cognitive Symbiont — MVP v3.6 Autonomous Edition
This bundle includes RAG memory, a local-first LLM adapter, a Streamlit homebase, script generation, initiative daemon (file/git/timer), Beliefs v1, and stubs for Voice/IDE.

## Initiative & Watchers (v3.6)
- Daemon: `python -m symbiont.cli initiative_daemon` (off by default).
- Watchers: file idle, git idle, and timer. Default trigger mode requires both idle and timer.
- Multi-repo: configure `initiative.watch_targets` inline or point `initiative.watch_config_path` to `configs/watch_targets.yaml` for per-repo triggers, idle timers, and `verify_rollback` flags.
- Auto hooks: `python -m symbiont.cli install-hooks --repo <path>` installs a git `pre-push` hook that runs `sym rag_reindex` to keep RAG fresh after each push.
- Autonomy mode: set `initiative.autonomy_mode` (`manual|guarded|full`) to control how Eternal cycles behave.
- Proposals: generates a one-step plan and a non-executing apply script plus a rollback script (one per repo configured).

### Quick Start (non-technical friendly)
1. (Optional) install the CLI locally: `pip install .` then run commands via the new `sym` entry point.
2. Open a terminal in the project folder and run `streamlit run app.py`.
3. Want a terminal-only smoke test? Run `sym quickstart` to execute a single guarded cycle (external web fetches are skipped by default) and preview the generated plan.
4. In the web app, click **Run a cycle**. Symbiont will write a short plan on the left and a matching script on the right.
5. Read the plan. If it sounds good, press **Run safely (guarded)** → **Confirm**. Symbiont runs the script, saves the log, and shows a success message.
6. Need a web reference? Use **Settings → Browser → Fetch to notes** and confirm; the note appears in “Sources” for the next plan.

Symbiont never changes files until you press **Confirm**, and every action (plans, scripts, logs) is written under `data/artifacts/`.

### Power Quick Start
- In the UI sidebar, toggle Initiative → Enabled and set minutes, or run once via button “Propose Now (ignore watchers)”.
- CLI: `python -m symbiont.cli initiative_once --force` to draft a proposal immediately.
- Rollback: a `rollback_*.sh` script is saved next to `apply_*.sh`.

### Hybrid LLM setup (optional)
- Default mode is **local**: Symbiont calls Ollama (e.g., `phi3:mini`). Make sure the `ollama` binary is installed and on your `PATH`; Symbiont logs a warning and falls back to the secondary provider if it cannot find the CLI.
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
- Set `max_tokens` in `configs/config.yaml` to enforce a shared token budget; orchestration cycles, crew runs, and graph executions will refuse further LLM calls once the budget is exhausted, and paused graphs carry the usage forward when you resume them.
- **Plugin manifest (Swarm preview)** – Copy `configs/plugins.example.yml` to `configs/plugins.yml` and flip `enabled: true` for entries you want Symbiont to auto-load. Each plugin entry names a Python module and optional callable, letting community crews stay modular without inflating the base install. Override the manifest path via `$SYMBIONT_PLUGINS_FILE` and inspect active entries anytime with `sym plugins-list`.
- **Crew orchestration (experimental)** – Define agents and crews in `configs/crews.yaml`. Run a crew with `python -m symbiont.cli crew_run quick_fix "Tidy the repo"`. Results land under `data/artifacts/crews/<crew>/crew_<timestamp>.json`; scripts still require approval via the guard dialogs.
- **Graph workflows (experimental)** – Describe branching workflows in YAML (e.g., `configs/graphs/quick_fix.yaml`) and run them with `python -m symbiont.cli run_graph configs/graphs/quick_fix.yaml "Fix lint"`. If a run pauses, resume via `python -m symbiont.cli graph_resume data/evolution/graph_state_<ts>.json`. Use `graph.parallel` groups to force sequential cohorts (every node in the array runs once before advancing, but any failure or block exits early to whatever `on_failure`/`on_block` targets you set). When an agent emits a handoff, capture the follow-up work in SQLite and unblock the graph with `python -m symbiont.cli graph-handoff-complete <state.json> --outcome success --result '{"verdict": "ok"}'` once the human task is finished.
- **Handoff notifications** – Optionally add `notifications.handoff_webhook_url=https://...` in `configs/config.yaml` to receive immediate POST callbacks whenever a graph blocks on human input. Lock down outbound hooks with `notifications.allow_domains` so Slack/PagerDuty integrations stay allowlisted.

## New Utilities (v3.6)
- Guarded actions: UI and CLI confirmations for script writes and runs; actions recorded in `audits`.
- Reflection + mutation: each cycle feeds `data/evolution/state.json`; `sym evolve_self --scope planner` queues guarded prompt tweaks (≤5% diff) saved under `data/artifacts/mutations/` after triple sandbox validation.
- Swarm evolution: enable `evolution.swarm_enabled=true` to spawn parallel belief variants, score via peer chats, and merge consensus claims (`sym swarm_evolve "belief: UI->prefers->dark_mode"`).
- Bias hunter: run `sym bias-check "topic or claim"` to weigh supportive vs. contrarian evidence and surface mitigation advice for confirmation bias.
- Rollback sandbox: `python -m symbiont.cli rollback-test data/artifacts/scripts/apply_*.sh` runs apply→rollback→apply in a temp checkout to guarantee idempotence before human approval.
- External RAG bridge: `python -m symbiont.cli rag-fetch-external "agentic AI"` hits arXiv + Semantic Scholar, caches under `data/external/`, and merges high-confidence triples into GraphRAG before the next cycle. Flip `retrieval.external.enabled` to `true` in `configs/config.yaml` to make this automatic for every orchestration cycle or graph run, and use `python -m symbiont.cli rag-cache` to inspect/clear cached fetches.
- Reflector meta-learning: cycle rewards quietly tune the planner repeat/empty thresholds so evolution triggers sooner when diversity or bullet quality dips.
- Evolution status: `python -m symbiont.cli evolution-status -n 3` prints live meta adjustments and the last few cycle outcomes, pulling from `data/evolution/state.json`.
- Watcher config: `python -m symbiont.cli watchers-config` shows the normalized repo watcher settings, including idle timers, trigger mode, and rollback verification.
- Diff preview: `python -m symbiont.cli script_diff data/artifacts/scripts/apply_*.sh` renders a git diff without touching the working tree; the VS Code command “Symbiont: Show Proposal Diff” mirrors the output in a webview.
- Rollbacks require an explicit `SYM_ROLLBACK_FORCE=1` environment variable to run destructive git resets; without it the guard refuses to proceed.
- Intent checkpoint: save an "intent summary" (Settings) to align future proposals.
- Browser (read-only): allowlisted fetches into `data/artifacts/notes/` with source URLs; `tools.network_access` must be true.
- GraphRAG-lite: add/query simple claims, resolve conflicts with confidence voting, and visualise triples in BigKit.
- Voice: cross-platform TTS via `pyttsx3`/`espeak` on Linux/Windows and `say` on macOS; STT falls back to `vosk-transcriber` if Whisper is absent.
- Query Oracle: `sym query_web "React hooks" --limit 3` plans allowlisted searches, stores notes in `data/artifacts/notes/`, and ingests belief triples.
- AI peer bridge: `sym peer_chat --prompt "Summarise migration risks"` simulates external conversations (stubbed unless configured).
- GitHub guard: `sym github_pr --title "Symbiont autopilot" --dry-run false` opens PRs under allowlisted owners using PyGitHub and stored tokens.
- **Observability metrics**: `sym metrics --port 8001` exposes Prometheus gauges for token usage, latency, and rogue scores; the Streamlit dashboard charts cumulative budgets and warns when limits are near.
- **Shadow observability**: `sym shadow_report`, `sym shadow_label --ingest`, `sym shadow_batch`, and `sym shadow_dashboard` capture guard/cycle clips into labeled datasets, append JSONL history under `data/artifacts/shadow/history.jsonl`, and generate governance dashboards (`systems/ShadowDashboard.md`, optional `systems/ShadowHistory.md`). Use `sym shadow_history --limit 5 --output systems/ShadowHistory.md` to review and export aggregated label trends.
- **Security helpers**: `sym rotate_credential ENV_KEY NEW_VALUE --env-file .env` rotates credentials with `audit_logs` tracking, and transcripts/notes now auto-redact emails, phone numbers, and obvious secrets.
- **Graph & crew templates**: reusable starters live under `configs/templates/graph_template.yaml` and `configs/templates/crew_template.yaml`, validated via new Pydantic schemas.
- MCP server (minimal) + CLI `install-hooks` for RAG automation.
- **Memory layers (experimental)**: configure `memory.layer` in `configs/config.yaml` or pass `--memory-layer` to `sym rag_*` commands to toggle `local`, `mem0`, or `letta` backends (stubs gracefully fall back to local when optional SDKs are absent).
- Remote memory setup:
  - Mem0: export `MEM0_API_KEY` (and optional `MEM0_API_HOST`, `MEM0_ORG_ID`, `MEM0_PROJECT_ID`) to sync `remember_preferences`/`recall_preferences` through the Mem0 API; set `MEM0_USER_ID`/`MEM0_SESSION_USER` (or override in `memory.mem0`) to pin remote user scopes.
  - Letta: export `LETTA_API_TOKEN`/`LETTA_TOKEN` plus `LETTA_PROJECT` (and optional `LETTA_BASE_URL`) to persist session state + preference blocks via Letta's Blocks API. Labels + env overrides are configurable under `memory.letta` in `configs/config.yaml`.
- **Foresight async hunts (v3.6)**: `python scripts/demo_async_foresight.py "emergent agentic trends"` blends arXiv, RSS, and optional Grok/Devin peer pings (rate-limited with jitter). Configure collaborators under `foresight.collaboration` in `configs/config.yaml`; enable the meta-foresight crew to trigger guardrail reviews automatically when relevance scores exceed the configured threshold. RLHF rewards are scrubbed and logged to `data/artifacts/rlhf/`, and you can inspect aggregates programmatically via `Orchestrator.train_from_rewards()`.

## Foresight Engine

- **Async hunt orchestrator**: `python scripts/demo_foresight_engine.py "multi-agent governance"` fans out arXiv, RSS, and peer collectors via asyncio with tenacity backoff. Artifacts land in `data/artifacts/foresight/hunts/` alongside Prometheus-style metrics exposed by visiting `?metrics=1` in BigKit.
- **Versioned memory**: Every hunt diffs triples into `data/artifacts/foresight/rag_diffs/`, enabling rollback and temporal inspection. The meta-foresight crew (`configs/crews/meta_foresight.yaml`) reflects on relevance drift and tunes future queries automatically.
- **Causal & RLHF guardrails**: Proposals flow through `ForesightSuggester`, combining DoWhy-style causal checks, zk-proof stubs (`data/artifacts/foresight/proofs/`), and a lightweight RLHF tuner that rewards high-signal hunts while down-weighting hype.
- **Adaptive simulations**: System dynamics utilities now early-stop when trajectories stabilise, keeping Pi 5 edge runs under 100 ms while still emitting stats and plots for governance dashboards.
- **System dynamics foresight (v4.1)**: run `python -m symbiont.cli run_graph configs/graphs/foresight_sd.yaml "Stress-test autonomy guard"` to trigger the new `dynamics_scout → sd_modeler → strategist` crew. A baseline simulation is recorded under `data/artifacts/graphs/simulations/`, plotted, and summarised in the BigKit governance dashboard (look for **Recent SD runs**) so you can inspect autonomy/rogue trajectories before executing scripts. Telemetry for each projection also lands in the SQLite `sd_runs` table; inspect it via `python -m symbiont.cli sd-runs --limit 3`.
- **Dynamics Weaver (hybrid SD+ABM)**: `python -m symbiont.cli dynamics-weaver "Model rogue drift in swarm"` runs the new hybrid crew that couples the macro SD engine with an agent-based noise layer. Results (plots + JSON) live under `data/artifacts/crews/dynamics_weaver/` and the summary appears both in the CLI output and the SD telemetry tables.
- **Rotating credentials & peer identities**: cloud LLM keys reload automatically (configurable `refresh_seconds`, default 1h) and swarm peers now carry UUIDs so governance logs can trace each agent’s input without long-lived secrets.
- **Pub/Sub hooks for initiative**: enable `initiative.pubsub` in `configs/config.yaml` to broadcast daemon events (memory JSONL or Redis) so distributed watchers can react without polling.
- **LLM judge for plans**: flip on `guard.judge.enabled` to have a secondary o1-style reviewer assign risk scores to plans; flags appear in CLI, swarm artifacts, and the UI.
- **Ready-to-use ops/research crews**: generic YAML templates (`configs/crews/ops_monitor.yaml`, `configs/crews/research_monitor.yaml`) provide plug-and-play monitors for reliability reviews without domain fine-tuning.
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
- Optional foresight extras: `pip install -r requirements-optional.txt`

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

# Cognitive Symbiont

Small, **verified** improvements to your repo. Symbiont finds a problem, fixes it in an isolated
git worktree, proves the fix (it parses, the problem is gone, nothing new broke, your tests still
pass), and only then hands it to you as a patch. Nothing touches your working tree until you say so.

```
$ sym propose
Trying: F401 unused import: 33 fix(es) in 27 file(s)
Running baseline checks: tests
  -> proposal #1: lint: 33/33 targeted issue(s) fixed, none introduced; check 'tests': passes
...
$ sym show 1      # evidence + diff
$ sym apply 1     # patch your working tree (no commit)
$ sym undo 1      # changed your mind
```

## Install

```bash
pip install git+https://github.com/Kubermench/cognitive-symbiont@v4-verified-fixes
```

Python 3.10+ and git. Run `sym` from anywhere inside the repo you want to improve.

## How it works

1. **Find**: `ruff` lint findings (your repo's ruff config if it has one, sensible rules otherwise),
   ordered with likely bugs (`F`, `B`) first and cosmetic modernisation (`UP`) last. Or a change
   you ask for: `sym ask "add type hints" src/util.py`.
2. **Fix** in a throwaway `git worktree` that holds your current state, uncommitted and untracked
   work included:
   - rules ruff can fix safely are fixed by ruff, with no LLM involved;
   - everything else goes to an LLM, which returns the whole updated file.
3. **Verify** before anything is kept:
   - Python files still parse;
   - the targeted findings went down and no new lint findings appeared;
   - every check that passed before the change still passes (your test suite, autodetected for
     pytest, or whatever you configure).
   A failed LLM attempt is retried once with the reason it failed.
4. **Hand over** a plain patch in `.symbiont/proposals/` (kept out of `git status`). `apply` and
   `undo` are `git apply` / `git apply -R`, and refuse to run if the files changed underneath.

## Commands

| command | what it does |
| --- | --- |
| `sym doctor` | shows the repo, detected checks, and whether the LLM answers |
| `sym scan` | lists tasks without changing anything |
| `sym propose [-n 3] [--no-llm]` | attempts tasks until `n` verified proposals exist |
| `sym ask "<instruction>" <file>` | one targeted LLM change, verified the same way |
| `sym list` / `sym show <id>` | review proposals and their evidence |
| `sym apply <id>` / `sym undo <id>` / `sym drop <id>` | act on a proposal |
| `sym init` | writes a starter `.symbiont.yml` |

## Configuration

Optional `.symbiont.yml` at the repo root (`sym init` writes one):

```yaml
llm:
  provider: ollama          # ollama | openai | anthropic | none
  model: qwen2.5-coder:7b
  # base_url: https://api.openai.com/v1   # any OpenAI-compatible server
  # api_key_env: OPENAI_API_KEY
checks:                     # must keep passing; autodetected (pytest) if omitted
  - name: tests
    run: .venv/bin/python -m pytest -q
limits:
  max_file_lines: 400       # bigger files are never sent to the LLM
```

Checks run inside the sandbox. Your repo's `.venv`, `venv` and `node_modules` are linked in, and
`$SYMBIONT_ROOT` points at the real repo if a command needs it.

**Memory:** a 7B model needs about 6 GB free; on an 8 GB machine it swaps and times out, so use
ruff-only mode or a hosted model there.

**Choosing a model:** small models (e.g. `phi3:mini`) mostly produce edits that fail
verification, so they get rejected rather than proposed. A code model of 7B or more locally, or a
hosted model, works far better. With `provider: none`, Symbiont still proposes every safe ruff fix.

## Weekly fix PRs on GitHub

Add this workflow to any repo. Each week Symbiont proposes fixes, applies each verified one on
top of the last, and opens (or updates) a single pull request whose description lists the
evidence for every fix.

```yaml
# .github/workflows/symbiont.yml
on:
  schedule: [{cron: "0 6 * * 1"}]
  workflow_dispatch:
permissions: {contents: write, pull-requests: write}
jobs:
  fixes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Kubermench/cognitive-symbiont@main
        with:
          setup-command: pip install -e ".[dev]"   # whatever makes your tests runnable
          # llm-provider: anthropic                # optional: also fix what ruff can't
          # llm-model: claude-sonnet-5-5
          # api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

The repo must allow it: **Settings → Actions → General → "Allow GitHub Actions to create and
approve pull requests"**. Without an LLM, only ruff's safe fixes are proposed.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

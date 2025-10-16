# Quickstart

Spin up Cognitive Symbiont Core in a fresh environment with the leanest possible footprint.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install .
sym init --lite --non-interactive
sym propose_here --goal "Draft a stack healthcheck script"
```

### Optional Extras

- Install features on demand:
  - Streamlit UI: `pip install .[ui]`
  - Foresight hunts: `pip install .[foresight]`
  - Observability metrics: `pip install .[observability]`
  - GitHub guardrails: `pip install .[github]`
  - Voice stubs: `pip install .[voice]`
  - Memory integrations (Mem0, Letta): `pip install .[memory]`
- Or pull everything: `pip install -r requirements-optional.txt`

### Smokescreen

Run a fast healthcheck after initialising the config and database:

```bash
python -m pytest tests/test_cli_init.py -q
```

If you are on Python 3.9, upgrade to 3.10+ or install the `eval_type_backport` package before running the full suite.

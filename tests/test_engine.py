from __future__ import annotations

from pathlib import Path

import pytest
from conftest import CALC, FIXED_CALC, FakeLLM, fenced, sh

from symbiont import config, engine
from symbiont.git import Sandbox
from symbiont.store import Store


def make_engine(root: Path, llm=None) -> engine.Engine:
    return engine.Engine(root, config.load(root), llm, log=lambda _msg: None)


def test_plan_splits_autofixable_and_manual(repo):
    tasks = engine.plan(repo, config.load(repo))
    kinds = {(t.kind, t.code or t.file) for t in tasks}
    assert ("lint-autofix", "F401") in kinds  # unused import: ruff can fix it
    assert ("lint-llm", "calc.py") in kinds  # bare except: needs judgement


def test_autofix_proposal_apply_and_undo(repo):
    eng = make_engine(repo)
    outcomes = eng.propose(engine.plan(repo, eng.cfg), limit=5)

    made = [o.proposal for o in outcomes if o.proposal]
    assert len(made) == 1
    prop = made[0]
    assert prop.kind == "lint-autofix" and prop.files == ["calc.py"]
    assert any("check 'tests': passes" in e for e in prop.evidence)
    # The LLM task is reported as skipped, not silently dropped.
    assert any(o.reason.startswith("needs an LLM") for o in outcomes if not o.proposal)

    assert (repo / "calc.py").read_text() == CALC, "proposing must not touch the working tree"
    store = Store(repo)
    store.apply(prop.id)
    assert "import os" not in (repo / "calc.py").read_text()
    assert store.get(prop.id).status == "applied"
    store.undo(prop.id)
    assert (repo / "calc.py").read_text() == CALC
    assert sh(repo, "git", "status", "--porcelain") == "", ".symbiont/ stays out of git status"


def test_existing_proposal_is_not_proposed_again(repo):
    eng = make_engine(repo)
    eng.propose(engine.plan(repo, eng.cfg), limit=5)
    again = make_engine(repo).propose(engine.plan(repo, eng.cfg), limit=5)
    assert not [o for o in again if o.proposal]


def test_llm_fix_is_verified_and_saved(repo):
    llm = FakeLLM(fenced(FIXED_CALC))
    eng = make_engine(repo, llm)
    task = next(t for t in engine.plan(repo, eng.cfg) if t.kind == "lint-llm")

    outcome = eng.attempt(task)

    assert outcome.proposal, outcome.reason
    assert "E722" in llm.prompts[0] and "except:" in llm.prompts[0]
    assert "except ZeroDivisionError:" in Store(repo).patch(outcome.proposal.id)


def test_llm_retry_after_syntax_error_gets_feedback(repo):
    llm = FakeLLM(fenced(FIXED_CALC.replace("return None", "return None)")), fenced(FIXED_CALC))
    eng = make_engine(repo, llm)
    task = next(t for t in engine.plan(repo, eng.cfg) if t.kind == "lint-llm")

    outcome = eng.attempt(task)

    assert outcome.proposal, outcome.reason
    assert "syntax error" in llm.prompts[1]


def test_change_that_breaks_tests_is_rejected(repo):
    broken = CALC.replace("return a / b", "return a * b")
    eng = make_engine(repo, FakeLLM(fenced(broken), fenced(broken)))

    outcome = eng.attempt(engine.ask_task("calc.py", "speed it up"))

    assert outcome.proposal is None
    assert "check 'tests' fails" in outcome.reason
    assert Store(repo).all() == []


def test_change_that_adds_lint_issues_is_rejected(repo):
    worse = "import sys\n" + CALC
    eng = make_engine(repo, FakeLLM(fenced(worse), fenced(worse)))
    outcome = eng.attempt(engine.ask_task("calc.py", "add an import"))
    assert outcome.proposal is None
    assert "introduced new lint issues: F401 in calc.py" in outcome.reason


def test_reply_without_code_is_rejected(repo):
    eng = make_engine(repo, FakeLLM("I cannot help", "still no"))
    outcome = eng.attempt(engine.ask_task("calc.py", "anything"))
    assert outcome.proposal is None and "no code block" in outcome.reason


def test_ask_change_is_saved(repo):
    documented = CALC.replace(
        "def divide(a, b):\n", 'def divide(a, b):\n    """Divide a by b."""\n'
    )
    eng = make_engine(repo, FakeLLM(fenced(documented)))
    outcome = eng.attempt(engine.ask_task("calc.py", "add a docstring to divide"))
    assert outcome.proposal, outcome.reason
    assert outcome.proposal.kind == "ask"


def test_sandbox_sees_uncommitted_and_untracked_work(repo):
    (repo / "calc.py").write_text(CALC + "\nVALUE = 1\n")
    (repo / "new.py").write_text("X = 2\n")
    with Sandbox(repo) as sb:
        assert "VALUE = 1" in (sb.path / "calc.py").read_text()
        assert (sb.path / "new.py").read_text() == "X = 2\n"
        assert sb.diff() == ""
        (sb.path / "new.py").write_text("X = 3\n")
        patch = sb.diff()
    assert "-X = 2" in patch and "+X = 3" in patch
    assert sh(repo, "git", "worktree", "list").count("\n") == 1, "sandbox worktree is cleaned up"


def test_apply_refuses_stale_patch(repo):
    eng = make_engine(repo)
    prop = next(o.proposal for o in eng.propose(engine.plan(repo, eng.cfg), limit=1) if o.proposal)
    (repo / "calc.py").write_text("print('rewritten')\n")
    with pytest.raises(ValueError, match="no longer applies"):
        Store(repo).apply(prop.id)


def test_removing_imports_tidies_import_order(repo):
    # Dropping the unused third-party import leaves a stray blank line, which is an I001 issue.
    (repo / "calc.py").write_text(
        "import json\nfrom pathlib import Path\n\nimport pytest\n\nprint(json, Path)\n"
    )
    sh(repo, "git", "commit", "-qam", "imports")
    eng = make_engine(repo)
    task = next(t for t in engine.plan(repo, eng.cfg) if t.code == "F401")
    outcome = eng.attempt(task)
    assert outcome.proposal, outcome.reason
    assert (
        Store(repo).patch(outcome.proposal.id).count("\n-") >= 2
    )  # the import and the blank line both go
    assert "import pytest" in Store(repo).patch(outcome.proposal.id)


def test_checks_see_repo_virtualenv_and_root(repo):
    (repo / ".gitignore").write_text(".venv/\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "marker").write_text("here")
    (repo / ".symbiont.yml").write_text(
        "checks:\n"
        "  - name: env\n"
        '    run: test -f .venv/marker && test -f "$SYMBIONT_ROOT/.venv/marker"\n'
    )
    sh(repo, "git", "add", "-A")
    sh(repo, "git", "commit", "-qm", "env")
    eng = make_engine(repo)
    assert all(r.passed for r in eng.baseline())


def test_check_artifacts_never_enter_the_patch(repo):
    (repo / ".symbiont.yml").write_text(
        "checks:\n  - name: noisy\n    run: echo junk > build-artifact.txt\n"
    )
    sh(repo, "git", "add", "-A")
    sh(repo, "git", "commit", "-qm", "noisy check")
    eng = make_engine(repo)
    task = next(t for t in engine.plan(repo, eng.cfg) if t.code == "F401")
    outcome = eng.attempt(task)
    assert outcome.proposal, outcome.reason
    assert outcome.proposal.files == ["calc.py"]
    assert "build-artifact" not in Store(repo).patch(outcome.proposal.id)

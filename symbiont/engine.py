"""Plan small tasks, attempt each in a sandbox, and keep only the ones that verify."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from . import checks, lint
from .config import Config
from .git import Sandbox, changed_files
from .llm import LLM, LLMError, extract_code
from .store import Proposal, Store

SYSTEM_PROMPT = (
    "You are a careful senior engineer making the smallest change that does the job. "
    "Reply with the complete updated file inside a single fenced code block and nothing else. "
    "Keep every other line, comment and behaviour exactly as it was."
)

LLM_ATTEMPTS = 2


class Rejected(Exception):
    """A fix attempt did not verify; the message says why."""


@dataclass
class Task:
    kind: str  # lint-autofix | lint-llm | ask
    title: str
    code: str | None = None  # lint-autofix: the rule fixed everywhere
    file: str | None = None  # lint-llm / ask: the one file edited
    findings: list[lint.Finding] = field(default_factory=list)
    instruction: str | None = None  # ask


@dataclass
class Outcome:
    task: Task
    proposal: Proposal | None
    reason: str


@cache
def _rule_name(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "rule", code, "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(proc.stdout)["name"].replace("-", " ")
    except (ValueError, KeyError, TypeError):
        return code


# Likely bugs first, cosmetic modernisation last.
_PRIORITY = ("F", "B", "E", "W", "SIM", "I", "UP")


def _priority(code: str) -> int:
    prefix = code.rstrip("0123456789")
    return _PRIORITY.index(prefix) if prefix in _PRIORITY else len(_PRIORITY)


def plan(root: Path, cfg: Config) -> list[Task]:
    """Turn the repo's lint findings into tasks: one per auto-fixable rule, one per file otherwise."""
    found = lint.findings(root, cfg)
    tasks: list[Task] = []

    fixable: dict[str, list[lint.Finding]] = defaultdict(list)
    manual: dict[str, list[lint.Finding]] = defaultdict(list)
    for f in found:
        (fixable[f.code] if f.fixable else manual[f.file]).append(f)

    for code, fs in sorted(fixable.items(), key=lambda kv: (_priority(kv[0]), -len(kv[1]), kv[0])):
        n_files = len({f.file for f in fs})
        tasks.append(
            Task(
                kind="lint-autofix",
                title=f"{code} {_rule_name(code)}: {len(fs)} fix(es) in {n_files} file(s)",
                code=code,
                findings=fs,
            )
        )
    for file, fs in sorted(
        manual.items(), key=lambda kv: (min(_priority(f.code) for f in kv[1]), -len(kv[1]), kv[0])
    ):
        codes = ", ".join(sorted({f.code for f in fs}))
        tasks.append(
            Task(
                kind="lint-llm",
                title=f"{file}: fix {len(fs)} issue(s) ({codes})",
                file=file,
                findings=fs,
            )
        )
    return tasks


def ask_task(file: str, instruction: str) -> Task:
    short = instruction if len(instruction) <= 60 else instruction[:57] + "..."
    return Task(kind="ask", title=f"{file}: {short}", file=file, instruction=instruction)


class Engine:
    def __init__(
        self, root: Path, cfg: Config, llm: LLM | None, log: Callable[[str], None] = print
    ):
        self.root = root
        self.cfg = cfg
        self.llm = llm
        self.log = log
        self.store = Store(root)
        self.checks = checks.detect(root, cfg)
        self._baseline: list[checks.Result] | None = None
        self._llm_down: str | None = None

    # -- baseline -------------------------------------------------------------
    def baseline(self) -> list[checks.Result]:
        if self._baseline is None:
            if self.checks:
                self.log(f"Running baseline checks: {', '.join(c.name for c in self.checks)}")
            with Sandbox(self.root) as sb:
                self._baseline = checks.run(
                    sb.path, self.checks, self.cfg.check_timeout, repo=self.root
                )
            for r in self._baseline:
                if not r.passed:
                    tail = "\n      ".join(r.output.strip().splitlines()[-5:])
                    self.log(
                        f"warning: check '{r.name}' already fails before any change, "
                        f"so it can't gate fixes. Last lines:\n      {tail}"
                    )
        return self._baseline

    # -- main loop ------------------------------------------------------------
    def propose(self, tasks: list[Task], limit: int, apply: bool = False) -> list[Outcome]:
        """Attempt tasks until `limit` proposals exist. With `apply`, each verified proposal is
        applied before the next attempt, so later fixes build on earlier ones."""
        open_titles = {p.title for p in self.store.all() if p.status in ("ready", "applied")}
        outcomes: list[Outcome] = []
        made = 0
        for task in tasks:
            if made >= limit:
                break
            if task.title in open_titles:
                continue
            if task.kind != "lint-autofix" and self.llm is None:
                outcomes.append(Outcome(task, None, "needs an LLM (llm.provider is 'none')"))
                continue
            if task.kind != "lint-autofix" and self._llm_down:
                outcomes.append(Outcome(task, None, f"LLM unavailable: {self._llm_down}"))
                continue
            self.log(f"Trying: {task.title}")
            outcome = self.attempt(task)
            verdict = f"proposal #{outcome.proposal.id}" if outcome.proposal else "rejected"
            self.log(f"  -> {verdict}: {outcome.reason}")
            outcomes.append(outcome)
            if outcome.proposal and apply:
                self.store.apply(outcome.proposal.id)
            made += outcome.proposal is not None
        return outcomes

    def attempt(self, task: Task) -> Outcome:
        baseline = self.baseline()
        tries = 1 if task.kind == "lint-autofix" else LLM_ATTEMPTS
        feedback: str | None = None
        reason = ""
        with Sandbox(self.root) as sb:
            assert sb.path is not None
            lint_before = lint.counts(lint.findings(sb.path, self.cfg))
            for _ in range(tries):
                sb.reset()
                try:
                    self._make_change(sb.path, task, feedback, lint_before)
                    evidence, files, patch = self._verify(sb, task, lint_before, baseline)
                except Rejected as exc:
                    reason = feedback = str(exc)
                    continue
                except LLMError as exc:
                    self._llm_down = str(exc)
                    return Outcome(task, None, str(exc))
                prop = self.store.add(task.title, task.kind, files, patch, evidence)
                return Outcome(task, prop, "; ".join(evidence))
        return Outcome(task, None, reason)

    # -- making a change --------------------------------------------------------
    def _make_change(self, wt: Path, task: Task, feedback: str | None, lint_before) -> None:
        if task.kind == "lint-autofix":
            assert task.code
            lint.autofix(wt, task.code)
            self._tidy(wt, lint_before)
            return
        assert task.file and self.llm is not None
        path = wt / task.file
        if not path.is_file():
            raise Rejected(f"{task.file} does not exist")
        original = path.read_text(encoding="utf-8")
        n_lines = original.count("\n") + 1
        if n_lines > self.cfg.max_file_lines:
            raise Rejected(f"{task.file} has {n_lines} lines (limit {self.cfg.max_file_lines})")

        if task.kind == "lint-llm":
            src = original.splitlines()
            issues = "\n".join(
                f"- line {f.line}: {f.code} {f.message}"
                + (f"\n    {src[f.line - 1].strip()}" if 0 < f.line <= len(src) else "")
                for f in task.findings
            )
            request = (
                f"Fix these lint issues in `{task.file}` without changing behaviour:\n{issues}"
            )
        else:
            request = f"Change `{task.file}` as follows: {task.instruction}"
        if feedback:
            request += f"\n\nA previous attempt was rejected: {feedback}\nAvoid that this time."
        prompt = f"{request}\n\nCurrent contents of `{task.file}`:\n```\n{original}```"

        reply = self.llm.complete(SYSTEM_PROMPT, prompt)
        code = extract_code(reply)
        if code is None:
            raise Rejected("the LLM reply contained no code block")
        if original.endswith("\n") and not code.endswith("\n"):
            code += "\n"
        if n_lines > 20 and code.count("\n") < 0.6 * n_lines:
            raise Rejected("the LLM returned a much shorter file (likely truncated)")
        path.write_text(code, encoding="utf-8")

    def _tidy(self, wt: Path, lint_before) -> None:
        """Let ruff clean up what a fix knocked loose, e.g. imports left unsorted by a removal."""
        changed = [f for f in changed_files(wt) if f.endswith(".py")]
        if not changed:
            return
        after = lint.findings(wt, self.cfg, changed)
        grown = lint.counts(after) - lint_before
        knock_on = {f.code for f in after if f.fixable and grown[(f.file, f.code)] > 0}
        for code in sorted(knock_on):
            lint.autofix(wt, code, changed)

    # -- verification -----------------------------------------------------------
    def _verify(
        self, sb: Sandbox, task: Task, lint_before, baseline: list[checks.Result]
    ) -> tuple[list[str], list[str], str]:
        """Return evidence plus the change itself, captured before checks can add artifacts."""
        assert sb.path is not None
        changed = sb.changed_files()
        if not changed:
            raise Rejected("nothing to change (an earlier fix may already have resolved it)")
        if task.kind != "lint-autofix" and changed != [task.file]:
            raise Rejected(f"changed files other than {task.file}: {', '.join(changed)}")

        for rel in changed:
            path = sb.path / rel
            if rel.endswith(".py") and path.exists():
                try:
                    ast.parse(path.read_text(encoding="utf-8"), filename=rel)
                except SyntaxError as exc:
                    raise Rejected(f"syntax error in {rel} line {exc.lineno}: {exc.msg}") from None

        evidence: list[str] = []
        lint_after = lint.counts(lint.findings(sb.path, self.cfg))
        introduced = {
            k: lint_after[k] - lint_before[k] for k in lint_after if lint_after[k] > lint_before[k]
        }
        if introduced:
            detail = ", ".join(f"{code} in {file}" for file, code in sorted(introduced))
            raise Rejected(f"introduced new lint issues: {detail}")
        if task.kind in ("lint-autofix", "lint-llm"):
            targets = {(f.file, f.code) for f in task.findings}
            before = sum(lint_before[k] for k in targets)
            after = sum(lint_after[k] for k in targets)
            if after >= before:
                raise Rejected("the targeted lint issues are still there")
            evidence.append(
                f"lint: {before - after}/{before} targeted issue(s) fixed, none introduced"
            )
        else:
            evidence.append("lint: no new issues")

        files, patch = changed, sb.diff()
        results = checks.run(sb.path, self.checks, self.cfg.check_timeout, repo=self.root)
        broken = checks.regressions(baseline, results)
        if broken:
            tail = broken[0].output.strip().splitlines()[-15:]
            raise Rejected(f"check '{broken[0].name}' fails after the change:\n" + "\n".join(tail))
        for r in results:
            evidence.append(
                f"check '{r.name}': {'passes' if r.passed else 'still failing (was failing before)'}"
            )
        if not results:
            evidence.append("no test checks configured or detected")
        return evidence, files, patch

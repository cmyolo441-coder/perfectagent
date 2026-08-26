"""JUDGE — deterministic verification subsystem.

A node cannot enter PASSED on the model's own testimony. Every claim is
checked against reality with deterministic predicates — real exit codes,
real file checks, real regex matches — and each result is sealed into the
event log as a 'judge.verdict' event. No LLM judging here: verification
is 100% deterministic.

Predicate dicts understood by Judge.check():
    {"type": "exit_code", "command": "pytest -q", "expect": 0, "timeout": 120}
    {"type": "file_exists", "path": "src/x.py"}
    {"type": "file_contains", "path": "src/x.py", "text": "def foo"}
    {"type": "file_matches", "path": "src/x.py", "pattern": r"def \\w+\\("}
    {"type": "command_output_contains", "command": "python -V", "text": "Python"}
    {"type": "ast_assert", "path": "src/x.py", "symbol": "verify_token",
     "has_parameter": "leeway", "kind": "def"}
    {"type": "diff_assert", "path": "src/x.py",
     "forbid": ["print\\(", "TODO"]}
    {"type": "file_unchanged", "path": "uv.lock", "baseline_hash": "…"}
    {"type": "tool_delta", "command": "mypy src", "delta": 0}
"""

from __future__ import annotations

import ast
import functools
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .kernel import EventLog, fold

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

EVIDENCE_LIMIT = 300  # max chars of evidence kept per verdict


@dataclass
class Verdict:
    """The sealed outcome of one deterministic predicate check."""
    passed: bool
    kind: str            # predicate type that was checked
    detail: str          # human-readable one-line result
    evidence: str = field(default="")  # minimal excerpt proving pass/fail

    def to_dict(self) -> dict:
        """JSON-serializable form — the 'judge.verdict' event payload."""
        return {"passed": self.passed, "kind": self.kind,
                "detail": self.detail, "evidence": self.evidence}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SHELL_CACHE: list[str] | None = None


def resolve_shell() -> list[str] | None:
    """Resolve a working POSIX shell, once, for predicate commands.

    On Windows, System32\\bash.exe is the WSL stub — it fails with 'no
    installed distributions' when WSL has no distro, silently breaking
    every shell predicate. Candidates (Git Bash first, then PATH bash/sh)
    are probed with a real `exit 0`; the first that works is cached. On
    POSIX it's just bash. Returns None when nothing runnable exists."""
    global _SHELL_CACHE
    if _SHELL_CACHE is not None:
        return _SHELL_CACHE or None
    candidates: list[str] = []
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            gdir = Path(git).parent.parent
            for rel in ("bin/bash.exe", "usr/bin/bash.exe",
                        "bin/sh.exe", "usr/bin/sh.exe"):
                candidates.append(str(gdir / rel))
        for name in ("bash.exe", "sh.exe"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
    else:
        candidates.append("bash")
    for cand in candidates:
        try:
            probe = subprocess.run([cand, "-c", "exit 0"],
                                   capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            _SHELL_CACHE = [cand, "-lc"]
            return _SHELL_CACHE
    _SHELL_CACHE = []
    return None


def _shell_command() -> list[str] | None:
    """Backwards-compatible alias for resolve_shell()."""
    return resolve_shell()


def _run_shell(command: str, timeout: int):
    """Run a predicate command in the resolved shell. Returns
    (returncode, combined_output) or None when no shell is available."""
    argv = _shell_command()
    if argv is None:
        return None
    proc = subprocess.run(argv + [command], capture_output=True,
                          text=True, timeout=timeout)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def _clip(text: str, limit: int = EVIDENCE_LIMIT) -> str:
    """Trim evidence down to a bounded excerpt."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _line_around(text: str, idx: int) -> str:
    """The line of text containing the character at idx."""
    start = text.rfind("\n", 0, idx) + 1
    end = text.find("\n", idx)
    return text[start: end if end != -1 else len(text)].strip()


def _as_timeout(value: object, default: int = 120) -> int:
    """Coerce a predicate timeout to a positive int."""
    try:
        t = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return t if t > 0 else default


def _guarded(kind: str) -> Callable:
    """Verdict helpers must never raise — wrap failures into failed verdicts."""
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Verdict:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                return Verdict(False, kind, f"{fn.__name__} error: {e}")
        return wrapper
    return decorate


# ---------------------------------------------------------------------------
# Deterministic checkers (pure module-level helpers)
# ---------------------------------------------------------------------------

@_guarded("exit_code")
def check_exit_code(command: str, expect: int = 0,
                    timeout: int = 120) -> Verdict:
    """Run command via bash and compare its exit code against expect."""
    try:
        expect = int(expect)
    except (TypeError, ValueError):
        return Verdict(False, "exit_code",
                       f"invalid expected exit code: {expect!r}")
    try:
        ran = _run_shell(command, timeout)
    except subprocess.TimeoutExpired:
        return Verdict(False, "exit_code",
                       f"timed out after {timeout}s: {command}")
    except OSError as e:
        return Verdict(False, "exit_code", f"cannot run command: {e}")
    if ran is None:
        return Verdict(False, "exit_code",
                       "no POSIX shell available — install Git Bash "
                       "(windows) or bash (posix)")
    rc, output = ran
    passed = rc == expect
    detail = f"'{command}' exited {rc}, expected {expect}"
    evidence = _clip(output)
    return Verdict(passed, "exit_code", detail, evidence)


@_guarded("file_exists")
def check_file_exists(path: str) -> Verdict:
    """Pass iff path exists on disk."""
    p = Path(path).expanduser()
    if p.exists():
        kind = "directory" if p.is_dir() else "file"
        return Verdict(True, "file_exists", f"{kind} exists: {p}", str(p))
    return Verdict(False, "file_exists", f"not found: {p}", str(p))


@_guarded("file_contains")
def check_file_contains(path: str, text: str) -> Verdict:
    """Pass iff literal text appears in the file at path."""
    p = Path(path).expanduser()
    if not p.is_file():
        return Verdict(False, "file_contains", f"file not found: {p}")
    content = p.read_text(errors="replace")
    idx = content.find(text)
    if idx < 0:
        return Verdict(False, "file_contains", f"{text!r} not found in {p}")
    line = content.count("\n", 0, idx) + 1
    return Verdict(True, "file_contains", f"{text!r} found at {p}:{line}",
                   _clip(_line_around(content, idx)))


@_guarded("file_matches")
def check_file_matches(path: str, pattern: str) -> Verdict:
    """Pass iff regex pattern matches anywhere in the file (MULTILINE, so
    ^ and $ match line boundaries)."""
    try:
        rx = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return Verdict(False, "file_matches",
                       f"invalid pattern {pattern!r}: {e}")
    p = Path(path).expanduser()
    if not p.is_file():
        return Verdict(False, "file_matches", f"file not found: {p}")
    content = p.read_text(errors="replace")
    m = rx.search(content)
    if m is None:
        return Verdict(False, "file_matches",
                       f"pattern {pattern!r} not found in {p}")
    line = content.count("\n", 0, m.start()) + 1
    return Verdict(True, "file_matches", f"pattern matched at {p}:{line}",
                   _clip(_line_around(content, m.start())))


@_guarded("command_output_contains")
def check_command_output_contains(command: str, text: str,
                                  timeout: int = 120) -> Verdict:
    """Run command via bash; pass iff text appears in its combined
    stdout+stderr output."""
    try:
        ran = _run_shell(command, timeout)
    except subprocess.TimeoutExpired:
        return Verdict(False, "command_output_contains",
                       f"timed out after {timeout}s: {command}")
    except OSError as e:
        return Verdict(False, "command_output_contains",
                       f"cannot run command: {e}")
    if ran is None:
        return Verdict(False, "command_output_contains",
                       "no POSIX shell available")
    rc, output = ran
    idx = output.find(text)
    if idx < 0:
        return Verdict(False, "command_output_contains",
                       f"{text!r} not in output of '{command}' "
                       f"(exit {rc})", _clip(output))
    return Verdict(True, "command_output_contains",
                   f"{text!r} found in output of '{command}'",
                   _clip(_line_around(output, idx)))


@_guarded("ast_assert")
def check_ast_assert(path: str, symbol: str, kind: str = "def",
                     has_parameter: str | None = None) -> Verdict:
    """Pass iff the AST of path defines `symbol` (def/class) — optionally
    with a parameter named has_parameter. Rung 2: structure proven."""
    p = Path(path).expanduser()
    if not p.is_file():
        return Verdict(False, "ast_assert", f"file not found: {p}")
    try:
        tree = ast.parse(p.read_text(errors="replace"))
    except SyntaxError as e:
        return Verdict(False, "ast_assert", f"{p} does not parse: {e}")
    targets = []
    for node in ast.walk(tree):
        if kind == "class" and isinstance(node, ast.ClassDef):
            targets.append(node)
        elif kind != "class" and isinstance(node, (ast.FunctionDef,
                                                   ast.AsyncFunctionDef)):
            targets.append(node)
    match = next((n for n in targets if n.name == symbol), None)
    if match is None:
        return Verdict(False, "ast_assert",
                       f"{kind} {symbol!r} not found in {p}")
    if has_parameter:
        if kind == "class" or not isinstance(match, (ast.FunctionDef,
                                                     ast.AsyncFunctionDef)):
            return Verdict(False, "ast_assert",
                           f"has_parameter is not supported for kind="
                           f"'class' ({symbol!r} is a class)")
        args = match.args
        names = ([a.arg for a in args.args] + [a.arg for a in args.kwonlyargs]
                 + ([args.vararg.arg] if args.vararg else [])
                 + ([args.kwarg.arg] if args.kwarg else []))
        if has_parameter not in names:
            return Verdict(False, "ast_assert",
                           f"{symbol}() has no parameter {has_parameter!r} "
                           f"(has: {', '.join(names)})",
                           _clip(f"line {match.lineno}"))
    return Verdict(True, "ast_assert",
                   f"{kind} {symbol!r} found at {p}:{match.lineno}"
                   + (f" with parameter {has_parameter!r}"
                      if has_parameter else ""),
                   _clip(f"line {match.lineno}"))


@_guarded("diff_assert")
def check_diff_assert(path: str, forbid: list | None = None,
                      require: list | None = None) -> Verdict:
    """Pass iff the file contains NONE of the forbid patterns and ALL of
    the require patterns (each a regex). Used for anti-clauses like 'no
    print( or TODO introduced'."""
    p = Path(path).expanduser()
    if not p.is_file():
        return Verdict(False, "diff_assert", f"file not found: {p}")
    content = p.read_text(errors="replace")
    for pat in forbid or []:
        try:
            m = re.search(str(pat), content, re.MULTILINE)
        except re.error as e:
            return Verdict(False, "diff_assert",
                           f"invalid forbid pattern {pat!r}: {e}")
        if m:
            line = content.count("\n", 0, m.start()) + 1
            return Verdict(False, "diff_assert",
                           f"forbidden pattern {pat!r} found at {p}:{line}",
                           _clip(_line_around(content, m.start())))
    for pat in require or []:
        try:
            m = re.search(str(pat), content, re.MULTILINE)
        except re.error as e:
            return Verdict(False, "diff_assert",
                           f"invalid require pattern {pat!r}: {e}")
        if not m:
            return Verdict(False, "diff_assert",
                           f"required pattern {pat!r} absent from {p}")
    return Verdict(True, "diff_assert",
                   f"{p}: no forbidden patterns, all required present")


@_guarded("file_unchanged")
def check_file_unchanged(path: str, baseline_hash: str) -> Verdict:
    """Pass iff the file's sha256 still equals baseline_hash — the
    anti-clause 'no new dependency' check against a lockfile."""
    p = Path(path).expanduser()
    if not p.is_file():
        return Verdict(False, "file_unchanged", f"file not found: {p}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != str(baseline_hash):
        return Verdict(False, "file_unchanged",
                       f"{p} changed: {actual[:12]} != {str(baseline_hash)[:12]}")
    return Verdict(True, "file_unchanged",
                   f"{p} unchanged ({actual[:12]})")


@_guarded("tool_delta")
def check_tool_delta(command: str, delta: int = 0,
                     timeout: int = 120) -> Verdict:
    """Run a checker (mypy, ruff, …) and pass iff its error-line count is
    <= delta. 'No NEW type errors' = delta 0 against a clean baseline."""
    try:
        ran = _run_shell(command, timeout)
    except subprocess.TimeoutExpired:
        return Verdict(False, "tool_delta",
                       f"timed out after {timeout}s: {command}")
    except OSError as e:
        return Verdict(False, "tool_delta", f"cannot run command: {e}")
    if ran is None:
        return Verdict(False, "tool_delta", "no POSIX shell available")
    rc, output = ran
    # count error-ish lines: "file:line: error" or "file:line:col: E501"
    errors = [ln for ln in output.splitlines()
              if re.search(r":\d+(:\d+)?:\s*(error|E\d{3}|F\d{3})", ln)]
    try:
        allowed = int(delta)
    except (TypeError, ValueError):
        allowed = 0
    if len(errors) > allowed:
        return Verdict(False, "tool_delta",
                       f"'{command}' reports {len(errors)} errors "
                       f"(allowed {allowed})", _clip("\n".join(errors[:5])))
    return Verdict(True, "tool_delta",
                   f"'{command}' reports {len(errors)} errors "
                   f"(allowed {allowed})")


# ---------------------------------------------------------------------------
# Judge — dispatch, emit, query
# ---------------------------------------------------------------------------

class Judge:
    """Deterministic verification over the event log.

    Every predicate is checked against reality and the verdict is sealed
    as a 'judge.verdict' event; state is recovered by folding the log.
    """

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def check(self, predicate: dict) -> Verdict:
        """Dispatch a predicate dict to the right deterministic checker,
        run it, build a Verdict, emit a 'judge.verdict' event, return it.
        Never raises."""
        verdict = self._evaluate(predicate)
        try:
            self.log.append("judge.verdict", verdict.to_dict())
        except Exception as e:
            verdict = Verdict(verdict.passed, verdict.kind,
                              f"{verdict.detail} (log append failed: {e})",
                              verdict.evidence)
        return verdict

    def check_all(self, predicates: list[dict]) -> tuple[bool, list[Verdict]]:
        """Run every predicate; return (all_passed, list_of_verdicts)."""
        verdicts = [self.check(p) for p in predicates]
        return all(v.passed for v in verdicts), verdicts

    def recent_verdicts(self, n: int = 10) -> list[dict]:
        """Last n verdict dicts from the fold, newest first."""
        if n <= 0:
            return []
        verdicts = fold(self.log).verdicts
        return list(reversed(verdicts[-n:]))

    # -- structured failure (§19.2) ------------------------------------------

    def failure(self, verdict: Verdict, context: str = "") -> dict:
        """Turn a failed verdict into a STRUCTURED Failure — never a raw
        traceback dumped into the prompt. This is the difference between an
        agent that debugs and an agent that flails."""
        location = ""
        m = re.search(r"([^\s:]+):(\d+)", verdict.detail)
        if m:
            location = f"{m.group(1)}:{m.group(2)}"
        return {
            "kind": "ASSERTION" if verdict.kind != "exit_code" else "EXCEPTION",
            "predicate": verdict.kind,
            "evidence": verdict.evidence[:EVIDENCE_LIMIT],
            "location": location,
            "detail": verdict.detail,
            "context": context,
            "suggested_next": "inspect the evidence, then change the "
                              "approach — do not retry the identical action",
        }

    # -- flake detection (§19.3) ----------------------------------------------

    def check_with_retry(self, predicate: dict, runs: int = 3) -> Verdict:
        """A failing test is re-run before being believed. If it passes on
        any retry it is a FLAKE: recorded as a project fact and excluded
        from pass criteria with a visible warning. Agents that treat flakes
        as real bugs waste enormous money chasing ghosts."""
        first = self.check(predicate)
        if first.passed:
            return first
        for _ in range(max(0, runs - 1)):
            retry = self.check(predicate)
            if retry.passed:
                self.log.append("fact.learned",
                                {"fact": f"FLAKE: {predicate.get('type')} "
                                         f"{predicate.get('command') or predicate.get('path') or ''} "
                                         "fails intermittently",
                                 "kind": "flake"},
                                actor="judge")
                return Verdict(True, "flake",
                               f"FLAKE detected (failed then passed): "
                               f"{first.detail}", first.evidence)
        return first

    # -- internal ----------------------------------------------------------

    def _evaluate(self, predicate: dict) -> Verdict:
        """Route one predicate dict to its checker. Never raises."""
        if not isinstance(predicate, dict):
            return Verdict(False, "invalid", "predicate must be a dict")
        ptype = predicate.get("type")
        if not isinstance(ptype, str) or not ptype:
            return Verdict(False, "invalid", "predicate missing 'type'")
        try:
            if ptype == "exit_code":
                command = predicate.get("command")
                if not isinstance(command, str) or not command.strip():
                    return Verdict(False, ptype, "predicate missing 'command'")
                return check_exit_code(command, predicate.get("expect", 0),
                                       _as_timeout(predicate.get("timeout")))
            if ptype == "file_exists":
                path = predicate.get("path")
                if not path:
                    return Verdict(False, ptype, "predicate missing 'path'")
                return check_file_exists(str(path))
            if ptype == "file_contains":
                path, text = predicate.get("path"), predicate.get("text")
                if not path:
                    return Verdict(False, ptype, "predicate missing 'path'")
                if text is None:
                    return Verdict(False, ptype, "predicate missing 'text'")
                # an empty search string vacuously matches at position 0 of
                # ANY content (str.find("") == 0) — without this guard a
                # predicate {"text": ""} would silently pass for every
                # file, including empty/missing ones, "proving" clauses
                # that search for nothing
                if not isinstance(text, str) or not text:
                    return Verdict(False, ptype,
                                   "predicate 'text' is empty")
                return check_file_contains(str(path), text)
            if ptype == "file_matches":
                path, pattern = predicate.get("path"), predicate.get("pattern")
                if not path:
                    return Verdict(False, ptype, "predicate missing 'path'")
                if pattern is None:
                    return Verdict(False, ptype,
                                   "predicate missing 'pattern'")
                if not isinstance(pattern, str) or not pattern:
                    return Verdict(False, ptype,
                                   "predicate 'pattern' is empty")
                return check_file_matches(str(path), pattern)
            if ptype == "command_output_contains":
                command, text = predicate.get("command"), predicate.get("text")
                if not isinstance(command, str) or not command.strip():
                    return Verdict(False, ptype, "predicate missing 'command'")
                if text is None:
                    return Verdict(False, ptype, "predicate missing 'text'")
                if not isinstance(text, str) or not text:
                    return Verdict(False, ptype,
                                   "predicate 'text' is empty")
                return check_command_output_contains(
                    command, str(text), _as_timeout(predicate.get("timeout")))
            if ptype == "ast_assert":
                path, symbol = predicate.get("path"), predicate.get("symbol")
                if not path:
                    return Verdict(False, ptype, "predicate missing 'path'")
                if not symbol:
                    return Verdict(False, ptype, "predicate missing 'symbol'")
                return check_ast_assert(str(path), str(symbol),
                                        str(predicate.get("kind", "def")),
                                        predicate.get("has_parameter"))
            if ptype == "diff_assert":
                path = predicate.get("path")
                if not path:
                    return Verdict(False, ptype, "predicate missing 'path'")
                return check_diff_assert(str(path),
                                         predicate.get("forbid"),
                                         predicate.get("require"))
            if ptype == "file_unchanged":
                path = predicate.get("path")
                baseline = predicate.get("baseline_hash")
                if not path:
                    return Verdict(False, ptype, "predicate missing 'path'")
                if not baseline:
                    return Verdict(False, ptype,
                                   "predicate missing 'baseline_hash'")
                return check_file_unchanged(str(path), str(baseline))
            if ptype == "tool_delta":
                command = predicate.get("command")
                if not isinstance(command, str) or not command.strip():
                    return Verdict(False, ptype, "predicate missing 'command'")
                return check_tool_delta(command, predicate.get("delta", 0),
                                        _as_timeout(predicate.get("timeout")))
        except Exception as e:  # backstop: verification must never raise
            return Verdict(False, ptype, f"judge error: {e}")
        return Verdict(False, ptype, "unknown predicate type")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        log = EventLog(tmp / "judge-selftest.jsonl")
        judge = Judge(log)

        sample = tmp / "sample.py"
        sample.write_text("def foo():\n    return 42\n")

        # passing predicates
        v = judge.check({"type": "file_exists", "path": str(sample)})
        assert v.passed and v.kind == "file_exists", v
        v = judge.check({"type": "file_contains", "path": str(sample),
                         "text": "return 42"})
        assert v.passed and "sample.py:2" in v.detail, v
        v = judge.check({"type": "file_matches", "path": str(sample),
                         "pattern": r"def \w+\("})
        assert v.passed, v
        v = judge.check({"type": "exit_code", "command": "true", "expect": 0})
        assert v.passed, v
        v = judge.check({"type": "command_output_contains",
                         "command": "echo fullagent-judge", "text": "judge"})
        assert v.passed, v
        v = judge.check({"type": "command_output_contains",
                         "command": "seq 1 500", "text": "42"})
        assert v.passed and len(v.evidence) <= EVIDENCE_LIMIT, v

        # failing predicates
        v = judge.check({"type": "file_exists", "path": str(tmp / "nope.py")})
        assert not v.passed, v
        v = judge.check({"type": "exit_code", "command": "false", "expect": 0})
        assert not v.passed, v
        v = judge.check({"type": "file_contains", "path": str(sample),
                         "text": "no such text"})
        assert not v.passed, v
        v = judge.check({"type": "totally_bogus"})
        assert not v.passed and v.kind == "totally_bogus", v
        assert v.detail == "unknown predicate type", v

        # check_all: mixed pass/fail
        ok, verdicts = judge.check_all([
            {"type": "file_exists", "path": str(sample)},
            {"type": "file_exists", "path": str(tmp / "nope.py")},
        ])
        assert ok is False and len(verdicts) == 2, (ok, verdicts)
        assert verdicts[0].passed and not verdicts[1].passed

        # ast_assert: symbol + parameter (rung 2)
        v = judge.check({"type": "ast_assert", "path": str(sample),
                         "symbol": "foo"})
        assert v.passed, v
        v = judge.check({"type": "ast_assert", "path": str(sample),
                         "symbol": "foo", "has_parameter": "nope"})
        assert not v.passed, v
        v = judge.check({"type": "ast_assert", "path": str(sample),
                         "symbol": "missing_fn"})
        assert not v.passed, v

        # diff_assert: forbid / require patterns
        v = judge.check({"type": "diff_assert", "path": str(sample),
                         "forbid": [r"print\("]})
        assert v.passed, v
        v = judge.check({"type": "diff_assert", "path": str(sample),
                         "forbid": [r"return 42"]})
        assert not v.passed, v
        v = judge.check({"type": "diff_assert", "path": str(sample),
                         "require": [r"def foo"]})
        assert v.passed, v

        # file_unchanged: baseline hash comparison
        import hashlib as _h
        base = _h.sha256(sample.read_bytes()).hexdigest()
        v = judge.check({"type": "file_unchanged", "path": str(sample),
                         "baseline_hash": base})
        assert v.passed, v
        v = judge.check({"type": "file_unchanged", "path": str(sample),
                         "baseline_hash": "0" * 64})
        assert not v.passed, v

        # tool_delta: error-line counting
        v = judge.check({"type": "tool_delta", "command": "true",
                         "delta": 0})
        assert v.passed, v

        # structured failure (§19.2)
        bad = judge.check({"type": "file_contains", "path": str(sample),
                           "text": "no such text"})
        f = judge.failure(bad)
        assert f["kind"] == "ASSERTION" and f["evidence"] is not None
        assert f["suggested_next"], f
        json.dumps(f)

        # flake detection (§19.3): a command that fails once then passes
        # (posix path — backslashes are eaten inside a bash -c string)
        flake_dir = Path(td).as_posix()
        flake_cmd = f"bash -c 'if [ ! -f {flake_dir}/flaked ]; then touch {flake_dir}/flaked; exit 1; fi; exit 0'"
        v = judge.check_with_retry(
            {"type": "exit_code", "command": flake_cmd, "expect": 0})
        assert v.passed and v.kind == "flake", v
        facts = [d for d in fold(log).facts if d.get("kind") == "flake"]
        assert facts, "flake must be recorded as a project fact"

        # verdicts are sealed into the fold, newest first, JSON-serializable
        state = fold(log)
        for d in state.verdicts:
            json.dumps(d)
            assert len(d["evidence"]) <= EVIDENCE_LIMIT
        recent = judge.recent_verdicts(3)
        assert len(recent) == 3
        assert judge.recent_verdicts(0) == []

    print("JUDGE SELF-TEST PASS")

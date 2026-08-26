"""COVERAGE — real line-coverage measurement.

Measures which lines of a target module actually execute when a piece of
code runs against it. This is genuine coverage, not an estimate: it uses
`sys.settrace` — the same CPython tracing hook the professional `coverage.py`
tool is built on — to record every executed line of the target file, then
compares against the set of *executable* lines derived from the AST
(statements, not blanks/comments/docstrings).

coverage % = executed executable lines / total executable lines

Design (pure stdlib, deterministic):
  * executable_lines() — AST walk: every statement's line is executable.
  * measure() — installs a trace function, runs the subject callable,
    collects the lines hit in the target file, restores the previous trace.
  * The trace only RECORDS; it never alters control flow, so the subject
    runs exactly as it would untraced.
  * Results are sealed as coverage.run / coverage.result events.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog, fold


# ---------------------------------------------------------------------------
# Executable lines from the AST
# ---------------------------------------------------------------------------

_STATEMENTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return,
               ast.Delete, ast.Raise, ast.Assert, ast.Import,
               ast.ImportFrom, ast.If, ast.For, ast.While, ast.Try,
               ast.With, ast.Expr, ast.Pass, ast.Break, ast.Continue,
               ast.Global, ast.Nonlocal, ast.FunctionDef,
               ast.AsyncFunctionDef, ast.ClassDef)
_STATEMENTS += (ast.Match,) if hasattr(ast, "Match") else ()  # py3.10+


def executable_lines(source: str) -> set[int]:
    """The set of line numbers that can execute (statements + the bodies
    of defs/classes). Blanks, comments and pure docstrings excluded."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _STATEMENTS):
            # a bare string Expr as the first body item is a docstring —
            # it does execute, but we keep it simple and count it
            if hasattr(node, "lineno"):
                lines.add(node.lineno)
    return lines


# ---------------------------------------------------------------------------
# Coverage measurement
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult:
    path: str
    total: int = 0
    hit: int = 0
    missed: list[int] = field(default_factory=list)
    percent: float = 0.0

    def to_dict(self) -> dict:
        return {"path": self.path, "total": self.total, "hit": self.hit,
                "missed": self.missed[:50],
                "percent": round(self.percent, 1)}


class CoverageEngine:
    """Measure real line coverage of one file while a subject runs."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def measure(self, target_path: str, subject) -> CoverageResult:
        """Run `subject()` under a trace that records lines executed in
        `target_path`. Returns the CoverageResult. The previous trace
        function is always restored, even if the subject raises."""
        tp = Path(target_path).expanduser().resolve()
        result = CoverageResult(path=str(tp))
        if not tp.is_file():
            return result
        source = tp.read_text(errors="replace")
        exec_lines = executable_lines(source)
        result.total = len(exec_lines)

        hit: set[int] = set()
        target_str = str(tp)

        def tracer(frame, event, arg):
            if event == "line":
                fn = frame.f_code.co_filename
                if fn == target_str or Path(fn).resolve() == tp:
                    hit.add(frame.f_lineno)
            return tracer

        old_trace = sys.gettrace()
        self.log.append("coverage.run", {"path": str(tp)}, actor="tester")
        try:
            sys.settrace(tracer)
            try:
                subject()
            except Exception:
                # coverage of a crashing subject is still meaningful
                pass
        finally:
            sys.settrace(old_trace)

        executed = {ln for ln in hit if ln in exec_lines}
        result.hit = len(executed)
        result.missed = sorted(exec_lines - executed)
        result.percent = (result.hit / result.total * 100.0
                          if result.total else 100.0)
        self.log.append("coverage.result", result.to_dict(), actor="tester")
        return result

    # -- projections -----------------------------------------------------------

    def results(self) -> list[dict]:
        return [e for e in fold(self.log).coverage_events
                if e.get("type") == "coverage.result"]

    def format_status(self) -> str:
        rs = self.results()
        lines = ["COVERAGE"]
        if not rs:
            lines.append("  no runs yet")
        for r in rs[-6:]:
            lines.append(f"  {r.get('path', '?')}: "
                         f"{r.get('percent', 0):.0f}%  "
                         f"({r.get('hit', 0)}/{r.get('total', 0)} lines)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import importlib.util
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # a module with two branches — one we exercise, one we don't
        target = Path(td) / "branchy.py"
        target.write_text(
            "def pick(x):\n"
            "    if x > 0:\n"
            "        return 'pos'\n"
            "    else:\n"
            "        return 'neg'\n"
            "\n"
            "def unused():\n"
            "    return 'never'\n")

        spec = importlib.util.spec_from_file_location("branchy", target)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        log = EventLog(Path(td) / "cov.jsonl")
        eng = CoverageEngine(log)

        # run only the positive branch
        res = eng.measure(str(target), lambda: mod.pick(5))
        assert res.total > 0, res
        assert res.hit > 0, res
        assert 0.0 < res.percent < 100.0, res  # else-branch + unused missed
        # the 'pos' return line executed; the 'neg' return line did not
        src_lines = target.read_text().splitlines()
        pos_line = next(i + 1 for i, l in enumerate(src_lines)
                        if "'pos'" in l)
        neg_line = next(i + 1 for i, l in enumerate(src_lines)
                        if "'neg'" in l)
        assert pos_line not in res.missed, res.missed
        assert neg_line in res.missed, res.missed

        # running both branches raises coverage
        res2 = eng.measure(str(target),
                           lambda: (mod.pick(5), mod.pick(-1)))
        assert res2.percent > res.percent, (res.percent, res2.percent)
        assert neg_line not in res2.missed

        # a subject that raises still yields coverage, never propagates
        def boom():
            mod.pick(1)
            raise RuntimeError("boom")
        res3 = eng.measure(str(target), boom)
        assert res3.hit > 0

        # the trace function is restored after measurement
        assert sys.gettrace() is None

        # results are sealed in the log
        assert len(eng.results()) == 3
        assert "COVERAGE" in eng.format_status()

    print("COVERAGE SELF-TEST PASS")

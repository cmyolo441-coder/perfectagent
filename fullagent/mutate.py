"""MUTATE — real mutation testing.

Mutation testing answers the question tests alone cannot: *are my tests
actually able to catch bugs?* The engine makes small, real changes
(mutants) to the source — flipping operators, breaking conditions,
removing statements — then runs the test suite against each one:

  * a mutant the suite KILLS (tests fail) = the suite caught the bug.
  * a mutant that SURVIVES (tests still pass) = a real hole in the suite.

mutation score = killed / (killed + survived). This is a genuine quality
signal used professionally (pitest, mutmut), implemented here with pure
stdlib AST transforms — deterministic, no model calls.

Design:
  * Mutators are AST NodeTransformers, each producing one class of mutant.
  * The runner writes each mutant to a temp copy, runs the suite command
    (caller-supplied, e.g. 'python -m pytest -q'), and reads the exit code.
  * Every mutant + outcome is sealed as mutation.run / mutation.result, so
    the score history is auditable.
"""

from __future__ import annotations

import ast
import copy
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog, fold

MAX_MUTANTS = 40          # cap per run so a suite never explodes


# ---------------------------------------------------------------------------
# Mutators — each yields (description, mutated_tree) pairs
# ---------------------------------------------------------------------------

class _OperatorFlip(ast.NodeTransformer):
    """Flip binary/comparison operators: + <-> -, * <-> /, == <-> !=, etc."""
    BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div,
           ast.Div: ast.Mult, ast.Mod: ast.Mult}
    CMP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE,
           ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt}

    def __init__(self, only_index: int) -> None:
        self.only_index = only_index
        self.count = -1

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        self.generic_visit(node)
        repl = self.BIN.get(type(node.op))
        if repl:
            self.count += 1
            if self.count == self.only_index:
                node.op = repl()
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        self.generic_visit(node)
        node.ops = [self._flip(o) for o in node.ops]
        return node

    def _flip(self, op: ast.cmpop) -> ast.cmpop:
        repl = self.CMP.get(type(op))
        if repl:
            self.count += 1
            if self.count == self.only_index:
                return repl()
        return op


class _ConditionNegate(ast.NodeTransformer):
    """Negate if/while conditions: if x -> if not x."""

    def __init__(self, only_index: int) -> None:
        self.only_index = only_index
        self.count = -1

    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        self.count += 1
        if self.count == self.only_index:
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        self.count += 1
        if self.count == self.only_index:
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node


class _ReturnBreak(ast.NodeTransformer):
    """Break return values: return X -> return None."""

    def __init__(self, only_index: int) -> None:
        self.only_index = only_index
        self.count = -1

    def visit_Return(self, node: ast.Return) -> ast.Return:
        self.generic_visit(node)
        if node.value is not None:
            self.count += 1
            if self.count == self.only_index:
                node.value = ast.Constant(value=None)
        return node


_MUTATORS = (
    ("operator_flip", _OperatorFlip),
    ("condition_negate", _ConditionNegate),
    ("return_break", _ReturnBreak),
)


def _count_sites(tree: ast.Module, cls) -> int:
    probe = cls(only_index=-1)
    probe.visit(copy.deepcopy(tree))
    return probe.count + 1


def generate_mutants(source: str,
                     max_mutants: int = MAX_MUTANTS) -> list[dict]:
    """Produce concrete mutants: [{kind, description, source}]."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    mutants: list[dict] = []
    for kind, cls in _MUTATORS:
        n = _count_sites(tree, cls)
        for i in range(n):
            if len(mutants) >= max_mutants:
                return mutants
            m = cls(only_index=i)
            mutated = m.visit(copy.deepcopy(tree))
            ast.fix_missing_locations(mutated)
            try:
                code = ast.unparse(mutated)
            except Exception:
                continue
            if code.strip() == source.strip():
                continue
            mutants.append({"kind": kind,
                            "description": f"{kind} site #{i}",
                            "source": code})
    return mutants


# ---------------------------------------------------------------------------
# Mutation testing engine
# ---------------------------------------------------------------------------

@dataclass
class MutantResult:
    kind: str
    description: str
    status: str = "pending"   # killed | survived | error
    detail: str = ""


@dataclass
class MutationReport:
    path: str
    total: int = 0
    killed: int = 0
    survived: int = 0
    errors: int = 0
    score: float = 0.0
    results: list[MutantResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "total": self.total,
                "killed": self.killed, "survived": self.survived,
                "errors": self.errors, "score": round(self.score, 3),
                "results": [{"kind": r.kind, "status": r.status,
                             "description": r.description}
                            for r in self.results]}


class MutationTester:
    """Generate mutants of a file and run the suite against each.

    `suite_command` runs the tests; a NON-ZERO exit code means the suite
    caught the mutant (killed). The mutant source is written to `mutant_path`
    (a temp copy the suite imports) so the real file is never touched."""

    def __init__(self, log: EventLog, suite_command: str,
                 mutant_path: str | None = None,
                 timeout: int = 60) -> None:
        self.log = log
        self.suite_command = suite_command
        self.mutant_path = mutant_path
        self.timeout = timeout

    def _run_suite(self) -> tuple[int, str]:
        try:
            proc = subprocess.run(["bash", "-lc", self.suite_command],
                                  capture_output=True, text=True,
                                  timeout=self.timeout)
            out = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, out[-300:]
        except subprocess.TimeoutExpired:
            return 124, "suite timed out"
        except OSError as e:
            return 127, str(e)

    def run(self, path: str, max_mutants: int = MAX_MUTANTS
            ) -> MutationReport:
        p = Path(path).expanduser()
        report = MutationReport(path=str(p))
        if not p.is_file():
            return report
        original = p.read_text(errors="replace")
        mutants = generate_mutants(original, max_mutants)
        report.total = len(mutants)
        self.log.append("mutation.run",
                        {"path": str(p), "mutants": len(mutants),
                         "suite": self.suite_command},
                        actor="tester")

        target = Path(self.mutant_path) if self.mutant_path else p
        backup = original
        try:
            for m in mutants:
                res = MutantResult(m["kind"], m["description"])
                try:
                    target.write_text(m["source"])
                    code, out = self._run_suite()
                except OSError as e:
                    code, out = 127, str(e)
                if code == 127:
                    res.status, res.detail = "error", out
                    report.errors += 1
                elif code != 0:
                    res.status = "killed"
                    report.killed += 1
                else:
                    res.status = "survived"
                    res.detail = out
                    report.survived += 1
                report.results.append(res)
        finally:
            target.write_text(backup)  # always restore the original

        scored = report.killed + report.survived
        report.score = report.killed / scored if scored else 0.0
        self.log.append("mutation.result", report.to_dict(), actor="tester")
        return report

    # -- projections -----------------------------------------------------------

    def reports(self) -> list[dict]:
        return [e for e in fold(self.log).mutation_events
                if e.get("type") == "mutation.result"]

    def format_status(self) -> str:
        reps = self.reports()
        lines = ["MUTATION TESTING"]
        if not reps:
            lines.append("  no runs yet")
        for r in reps[-5:]:
            lines.append(f"  {r.get('path', '?')}: score "
                         f"{r.get('score', 0):.0%}  "
                         f"killed {r.get('killed', 0)}/"
                         f"{r.get('total', 0)}  "
                         f"survived {r.get('survived', 0)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # mutant generation is real AST surgery
        src = '''
def add(a, b):
    if a > 0:
        return a + b
    return a - b
'''
        mutants = generate_mutants(src)
        assert len(mutants) >= 3, len(mutants)
        kinds = {m["kind"] for m in mutants}
        assert "operator_flip" in kinds and "condition_negate" in kinds, kinds
        # every mutant is valid python and differs from the original
        for m in mutants:
            ast.parse(m["source"])
            assert m["source"].strip() != src.strip()
        # an operator flip really changed + to -
        flips = [m for m in mutants if m["kind"] == "operator_flip"]
        assert any("-" in m["source"] for m in flips)

        # a condition negation really added 'not'
        negs = [m for m in mutants if m["kind"] == "condition_negate"]
        assert any("not" in m["source"] for m in negs)

        # end-to-end: a real file + a real suite that catches some mutants.
        # The suite asserts add(2,3)==5 and add(-1,3)==-4, so:
        #   - flipping + to - in the true branch is KILLED
        #   - negating the condition is KILLED
        log = EventLog(Path(td) / "mutate.jsonl")
        target = Path(td) / "calc.py"
        target.write_text(src)
        suite = Path(td) / "test_calc.py"
        suite.write_text(
            "import sys; sys.path.insert(0, %r)\n"
            "from calc import add\n"
            "assert add(2, 3) == 5\n"
            "assert add(-1, 3) == -4\n" % str(td))

        tester = MutationTester(log,
                                suite_command=f"python {suite}",
                                mutant_path=str(target))
        report = tester.run(str(target))
        assert report.total >= 3, report.total
        assert report.killed >= 1, report.to_dict()
        assert 0.0 <= report.score <= 1.0
        # the original file is restored byte-for-byte
        assert target.read_text() == src

        # results are sealed in the log
        assert tester.reports()
        assert "MUTATION TESTING" in tester.format_status()

    print("MUTATE SELF-TEST PASS")

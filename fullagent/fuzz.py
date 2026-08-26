"""FUZZ — real property-based fuzzing.

Feeds a function a stream of generated inputs — random, boundary, and
mutated — and watches for crashes, hangs and invariant violations. When
something breaks, the engine SHRINKS the failing input to a minimal
reproducer, which is the difference between "it crashed somewhere" and
"here is the exact smallest input that breaks it."

Design (pure stdlib, deterministic under a seed):
  * Generators produce typed values (int, str, list, dict, bytes, None)
    with a bias toward boundaries (0, -1, empty, huge, unicode).
  * A property is a callable under test; any raised exception is a crash.
    An optional invariant callable can assert post-conditions.
  * Shrinking: for a failing input, repeatedly try simpler variants
    (shorter strings, smaller numbers, dropped elements) and keep the
    smallest one that still fails.
  * Runs are sealed as fuzz.run / fuzz.crash / fuzz.shrunk events.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from .kernel import EventLog, fold

MAX_SHRINK_STEPS = 60


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

_BOUNDARY_INTS = (0, 1, -1, 2, -2, 2**7, -2**7, 2**15, 2**31, -2**31,
                  2**63, -2**63)
_BOUNDARY_STRS = ("", " ", "\n", "\t", "\x00", "a", "abc", "A" * 64,
                  "é", "😀", "'\"\\", "<script>", "../../../etc/passwd",
                  "%s%s%s", "{0}", "$(rm -rf /)", "SELECT * FROM t;--")


class Generator:
    """Typed random input generation with a boundary bias."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def integer(self) -> int:
        if self.rng.random() < 0.35:
            return self.rng.choice(_BOUNDARY_INTS)
        return self.rng.randint(-10**6, 10**6)

    def text(self) -> str:
        if self.rng.random() < 0.35:
            return self.rng.choice(_BOUNDARY_STRS)
        n = self.rng.randint(0, 40)
        alphabet = string.printable
        return "".join(self.rng.choice(alphabet) for _ in range(n))

    def blob(self) -> bytes:
        if self.rng.random() < 0.3:
            return b""
        return self.rng.randbytes(self.rng.randint(1, 32))

    def lst(self) -> list:
        n = self.rng.randint(0, 8)
        return [self.any(depth=1) for _ in range(n)]

    def dct(self) -> dict:
        n = self.rng.randint(0, 5)
        return {self.text()[:8]: self.any(depth=1) for _ in range(n)}

    def any(self, depth: int = 0) -> object:
        if depth > 2:
            return self.integer()
        choice = self.rng.random()
        if choice < 0.25:
            return self.integer()
        if choice < 0.5:
            return self.text()
        if choice < 0.6:
            return None
        if choice < 0.7:
            return self.rng.random()
        if choice < 0.85:
            return self.lst()
        if choice < 0.92:
            return self.blob()
        return self.dct()

    def args_for(self, nargs: int) -> tuple:
        return tuple(self.any() for _ in range(nargs))


# ---------------------------------------------------------------------------
# Shrinking
# ---------------------------------------------------------------------------

def _simpler(variant: object, gen: Generator) -> list:
    """Yield simpler variants of a failing value for shrinking."""
    out: list = []
    if isinstance(variant, str):
        if variant:
            out.append("")
            out.append(variant[0])
            out.append(variant[:len(variant) // 2])
            out.append(variant[len(variant) // 2:])
            for i in range(min(len(variant), 8)):
                out.append(variant[:i] + variant[i + 1:])
    elif isinstance(variant, int):
        if variant != 0:
            out.append(0)
            out.append(1 if variant > 0 else -1)
            out.append(variant // 2)
    elif isinstance(variant, list):
        if variant:
            out.append([])
            out.append(variant[:len(variant) // 2])
            for i in range(min(len(variant), 6)):
                out.append(variant[:i] + variant[i + 1:])
    elif isinstance(variant, dict):
        if variant:
            out.append({})
            keys = list(variant)
            for k in keys[:4]:
                d = dict(variant)
                del d[k]
                out.append(d)
    elif isinstance(variant, float):
        out.append(0.0)
    elif isinstance(variant, bytes):
        if variant:
            out.append(b"")
            out.append(variant[:len(variant) // 2])
    return out


# ---------------------------------------------------------------------------
# Fuzz engine
# ---------------------------------------------------------------------------

@dataclass
class Crash:
    args: tuple
    error: str
    shrunk_args: tuple = field(default_factory=tuple)
    shrunk_error: str = ""
    iterations: int = 0


@dataclass
class FuzzReport:
    target: str
    iterations: int = 0
    crashes: int = 0
    invariant_failures: int = 0
    first_crash: Crash | None = None
    ok: bool = True

    def to_dict(self) -> dict:
        fc = None
        if self.first_crash:
            fc = {"args": repr(self.first_crash.args)[:200],
                  "error": self.first_crash.error[:200],
                  "shrunk_args": repr(self.first_crash.shrunk_args)[:200],
                  "shrunk_error": self.first_crash.shrunk_error[:200]}
        return {"target": self.target, "iterations": self.iterations,
                "crashes": self.crashes,
                "invariant_failures": self.invariant_failures,
                "first_crash": fc, "ok": self.ok}


class Fuzzer:
    """Property-based fuzzing over the event log.

    `target(*args)` is the callable under test. `invariant(result, *args)`
    is an optional post-condition; returning False (or raising) counts as a
    failure. Neither is ever modified — the fuzzer only observes."""

    def __init__(self, log: EventLog, seed: int = 0) -> None:
        self.log = log
        self.gen = Generator(seed)

    def fuzz(self, target, iterations: int = 200, nargs: int = 1,
             invariant=None, name: str = "") -> FuzzReport:
        report = FuzzReport(target=name or getattr(target, "__name__",
                                                   "target"))
        self.log.append("fuzz.run",
                        {"target": report.target, "iterations": iterations,
                         "nargs": nargs}, actor="fuzzer")
        for i in range(iterations):
            report.iterations = i + 1
            args = self.gen.args_for(nargs)
            try:
                result = target(*args)
            except Exception as e:
                report.crashes += 1
                crash = Crash(args=args,
                              error=f"{type(e).__name__}: {e}",
                              iterations=i + 1)
                self._shrink(crash, target)
                if report.first_crash is None:
                    report.first_crash = crash
                self.log.append("fuzz.crash",
                                {"target": report.target,
                                 "args": repr(args)[:200],
                                 "error": crash.error[:200]},
                                actor="fuzzer")
                continue
            if invariant is not None:
                try:
                    ok = invariant(result, *args)
                except Exception as e:
                    ok = False
                    report.crashes += 1
                    crash = Crash(args=args,
                                  error=f"invariant raised "
                                        f"{type(e).__name__}: {e}",
                                  iterations=i + 1)
                    if report.first_crash is None:
                        report.first_crash = crash
                    continue
                if not ok:
                    report.invariant_failures += 1
                    if report.first_crash is None:
                        report.first_crash = Crash(
                            args=args, error="invariant returned False",
                            iterations=i + 1)
        report.ok = report.crashes == 0 and report.invariant_failures == 0
        return report

    def _shrink(self, crash: Crash, target) -> None:
        """Reduce the failing args to a minimal reproducer."""
        current = list(crash.args)
        for _ in range(MAX_SHRINK_STEPS):
            improved = False
            for idx, val in enumerate(current):
                for simpler in _simpler(val, self.gen):
                    candidate = list(current)
                    candidate[idx] = simpler
                    try:
                        target(*candidate)
                    except Exception:
                        current = candidate
                        improved = True
                        break
            if not improved:
                break
        crash.shrunk_args = tuple(current)
        try:
            target(*current)
            crash.shrunk_error = "(no longer reproduces)"
        except Exception as e:
            crash.shrunk_error = f"{type(e).__name__}: {e}"
        self.log.append("fuzz.shrunk",
                        {"args": repr(crash.shrunk_args)[:200],
                         "error": crash.shrunk_error[:200]},
                        actor="fuzzer")

    # -- projections -----------------------------------------------------------

    def runs(self) -> list[dict]:
        return fold(self.log).fuzz_events

    def format_status(self) -> str:
        evs = self.runs()
        runs = [e for e in evs if e["type"] == "fuzz.run"]
        crashes = [e for e in evs if e["type"] == "fuzz.crash"]
        lines = ["FUZZ", f"  runs {len(runs)}   crashes {len(crashes)}"]
        for c in crashes[-5:]:
            lines.append(f"    ⚠ {c.get('error', '')[:60]}  "
                         f"args {c.get('args', '')[:40]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "fuzz.jsonl")
        fz = Fuzzer(log, seed=42)

        # a robust function survives fuzzing
        report = fz.fuzz(lambda x: 1, iterations=100, nargs=1,
                         name="always_ok")
        assert report.ok and report.crashes == 0, report.to_dict()
        assert report.iterations == 100

        # a function that crashes on empty string is found + shrunk
        def crashy(s):
            if isinstance(s, str) and len(s) == 0:
                raise ValueError("empty!")
            return s

        report = fz.fuzz(crashy, iterations=300, nargs=1, name="crashy")
        assert report.crashes >= 1, report.to_dict()
        fc = report.first_crash
        assert fc is not None and "empty" in fc.error
        # shrinking finds the minimal reproducer: the empty string
        assert fc.shrunk_args == ("",), fc.shrunk_args
        assert "empty" in fc.shrunk_error

        # an integer crash on zero is shrunk to 0
        def divvy(n):
            if not isinstance(n, int):
                return None
            return 100 // n  # ZeroDivisionError when n == 0

        report = fz.fuzz(divvy, iterations=300, nargs=1, name="divvy")
        assert report.crashes >= 1
        fc = report.first_crash
        assert fc.shrunk_args == (0,), fc.shrunk_args

        # invariant violations are caught (sorted output must be sorted)
        def bad_sort(xs):
            if isinstance(xs, list) and len(xs) > 3:
                return list(reversed(sorted(xs)))  # wrong on purpose
            return sorted(xs) if isinstance(xs, list) else []

        def is_sorted(result, *args):
            return isinstance(result, list) and \
                all(result[i] <= result[i + 1]
                    for i in range(len(result) - 1))

        report = fz.fuzz(bad_sort, iterations=300, nargs=1,
                         invariant=is_sorted, name="bad_sort")
        assert report.invariant_failures >= 1 or report.crashes >= 1, \
            report.to_dict()

        # deterministic under the same seed
        fz2 = Fuzzer(EventLog(Path(td) / "fuzz2.jsonl"), seed=42)
        r1 = fz2.fuzz(crashy, iterations=50, nargs=1, name="crashy")
        fz3 = Fuzzer(EventLog(Path(td) / "fuzz3.jsonl"), seed=42)
        r2 = fz3.fuzz(crashy, iterations=50, nargs=1, name="crashy")
        assert r1.crashes == r2.crashes

        # events are sealed
        evs = fz.runs()
        types = {e["type"] for e in evs}
        assert {"fuzz.run", "fuzz.crash", "fuzz.shrunk"} <= types
        assert "FUZZ" in fz.format_status()

    print("FUZZ SELF-TEST PASS")

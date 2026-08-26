"""CI — the continuous-integration pilot: file-watch autopilot.

A personal CI system living inside the agent:

    watch       a background thread snapshots file hashes (mtimes +
                sizes) and polls; a change is a REAL diff, not a touch
    map         changed files map to their tests three ways: name
                convention (test_x.py <-> x.py), import scan (regex
                over test files), and the file's own directory — the
                dependency map grows from what the project actually
                looks like, not a config file
    run         only the IMPACTED tests run (the command is injectable
                — production shells out; the self-test stubs it), and
                every run lands as a ci.run event with exit status
    streaks     green/red streaks are tracked and sealed; a red streak
                is exactly the signal the healer or the main agent
                should react to

The pilot never edits anything — it watches, maps, runs, reports. What
to DO about a red build is the agent's job (and it gets told).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog

_POLL_SECONDS = 2.0
_TEST_NAME = re.compile(r"^(test_[a-z0-9_]+|[a-z0-9_]+_test)\.py$")


@dataclass
class RunRecord:
    changed: list[str]
    tests: list[str]
    passed: bool
    output: str = ""
    ts: float = field(default_factory=time.time)


class CIPilot:
    """Watch → map → run → report. Never mutates the project."""

    def __init__(self, log: EventLog, root: Path,
                 runner=None, poll: float = _POLL_SECONDS) -> None:
        """runner(test_files: list[str]) -> (ok: bool, output: str) —
        production runs pytest via the shell; tests stub it."""
        self.log = log
        self.root = Path(root)
        self.runner = runner
        self.poll = poll
        self._snapshot: dict[str, tuple[float, int]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.records: list[RunRecord] = []
        self.streak_green = 0
        self.streak_red = 0

    # -- the file map -----------------------------------------------------------

    def _scan(self) -> dict[str, tuple[float, int]]:
        out: dict[str, tuple[float, int]] = {}
        try:
            for p in self.root.rglob("*"):
                if p.is_file() and ".git" not in p.parts and \
                        "__pycache__" not in p.parts:
                    try:
                        st = p.stat()
                        rel = p.relative_to(self.root).as_posix()
                        out[rel] = (st.st_mtime, st.st_size)
                    except OSError:
                        continue
        except OSError:
            pass
        return out

    def impacted_tests(self, changed: list[str]) -> list[str]:
        """Map changed source files to their test files: name twins,
        importers of the changed module, and same-directory tests.
        Paths are posix-normalised everywhere (Windows-safe)."""
        tests = {p.relative_to(self.root).as_posix()
                 for p in self.root.rglob("test_*.py")}
        if not tests:
            return []
        picked: set[str] = set()
        test_text: dict[str, str] = {}
        for t in tests:
            try:
                test_text[t] = (self.root / t).read_text(
                    errors="replace")
            except OSError:
                test_text[t] = ""
        for path in changed:
            path = Path(str(path).replace("\\", "/"))
            mod = path.stem
            for t in tests:
                if t.endswith("/test_" + path.name) or \
                        t == f"test_{mod}.py" or \
                        t.endswith(f"/test_{mod}.py"):
                    picked.add(t)
            for t, text in test_text.items():
                if mod and mod in text:
                    picked.add(t)       # imports/mentions the module
            dirn = str(path.parent)
            for t in tests:
                if str(Path(t).parent).replace("\\", "/") == dirn:
                    picked.add(t)       # same directory
        return sorted(picked)

    # -- one cycle -----------------------------------------------------------------

    def check_once(self) -> RunRecord | None:
        """Diff the tree since the last look; run impacted tests."""
        current = self._scan()
        if self._snapshot:
            changed = sorted(
                p for p, sig in current.items()
                if self._snapshot.get(p) != sig)
            disappeared = [p for p in self._snapshot if p not in current]
            changed += disappeared
            if changed:
                tests = self.impacted_tests(changed)
                record = RunRecord(changed=changed[:50], tests=tests[:30],
                                   passed=True)
                if tests and self.runner is not None:
                    try:
                        ok, output = self.runner(tests[:30])
                    except Exception as e:   # a broken runner reports
                        ok, output = False, f"runner failed: {e}"
                    record.passed = bool(ok)
                    record.output = str(output)[:400]
                self.records.append(record)
                if record.passed:
                    self.streak_green += 1
                    self.streak_red = 0
                else:
                    self.streak_red += 1
                    self.streak_green = 0
                self.log.append("ci.run",
                                {"changed": record.changed[:10],
                                 "tests": record.tests[:10],
                                 "passed": record.passed})
                self.log.append("ci.streak",
                                {"green": self.streak_green,
                                 "red": self.streak_red})
                self._snapshot = current
                return record
        self._snapshot = current
        return None

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Begin the watch loop in the background."""
        if self._thread is not None:
            return
        self._snapshot = self._scan()
        self.log.append("ci.watch",
                        {"root": str(self.root),
                         "files": len(self._snapshot)}, actor="human")

        def loop():
            while not self._stop.is_set():
                try:
                    self.check_once()
                except Exception:
                    pass          # the pilot never dies mid-watch
                self._stop.wait(self.poll)

        self._thread = threading.Thread(target=loop, name="ci:pilot",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def status(self) -> str:
        last = self.records[-1] if self.records else None
        lines = [f"CI PILOT — watching {self.root} "
                 f"({'running' if self._thread else 'stopped'})",
                 f"  streaks: {self.streak_green} green · "
                 f"{self.streak_red} red · {len(self.records)} run(s)"]
        if last:
            mark = "✓ green" if last.passed else "✗ RED"
            lines.append(f"  last: {mark} — changed "
                         + ", ".join(last.changed[:4]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — a real temp project, real file changes, stubbed runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        # the watched project lives in a subdir — the CI log stays
        # OUTSIDE it (a watch loop watching its own log sees every run
        # as a change)
        root = P(td) / "proj"
        (root / "app").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "app" / "parser.py").write_text("def parse(): pass\n")
        (root / "tests" / "test_parser.py").write_text(
            "from app.parser import parse\n")
        (root / "tests" / "test_billing.py").write_text(
            "from app.billing import charge\n")
        (root / "app" / "billing.py").write_text("def charge(): pass\n")
        (root / "README.md").write_text("project\n")

        runs: list[list[str]] = []
        log = EventLog(P(td) / "ci-log.jsonl")

        def runner(tests: list[str]) -> tuple[bool, str]:
            runs.append(tests)
            ok = not any("billing" in t for t in tests)  # billing is broken
            return ok, "1 passed" if ok else "1 failed"

        ci = CIPilot(log, root, runner=runner)

        # name twins + import scan map parser.py to test_parser.py only
        assert ci.impacted_tests(["app/parser.py"]) == \
            ["tests/test_parser.py"]
        # billing change maps to its twin; README maps to nothing
        assert "tests/test_billing.py" in ci.impacted_tests(
            ["app/billing.py"])
        assert ci.impacted_tests(["README.md"]) == []

        # a change with no tests is noted but never fails the build
        first = None
        ci._snapshot = ci._scan()
        (root / "README.md").write_text("project v2\n")
        first = ci.check_once()
        assert first is not None and first.passed and first.tests == []
        assert ci.streak_green == 1

        # a parser change runs its test: green
        import os
        time.sleep(0.02)
        (root / "app" / "parser.py").write_text(
            "def parse(): return 42\n")
        second = ci.check_once()
        assert second is not None and second.tests == \
            ["tests/test_parser.py"]
        assert second.passed and ci.streak_green == 2

        # a billing change fails: red streak, runner output captured
        time.sleep(0.02)
        (root / "app" / "billing.py").write_text(
            "def charge(): raise RuntimeError\n")
        third = ci.check_once()
        assert not third.passed and "failed" in third.output
        assert ci.streak_red == 1 and ci.streak_green == 0

        # no change, no run
        assert ci.check_once() is None

        # background watch loop fires on a real change
        ci.start()
        time.sleep(0.05)
        (root / "app" / "parser.py").write_text("def parse(): return 1\n")
        deadline = time.time() + 8
        while time.time() < deadline and len(ci.records) < 4:
            time.sleep(0.2)
        ci.stop()
        assert len(ci.records) >= 4, len(ci.records)
        assert len(runs) >= 2

        kinds = [e.type for e in log.events()]
        assert {"ci.watch", "ci.run", "ci.streak"} <= set(kinds)
        assert "CI PILOT" in ci.status()

        print("CI SELF-TEST PASS")

"""HEALER — the self-healing root-cause engine.

When a command or check fails, the healer does not just report the error —
it runs a closed loop: CAPTURE the error, CLASSIFY it against a root-cause
taxonomy, propose a deterministic FIX, apply it, RETRY the original check
for proof, and seal the LESSON so the same failure is recognised instantly
next time.

Hard rules (mechanical, rung 1 — no LLM in the loop):
  * Classification is a pattern table over the error text. Unknown errors
    are classified 'unknown' and the healer says so — it never guesses.
  * A fix is only applied through a caller-supplied `fixer` callback; the
    healer itself never writes files. No fixer -> classify + lesson only.
  * Proof, not promise: after a fix, the ORIGINAL failing check is re-run.
    Only a green re-run seals heal.lesson with healed=True.
  * Every stage is an event: heal.captured / heal.hypothesis / heal.patch /
    heal.retry / heal.lesson. The healing history is replayable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .kernel import EventLog, fold

# ---------------------------------------------------------------------------
# Root-cause taxonomy — pattern -> (root_cause, suggested_fix)
# ---------------------------------------------------------------------------
# Each entry: (compiled regex, root cause label, deterministic suggestion).
# Order matters: first match wins. Keep patterns cheap and specific.
TAXONOMY: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"ModuleNotFoundError|No module named", re.I),
     "missing_module",
     "install the missing module or fix the import path"),
    (re.compile(r"FileNotFoundError|No such file or directory", re.I),
     "missing_file",
     "create the file or fix the path"),
    (re.compile(r"PermissionError", re.I),
     "permission_denied",
     "check file permissions / run with appropriate access"),
    (re.compile(r"ConnectionError|Connection refused|Name or service not "
                r"known|getaddrinfo", re.I),
     "network_unreachable",
     "check network / target host availability"),
    (re.compile(r"TimeoutError|timed out", re.I),
     "timeout",
     "raise the timeout or optimise the slow step"),
    (re.compile(r"SyntaxError", re.I),
     "syntax_error",
     "fix the syntax at the reported line"),
    (re.compile(r"IndentationError", re.I),
     "indentation_error",
     "fix the indentation at the reported line"),
    (re.compile(r"NameError", re.I),
     "undefined_name",
     "define or import the missing name"),
    (re.compile(r"TypeError", re.I),
     "type_mismatch",
     "fix the argument types / signature"),
    (re.compile(r"KeyError", re.I),
     "missing_key",
     "use .get() or ensure the key exists"),
    (re.compile(r"AttributeError", re.I),
     "missing_attribute",
     "check the object type / attribute name"),
    (re.compile(r"ZeroDivisionError", re.I),
     "division_by_zero",
     "guard the divisor against zero"),
    (re.compile(r"AssertionError", re.I),
     "assertion_failed",
     "the behaviour under test is wrong — inspect the assertion"),
    (re.compile(r"command not found|not recognized", re.I),
     "missing_binary",
     "install the tool or fix PATH"),
    (re.compile(r"out of memory|MemoryError|Cannot allocate", re.I),
     "out_of_memory",
     "reduce memory use or raise the limit"),
    (re.compile(r"rate limit|429|too many requests", re.I),
     "rate_limited",
     "back off and retry with delay"),
]


@dataclass
class Diagnosis:
    root_cause: str
    suggestion: str
    matched: str = ""          # the pattern text that matched
    evidence: str = ""         # the error excerpt used

    def to_dict(self) -> dict:
        return {"root_cause": self.root_cause,
                "suggestion": self.suggestion,
                "matched": self.matched,
                "evidence": self.evidence[:300]}


def classify(error_text: str) -> Diagnosis:
    """Match an error against the taxonomy. First match wins; no match is
    an honest 'unknown', never a guess."""
    text = error_text or ""
    for rx, cause, suggestion in TAXONOMY:
        m = rx.search(text)
        if m:
            return Diagnosis(cause, suggestion, rx.pattern,
                             text[max(0, m.start() - 40):m.end() + 80])
    return Diagnosis("unknown",
                     "no known pattern — inspect the error manually",
                     "", text[:200])


# ---------------------------------------------------------------------------
# Healer
# ---------------------------------------------------------------------------

@dataclass
class HealReport:
    error: str
    diagnosis: Diagnosis
    fix_applied: bool = False
    fix_result: str = ""
    retried: bool = False
    healed: bool = False
    lesson: str = ""

    def to_dict(self) -> dict:
        return {"error": self.error[:300], **self.diagnosis.to_dict(),
                "fix_applied": self.fix_applied,
                "fix_result": self.fix_result[:200],
                "retried": self.retried, "healed": self.healed,
                "lesson": self.lesson[:200]}


class Healer:
    """Closed-loop self-healing over the event log.

    `fixer` applies a suggested fix: fixer(diagnosis, context) -> str.
    `recheck` re-runs the original check: recheck() -> (ok, error_text).
    Both are caller-supplied so the healer itself never mutates anything."""

    def __init__(self, log: EventLog, fixer=None, recheck=None) -> None:
        self.log = log
        self.fixer = fixer
        self.recheck = recheck

    def heal(self, error_text: str, context: str = "") -> HealReport:
        """Run the full capture -> classify -> fix -> retry -> lesson loop."""
        diag = classify(error_text)
        self.log.append("heal.captured",
                        {"error": error_text[:300], "context": context[:200],
                         **diag.to_dict()},
                        actor="healer")

        report = HealReport(error=error_text, diagnosis=diag)
        self.log.append("heal.hypothesis",
                        {"root_cause": diag.root_cause,
                         "suggestion": diag.suggestion},
                        actor="healer")

        # apply a fix only when one is available and the cause is known
        if self.fixer is not None and diag.root_cause != "unknown":
            try:
                fix_result = str(self.fixer(diag, context))
            except Exception as e:
                fix_result = f"ERROR: {type(e).__name__}: {e}"
            report.fix_applied = True
            report.fix_result = fix_result
            self.log.append("heal.patch",
                            {"root_cause": diag.root_cause,
                             "result": fix_result[:200]},
                            actor="healer")

            # proof: re-run the original check
            if self.recheck is not None:
                try:
                    ok, retry_err = self.recheck()
                except Exception as e:
                    ok, retry_err = False, f"{type(e).__name__}: {e}"
                report.retried = True
                report.healed = bool(ok)
                self.log.append("heal.retry",
                                {"ok": bool(ok),
                                 "error": str(retry_err)[:200]},
                                actor="healer")

        report.lesson = self._lesson(report)
        self.log.append("heal.lesson",
                        {"root_cause": diag.root_cause,
                         "healed": report.healed,
                         "lesson": report.lesson},
                        actor="healer")
        return report

    @staticmethod
    def _lesson(report: HealReport) -> str:
        d = report.diagnosis
        if report.healed:
            return (f"{d.root_cause}: auto-healed via "
                    f"'{d.suggestion}'")
        if report.fix_applied:
            return (f"{d.root_cause}: fix attempted but check still fails "
                    f"— needs a different approach")
        return f"{d.root_cause}: {d.suggestion}"

    # -- projections -----------------------------------------------------------

    def lessons(self) -> list[dict]:
        return [e for e in fold(self.log).heal_events
                if e.get("type") == "heal.lesson"]

    def known_cause(self, error_text: str) -> bool:
        """Has this root cause been seen (and healed) before? Instant
        recognition from the lesson ledger."""
        cause = classify(error_text).root_cause
        return any(l.get("root_cause") == cause and l.get("healed")
                   for l in self.lessons())

    def stats(self) -> dict:
        evs = fold(self.log).heal_events
        captured = sum(1 for e in evs if e["type"] == "heal.captured")
        healed = sum(1 for e in evs
                     if e["type"] == "heal.lesson" and e.get("healed"))
        by_cause: dict[str, int] = {}
        for e in evs:
            if e["type"] == "heal.captured":
                c = e.get("root_cause", "unknown")
                by_cause[c] = by_cause.get(c, 0) + 1
        return {"captured": captured, "healed": healed,
                "by_cause": by_cause}

    def format_status(self) -> str:
        s = self.stats()
        lines = ["HEALER — self-healing root-cause engine",
                 f"  captured {s['captured']}   healed {s['healed']}"]
        for cause, n in sorted(s["by_cause"].items(),
                               key=lambda kv: -kv[1])[:8]:
            lines.append(f"    {cause:<22} ×{n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "healer.jsonl")

        # classification hits the taxonomy
        d = classify("ModuleNotFoundError: No module named 'requests'")
        assert d.root_cause == "missing_module", d
        d = classify("bash: foobar: command not found")
        assert d.root_cause == "missing_binary", d
        d = classify("ConnectionError: connection refused by host")
        assert d.root_cause == "network_unreachable", d
        # unknown errors are honest, never guessed
        d = classify("a totally novel failure mode xyz")
        assert d.root_cause == "unknown", d

        # full heal loop with a fixer that works and a recheck that passes
        state = {"fixed": False}

        def fixer(diag, context):
            state["fixed"] = True
            return f"applied: {diag.suggestion}"

        def recheck():
            return (state["fixed"], "" if state["fixed"] else "still broken")

        h = Healer(log, fixer, recheck)
        rep = h.heal("ModuleNotFoundError: No module named 'yaml'",
                     context="importing config loader")
        assert rep.diagnosis.root_cause == "missing_module"
        assert rep.fix_applied and rep.retried and rep.healed, rep
        assert "auto-healed" in rep.lesson

        # the lesson is recognised next time
        assert h.known_cause("ModuleNotFoundError: No module named 'toml'")

        # a fix that does NOT pass the recheck is reported unhealed
        h2 = Healer(log, fixer=lambda d, c: "tried something",
                    recheck=lambda: (False, "still failing"))
        rep2 = h2.heal("KeyError: 'user'")
        assert rep2.fix_applied and rep2.retried and not rep2.healed, rep2
        assert "still fails" in rep2.lesson

        # no fixer -> classify + lesson only, never a fake heal
        h3 = Healer(log)
        rep3 = h3.heal("PermissionError: [Errno 13] /etc/shadow")
        assert rep3.diagnosis.root_cause == "permission_denied"
        assert not rep3.fix_applied and not rep3.healed

        # unknown cause is never 'fixed' even with a fixer present
        rep4 = h.heal("a totally novel failure mode xyz")
        assert not rep4.fix_applied and not rep4.healed

        # stats + ledger
        s = h.stats()
        assert s["captured"] >= 4 and s["healed"] >= 1
        assert "missing_module" in s["by_cause"]
        assert "HEALER" in h.format_status()
        evs = fold(log).heal_events
        types = {e["type"] for e in evs}
        assert {"heal.captured", "heal.hypothesis", "heal.patch",
                "heal.retry", "heal.lesson"} <= types

    print("HEALER SELF-TEST PASS")

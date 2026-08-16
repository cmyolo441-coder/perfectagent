"""HOMEO — homeostasis: the agent that maintains itself.

A biological body does not wait to die of a fever — it regulates. The
agent's subsystems get the same treatment:

    vitals      mechanical measurements over a sliding window of the
                event log: tool error rate, mean tool latency, loop
                alerts per hour, context churn (compactions per turn),
                event-log growth, budget events
    rules       each vital has a healthy RANGE; a breach is a SYMPTOM
                with a named repair (re-seal prompts, restart the
                daemon, consolidate the brain, clear speculative
                caches, compact the log view) — every repair is an
                injectable callable, so the core stays pure and the
                self-test runs fully offline
    check       run vitals → breach rules → fire repairs → re-check
                the affected vital after the repair (did it help?).
                A repair that does not move its vital is recorded as
                ineffective — homeostasis is measured, not assumed

Every check and repair is sealed (homeo.check / homeo.repair); a
healthy system seals a quiet all-clear and touches nothing.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from .kernel import EventLog

_WINDOW = 120              # events considered "now"
_ERROR_RATE_MAX = 0.35     # more than this = symptom
_LATENCY_MAX_MS = 20_000   # mean tool duration ceiling
_LOOP_ALERTS_MAX = 3       # loop alerts in the window


@dataclass
class Vital:
    name: str
    value: float
    healthy: bool
    unit: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 3),
                "healthy": self.healthy, "unit": self.unit}


@dataclass
class RepairRecord:
    symptom: str
    action: str
    helped: bool
    before: float
    after: float


@dataclass
class CheckReport:
    vitals: list[Vital] = field(default_factory=list)
    repairs: list[RepairRecord] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(v.healthy for v in self.vitals)

    def format(self) -> str:
        lines = ["HOMEOSTASIS — " +
                 ("ALL VITALS NORMAL" if self.healthy else
                  f"{sum(1 for v in self.vitals if not v.healthy)} "
                  f"SYMPTOM(S)")]
        for v in self.vitals:
            icon = "✓" if v.healthy else "✗"
            lines.append(f"  {icon} {v.name:<16} {v.value:.3f}"
                         f"{v.unit}")
        for r in self.repairs:
            outcome = "helped" if r.helped else "NO EFFECT"
            lines.append(f"  🔧 {r.action} for {r.symptom} — {outcome} "
                         f"({r.before:.2f} → {r.after:.2f})")
        return "\n".join(lines)


class Homeostasis:
    """Vitals → rules → repairs → re-measure."""

    def __init__(self, log: EventLog,
                 repairs: dict[str, Callable[[], bool]] | None = None
                 ) -> None:
        """repairs: symptom name -> callable returning True if it ran.
        Defaults are wired by the Agent (re-seal, restart, sleep…)."""
        self.log = log
        self.repairs = repairs or {}
        self.last_report: CheckReport | None = None

    # -- vitals ----------------------------------------------------------------

    def vitals(self) -> list[Vital]:
        """Measure the subsystem health over the recent window."""
        events = self.log.events()[-_WINDOW:]
        tool_results = [e for e in events if e.type == "tool.result"]
        errors = sum(1 for e in tool_results
                     if e.data.get("status") == "error")
        error_rate = errors / len(tool_results) if tool_results else 0.0

        durations = [float(e.data.get("duration", 0) or 0)
                     for e in tool_results]
        mean_ms = (statistics.fmean(durations) * 1000
                   if durations else 0.0)

        loop_alerts = sum(1 for e in events if e.type == "loop.alert")
        compactions = sum(1 for e in events
                          if e.type == "context.compacted")
        budget_breaches = sum(1 for e in events
                              if e.type == "budget.event"
                              and e.data.get("kind") == "exceeded")

        return [
            Vital("tool_error_rate", error_rate,
                  error_rate <= _ERROR_RATE_MAX),
            Vital("tool_latency_ms", mean_ms, mean_ms <=
                  _LATENCY_MAX_MS, "ms"),
            Vital("loop_alerts", float(loop_alerts),
                  loop_alerts <= _LOOP_ALERTS_MAX),
            Vital("context_churn", float(compactions),
                  compactions <= 10),
            Vital("budget_breaches", float(budget_breaches),
                  budget_breaches == 0),
        ]

    # -- the loop ------------------------------------------------------------------

    def check_and_repair(self) -> CheckReport:
        """One homeostatic cycle: measure, repair what breaches,
        re-measure what was repaired."""
        report = CheckReport(vitals=self.vitals())
        vital_by_name = {v.name: v for v in report.vitals}
        for vital in report.vitals:
            if vital.healthy:
                continue
            repair = self.repairs.get(vital.name)
            if repair is None:
                continue                     # no known cure — report it
            try:
                ran = bool(repair())
            except Exception:
                ran = False
            after = self._recheck(vital.name)
            helped = ran and after is not None and after.healthy
            record = RepairRecord(symptom=vital.name,
                                  action=repair.__name__ or "repair",
                                  helped=helped, before=vital.value,
                                  after=after.value if after else
                                  vital.value)
            report.repairs.append(record)
            self.log.append("homeo.repair",
                            {"symptom": record.symptom,
                             "action": record.action,
                             "helped": helped,
                             "before": round(record.before, 3),
                             "after": round(record.after, 3)})
            if after is not None:
                vital_by_name[vital.name] = after
        report.vitals = list(vital_by_name.values())
        self.last_report = report
        self.log.append("homeo.check",
                        {"healthy": report.healthy,
                         "vitals": [v.to_dict() for v in report.vitals],
                         "repairs": len(report.repairs)}, actor="kernel")
        return report

    def _recheck(self, vital_name: str) -> Vital | None:
        for v in self.vitals():
            if v.name == vital_name:
                return v
        return None


# ---------------------------------------------------------------------------
# Self-test — planted symptoms trigger the right cures, offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "homeo.jsonl")

        # a healthy system: quiet all-clear, nothing touched
        log.append("user.message", {"text": "hi"})
        log.append("tool.call", {"name": "read_file", "args": {}})
        log.append("tool.result", {"name": "read_file", "status": "done",
                                   "duration": 0.4})
        fired: list[str] = []
        homeo = Homeostasis(log, repairs={
            "tool_error_rate": lambda: fired.append("reseed") or True,
            "loop_alerts": lambda: fired.append("restart_daemon")
            or True,
            "tool_latency_ms": lambda: fired.append("warm_caches")
            or True,
        })
        healthy = homeo.check_and_repair()
        assert healthy.healthy and not healthy.repairs and not fired

        # plant symptoms: errors + loop alerts + latency
        for i in range(10):
            log.append("tool.result",
                       {"name": "run_command",
                        "status": "error" if i < 5 else "done",
                        "duration": 30.0})
        for _ in range(5):
            log.append("loop.alert", {"kind": "exact_repeat"})
        report = homeo.check_and_repair()
        symptoms = {v.name for v in report.vitals if not v.healthy}
        assert "tool_error_rate" in symptoms, symptoms
        assert "loop_alerts" in symptoms, symptoms
        assert "tool_latency_ms" in symptoms, symptoms
        # the wired cures fired, once per symptom
        assert sorted(fired) == ["reseed", "restart_daemon",
                                 "warm_caches"], fired
        assert len(report.repairs) == 3
        # the error-rate repair "worked" (data unchanged, but the
        # record shows the re-measured value)
        rec = next(r for r in report.repairs
                   if r.symptom == "tool_error_rate")
        assert rec.before == rec.after          # honest re-measure

        # an unwired symptom is reported, never fabricated
        log2 = EventLog(Path(td) / "h2.jsonl")
        for i in range(8):
            log2.append("tool.result", {"name": "x",
                                        "status": "error",
                                        "duration": 0.1})
        log2.append("budget.event", {"kind": "exceeded"})
        h2 = Homeostasis(log2)                  # no repairs wired
        rep2 = h2.check_and_repair()
        assert not rep2.healthy and not rep2.repairs   # reported only

        # a crashing repair never kills the check
        def bad_repair():
            raise RuntimeError("cure exploded")

        h3 = Homeostasis(log2, repairs={"tool_error_rate": bad_repair})
        rep3 = h3.check_and_repair()
        assert not rep3.healthy                 # still reports vitals
        rec3 = next((r for r in rep3.repairs
                     if r.symptom == "tool_error_rate"), None)
        assert rec3 is not None and not rec3.helped

        # events + formatting
        kinds = {e.type for e in log.events()}
        assert {"homeo.check", "homeo.repair"} <= kinds
        assert "HOMEOSTASIS" in report.format()
        assert "ALL VITALS NORMAL" in healthy.format()

        print("HOMEOSTASIS SELF-TEST PASS")

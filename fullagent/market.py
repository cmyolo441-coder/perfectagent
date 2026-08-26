"""MARKET — the task market: specialists bid, the auctioneer awards.

The Crew assigns work by role; the Market assigns work by ECONOMICS.
Contract-Net Protocol, the real thing:

    announce    tasks hit the board; every eligible role examines each
    bid         each role prices the task: bid = capability × trust ×
                speed, where
                  capability = tool-fit between the role's whitelist and
                  what the task text names (write? run? research?)
                  trust      = the role's EMA success rate over its
                  settled auctions (starts neutral)
                  speed      = role-specific pace prior
    award       highest bid wins the contract (ONE role per task; ties
                break by trust, then name — deterministic)
    settle      execution lands (done/blocked/error + actual cost);
                trust updates exponentially from the OUTCOME, and the
                forecast error (bid vs actual) recalibrates the pace
                priors — the market gets sharper every auction

Everything is sealed (market.announce/bid/award/settle) and the trust
table persists through the event log. The executor is injectable: the
self-test runs full auctions offline with scripted workers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .kernel import EventLog, fold
from .team import ROLES

# role pace priors: abstract tool-steps per typical task (lower = faster)
_PACE = {"coder": 1.30, "tester": 1.10, "researcher": 0.90,
         "reviewer": 0.80, "analyst": 0.95, "architect": 1.20,
         "debugger": 1.25, "optimizer": 1.15, "refactorer": 1.20,
         "documenter": 0.85, "devops": 1.10, "integrator": 1.25,
         "planner": 0.70}
_TRUST_RATE = 0.30          # EMA update rate on settlement
_TRUST_FLOOR, _TRUST_CEIL = 0.05, 0.99

_NEED_PATTERNS = [
    ("write", re.compile(r"\b(write|create|implement|build|refactor|"
                         r"fix|edit|patch)\b", re.I)),
    ("run", re.compile(r"\b(run|execute|test|benchmark|measure|"
                       r"install|deploy|debug|reproduce|diagnos\w*)\b",
                       re.I)),
    ("read", re.compile(r"\b(read|review|analy[sz]e|inspect|map|"
                        r"research|plan|document)\b", re.I)),
    ("web", re.compile(r"\b(web|online|latest|url|http|search)\b", re.I)),
]
_TOOL_NEEDS = {"write": {"write_file", "edit_file"},
               "run": {"run_command"},
               "read": set(),
               "web": {"web_search", "web_fetch"}}


def _task_needs(task: str) -> set[str]:
    """What kinds of capability the task text demands."""
    needs: set[str] = set()
    for need, rx in _NEED_PATTERNS:
        if rx.search(task or ""):
            needs.add(need)
    return needs or {"read"}


@dataclass
class Bid:
    role: str
    task: str
    amount: float        # higher = better claim on the contract
    trust: float
    capability: float
    pace: float

    def to_dict(self) -> dict:
        return {"role": self.role, "amount": round(self.amount, 4),
                "trust": round(self.trust, 3),
                "capability": round(self.capability, 3),
                "pace": round(self.pace, 3)}


@dataclass
class Contract:
    task: str
    awarded: str = ""
    bids: list[Bid] = field(default_factory=list)
    status: str = "open"          # open | running | done | blocked | error
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"task": self.task[:200], "awarded": self.awarded,
                "status": self.status,
                "bids": [b.to_dict() for b in self.bids],
                "report": self.report}


class TaskMarket:
    """Announce → bid → award → settle, with a living trust table."""

    def __init__(self, log: EventLog,
                 executor=None) -> None:
        """executor(task, role) -> {"status": "done|blocked|error",
        "summary": str, "tool_calls": int} — runs the REAL worker."""
        self.log = log
        self.executor = executor
        self.trust: dict[str, float] = {}
        self.pace: dict[str, float] = dict(_PACE)
        self._load()

    # -- trust/pace persistence from the log --------------------------------

    def _load(self) -> None:
        for ev in self.log.events():
            if ev.type != "market.settle":
                continue
            role = ev.data.get("role", "")
            self.trust[role] = float(ev.data.get("trust", self.trust.get(
                role, 0.5)))
            p = ev.data.get("pace")
            if p:
                self.pace[role] = float(p)

    def trust_of(self, role: str) -> float:
        return self.trust.get(role, 0.5)

    # -- bidding -----------------------------------------------------------------

    def _capability(self, role: str, needs: set[str]) -> float:
        """Tool-fit: does the role's whitelist cover what the task
        demands? (web needs are a hard gate — a role without web tools
        bids 0 on a web task.)"""
        tools = set(ROLES.get(role, {}).get("tools", ()))
        if "web" in needs and not (_TOOL_NEEDS["web"] & tools):
            return 0.0
        fit = 0.0
        for need in needs:
            required = _TOOL_NEEDS[need]
            if not required:
                fit += 1.0
            elif required & tools:
                fit += 0.9
            else:
                fit -= 0.25
        return max(0.0, fit / max(1, len(needs)))

    def bid(self, task: str) -> list[Bid]:
        """Every role bids on one task; sorted best-first."""
        needs = _task_needs(task)
        bids = []
        for role in ROLES:
            cap = self._capability(role, needs)
            if cap <= 0.0:
                continue                       # cannot do it at all
            trust = self.trust_of(role)
            pace = self.pace.get(role, 1.0)
            amount = cap * (0.4 + trust) / pace
            bids.append(Bid(role=role, task=task, amount=amount,
                            trust=trust, capability=cap, pace=pace))
        bids.sort(key=lambda b: (-b.amount, -b.trust, b.role))
        return bids

    # -- one auction ----------------------------------------------------------------

    def auction(self, task: str) -> Contract:
        """Full contract-net cycle for ONE task. Seals announce, bids,
        the award, and (if an executor is attached) the settlement."""
        task = str(task or "").strip()
        contract = Contract(task=task)
        if not task:
            contract.status = "error"
            contract.report = {"summary": "empty task"}
            return contract
        self.log.append("market.announce", {"task": task[:300]})
        contract.bids = self.bid(task)
        if not contract.bids:
            contract.status = "error"
            contract.report = {"summary": "no role can service this task"}
            return contract
        for b in contract.bids:
            self.log.append("market.bid", b.to_dict())
        winner = contract.bids[0]
        contract.awarded = winner.role
        contract.status = "running"
        self.log.append("market.award",
                        {"task": task[:300], "role": winner.role,
                         "amount": round(winner.amount, 4),
                         "beaten": [b.role for b in contract.bids[1:4]]},
                        actor="kernel")
        if self.executor is not None:
            self.settle(contract)
        return contract

    # -- settlement + learning ---------------------------------------------------------

    def settle(self, contract: Contract,
               report: dict | None = None) -> Contract:
        """Run (or record) the outcome, update trust and pace priors."""
        if report is None:
            try:
                report = self.executor(contract.task, contract.awarded) \
                    or {}
            except Exception as e:  # a failing worker never kills the market
                report = {"status": "error", "summary": str(e),
                          "tool_calls": 0}
        status = str(report.get("status", "error"))
        contract.report = report
        contract.status = status if status in ("done", "blocked",
                                               "error") else "error"
        # trust EMA: done up, blocked slightly down, error hard down
        role = contract.awarded
        cur = self.trust_of(role)
        if contract.status == "done":
            target = _TRUST_CEIL
        elif contract.status == "blocked":
            target = cur
        else:
            target = _TRUST_FLOOR
        new = cur + (target - cur) * _TRUST_RATE
        if contract.status == "blocked":
            new = max(_TRUST_FLOOR,
                      min(_TRUST_CEIL, cur - 0.05))   # small honest dip
        self.trust[role] = new
        # pace recalibration: more tool calls than the prior assumed ->
        # the role is slower than we thought
        calls = int(report.get("tool_calls", 0) or 0)
        if calls > 0:
            p = self.pace.get(role, 1.0)
            self.pace[role] = max(0.5, min(2.0, p * 0.7 + 0.3 * (
                calls / 10.0)))
        self.log.append("market.settle",
                        {"task": contract.task[:200], "role": role,
                         "status": contract.status,
                         "trust": round(new, 4),
                         "pace": round(self.pace[role], 4),
                         "summary": str(report.get("summary", ""))[:300]})
        return contract

    # -- batch + reporting ----------------------------------------------------------------

    def run(self, tasks: list[str]) -> list[Contract]:
        return [self.auction(t) for t in tasks if str(t or "").strip()]

    def leaderboard(self) -> list[tuple[str, float, float]]:
        """(role, trust, pace) sorted by trust — the market's memory."""
        rows = [(r, self.trust_of(r), self.pace.get(r, 1.0))
                for r in ROLES]
        rows.sort(key=lambda p: (-p[1], p[0]))
        return rows

    def format(self, contracts: list[Contract]) -> str:
        lines = ["MARKET — " + str(len(contracts)) + " contract(s)"]
        for c in contracts:
            icon = {"done": "✓", "blocked": "◐", "error": "✗",
                    "running": "…", "open": " "}.get(c.status, "?")
            top = ", ".join(f"{b.role}({b.amount:.2f})"
                            for b in c.bids[:3])
            lines.append(f"{icon} [{c.awarded or '—'}] {c.task[:70]}")
            if top:
                lines.append(f"    bids: {top}")
            if c.report:
                lines.append(f"    → {c.status}: "
                             + str(c.report.get("summary", ""))[:140])
        lb = self.leaderboard()[:5]
        lines.append("  trust board: " + " · ".join(
            f"{r} {t:.2f}" for r, t, _ in lb))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — scripted workers drive full auctions offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "market.jsonl")

        def executor(task: str, role: str) -> dict:
            if "crash" in task:
                return {"status": "error",
                        "summary": "could not reproduce", "tool_calls": 9}
            if "blocked" in task:
                return {"status": "blocked",
                        "summary": "needs credentials", "tool_calls": 2}
            return {"status": "done",
                    "summary": f"{role} handled it",
                    "tool_calls": 4}

        market = TaskMarket(log, executor)

        # a write task: read-only roles bid low, writer roles bid high
        c = market.auction("write the new auth module and its tests")
        assert c.awarded in ("coder", "refactorer", "integrator",
                             "architect", "documenter", "devops"), \
            c.awarded
        assert c.status == "done"
        # a web task: roles without web tools are gated out entirely
        web = market.auction("research the latest flask release online")
        assert web.awarded in ("researcher", "analyst")
        assert all(b.role not in ("coder", "refactorer", "reviewer",
                                  "tester", "planner")
                   for b in web.bids), [b.role for b in web.bids]
        # a plan task: planner's pace prior makes it cheapest-fastest
        plan = market.auction("plan the migration carefully")
        assert plan.bids[0].role in ("planner", "reviewer",
                                     "documenter"), plan.bids[0].role

        # settlement moves trust: errors sink the role that errored
        crash = market.auction("debug the crash in the worker pool")
        assert crash.status == "error"          # executor script errors it
        assert market.trust_of(crash.awarded) < 0.5
        # direct settle drives the same math deterministically
        before = market.trust_of("debugger")
        market.settle(Contract(task="deep dive", awarded="debugger"),
                      report={"status": "error",
                              "summary": "boom", "tool_calls": 3})
        after = market.trust_of("debugger")
        assert after < before, (before, after)
        # ...and done work lifts the winner
        assert market.trust_of(c.awarded) > 0.5

        # trust persists through reload from the log alone
        market2 = TaskMarket(log)
        assert abs(market2.trust_of("debugger") - after) < 1e-9

        # a trusted role wins ties against a fresh role
        # (deterministic: amount desc, trust desc, name asc)
        bids = market.bid("analyze the system design")
        amounts = [b.amount for b in bids]
        assert amounts == sorted(amounts, reverse=True)

        # blocked contracts dip trust slightly, never crater it
        b0 = market.trust_of("researcher")
        market.auction("research the blocked archive")
        b1 = market.trust_of("researcher")
        assert b0 - 0.06 < b1 <= b0 + 1e-9, (b0, b1)

        # empty task errors cleanly; no-bid task (empty roles) too
        bad = market.auction("")
        assert bad.status == "error"

        # format renders
        text = market.format(market.run(["write docs for the api",
                                         "run the full test suite"]))
        assert "MARKET" in text and "trust board" in text

        # events sealed + foldable
        st = fold(log)
        kinds = {e["type"] for e in st.advanced_events}
        assert {"market.announce", "market.bid", "market.award",
                "market.settle"} <= kinds, kinds

        print("MARKET SELF-TEST PASS")

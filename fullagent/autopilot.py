"""AutoPilot — the self-enabling routing brain.

The agent decides FOR ITSELF, before every turn, which powers the task
needs — and enables them on its own:

  * GOAL MODE      — the request is a verifiable mission ("fix", "add",
                     "make X pass") -> auto-draft a machine-checkable
                     goal contract.
  * REAL-TIME WEB  — the request needs live data ("latest", "today",
                     "current", prices, news) -> steer the turn to use
                     web_search for real-time facts.

Routing is deterministic Python (rung 1-2 of the Determinism Ladder):
fast, free, and explainable. Every decision is logged as an
'autopilot.route' event and surfaced in the TUI, so nothing the agent
enables is ever hidden (axiom A7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import EventLog

# -- trigger vocabularies ----------------------------------------------------

_WEB_TRIGGERS = (
    "latest", "today", "current", "right now", "real time", "real-time",
    "news", "headline", "price", "stock", "weather", "score", "release",
    "version", "who won", "update on", "breaking", "live", "now",
    "aaj", "abhi", "taaza", "naya", "sabse naya", "rate", "exchange",
)

_GOAL_VERBS = (
    "fix", "implement", "add", "create", "build", "make", "ensure",
    "refactor", "write", "develop", "set up", "setup", "migrate",
    "convert", "optimize", "optimise", "banao", "thik karo", "likho",
)

_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which", "is ", "are ",
    "do ", "does ", "can ", "kya", "kaun", "kab", "kahan", "kyu", "kaise",
)

# segment-free trigger regex — word-boundary match so "rate" never
# fires inside "generate"
_WEB_TRIGGER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _WEB_TRIGGERS) + r")\b")


@dataclass
class RouteDecision:
    """What the AutoPilot enabled for this turn, and why."""
    suggest_goal: bool = False
    goal_statement: str = ""
    goal_clauses: list[dict] = field(default_factory=list)
    use_web: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.suggest_goal or self.use_web

    def summary(self) -> str:
        bits = []
        if self.suggest_goal:
            bits.append(f"⚡ goal mode: {len(self.goal_clauses)} clause(s) "
                        "auto-drafted")
        if self.use_web:
            bits.append("⚡ real-time web mode")
        return "  ·  ".join(bits) if bits else ""


class AutoPilot:
    """Deterministic pre-turn router. Never calls a model itself."""

    def __init__(self, log: EventLog, enabled: bool = True) -> None:
        self.log = log
        self.enabled = enabled

    # -- public -------------------------------------------------------------

    def route(self, text: str, goal_active: bool,
              autonomy: int = 3) -> RouteDecision:
        """Inspect the user message and decide what to enable."""
        d = RouteDecision()
        if not self.enabled or not text.strip():
            return d

        low = text.lower()
        is_question = (text.strip().endswith("?")
                       or low.startswith(_QUESTION_STARTERS))

        # 1. real-time web — questions about live data
        hits = _WEB_TRIGGER_RE.findall(low)
        if hits:
            d.use_web = True
            d.reasons.append("real-time web: live-data trigger(s) "
                             + ", ".join(repr(h) for h in hits[:3]))

        # 2. goal mode — a verifiable mission, not a question
        if not goal_active and not is_question:
            if any(v in low for v in _GOAL_VERBS):
                d.suggest_goal = True
                d.goal_statement = text.strip()[:100]
                d.goal_clauses = self._draft_clauses(text)
                d.reasons.append("goal mode: mission verb detected, "
                                 f"{len(d.goal_clauses)} clause(s) drafted")

        if d.active:
            self.log.append("autopilot.route",
                            {"suggest_goal": d.suggest_goal,
                             "use_web": d.use_web,
                             "reasons": d.reasons},
                            actor="system")
        return d

    # -- goal drafting (rung 1-3) ----------------------------------------------

    def _draft_clauses(self, text: str) -> list[dict]:
        """Derive machine-checkable clauses from the request itself.

        Only predicates we can actually construct are added; anything else
        becomes an advisory clause a human can prove. A goal that cannot
        be failed is not a goal — so we always try for a real predicate."""
        low = text.lower()
        clauses: list[dict] = []

        # tests mentioned -> find the real test command on disk (rung 3)
        if any(w in low for w in ("test", "pytest", "suite")):
            cmd = self._detect_test_command()
            if cmd:
                clauses.append({
                    "id": "C1", "text": "test suite passes", "weight": 1.0,
                    "proof": {"type": "exit_code", "command": cmd,
                              "expect": 0}})

        # explicit paths mentioned -> they must exist when done
        paths = re.findall(r"(?:^|\s)([./~][\w./~-]+\.\w{1,6})", text)
        for i, p in enumerate(paths[:2]):
            clauses.append({
                "id": f"C{len(clauses) + 1}",
                "text": f"artifact exists: {p}", "weight": 1.0,
                "proof": {"type": "file_exists", "path": p}})

        if not clauses:
            # nothing machine-checkable could be derived — advisory clause,
            # provable by the human or by a later /goal prove
            clauses.append({"id": "C1", "text": text.strip()[:80],
                            "weight": 1.0, "advisory": True})
        # normalise ids
        for i, c in enumerate(clauses, 1):
            c["id"] = f"C{i}"
        return clauses

    @staticmethod
    def _detect_test_command() -> str | None:
        """Probe the cwd for a real test runner (deterministic, rung 3)."""
        cwd = Path.cwd()
        if (cwd / "pytest.ini").exists() or (cwd / "pyproject.toml").exists() \
                or (cwd / "tests").is_dir() or (cwd / "test").is_dir():
            return "python -m pytest -q"
        if any(cwd.glob("test_*.py")) or any(cwd.glob("*_test.py")):
            return "python -m pytest -q"
        if (cwd / "package.json").exists():
            return "npm test"
        return None


if __name__ == "__main__":
    import tempfile

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EventLog(Path(td) / "autopilot-test.jsonl")
            ap = AutoPilot(log)

            # goal mode: mission verb
            d = ap.route("fix the login bug in auth.py", goal_active=False)
            assert d.suggest_goal, d
            assert d.goal_clauses and d.goal_clauses[0]["id"] == "C1", d

            # questions never trigger goal mode
            d = ap.route("what is the best way to fix a login bug?",
                          goal_active=False)
            assert not d.suggest_goal, d

            # an active goal suppresses auto-drafting
            d = ap.route("implement the new parser now", goal_active=True)
            assert not d.suggest_goal, d

            # real-time web triggers
            d = ap.route("what is the latest news about AI today?",
                          goal_active=False)
            assert d.use_web, d
            d = ap.route("tell me the current bitcoin price",
                          goal_active=False)
            assert d.use_web, d

            # plain chat: nothing enabled
            d = ap.route("hello, how are you", goal_active=False)
            assert not d.active, d

            # disabled autopilot routes nothing
            ap.enabled = False
            d = ap.route("fix everything and also test everything please",
                          goal_active=False)
            assert not d.active, d
            ap.enabled = True

            # route events are sealed in the log
            st_events = [e for e in log.events()
                          if e.type == "autopilot.route"]
            assert len(st_events) >= 1

            # summary renders
            d = ap.route("build the parser module", goal_active=False)
            assert "goal mode" in d.summary()

            print("AUTOPILOT SELF-TEST PASS")

    _self_test()

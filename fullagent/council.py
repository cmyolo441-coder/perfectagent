"""COUNCIL — multi-agent adversarial debate.

For high-stakes decisions, one opinion is not enough. The Council convenes
a structured debate over a question:

  * THESIS      — argues FOR the proposition.
  * ANTITHESIS  — argues AGAINST it, and must attack the thesis's weakest
                  point (not just restate a contrary view).
  * SYNTHESIS   — a blind judge: it sees ONLY the two arguments, never the
                  question's framing, never the speakers' identities, and
                  decides on argument strength alone. (Blind review is the
                  point — a judge that sees the defence is not a judge.)

The verdict is sealed as council.verdict with the winning side, the
deciding reason, and a confidence score. Positions are sealed as
council.position events, so every debate is replayable and auditable.

Design:
  * `speaker` produces one position: speaker(role, brief) -> str. In the
    agent it is bound to a model call through the Mastermind gate; in tests
    it is a stub. The council itself is deterministic orchestration.
  * Positions run one at a time (thesis, then antithesis) — no parallel
    subagents.
  * The synthesis prompt is built mechanically from the two positions and
    carries NO other context — the blindness is structural.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .kernel import EventLog, fold

ROLES = ("thesis", "antithesis")
MAX_POSITION_CHARS = 1200

_THESIS_BRIEF = (
    "You are the THESIS in a structured debate. Argue FOR the proposition "
    "below. Make the strongest honest case: concrete evidence, mechanisms, "
    "trade-offs in your favour. <= 150 words. End with a line exactly:\n"
    "STRONGEST POINT: <one sentence>")

_ANTITHESIS_BRIEF = (
    "You are the ANTITHESIS in a structured debate. Argue AGAINST the "
    "proposition below AND attack the strongest argument FOR it. Concrete "
    "risks, failure modes, costs. <= 150 words. End with a line exactly:\n"
    "STRONGEST POINT: <one sentence>")

_SYNTHESIS_BRIEF = (
    "You are a BLIND JUDGE in a debate. You see only two anonymised "
    "arguments — you do not know the question's framing or who wrote "
    "what. Decide purely on argument strength: evidence, specificity, "
    "and how well each side rebuts the other.\n\n"
    "ARGUMENT A:\n{thesis}\n\nARGUMENT B:\n{antithesis}\n\n"
    "Reply in EXACTLY this form:\n"
    "WINNER: A | B | DRAW\n"
    "CONFIDENCE: <0-100>%\n"
    "REASON: <one or two sentences>")

_WINNER_RE = re.compile(r"winner:\s*(A|B|DRAW)", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence:\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Position:
    role: str
    text: str = ""
    strongest_point: str = ""
    ok: bool = True
    error: str = ""


@dataclass
class Verdict:
    question: str
    winner: str = ""          # thesis | antithesis | draw
    confidence: float = 0.0
    reason: str = ""
    positions: dict = field(default_factory=dict)
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {"question": self.question[:300], "winner": self.winner,
                "confidence": round(self.confidence, 1),
                "reason": self.reason[:300],
                "positions": {k: v[:MAX_POSITION_CHARS]
                              for k, v in self.positions.items()},
                "ok": self.ok, "error": self.error[:200]}


def _strongest_point(text: str) -> str:
    m = re.search(r"strongest point:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_synthesis(text: str) -> tuple[str, float, str]:
    """(winner A|B|DRAW, confidence, reason) from the judge's reply."""
    wm = _WINNER_RE.search(text)
    winner = wm.group(1).upper() if wm else "DRAW"
    # The judge has to write "CONFIDENCE: NN%" or the parse fails. The
    # previous default of 50.0 was a coin-flip — a non-compliant judge
    # got its verdict trusted at exactly the bar of "no information".
    # A missing or unparseable confidence is genuinely "unknown", not
    # 50/50, so we default to 0.0 (the verdict is still recorded, but
    # the caller can see there's no signal and act accordingly).
    cm = _CONF_RE.search(text)
    conf = float(cm.group(1)) if cm else 0.0
    rm = re.search(r"reason:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = rm.group(1).strip() if rm else ""
    return winner, min(100.0, max(0.0, conf)), reason[:300]


# ---------------------------------------------------------------------------
# Council
# ---------------------------------------------------------------------------

class Council:
    """Convene adversarial debates over the event log.

    `speaker(role, brief) -> str` produces one position. The synthesis
    brief is built from the two positions alone — structurally blind."""

    def __init__(self, log: EventLog, speaker=None,
                 timeout: float = 120.0) -> None:
        self.log = log
        self.speaker = speaker
        self.timeout = timeout

    def convene(self, question: str) -> Verdict:
        """Run one full debate: thesis, then antithesis (serial), then the
        blind synthesis. Never raises — failures land in Verdict.error."""
        council_id = f"council-{self.log.head() + 1}"
        self.log.append("council.convened",
                        {"council_id": council_id,
                         "question": question[:300]},
                        actor="council")
        verdict = Verdict(question=question)
        if self.speaker is None:
            verdict.ok = False
            verdict.error = "no speaker attached"
            self._seal_verdict(council_id, verdict)
            return verdict

        briefs = {"thesis": f"{_THESIS_BRIEF}\n\nPROPOSITION: {question}",
                  "antithesis": f"{_ANTITHESIS_BRIEF}\n\n"
                                f"PROPOSITION: {question}"}

        def _speak(role: str) -> Position:
            try:
                text = str(self.speaker(role, briefs[role])).strip()
                if not text:
                    return Position(role, ok=False, error="empty position")
                return Position(role, text=text[:MAX_POSITION_CHARS],
                                strongest_point=_strongest_point(text))
            except Exception as e:
                return Position(role, ok=False,
                                error=f"{type(e).__name__}: {e}")

        positions = [_speak(role) for role in ROLES]

        for p in positions:
            self.log.append("council.position",
                            {"council_id": council_id, "role": p.role,
                             "text": p.text[:MAX_POSITION_CHARS],
                             "strongest_point": p.strongest_point,
                             "ok": p.ok, "error": p.error},
                            actor=f"council:{p.role}")
            if p.ok:
                verdict.positions[p.role] = p.text

        failed = [p for p in positions if not p.ok]
        if len(failed) == 2:
            verdict.ok = False
            verdict.error = "; ".join(f"{p.role}: {p.error}" for p in failed)
            self._seal_verdict(council_id, verdict)
            return verdict
        if len(failed) == 1:
            # one side silent -> the other wins by default, low confidence
            winner_role = next(p.role for p in positions if p.ok)
            verdict.winner = winner_role
            verdict.confidence = 30.0
            verdict.reason = (f"{failed[0].role} failed "
                              f"({failed[0].error}); default to the side "
                              "that argued")
            self._seal_verdict(council_id, verdict)
            return verdict

        # blind synthesis: only the two arguments, anonymised as A/B
        synthesis_brief = _SYNTHESIS_BRIEF.format(
            thesis=verdict.positions["thesis"],
            antithesis=verdict.positions["antithesis"])
        try:
            raw = str(self.speaker("synthesis", synthesis_brief))
        except Exception as e:
            verdict.ok = False
            verdict.error = f"synthesis failed: {type(e).__name__}: {e}"
            self._seal_verdict(council_id, verdict)
            return verdict

        side, conf, reason = _parse_synthesis(raw)
        verdict.winner = {"A": "thesis", "B": "antithesis"}.get(side, "draw")
        verdict.confidence = conf
        verdict.reason = reason
        self._seal_verdict(council_id, verdict)
        return verdict

    def _seal_verdict(self, council_id: str, verdict: Verdict) -> None:
        self.log.append("council.verdict",
                        {"council_id": council_id, **verdict.to_dict()},
                        actor="council")

    # -- projections -----------------------------------------------------------

    def verdicts(self) -> list[dict]:
        return [e for e in fold(self.log).council_events
                if e.get("type") == "council.verdict"]

    def format_status(self) -> str:
        vs = self.verdicts()
        lines = ["COUNCIL — adversarial debate",
                 f"  debates decided: {len(vs)}"]
        for v in vs[-6:]:
            lines.append(f"    {v.get('winner', '?'):<11} "
                         f"conf {v.get('confidence', 0):.0f}%  "
                         f"{v.get('question', '')[:48]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "council.jsonl")

        # scripted speaker: thesis strong, antithesis weak, judge picks A
        def speaker(role: str, brief: str) -> str:
            if role == "thesis":
                return ("Evidence shows the change cuts latency 40%.\n"
                        "STRONGEST POINT: measured 40% latency reduction")
            if role == "antithesis":
                return ("It might be risky maybe.\n"
                        "STRONGEST POINT: vague concern")
            if role == "synthesis":
                # the synthesis brief must be blind — no proposition text
                assert "PROPOSITION" not in brief, "synthesis is not blind"
                return ("WINNER: A\nCONFIDENCE: 82%\n"
                        "REASON: A cites measurements; B is vague.")
            raise AssertionError(f"unexpected role {role}")

        c = Council(log, speaker)
        v = c.convene("should we migrate the database this week")
        assert v.ok, v.error
        assert v.winner == "thesis", v.winner
        assert abs(v.confidence - 82.0) < 1e-9
        assert "measurements" in v.reason
        assert set(v.positions) == {"thesis", "antithesis"}

        # judge picks B -> antithesis wins
        def speaker_b(role: str, brief: str) -> str:
            if role == "synthesis":
                return "WINNER: B\nCONFIDENCE: 61%\nREASON: B rebuts A."
            return f"{role} argues its side.\nSTRONGEST POINT: a point"

        v2 = Council(log, speaker_b).convene("adopt framework X")
        assert v2.winner == "antithesis" and v2.confidence == 61.0, v2

        # a DRAW verdict maps through
        def speaker_draw(role: str, brief: str) -> str:
            if role == "synthesis":
                return "WINNER: DRAW\nCONFIDENCE: 50%\nREASON: even."
            return f"{role} case\nSTRONGEST POINT: p"

        assert Council(log, speaker_draw).convene("q").winner == "draw"

        # one silent side -> default win at low confidence, no crash
        def speaker_flaky(role: str, brief: str) -> str:
            if role == "antithesis":
                raise RuntimeError("model timeout")
            if role == "synthesis":
                return "WINNER: A\nCONFIDENCE: 90%\nREASON: x"
            return "thesis stands\nSTRONGEST POINT: p"

        v3 = Council(log, speaker_flaky).convene("flaky debate")
        assert v3.ok and v3.winner == "thesis", v3
        assert v3.confidence == 30.0 and "failed" in v3.reason

        # both sides silent -> honest failure, never a fake verdict
        def speaker_dead(role: str, brief: str) -> str:
            raise RuntimeError("api down")

        v4 = Council(log, speaker_dead).convene("dead debate")
        assert not v4.ok and "thesis" in v4.error and v4.winner == ""

        # no speaker attached -> clean error
        v5 = Council(log).convene("no speaker")
        assert not v5.ok and v5.error == "no speaker attached"

        # everything is sealed and replayable
        vs = c.verdicts()
        assert len(vs) == 6
        evs = fold(log).council_events
        types = {e["type"] for e in evs}
        assert {"council.convened", "council.position",
                "council.verdict"} <= types
        assert "COUNCIL" in c.format_status()

    print("COUNCIL SELF-TEST PASS")

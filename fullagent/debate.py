"""DEBATE — Mixture-of-Agents tournament with calibrated fusion.

One model's answer is a guess; a tournament of models that must survive
each other's critiques is an argument. Three rounds, all real model
calls:

    PROPOSAL    every participant answers the question independently
                (blind — nobody sees another's answer yet)
    CHALLENGE   every participant sees ALL other proposals and attacks
                the weakest claims (their own included)
    REVISION    every participant revises its own answer in light of
                every critique

Fusion is deterministic (rung 1): every final answer is embedded as a
sparse token vector; answers cluster by pairwise cosine similarity; each
cluster's weight = Σ calibration(model) × internal-consistency; the
champion cluster's strongest member becomes the verdict, carrying the
full dissent record.

CALIBRATION is the memory of the tournament: every participant starts at
0.5 trust; when a verdict is later confirmed or refuted (confirm() /
refute() from the human or the judge), every participant on the winning
side gains, the losing side decays (exponential update, bounded). Over
time the tournament learns WHICH models to believe about WHAT.

The speaker is injectable — the self-test runs a full 3-round tournament
with scripted positions, offline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .kernel import EventLog, fold

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TRUST_MIN, _TRUST_MAX = 0.05, 0.95
_TRUST_RATE = 0.25          # exponential update rate
_CLUSTER_THRESHOLD = 0.45   # cosine similarity for same-cluster


def _vector(text: str) -> dict[str, int]:
    """Sparse bag-of-words vector (token -> count)."""
    out: dict[str, int] = {}
    for tok in _TOKEN_RE.findall(str(text).lower()):
        out[tok] = out.get(tok, 0) + 1
    return out


def cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Position:
    model_id: str
    answer: str = ""
    critique: str = ""
    revised: str = ""
    cluster: int = -1


@dataclass
class DebateResult:
    question: str
    verdict: str = ""
    champion_model: str = ""
    clusters: list[list[str]] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    dissent: list[str] = field(default_factory=list)
    rounds: int = 0

    def to_dict(self) -> dict:
        return {"question": self.question, "verdict": self.verdict[:600],
                "champion_model": self.champion_model,
                "clusters": self.clusters, "rounds": self.rounds,
                "dissent": self.dissent[:4],
                "participants": [p.model_id for p in self.positions]}


class DebateTournament:
    """Blind proposals → mutual critique → revision → calibrated fusion."""

    def __init__(self, log: EventLog, speaker, models: list[str]) -> None:
        """speaker(model_id, prompt) -> str. models: the participant ids
        (any names the speaker can route on)."""
        self.log = log
        self.speaker = speaker
        self.models = models or []
        self.trust: dict[str, float] = {}
        self._champion_cluster: set[str] = set()
        self._load_trust()

    # -- calibration persistence (from the event log — the only source) ----

    def _load_trust(self) -> None:
        for ev in self.log.events():
            if ev.type == "debate.calibration":
                self.trust[ev.data.get("model", "")] = float(
                    ev.data.get("trust", 0.5))

    def calibration(self, model_id: str) -> float:
        return self.trust.get(model_id, 0.5)

    def _update(self, model_id: str, won: bool) -> float:
        cur = self.calibration(model_id)
        target = _TRUST_MAX if won else _TRUST_MIN
        new = cur + (target - cur) * _TRUST_RATE
        new = max(_TRUST_MIN, min(_TRUST_MAX, new))
        self.trust[model_id] = new
        self.log.append("debate.calibration",
                        {"model": model_id, "trust": round(new, 4),
                         "direction": "up" if won else "down"})
        return new

    # -- the tournament ------------------------------------------------------------

    def run(self, question: str, rounds: int = 3) -> DebateResult:
        """Full 3-round tournament (rounds<3 degrades gracefully: 2 =
        no revision, 1 = plain parallel sampling)."""
        question = str(question or "").strip()
        result = DebateResult(question=question, rounds=0)
        if not question or not self.models:
            return result
        positions = [Position(model_id=m) for m in self.models]

        # round 1 — blind proposals
        for p in positions:
            p.answer = self.speaker(p.model_id, (
                "Answer this question directly and concisely. Stand "
                "alone: you will defend it in a tournament.\n\n"
                f"QUESTION: {question}"))
        result.rounds = 1
        self.log.append("debate.round",
                        {"n": 1, "kind": "proposal",
                         "answers": {p.model_id: p.answer[:200]
                                     for p in positions}})

        if rounds >= 2 and len(positions) >= 2:
            # round 2 — mutual critique (everybody sees everything)
            others = "\n\n".join(
                f"[{p.model_id}] says: {p.answer[:600]}"
                for p in positions)
            for p in positions:
                p.critique = self.speaker(p.model_id, (
                    f"QUESTION: {question}\n\nThe tournament's "
                    f"proposals:\n{others}\n\nYou are [{p.model_id}]. "
                    f"Attack the weakest claims above — including your "
                    "own. Name concrete errors, missing cases, bad "
                    "assumptions. Max 120 words."))
            result.rounds = 2
            self.log.append("debate.round",
                            {"n": 2, "kind": "critique",
                             "critiques": {p.model_id: p.critique[:200]
                                           for p in positions}})

        if rounds >= 3 and len(positions) >= 2:
            # round 3 — revision under fire (pointless without critiques)
            all_critiques = "\n\n".join(
                f"[{p.model_id}] critiques: {p.critique[:400]}"
                for p in positions)
            for p in positions:
                p.revised = self.speaker(p.model_id, (
                    f"QUESTION: {question}\n\nYour original answer: "
                    f"{p.answer[:600]}\n\nThe tournament's critiques:\n"
                    f"{all_critiques}\n\nRevise YOUR answer. Keep what "
                    "survived the critique, fix what did not. Final "
                    "answer only."))
            result.rounds = 3
            self.log.append("debate.round",
                            {"n": 3, "kind": "revision",
                             "revised": {p.model_id: p.revised[:200]
                                         for p in positions}})

        result.positions = positions
        finals = [(p, p.revised or p.answer) for p in positions]
        result.verdict, result.champion_model, result.clusters, \
            result.dissent = self._fuse(finals)
        # remember the winning cluster so confirm() can credit every
        # member that argued the winning position, not just the champion
        self._champion_cluster = (
            set(result.clusters[0]) if result.clusters
            else {result.champion_model})
        self.log.append("debate.verdict", result.to_dict(),
                        actor="kernel")
        return result

    # -- deterministic fusion ----------------------------------------------------------

    def _fuse(self, finals: list[tuple[Position, str]]
              ) -> tuple[str, str, list[list[str]], list[str]]:
        """Cluster the final answers by cosine similarity; weight each
        cluster by Σ calibration; champion = strongest member of the
        strongest cluster. Dissent = best answer outside the champion
        cluster."""
        vecs = [(p, text, _vector(text)) for p, text in finals]

        # single-link clustering over the similarity threshold
        clusters: list[list[int]] = []
        for i in range(len(vecs)):
            placed = False
            for c in clusters:
                if any(cosine(vecs[i][2], vecs[j][2])
                       >= _CLUSTER_THRESHOLD for j in c):
                    c.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        def cluster_weight(c: list[int]) -> float:
            # calibration mass × agreement tightness (mean pairwise sim)
            mass = sum(self.calibration(vecs[i][0].model_id) for i in c)
            if len(c) < 2:
                return mass
            sims = [cosine(vecs[a][2], vecs[b][2])
                    for a in c for b in c if a < b]
            tight = sum(sims) / len(sims) if sims else 1.0
            return mass * (0.5 + tight / 2)

        clusters.sort(key=cluster_weight, reverse=True)
        champion_cluster = clusters[0]
        for i in champion_cluster:
            vecs[i][0].cluster = 0
        for ci, c in enumerate(clusters[1:], 1):
            for i in c:
                vecs[i][0].cluster = ci

        # champion member: highest calibration in the winning cluster
        best = max(champion_cluster,
                   key=lambda i: self.calibration(vecs[i][0].model_id))
        verdict = vecs[best][1]
        dissent: list[str] = []
        if len(clusters) > 1:
            other = clusters[1]
            top_other = max(
                other, key=lambda i: self.calibration(
                    vecs[i][0].model_id))
            dissent = [f"[{vecs[top_other][0].model_id}] dissents: "
                       + vecs[top_other][1][:300]]
        names = [[vecs[i][0].model_id for i in c] for c in clusters]
        return verdict, vecs[best][0].model_id, names, dissent

    # -- outcome feedback ----------------------------------------------------------------

    def confirm(self, verdict_model: str) -> dict:
        """The verdict's cluster was RIGHT: its members gain trust, the
        dissenters decay. verdict_model = the champion from the result."""
        winners = self._champion_cluster or {verdict_model}
        if verdict_model not in winners:
            # confirming an off-cluster model: credit just that model
            winners = {verdict_model}
        for m in self.models:
            self._update(m, won=(m in winners))
        return dict(self.trust)

    def refute(self, dissent_model: str) -> dict:
        """The dissent was right instead: flip the flow of trust."""
        for m in self.models:
            self._update(m, won=(m == dissent_model))
        return dict(self.trust)

    # -- reporting --------------------------------------------------------------------------

    def format(self, result: DebateResult) -> str:
        lines = [f"DEBATE VERDICT — {result.question}",
                 f"  champion: [{result.champion_model}] "
                 f"({result.rounds} rounds, "
                 f"{len(result.positions)} participants)",
                 f"  clusters: " + " | ".join(
                     ", ".join(c) for c in result.clusters)]
        lines.append("  VERDICT: " + result.verdict[:1200])
        for d in result.dissent:
            lines.append(f"  ⚠ dissent — {d}")
        lines.append("  calibration: " + ", ".join(
            f"{m}={self.calibration(m):.2f}" for m in self.models))
        lines.append("  feedback: /debate confirm|refute <model-id>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — a scripted full tournament, offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "debate.jsonl")

        calls: list[tuple[str, str]] = []

        def speaker(model_id: str, prompt: str) -> str:
            calls.append((model_id, prompt[:40]))
            # three-persona script: two agree (rust), one dissents (go)
            if model_id == "modelA":
                if "Attack" in prompt:
                    return "the go answer ignores memory safety"
                if "Revise" in prompt:
                    return "rust is memory-safe and fast for systems"
                return "rust is the right choice for systems work"
            if model_id == "modelB":
                if "Attack" in prompt:
                    return "rust has a steep learning curve"
                if "Revise" in prompt:
                    return "rust remains right despite the curve"
                return "rust suits performance-critical systems"
            # modelC dissents
            if "Attack" in prompt:
                return "rust compile times hurt iteration speed"
            if "Revise" in prompt:
                return "go ships faster for web services"
            return "go is more practical for most teams"

        tour = DebateTournament(log, speaker,
                                ["modelA", "modelB", "modelC"])
        assert tour.calibration("modelA") == 0.5

        result = tour.run("rust or go for a new backend?", rounds=3)
        # all three rounds ran for all three models
        assert result.rounds == 3
        assert len(calls) == 3 * 3, len(calls)
        # the two rust answers cluster together; the champion comes from
        # the majority cluster
        assert len(result.clusters) >= 2, result.clusters
        assert result.champion_model in ("modelA", "modelB"), \
            result.champion_model
        assert "rust" in result.verdict.lower()
        assert result.dissent and "dissent" in result.dissent[0]

        # calibration feedback moves trust, bounded, and persists
        trust = tour.confirm(result.champion_model)
        assert trust[result.champion_model] > 0.5
        assert trust["modelC"] < 0.5
        # reload from the log — calibration survives restarts
        tour2 = DebateTournament(log, speaker,
                                 ["modelA", "modelB", "modelC"])
        assert abs(tour2.calibration(result.champion_model)
                   - trust[result.champion_model]) < 1e-6
        assert abs(tour2.calibration("modelC") - trust["modelC"]) < 1e-6

        # weighted fusion: a high-trust dissenter still loses to the
        # majority, but a confirmed dissenter changes nothing structural
        # (fusion is deterministic — trust only weights, never vetoes)
        tour.refute("modelC")
        assert tour.calibration("modelC") > trust["modelC"]

        # single participant: no critique round needed
        solo = DebateTournament(log, speaker, ["modelA"])
        r1 = solo.run("solo question", rounds=3)
        assert r1.rounds == 1 and r1.champion_model == "modelA"

        # cosine + vector sanity
        assert math.isclose(cosine(_vector("a b c"),
                                   _vector("a b c")), 1.0)
        assert cosine(_vector("a b"), _vector("x y")) == 0.0

        # empty inputs are clean
        assert tour.run("").rounds == 0
        empty = DebateTournament(log, speaker, [])
        assert empty.run("q").rounds == 0

        # events sealed + foldable
        st = fold(log)
        kinds = {e["type"] for e in st.advanced_events}
        assert {"debate.round", "debate.verdict",
                "debate.calibration"} <= kinds, kinds

        print("DEBATE SELF-TEST PASS")

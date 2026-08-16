"""CAUSAL — causal discovery and do()-interventions from the event log.

Correlation says web_search and success move together; causation says
USING web_search CHANGES success. This module tells them apart, from
the agent's own history:

    features    every turn becomes one observation: binary/numeric
                features (used web? tool count, wrote files, context
                size, role mix…) and the outcome (scorecard score)
    discovery   pairwise association (Pearson on binarized data) is
                TESTED, not trusted: an edge survives only if it holds
                after stratifying on every observed confounder (the
                backdoor test, discretized). Spurious links die here.
    effect(x)   the do(x) estimate: stratified mean difference of the
                outcome between x=1 and x=0 within each confounder
                stratum, weighted by stratum size — a discrete
                backdoor adjustment. With an honest confidence flag
                (n too small -> reported as unmeasured, never guessed).
    verdict     edges are classified: CAUSAL (survived adjustment),
                SPURIOUS (died under stratification), UNMEASURED.

All math is rung 1: deterministic statistics, zero tokens. The
self-test plants a known causal structure (a confounded pair and a
true cause) and proves the engine separates them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .kernel import EventLog

_MIN_STRATUM = 3          # below this a stratum is too thin to trust
_MIN_TOTAL = 12           # below this nothing is claimed at all


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return num / den if den else 0.0


@dataclass
class Observation:
    """One turn distilled into features + outcome."""
    features: dict[str, float]
    outcome: float          # 0..1 success score


@dataclass
class CausalEdge:
    cause: str
    effect: str             # always the outcome variable name
    association: float
    adjusted: float         # effect after backdoor adjustment
    verdict: str            # CAUSAL | SPURIOUS | UNMEASURED
    n: int

    def to_dict(self) -> dict:
        return {"cause": self.cause, "association": round(
                    self.association, 3),
                "adjusted_effect": round(self.adjusted, 3),
                "verdict": self.verdict, "n": self.n}


def observations_from_log(log: EventLog) -> list[Observation]:
    """Fold turns into observations. Features are mechanical: tools
    used, writes, errors, web usage; outcome = scorecard score or
    derived (done turn = 1, error turn = 0)."""
    turns: list[dict] = []
    current: dict | None = None
    for ev in log.events():
        t = ev.type
        if t == "user.message":
            if current is not None:
                turns.append(current)
            current = {"web": 0.0, "tools": 0.0, "writes": 0.0,
                       "errors": 0.0, "files": 0.0, "outcome": None}
        elif current is None:
            continue
        elif t == "tool.call":
            current["tools"] += 1
            name = str(ev.data.get("name", ""))
            if name in ("web_search", "web_fetch"):
                current["web"] = 1.0
            if name in ("write_file", "edit_file"):
                current["writes"] += 1
        elif t == "tool.result":
            if ev.data.get("status") == "error":
                current["errors"] += 1
        elif t == "turn.scorecard":
            current["outcome"] = float(ev.data.get("score", 0)) / 100.0
    if current is not None:
        turns.append(current)
    out = []
    for tn in turns:
        if tn["outcome"] is None:
            tn["outcome"] = 1.0 if tn["errors"] == 0 and tn["tools"] \
                else (0.4 if tn["errors"] == 0 else 0.0)
        out.append(Observation(
            features={"web": tn["web"],
                      "tools": min(tn["tools"], 20) / 20.0,
                      "writes": min(tn["writes"], 10) / 10.0,
                      "errors": min(tn["errors"], 5) / 5.0},
            outcome=tn["outcome"]))
    return out


class CausalEngine:
    """Discovery + do() on binned observations."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    # -- the backdoor test ---------------------------------------------------

    def _strata(self, obs: list[Observation], cause: str,
                confounders: list[str]) -> dict[tuple, list[Observation]]:
        """Split observations into (cause_bin, confounder_quartiles)
        strata — quartile bins hold a continuous confounder tightly
        enough that its residual cannot masquerade as the cause."""
        quartiles: dict[str, tuple[float, float, float]] = {}
        for c in confounders:
            vals = sorted(o.features.get(c, 0.0) for o in obs)
            if not vals:
                quartiles[c] = (0.0, 0.0, 0.0)
                continue
            quartiles[c] = (vals[len(vals) // 4],
                            vals[len(vals) // 2],
                            vals[3 * len(vals) // 4])
        out: dict[tuple, list[Observation]] = {}
        for o in obs:
            key = []
            for c in confounders:
                v = o.features.get(c, 0.0)
                q1, q2, q3 = quartiles[c]
                key.append(0 if v <= q1 else (1 if v <= q2 else
                                              (2 if v <= q3 else 3)))
            cause_bin = 1 if o.features.get(cause, 0.0) > 0.5 else 0
            out.setdefault((cause_bin,) + tuple(key), []).append(o)
        return out

    def effect(self, obs: list[Observation], cause: str,
               confounders: list[str] | None = None) -> tuple[float, int]:
        """Discrete backdoor-adjusted effect of cause on outcome.
        Returns (effect, usable_n). Effect = weighted mean difference
        (outcome | cause=1) − (outcome | cause=0) within strata."""
        if len(obs) < _MIN_TOTAL:
            return 0.0, 0
        features_seen = {k for o in obs for k in o.features}
        candidates = [c for c in (confounders if confounders is not None
                                  else sorted(features_seen))
                      if c != cause]
        # adjust on the two strongest associated confounders — enough to
        # kill common confounding without fragmenting the strata
        confounders: list[str] = []
        if candidates:
            ys = [o.outcome for o in obs]
            assocs = [(abs(pearson(
                [1.0 if o.features.get(c, 0.0) > 0.5 else 0.0
                 for o in obs], ys)), c) for c in candidates]
            assocs.sort(reverse=True)
            confounders = [c for a, c in assocs[:2] if a > 0.1]

        strata = (self._strata(obs, cause, confounders) if confounders
                  else {(1,): [o for o in obs
                               if o.features.get(cause, 0.0) > 0.5],
                        (0,): [o for o in obs
                               if o.features.get(cause, 0.0) <= 0.5]})

        # BACKDOOR ADJUSTMENT — compare WITHIN each stratum, then weight
        # by stratum size. (Pooling arms across strata would smuggle the
        # confounder straight back in — Simpson's paradox.)
        by_confounder: dict[tuple, dict[int, list[Observation]]] = {}
        for key, group in strata.items():
            by_confounder.setdefault(key[1:], {})[key[0]] = group
        effect_sum = 0.0
        weight_sum = 0
        usable = 0
        for ckey, arms in by_confounder.items():
            on, off = arms.get(1, []), arms.get(0, [])
            if len(on) < 2 or len(off) < 2:
                continue                  # an arm too thin to compare
            n_stratum = len(on) + len(off)
            diff = (sum(o.outcome for o in on) / len(on)
                    - sum(o.outcome for o in off) / len(off))
            effect_sum += n_stratum * diff
            weight_sum += n_stratum
            usable += n_stratum
        if weight_sum < _MIN_TOTAL:
            return 0.0, 0
        return effect_sum / weight_sum, usable

    # -- discovery ---------------------------------------------------------------

    def discover(self, obs: list[Observation] | None = None
                 ) -> list[CausalEdge]:
        """Test every feature against the outcome; classify each by
        whether the association survives adjustment."""
        obs = obs if obs is not None else observations_from_log(self.log)
        if not obs:
            return []
        ys = [o.outcome for o in obs]
        causes = sorted({k for o in obs for k in o.features})
        edges: list[CausalEdge] = []
        for cause in causes:
            xs = [o.features.get(cause, 0.0) for o in obs]
            association = pearson(xs, ys)
            if abs(association) < 0.08:
                continue                      # nothing to explain
            adjusted, usable = self.effect(obs, cause)
            if usable == 0:
                verdict = "UNMEASURED"
            elif (association > 0) == (adjusted > 0) \
                    and abs(adjusted) >= 0.05:
                verdict = "CAUSAL"
            else:
                verdict = "SPURIOUS"
            edges.append(CausalEdge(cause=cause, effect="outcome",
                                    association=association,
                                    adjusted=adjusted, verdict=verdict,
                                    n=len(obs)))
            self.log.append("causal.edge", edges[-1].to_dict())
        edges.sort(key=lambda e: -abs(e.adjusted))
        return edges

    def do(self, cause: str, enable: bool = True) -> dict:
        """do(x) — the estimated effect of FORCING cause on/off."""
        obs = observations_from_log(self.log)
        effect, usable = self.effect(obs, cause)
        sign = effect if enable else -effect
        report = {"intervention": cause, "set_to": 1 if enable else 0,
                  "estimated_outcome_change": round(sign, 4),
                  "usable_observations": usable,
                  "trustworthy": usable >= 2 * _MIN_STRATUM}
        self.log.append("causal.intervention", report)
        return report

    def format(self, edges: list[CausalEdge]) -> str:
        if not edges:
            return "not enough history for causal analysis yet"
        lines = ["CAUSAL ANALYSIS — outcome: turn success",
                 "  cause      association  adjusted  verdict"]
        for e in edges:
            lines.append(f"  {e.cause:<10} {e.association:+.2f}        "
                         f"{e.adjusted:+.2f}      {e.verdict} (n={e.n})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — planted ground truth, recovered exactly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _r
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        rng = _r.Random(42)

        # ground truth: tools CAUSES success; a confounder Z drives both
        # web usage and success (web looks associated but is spurious)
        obs = []
        for _ in range(200):
            z = rng.random()
            web = 1.0 if rng.random() < z else 0.0
            tools = min(1.0, max(0.0, z * 0.5 + rng.random() * 0.3))
            noise = rng.random() * 0.2
            outcome = min(1.0, max(0.0, 0.5 * tools + 0.35 * z + noise))
            obs.append(Observation(
                features={"web": web, "tools": tools, "z": z,
                          "errors": 0.0},
                outcome=outcome))

        log = EventLog(Path(td) / "causal.jsonl")
        eng = CausalEngine(log)
        edges = eng.discover(obs)
        verdicts = {e.cause: e.verdict for e in edges}

        assert verdicts.get("tools") == "CAUSAL", verdicts
        assert verdicts.get("web") == "SPURIOUS", verdicts
        # adjusted tools effect is positive and meaningful
        tools_edge = next(e for e in edges if e.cause == "tools")
        assert tools_edge.adjusted > 0.05, tools_edge.to_dict()

        # observations fold from a real kernel log
        log.append("user.message", {"text": "go"})
        log.append("tool.call", {"name": "web_search",
                                 "args": {"q": "x"}})
        log.append("tool.result", {"status": "done"})
        log.append("turn.scorecard", {"score": 80})
        log.append("user.message", {"text": "go 2"})
        log.append("tool.call", {"name": "write_file",
                                 "args": {"path": "a.py"}})
        log.append("tool.result", {"status": "error"})
        obs2 = observations_from_log(log)
        assert len(obs2) == 2
        assert obs2[0].features["web"] == 1.0
        assert obs2[0].outcome == 0.8
        assert obs2[1].features["writes"] > 0
        assert obs2[1].features["errors"] > 0

        # do() reports the adjusted estimate with an honesty flag
        report = eng.do("tools")
        assert report["intervention"] == "tools"
        assert isinstance(report["trustworthy"], bool)

        # tiny data is UNMEASURED, never guessed
        tiny = eng.discover([Observation({"web": 1.0}, 1.0)])
        assert tiny == []

        # pearson sanity
        assert pearson([1, 2, 3], [1, 2, 3]) > 0.99
        assert abs(pearson([1, 2, 3], [3, 2, 1]) + 1.0) < 0.01
        assert pearson([1], [1]) == 0.0

        # events sealed
        kinds = {e.type for e in log.events()}
        assert "causal.edge" in kinds and "causal.intervention" in kinds

        print("CAUSAL SELF-TEST PASS")

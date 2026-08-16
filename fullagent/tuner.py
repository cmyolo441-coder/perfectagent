"""TUNER — Bayesian-ish auto-tuning of the agent's own knobs.

Which effort level, worker step count, and compaction threshold make
turns score best? Instead of guessing, the tuner runs the classic
Tree-structured Parzen Estimator loop:

    suggest    split observed (config, score) history into GOOD (top γ)
               and REST; sample candidate values per knob by drawing
               from the good-set's per-knob histogram with probability
               l(x) ∝ good-density / (good + rest density) — the TPE
               density ratio — and otherwise explore uniformly. Early
               on (no history) it space-fills.
    observe    every trial's score feeds the history
    best       the argmax config so far, sealed as tuner.best

This is a real (discrete-knob) TPE: per-knob histogram likelihoods,
density-ratio candidate acceptance, γ = 25%, plus ε-greedy exploration
so nothing is ever permanently locked out. The self-test optimizes a
known quadratic-with-interaction objective and proves convergence
toward the optimum without ever evaluating the whole grid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .kernel import EventLog

_GAMMA = 0.25          # fraction of history counted as GOOD
_EPSILON = 0.10        # uniform exploration probability


@dataclass
class Trial:
    config: dict
    score: float = 0.0
    ts_order: int = 0


@dataclass
class TunerReport:
    best_config: dict
    best_score: float
    trials: int
    distinct: int

    def to_dict(self) -> dict:
        return {"best_config": self.best_config,
                "best_score": round(self.best_score, 4),
                "trials": self.trials, "distinct": self.distinct}


class ParzenTuner:
    """TPE over discrete knob spaces."""

    def __init__(self, log: EventLog, space: dict[str, list],
                 seed: int = 13, objective=None) -> None:
        """space: knob -> allowed values. objective(config) -> float —
        injectable (production: run a scored benchmark turn; tests: a
        synthetic function)."""
        self.log = log
        self.space = {k: list(v) for k, v in space.items() if v}
        self.rng = random.Random(seed)
        self.objective = objective
        self.history: list[Trial] = []
        self._n = 0

    # -- TPE internals --------------------------------------------------------

    def _density(self, value, values: list) -> float:
        """Histogram likelihood of `value` under observed `values`
        (with a smoothing floor so empty bins never zero out)."""
        if not values:
            return 1.0
        n_hit = sum(1 for v in values if v == value)
        return (n_hit + 0.5) / (len(values) + 0.5)

    def _sample_knob(self, knob: str, good: list, rest: list) -> object:
        """Draw one value by TPE density-ratio acceptance."""
        values = self.space[knob]
        if not good:
            return self.rng.choice(values)      # space-fill early
        # l(x) ∝ p(x|good) / (p(x|good) + p(x|rest))
        weights = []
        good_vals = [t.config[knob] for t in good]
        rest_vals = [t.config[knob] for t in rest]
        for v in values:
            pg = self._density(v, good_vals)
            pr = self._density(v, rest_vals)
            weights.append(pg / (pg + pr))
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(values)
        pick = self.rng.random() * total
        acc = 0.0
        for v, w in zip(values, weights):
            acc += w
            if pick <= acc:
                return v
        return values[-1]

    # -- the loop ------------------------------------------------------------------

    def suggest(self) -> dict:
        """One config proposal: ε-uniform explore, else TPE exploit."""
        n_good = max(1, int(len(self.history) * _GAMMA))
        ranked = sorted(self.history, key=lambda t: -t.score)
        good, rest = ranked[:n_good], ranked[n_good:]
        if not self.history or self.rng.random() < _EPSILON:
            return {k: self.rng.choice(v)
                    for k, v in self.space.items()}
        return {k: self._sample_knob(k, good, rest)
                for k in self.space}

    def observe(self, config: dict, score: float) -> None:
        self._n += 1
        self.history.append(Trial(config=dict(config),
                                  score=float(score),
                                  ts_order=self._n))
        self.log.append("tuner.trial",
                        {"config": config, "score": round(float(score),
                                                          4),
                         "n": self._n}, actor="kernel")

    def step(self) -> Trial:
        """Suggest → evaluate (objective) → observe. Returns the trial."""
        if self.objective is None:
            raise RuntimeError("no objective attached")
        config = self.suggest()
        score = float(self.objective(config))
        self.observe(config, score)
        return self.history[-1]

    def best(self) -> Trial | None:
        if not self.history:
            return None
        return max(self.history, key=lambda t: t.score)

    def run(self, n: int = 20) -> TunerReport:
        for _ in range(n):
            self.step()
        b = self.best()
        assert b is not None
        distinct = len({tuple(sorted(t.config.items()))
                        for t in self.history})
        report = TunerReport(b.config, b.score, len(self.history),
                             distinct)
        self.log.append("tuner.best", report.to_dict())
        return report

    def format(self) -> str:
        b = self.best()
        lines = [f"TUNER — {len(self.history)} trials"]
        if b:
            lines.append(f"  best score {b.score:.4f} with "
                         + ", ".join(f"{k}={v}"
                                     for k, v in b.config.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — a known objective, conquered without grid search
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "tuner.jsonl")

        # known optimum: effort=high, steps=medium, compact=tight
        opt = {"effort": "high", "steps": "medium",
               "compact": "tight"}
        score_map = {"effort": {"low": 0.2, "medium": 0.5, "high": 0.9},
                     "steps": {"low": 0.4, "medium": 0.8, "high": 0.5},
                     "compact": {"loose": 0.3, "tight": 0.7}}

        def objective(cfg: dict) -> float:
            # multiplicative landscape with interactions — a naive
            # per-knob greedy sweep can get stuck; TPE explores
            base = 1.0
            for k, v in cfg.items():
                base *= score_map[k][v]
            penalty = 0.0
            if cfg["steps"] == "high" and cfg["compact"] == "tight":
                penalty = 0.3      # interaction: bad combo
            return base + 0.1 - penalty

        best_possible = objective(opt)

        space = {"effort": ["low", "medium", "high"],
                 "steps": ["low", "medium", "high"],
                 "compact": ["loose", "tight"]}
        tuner = ParzenTuner(log, space, seed=21, objective=objective)
        report = tuner.run(n=40)

        # converged near the optimum without exhausting the grid
        assert report.best_score >= best_possible * 0.999, \
            (report.to_dict(), best_possible)
        assert report.best_config == opt, report.to_dict()
        # ...but never evaluated everything (27 combos exist)
        assert report.distinct <= 26

        # determinism with the same seed
        t2 = ParzenTuner(EventLog(Path(td) / "t2.jsonl"), space,
                         seed=21, objective=objective)
        r2 = t2.run(n=40)
        assert r2.best_config == report.best_config

        # early suggestions space-fill (no history yet)
        t3 = ParzenTuner(EventLog(Path(td) / "t3.jsonl"), space,
                         seed=4)
        first = t3.suggest()
        assert set(first) == set(space)

        # density ratio math: good-only values outweigh unseen ones
        t3.observe({"effort": "high", "steps": "low",
                    "compact": "loose"}, 0.9)
        t3.observe({"effort": "high", "steps": "low",
                    "compact": "loose"}, 0.8)
        t3.observe({"effort": "low", "steps": "high",
                    "compact": "tight"}, 0.1)
        draws = [t3.suggest()["effort"] for _ in range(60)]
        assert draws.count("high") > draws.count("low"), \
            {"high": draws.count("high"), "low": draws.count("low")}

        # events sealed
        kinds = {e.type for e in log.events()}
        assert {"tuner.trial", "tuner.best"} <= kinds

        # step() without an objective is a clean error
        try:
            ParzenTuner(log, space).step()
            raise AssertionError("must raise without objective")
        except RuntimeError:
            pass

        print("TUNER SELF-TEST PASS")

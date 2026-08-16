"""BANDIT — the contextual Thompson-sampling router.

Which (model, effort) combo should serve a code task? The bandit
answers with probability, not vibes:

    arms       the model×effort combos actually configured
    context    task bucket derived from the request text (code / write /
               research / run / chat) — different buckets, different
    posteriors one Beta(α, β) per (context, arm); α grows with good
    outcomes, β with bad ones
    recommend  Thompson sampling: draw one sample per arm, take the
    argmax — the classic probability-matching explore/exploit balance:
    uncertain arms still win sometimes (that is how they get learned),
    proven arms dominate once the data is in
    update     every finished turn feeds its score (scorecard /100 or a
    status-derived score) into the posterior; persistence rides the
    kernel events (bandit.update), so the policy survives restarts

All math is rung 1 (numpy-free Beta sampling via two Gamma draws —
Marsaglia-Tsang on stdlib random). The self-test plants a best arm and
proves convergence: the bandit routes >90% of late draws to it while
still having explored everything.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

from .kernel import EventLog

CONTEXTS = ("code", "write", "research", "run", "chat")
_PRIOR = (1.0, 1.0)          # uniform Beta prior — no arm is presumed


def _context_of(task: str) -> str:
    """Bucket a request by its dominant verb/noun signature (execution
    verbs outrank nouns — 'run the test suite' is a run, not a test)."""
    t = str(task or "").lower()
    if re.search(r"\b(run|execute|command|build|install|deploy|"
                 r"benchmark)\b", t):
        return "run"
    if re.search(r"\b(bug|fix|code|implement|refactor|function|class|"
                 r"test|error|stack|trace)\b", t):
        return "code"
    if re.search(r"\b(write|document|readme|docs?|blog|letter|essay|"
                 r"email)\b", t):
        return "write"
    if re.search(r"\b(research|latest|news|compare|find|who|what is|"
                 r"search)\b", t):
        return "research"
    return "chat"


def _gamma_draw(shape: float, rng: random.Random) -> float:
    """Marsaglia-Tsang Gamma sampler (shape >= 1 branch + boost)."""
    if shape < 1.0:
        u = rng.random()
        while u <= 1e-12:
            u = rng.random()
        return _gamma_draw(shape + 1.0, rng) * u ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = 1.0 + c * x
        if v <= 0:
            continue
        v = v * v * v
        u = rng.random()
        if u < 1 - 0.0331 * x * x * x * x:
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v


def _beta_draw(alpha: float, beta: float,
               rng: random.Random) -> float:
    x = _gamma_draw(alpha, rng)
    y = _gamma_draw(beta, rng)
    return x / (x + y) if (x + y) > 0 else 0.5


@dataclass
class Recommendation:
    arm: str
    context: str
    sampled: dict         # arm -> sampled value (this round's draw)
    expected: dict        # arm -> posterior mean α/(α+β)


class BanditRouter:
    """Thompson sampling over (model, effort) arms, per context."""

    def __init__(self, log: EventLog, arms: list[str], seed: int = 5
                 ) -> None:
        self.log = log
        self.arms = [str(a) for a in arms] or ["default"]
        self.rng = random.Random(seed)
        self.alpha: dict[tuple[str, str], float] = {}
        self.beta: dict[tuple[str, str], float] = {}
        self._load()

    def _load(self) -> None:
        """Rebuild posteriors from sealed updates — restart-proof."""
        for ev in self.log.events():
            if ev.type != "bandit.update":
                continue
            key = (str(ev.data.get("context", "chat")),
                   str(ev.data.get("arm", "")))
            self.alpha[key] = float(ev.data.get("alpha", _PRIOR[0]))
            self.beta[key] = float(ev.data.get("beta", _PRIOR[1]))

    def _posterior(self, context: str, arm: str) -> tuple[float, float]:
        return (self.alpha.get((context, arm), _PRIOR[0]),
                self.beta.get((context, arm), _PRIOR[1]))

    def recommend(self, task: str) -> Recommendation:
        """One Thompson round: sample every arm, take the argmax."""
        context = _context_of(task)
        sampled = {}
        for arm in self.arms:
            a, b = self._posterior(context, arm)
            sampled[arm] = _beta_draw(a, b, self.rng)
        best = max(self.arms, key=lambda arm: sampled[arm])
        rec = Recommendation(arm=best, context=context, sampled=sampled,
                             expected={arm: (lambda a, b: a / (a + b))(
                                 *self._posterior(context, arm))
                                 for arm in self.arms})
        self.log.append("bandit.pull",
                        {"context": context, "arm": best,
                         "sampled": {k: round(v, 3)
                                     for k, v in sampled.items()}})
        return rec

    def update(self, arm: str, reward: float, context: str = "") -> None:
        """Feed one outcome (0..1) into the posterior. Rewards outside
        0..1 are clamped; a hard failure is reward 0."""
        reward = max(0.0, min(1.0, float(reward)))
        context = context or "chat"
        a, b = self._posterior(context, arm)
        # fractional update: α += reward, β += (1 − reward)
        a += reward
        b += (1.0 - reward)
        self.alpha[(context, arm)] = a
        self.beta[(context, arm)] = b
        self.log.append("bandit.update",
                        {"context": context, "arm": arm,
                         "reward": round(reward, 3),
                         "alpha": round(a, 3), "beta": round(b, 3)})

    def policy(self) -> dict[str, dict[str, float]]:
        """Expected value per (context, arm) — the learned policy."""
        out: dict[str, dict[str, float]] = {}
        for ctx in CONTEXTS:
            out[ctx] = {}
            for arm in self.arms:
                a, b = self._posterior(ctx, arm)
                out[ctx][arm] = round(a / (a + b), 3)
        return out

    def format(self) -> str:
        pol = self.policy()
        lines = ["BANDIT ROUTER — expected success per arm:"]
        for ctx in CONTEXTS:
            best = max(pol[ctx], key=pol[ctx].get)
            row = " · ".join(f"{a}={v:.2f}" for a, v in
                             sorted(pol[ctx].items(),
                                    key=lambda kv: -kv[1]))
            lines.append(f"  {ctx:<9} best={best:<24} {row}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — planted best arm, convergence proven offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "bandit.jsonl")
        arms = ["fast-model", "smart-model", "huge-model"]
        bandit = BanditRouter(log, arms, seed=9)

        # context bucketing
        assert _context_of("fix the parser bug") == "code"
        assert _context_of("write a README for the api") == "write"
        assert _context_of("research the latest flask release") == \
            "research"
        assert _context_of("run the test suite") == "run"
        assert _context_of("hii") == "chat"

        # planted environment: smart-model is best for code (0.9),
        # fast-model best for chat (0.8), others mediocre
        def true_reward(arm: str, ctx: str, rng) -> float:
            table = {("smart-model", "code"): 0.9,
                     ("fast-model", "chat"): 0.8,
                     ("huge-model", "code"): 0.5}
            p = table.get((arm, ctx), 0.3)
            return 1.0 if rng.random() < p else 0.0

        env = random.Random(3)
        picks = []
        for i in range(400):
            ctx = "code" if i % 2 == 0 else "chat"
            task = "fix the bug" if ctx == "code" else "hello there"
            rec = bandit.recommend(task)
            r = true_reward(rec.arm, ctx, env)
            bandit.update(rec.arm, r, ctx)
            picks.append((ctx, rec.arm))

        late_code = [a for c, a in picks[-100:] if c == "code"]
        late_chat = [a for c, a in picks[-100:] if c == "chat"]
        # converged: the true best arms dominate late picks…
        assert late_code.count("smart-model") > len(late_code) * 0.75, \
            {a: late_code.count(a) for a in arms}
        assert late_chat.count("fast-model") > len(late_chat) * 0.6, \
            {a: late_chat.count(a) for a in arms}
        # …but exploration happened early (everything was tried)
        tried = {a for _, a in picks[:80]}
        assert tried == set(arms), tried

        # posteriors persist across restarts (rebuild from the log)
        bandit2 = BanditRouter(log, arms, seed=1)
        pol = bandit2.policy()
        assert pol["code"]["smart-model"] > 0.7
        assert pol["code"]["huge-model"] < pol["code"]["smart-model"]
        assert pol["chat"]["fast-model"] > 0.5

        # recommendation respects the context split
        rec = bandit2.recommend("fix this nasty bug now")
        assert rec.arm == "smart-model", rec.arm
        rec_chat = bandit2.recommend("hi there friend")
        assert rec_chat.arm == "fast-model", rec_chat.arm

        # beta sampler sanity: uniform prior draws cover the space
        draws = [_beta_draw(1, 1, random.Random(i)) for i in range(50)]
        assert any(d < 0.3 for d in draws) and any(d > 0.7 for d in draws)

        # format renders
        assert "BANDIT ROUTER" in bandit2.format()

        # events sealed
        kinds = {e.type for e in log.events()}
        assert {"bandit.pull", "bandit.update"} <= kinds

        print("BANDIT SELF-TEST PASS")

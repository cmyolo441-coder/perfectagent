"""EVOLUTION — the self-improvement engine for role briefs.

The agent gets better at its own job, mechanically:

    assess    each worker/team report carries a status (done/blocked/
              error) and tool-call counts; per ROLE, fitness is the
              trailing success rate over recent runs
    mutate    the weakest role's brief is rewritten by the model into
              K candidate variants (the mutator is INJECTABLE — the
              self-test evolves deterministically offline)
    evaluate  every candidate runs the SAME benchmark task as a real
              worker; the scorer grades the outcome deterministically
    select    the champion must beat the incumbent's score by a margin;
              ties keep the incumbent (no drift without evidence)
    deploy    the winner is registered through systemprompt.register()
              and re-sealed by the Mastermind vault on its next resolve
              — the deployment lineage is a sealed evolution.deployed
              event, and the full old text lives in the event log, so
              rollback is one command, forever auditable

Guardrails (mechanical, not advice):
  * only WORKER briefs evolve — 'main' and 'master' are never touched
  * at most ONE role evolves per generation
  * a failed evaluation never deploys anything
  * generations are capped per session (evolution is deliberate work)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from . import systemprompt
from .kernel import EventLog, fold

# a candidate must beat the incumbent by this margin to deploy
DEPLOY_MARGIN = 0.10
MAX_GENERATIONS_PER_SESSION = 5
EVAL_WINDOW = 12          # recent reports per role for fitness
BENCHMARK_TIMEOUT = 240.0

_EVOLVABLE = tuple(systemprompt.ROLE_BRIEFS)   # worker roles only


@dataclass
class Generation:
    """One completed evolution attempt (deployed or not)."""
    gen: int
    role: str
    incumbent_score: float
    champion: str = ""
    champion_score: float = 0.0
    deployed: bool = False
    reason: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"gen": self.gen, "role": self.role,
                "incumbent_score": round(self.incumbent_score, 3),
                "champion": self.champion[:200],
                "champion_score": round(self.champion_score, 3),
                "deployed": self.deployed, "reason": self.reason}


def default_benchmark(role: str) -> str:
    """A fixed, role-neutral benchmark task per role — the same yardstick
    every generation, so scores are comparable across time."""
    return (f"EVOLUTION BENCHMARK [{role}]: survey the current directory "
            f"tree, then produce your standard work product for the role "
            f"{role} on what you find. Final line must be "
            f"'STATUS: DONE' or 'STATUS: BLOCKED' plus a SUMMARY.")


class EvolutionEngine:
    """Mutate → evaluate → select → deploy, one role per generation."""

    def __init__(self, log: EventLog, mutator: Callable[[str, str, int],
                                                        list[str]],
                 evaluator: Callable[[str, str], tuple[str, float]],
                 benchmark: Callable[[str], str] = default_benchmark,
                 margin: float = DEPLOY_MARGIN) -> None:
        """mutator(role, incumbent_brief, k) -> k candidate briefs.
        evaluator(role, candidate_brief) -> (final_reply, score 0..1) —
        the evaluator runs a REAL worker with the candidate brief."""
        self.log = log
        self.mutator = mutator
        self.evaluator = evaluator
        self.benchmark = benchmark
        self.margin = margin
        self.generations = 0

    # -- fitness from history (rung 1, deterministic) ------------------------

    def fitness(self) -> dict[str, float]:
        """Trailing success rate per role from sealed worker reports —
        done=1.0, blocked=0.4, error=0.0, newest reports weigh double."""
        scores: dict[str, list[float]] = {}
        events = [e.data for e in self.log.events()
                  if e.type == "crew.done"]
        for d in events[-EVAL_WINDOW * len(_EVOLVABLE):]:
            role = str(d.get("role", "")).strip()
            if role not in _EVOLVABLE:
                continue
            status = d.get("status") or d.get("state") or ""
            score = {"done": 1.0, "blocked": 0.4}.get(status, 0.0)
            scores.setdefault(role, []).append(score)
        out: dict[str, float] = {}
        for role, vals in scores.items():
            weighted = [v * (2.0 if i >= len(vals) // 2 else 1.0)
                        for i, v in enumerate(vals)]
            out[role] = sum(weighted) / sum(
                2.0 if i >= len(vals) // 2 else 1.0
                for i in range(len(vals)))
        return out

    def weakest_role(self) -> str | None:
        """The evolvable role with the worst trailing fitness (and at
        least one recorded run — a role with no history has nothing to
        improve on yet)."""
        fit = self.fitness()
        if not fit:
            return None
        return min(fit, key=lambda r: fit[r])

    # -- one generation -------------------------------------------------------

    def evolve(self, role: str | None = None) -> Generation:
        """Run ONE generation: mutate the weakest (or given) role's brief,
        evaluate every candidate on the fixed benchmark, deploy the
        champion only if it clears the incumbent by the margin."""
        if self.generations >= MAX_GENERATIONS_PER_SESSION:
            gen = Generation(self.generations, role or "?", 0.0,
                             reason="generation cap reached for this "
                                    "session")
            return gen
        role = role or self.weakest_role()
        if role is None or role not in _EVOLVABLE:
            return Generation(self.generations, role or "?", 0.0,
                              reason="no role history to evolve on yet")
        self.generations += 1
        incumbent = systemprompt.ROLE_BRIEFS[role]
        incumbent_reply, incumbent_score = self.evaluator(
            role, incumbent)

        candidates = [c for c in self.mutator(role, incumbent, 3)
                      if c and c != incumbent]
        gen = Generation(self.generations, role, incumbent_score)
        best_text, best_score = "", incumbent_score
        for cand in candidates:
            try:
                _, score = self.evaluator(role, cand)
            except Exception:            # a bad candidate never kills a run
                continue
            if score > best_score:
                best_text, best_score = cand, score
        if not best_text or best_score < incumbent_score + self.margin:
            gen.reason = (f"champion {best_score:.2f} did not clear "
                          f"incumbent {incumbent_score:.2f} + margin "
                          f"{self.margin:.2f} — incumbent kept")
            self.log.append("evolution.generation", gen.to_dict(),
                            actor="kernel")
            return gen

        gen.champion = best_text
        gen.champion_score = best_score
        gen.deployed = True
        gen.reason = f"champion cleared incumbent by " \
                     f"{best_score - incumbent_score:.2f}"
        # update the brief FIRST, then re-register the built worker
        # prompt — the Mastermind vault re-seals it on next resolve
        systemprompt.ROLE_BRIEFS[role] = best_text
        systemprompt.register(f"worker:{role}",
                              systemprompt.worker(role))
        self.log.append("evolution.deployed",
                        {"gen": gen.gen, "role": role,
                         "old": incumbent, "new": best_text,
                         "score": best_score}, actor="kernel")
        self.log.append("evolution.generation", gen.to_dict(),
                        actor="kernel")
        return gen

    # -- rollback + history ----------------------------------------------------

    def rollback(self, role: str) -> str:
        """Restore the brief recorded in the LAST evolution.deployed
        event for this role. Restores the vault copy on next resolve."""
        deploys = [e for e in self.log.events()
                   if e.type == "evolution.deployed"
                   and e.data.get("role") == role]
        if not deploys:
            return f"no deployed evolution for role {role!r} — nothing " \
                   "to roll back"
        old = str(deploys[-1].data.get("old", ""))
        if not old:
            return "deployed event carried no old text — cannot roll back"
        systemprompt.ROLE_BRIEFS[role] = old
        systemprompt.register(f"worker:{role}",
                              systemprompt.worker(role))
        self.log.append("evolution.rollback", {"role": role}, actor="human")
        return f"rolled back {role} to its pre-evolution brief"

    def history(self) -> list[dict]:
        return [e.data for e in self.log.events()
                if e.type in ("evolution.generation", "evolution.deployed",
                              "evolution.rollback")]

    def format(self, gen: Generation) -> str:
        head = (f"GENERATION {gen.gen} — role: {gen.role} · "
                f"{'DEPLOYED ✓' if gen.deployed else 'kept incumbent'}")
        body = (f"  incumbent {gen.incumbent_score:.2f} · champion "
                f"{gen.champion_score:.2f}")
        return "\n".join((head, body, f"  {gen.reason}"))


# ---------------------------------------------------------------------------
# Self-test — deterministic offline mutator/evaluator drive the full loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "evolution-test.jsonl")

        # seed history: coder failed 3 of 4 recent runs -> weakest role
        for i in range(4):
            log.append("crew.done",
                       {"role": "coder", "state": "error" if i < 3
                        else "done", "task": f"t{i}"})
        for i in range(4):
            log.append("crew.done",
                       {"role": "tester", "state": "done", "task": f"t{i}"})

        eval_calls: list[str] = []

        def mutator(role: str, incumbent: str, k: int) -> list[str]:
            return [f"{incumbent} [evolved variant {i}]"
                    for i in range(k)]

        def evaluator(role: str, brief: str) -> tuple[str, float]:
            eval_calls.append(brief)
            # the evolved variant with the magic word scores higher
            score = 0.95 if "[evolved variant" in brief else 0.60
            return f"STATUS: DONE\nSUMMARY: ran with {brief[:30]}", score

        eng = EvolutionEngine(log, mutator, evaluator)

        fit = eng.fitness()
        assert fit["coder"] < fit["tester"], fit
        assert eng.weakest_role() == "coder"

        original_brief = systemprompt.ROLE_BRIEFS["coder"]
        gen = eng.evolve()                     # evolves the weakest: coder
        assert gen.role == "coder" and gen.deployed, gen.to_dict()
        assert gen.champion_score == 0.95 and gen.incumbent_score == 0.60
        # incumbent evaluated once + every candidate once
        assert len(eval_calls) == 4, eval_calls
        # the deployed text is live in the registry and the vault copy
        assert systemprompt.ROLE_BRIEFS["coder"] != original_brief
        assert "evolved variant" in systemprompt.ROLE_BRIEFS["coder"]
        types = [e.type for e in log.events()]
        assert "evolution.deployed" in types
        # old text is preserved in the event -> rollback restores it
        msg = eng.rollback("coder")
        assert "rolled back" in msg
        assert systemprompt.ROLE_BRIEFS["coder"] == original_brief
        assert eng.rollback("tester").startswith("no deployed")

        # a challenger that fails the margin never deploys
        def weak_mutator(role, incumbent, k):
            return ["slightly different but not better"] * k

        def flat_eval(role, brief):
            return "STATUS: DONE", 0.62       # beats 0.60 but < 0.70

        eng2 = EvolutionEngine(log, weak_mutator, flat_eval)
        gen2 = eng2.evolve("coder")
        assert not gen2.deployed and "margin" in gen2.reason
        assert systemprompt.ROLE_BRIEFS["coder"] == original_brief

        # generation cap protects against runaway evolution
        eng3 = EvolutionEngine(log, mutator, evaluator)
        eng3.generations = MAX_GENERATIONS_PER_SESSION
        gen3 = eng3.evolve()
        assert not gen3.deployed and "cap" in gen3.reason

        # no history -> no evolution target
        log2 = EventLog(Path(td) / "evolution-empty.jsonl")
        eng4 = EvolutionEngine(log2, mutator, evaluator)
        assert eng4.weakest_role() is None
        assert "no role history" in eng4.evolve().reason

        # fold carries the lineage
        st = fold(log)
        assert any(e["type"] == "evolution.deployed"
                   for e in st.advanced_events)

        print("EVOLUTION SELF-TEST PASS")

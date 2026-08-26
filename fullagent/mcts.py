"""MCTS — Tree-of-Agents: Monte Carlo Tree Search over strategies.

When a goal admits several strategies (which role attacks which item,
in what approach), a linear agent picks one and hopes. A tree-search
agent EXPLORES the strategy space:

    select      descend the tree by UCB1 (exploit proven value, explore
                uncertain branches — the exploration constant balances)
    expand      add one untried strategy choice at the frontier
    simulate    a cheap rollout completes the assignment randomly and
                scores it with the injected evaluator
    backprop    the score flows up the selection path

The evaluator is injectable: production scores a real (cheap) worker
run; the self-test scores a synthetic objective with a known optimum —
and proves the search finds it.

Nodes are strategy assignments: {item_index: strategy_id}. The search
is fully deterministic when seeded; every iteration is sealed so a
search is replayable and auditable (mcts.search event carries the
final tree stats).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .kernel import EventLog

_EXPLORATION = 1.41              # sqrt(2) — the classic UCB constant
_MAX_CHILDREN_EXPANSION = 8


@dataclass
class StrategyNode:
    """One node = a partial assignment of items to strategies."""
    assignment: dict = field(default_factory=dict)   # item -> strategy
    parent: "StrategyNode | None" = None
    children: list = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried: list = field(default_factory=list)      # (item, strategy)

    def ucb1(self, exploration: float = _EXPLORATION) -> float:
        if self.visits == 0:
            return math.inf
        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits
        return (self.value / self.visits
                + exploration * math.sqrt(
                    math.log(self.parent.visits) / self.visits))

    def best_child(self) -> "StrategyNode":
        return max(self.children, key=lambda c: c.ucb1())


@dataclass
class SearchReport:
    best_assignment: dict
    best_score: float
    iterations: int
    nodes: int
    depth: int
    elapsed_ms: int

    def to_dict(self) -> dict:
        return {"best_assignment": self.best_assignment,
                "best_score": round(self.best_score, 4),
                "iterations": self.iterations, "nodes": self.nodes,
                "depth": self.depth, "elapsed_ms": self.elapsed_ms}


class TreeSearch:
    """UCB1 Monte Carlo Tree Search over item→strategy assignments."""

    def __init__(self, log: EventLog, evaluator, seed: int = 7) -> None:
        """evaluator(assignment: dict) -> float in [0, 1]. The rollout
        scorer — cheap by design (the point of simulation)."""
        self.log = log
        self.evaluator = evaluator
        self.rng = random.Random(seed)

    def search(self, items: list[str], strategies: list[str],
               iterations: int = 200,
               deadline_s: float = 30.0) -> SearchReport:
        """Find the best item→strategy assignment. items: work item
        labels; strategies: candidate approach ids."""
        items = list(items or [])
        strategies = list(strategies or [])
        t0 = time.monotonic()
        if not items or not strategies:
            return SearchReport({}, 0.0, 0, 0, 0, 0)

        root = StrategyNode(untried=[(0, s) for s in strategies])
        best: dict = {}
        best_score = 0.0  # evaluator range is [0,1] — never leak a sentinel
        it = 0
        while it < iterations and time.monotonic() - t0 < deadline_s:
            it += 1
            node = root

            # SELECT — descend fully-expanded nodes by UCB1
            while not node.untried and node.children:
                node = node.best_child()

            # EXPAND — one untried (item, strategy) choice
            if node.untried:
                idx = self.rng.randrange(len(node.untried))
                item, strat = node.untried.pop(idx)
                child = StrategyNode(
                    assignment={**node.assignment, item: strat},
                    parent=node,
                    untried=[(item + 1, s) for s in strategies
                             ] if item + 1 < len(items) else [])
                node.children.append(child)
                node = child

            # SIMULATE — random completion of the remaining items
            assignment = dict(node.assignment)
            for i in range(len(assignment), len(items)):
                assignment[i] = self.rng.choice(strategies)
            score = self.evaluator(assignment)

            # BACKPROPAGATE
            walk = node
            while walk is not None:
                walk.visits += 1
                walk.value += score
                walk = walk.parent

            if score > best_score:
                best_score, best = score, dict(assignment)

        depth = max((len(n.assignment) for _ in
                     [0] for n in _walk(root)), default=0)
        nodes = sum(1 for _ in _walk(root))
        report = SearchReport(best_assignment=best,
                              best_score=best_score, iterations=it,
                              nodes=nodes, depth=depth,
                              elapsed_ms=int((time.monotonic() - t0)
                                             * 1000))
        self.log.append("mcts.search", report.to_dict(), actor="kernel")
        return report


def _walk(root: StrategyNode):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


# ---------------------------------------------------------------------------
# Self-test — a synthetic objective with a known global optimum
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "mcts.jsonl")

        # objective: item i MUST take strategy i (4 items, 4 strategies)
        # optimum = 1.0; each correct assignment contributes 1/n
        items = ["parser", "docs", "tests", "deploy"]
        strategies = ["A", "B", "C", "D"]
        optimal = {i: strategies[i] for i in range(4)}

        def evaluator(assignment: dict) -> float:
            correct = sum(1 for i in range(len(items))
                          if assignment.get(i) == optimal[i])
            return correct / len(items)

        ts = TreeSearch(log, evaluator, seed=11)
        report = ts.search(items, strategies, iterations=400,
                           deadline_s=10.0)

        assert report.best_score == 1.0, report.to_dict()
        assert report.best_assignment == optimal, report.to_dict()
        assert report.iterations >= 100 and report.nodes > 4
        # the tree actually branched (exploration happened)
        assert report.depth == 4, report.to_dict()

        # determinism: same seed, same search
        ts2 = TreeSearch(EventLog(Path(td) / "m2.jsonl"), evaluator,
                         seed=11)
        r2 = ts2.search(items, strategies, iterations=400)
        assert r2.best_assignment == report.best_assignment

        # deadline is respected
        slow = TreeSearch(log, evaluator, seed=3)
        r3 = slow.search(items * 10, strategies * 2,
                         iterations=10_000_000, deadline_s=0.5)
        assert r3.elapsed_ms < 1500, r3.elapsed_ms

        # empty inputs are clean
        empty = TreeSearch(log, evaluator).search([], ["A"], 10)
        assert empty.best_assignment == {} and empty.best_score == 0.0

        # UCB1 math: unvisited nodes are infinitely attractive
        from fullagent.mcts import StrategyNode
        parent = StrategyNode(visits=10, value=6.0)
        fresh = StrategyNode(parent=parent)
        assert fresh.ucb1() == math.inf
        proven = StrategyNode(parent=parent, visits=9, value=6.0)
        assert proven.ucb1() > 0
        parent.children = [fresh, proven]
        assert parent.best_child() is fresh

        st = log.events()
        assert any(e.type == "mcts.search" for e in st)

        print("MCTS SELF-TEST PASS")

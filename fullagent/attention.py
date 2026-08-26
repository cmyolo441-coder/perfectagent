"""ATTENTION — the attention economy: context sections compete for
tokens.

Free context is why agents bloat: every subsystem piles its text in
"just in case". Here, every context section PAYS for its tokens in an
auction, every turn:

    bid         each section's value is computed mechanically:
                relevance (token overlap with the current request) ×
                recency (fresh sections beat stale ones) × source
                priority (goal > constitution > memory > web — the
                constitution always wins ties) — normalized to [0, 1]
    allocate    the char budget distributes proportionally to bids,
                under two constraints: every section keeps a FLOOR
                (never starved to zero — a silent context is a lie)
                and no section exceeds a CEILING (no monopolist). When
                the budget cannot cover the floors, the floors win and
                the over-budget is reported honestly.
    enforce     sections are trimmed (head+tail) to their allocation
                with a visible [...trimmed by the attention auction]
                marker — the model KNOWS what it did not get

Every auction is sealed (attention.auction) with the full allocation
table — the economics of attention are auditable, per turn.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .kernel import EventLog

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# source priority: constitutional documents outrank ephemera
_SOURCE_PRIORITY = {"goal": 1.30, "constitution": 1.25, "memory": 1.0,
                    "brain": 0.95, "web": 0.85, "compacted": 0.6}
_FLOOR_FRAC = 0.06          # each section's minimum share of budget
_CEIL_FRAC = 0.45           # no section may take more than this
_KEEP_HEAD = 0.6            # when trimming: keep 60% head, 40% tail


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


@dataclass
class Allocation:
    section: str
    chars: int
    limit: int
    bid: float
    trimmed: bool

    def to_dict(self) -> dict:
        return {"section": self.section, "chars": self.chars,
                "limit": self.limit, "bid": round(self.bid, 3),
                "trimmed": self.trimmed}


@dataclass
class AuctionResult:
    budget: int
    used: int
    allocations: list[Allocation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"budget": self.budget, "used": self.used,
                "allocations": [a.to_dict() for a in
                                self.allocations]}


class AttentionEconomy:
    """Proportional-share auction over context sections."""

    def __init__(self, log: EventLog, budget_chars: int = 12_000) -> None:
        self.log = log
        self.budget_chars = budget_chars
        self.last: AuctionResult | None = None

    # -- bids ------------------------------------------------------------------

    def bid(self, section: str, text: str, query: str,
            fresh_ts: float | None = None) -> float:
        """Mechanical value of one section for this turn."""
        relevance = 0.0
        q = _tokens(query)
        if q:
            t = _tokens(text)
            overlap = len(q & t)
            relevance = overlap / (len(q) + 2)
        recency = 1.0
        if fresh_ts is not None:
            age_h = max(0.0, time.time() - fresh_ts) / 3600.0
            recency = 1.0 / (1.0 + age_h / 24.0)   # halves per ~day
        priority = _SOURCE_PRIORITY.get(section, 0.9)
        return max(0.0, min(1.0, (0.25 + 0.75 * relevance) * recency
                            * priority))

    # -- the auction ---------------------------------------------------------------

    def allocate(self, sections: dict[str, str], query: str = "",
                 fresh: dict[str, float] | None = None
                 ) -> AuctionResult:
        """Run the auction; returns allocations AND the enforced
        (trimmed) sections via enforce()."""
        budget = self.budget_chars
        names = [s for s in sections if str(sections[s]).strip()]
        result = AuctionResult(budget=budget, used=0)
        if not names:
            self.last = result
            return result

        bids = {s: self.bid(s, sections[s], query,
                            (fresh or {}).get(s)) for s in names}
        total_bid = sum(bids.values()) or 1.0
        floor = int(budget * _FLOOR_FRAC)
        ceil = int(budget * _CEIL_FRAC)

        # proportional share, clamped to [floor, ceil]; iterate the
        # surplus back to unclamped sections (water-filling): a section
        # pinned at its ceiling frees its excess for the next round until
        # nobody new clamps
        limits: dict[str, int] = {}
        fixed: set[str] = set()          # pinned at the ceiling this round

        def _pinned_total() -> int:
            return sum(limits[s] for s in fixed)

        for _ in range(6):               # converges fast
            pool = [s for s in names if s not in fixed]
            pool_bid = sum(bids[s] for s in pool) or 1.0
            avail = budget - _pinned_total()
            changed = False
            for s in sorted(pool):
                want = int(avail * bids[s] / pool_bid)
                if want >= ceil:
                    limits[s] = ceil
                    fixed.add(s)
                    changed = True
                else:
                    limits[s] = max(floor, min(ceil, want))
            if not changed:
                break

        # floors may have pushed the total over budget — shave the
        # largest allocations back down toward their floor
        over = sum(limits.values()) - budget
        for s in sorted(limits, key=lambda k: -limits[k]):
            if over <= 0:
                break
            shave = min(over, limits[s] - floor)
            if shave > 0:
                limits[s] -= shave
                over -= shave

        for s in names:
            text = str(sections[s])
            limit = limits.get(s, floor)
            trimmed = len(text) > limit
            result.allocations.append(Allocation(
                section=s, chars=min(len(text), limit), limit=limit,
                bid=bids[s], trimmed=trimmed))
        result.used = sum(a.chars for a in result.allocations)
        self.last = result
        self.log.append("attention.auction", result.to_dict(),
                        actor="kernel")
        return result

    # -- enforcement ---------------------------------------------------------------

    def enforce(self, sections: dict[str, str], result: AuctionResult
                ) -> dict[str, str]:
        """Apply the allocations: trim every over-budget section,
        visibly (the model sees what was cut and why)."""
        limits = {a.section: a.limit for a in result.allocations}
        out: dict[str, str] = {}
        for name, text in sections.items():
            limit = limits.get(name, len(text))
            text = str(text)
            if len(text) <= limit:
                out[name] = text
                continue
            # a zero/negative limit means EVERYTHING is cut — text[-0:]
            # would hand back the whole body, so guard the slices
            if limit <= 0:
                out[name] = (f"\n[…trimmed by the attention auction — "
                             f"{len(text):,} chars wanted, 0 allocated]")
                continue
            head = int(limit * _KEEP_HEAD)
            tail = limit - head
            trimmed = text[:head]
            if tail > 0:
                trimmed += "\n…\n" + text[-tail:]
            out[name] = (trimmed
                         + f"\n[…trimmed by the attention auction — "
                           f"{len(text):,} chars wanted, {limit:,} "
                           f"allocated]")
        return out

    def format_last(self) -> str:
        if self.last is None:
            return "no auction has run yet"
        r = self.last
        lines = [f"ATTENTION AUCTION — budget {r.budget:,} chars, "
                 f"used {r.used:,}"]
        for a in r.allocations:
            bar = "█" * max(1, int(a.limit / r.budget * 20))
            cut = " ✂" if a.trimmed else ""
            lines.append(f"  {a.section:<14} bid {a.bid:.2f} → "
                         f"{a.limit:,}{cut} {bar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — the auction's whole mechanics, offline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "attention.jsonl")
        eco = AttentionEconomy(log, budget_chars=10_000)

        sections = {
            "goal": "ship the parser v2 with property tests and docs "
                    "covering the tokenizer rewrite",
            "constitution": "standing rules that barely change at all "
                            "and are mostly always the same words here",
            "memory": "unrelated pasta recipes and old notes about a "
                      "printer",
            "web": "live cricket scores and weather for a city "
                   "somewhere",
        }
        result = eco.allocate(sections, query="ship the parser v2 "
                                              "tokenizer rewrite")
        by = {a.section: a for a in result.allocations}

        # relevance rules: goal gets the most, web the least
        assert by["goal"].limit > by["web"].limit, \
            {a.section: a.limit for a in result.allocations}
        assert by["memory"].limit > by["web"].limit
        # floors respected — nothing starves
        floor = int(10_000 * _FLOOR_FRAC)
        assert all(a.limit >= floor for a in result.allocations)
        # no monopolist
        assert all(a.limit <= int(10_000 * _CEIL_FRAC)
                   for a in result.allocations)
        # total under budget
        assert result.used <= 10_000

        # enforcement trims oversized sections VISIBLY
        fat = dict(sections)
        fat["goal"] = "ship the parser v2. " * 2000      # ~56k chars
        big = eco.allocate(fat, query="parser v2")
        enforced = eco.enforce(fat, big)
        assert "attention auction" in enforced["goal"]
        assert len(enforced["goal"]) <= big.allocations[0].limit + 200
        # small sections pass through untouched
        assert enforced["web"] == fat["web"]

        # recency: a stale section loses to a fresh identical one
        r_fresh = eco.allocate({"memory": "parser notes"},
                               query="parser",
                               fresh={"memory": time.time()})
        fresh_bid = eco.bid("memory", "parser notes", "parser",
                            time.time())
        stale_bid = eco.bid("memory", "parser notes", "parser",
                            time.time() - 7 * 86400)
        assert stale_bid < fresh_bid

        # empty sections are dropped; empty input is clean
        assert eco.allocate({}).allocations == []
        # the auction is sealed and auditable
        kinds = [e.type for e in log.events()]
        assert kinds.count("attention.auction") >= 2
        assert "ATTENTION AUCTION" in eco.format_last()

        print("ATTENTION SELF-TEST PASS")

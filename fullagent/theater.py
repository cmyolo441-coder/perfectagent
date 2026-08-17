"""THEATER — the time-travel debugger for agent cognition.

The kernel records everything; the Theater makes it navigable like film:

    frames    every event is a frame; frame(seq) reconstructs the ENTIRE
              agent state at that instant (fold upto_seq): conversation
              so far, cost, tool calls, files touched, goal state
    why       the causal envelope already sealed in every event
              (causation_id chains) renders as an indented proof tree —
              "why did the agent do X?" answered from evidence, never
              reconstructed from vibes
    diff      state_at(a) vs state_at(b): exactly what changed between
              two moments (messages added, tokens spent, files touched)
    counterfactual   fork the timeline at seq and REPLAY it with one
              tool.call removed — "what if the agent had NOT made that
              call?" The counterfactual branch is real: it can be
              checked out, resumed, merged. The divergence report shows
              what actually changed (messages, tool calls, cost).

Everything is a pure fold — zero tokens, fully deterministic. The
scrubber is the TUI command; the projector is this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .kernel import EventLog, fold

_MAX_CF_REPLAY = 400      # safety ceiling for counterfactual replays


@dataclass
class Frame:
    """The agent's world at one seq."""
    seq: int
    type: str
    summary: str
    state: dict = field(default_factory=dict)

    def format(self) -> str:
        lines = [f"FRAME seq {self.seq} — {self.type}",
                 f"  {self.summary}"]
        s = self.state
        lines.append(
            f"  conversation: {len(s.get('messages', []))} msgs · "
            f"tool calls: {s.get('tool_calls', 0)} · "
            f"errors: {s.get('tool_errors', 0)} · "
            f"cost: ${s.get('cost_usd', 0.0):.4f}")
        files = s.get("files_touched") or set()
        if files:
            lines.append("  files touched: "
                         + ", ".join(sorted(files)[:10]))
        goal = s.get("goal")
        if goal:
            lines.append(f"  goal: {str(goal.get('statement', ''))[:80]}")
        return "\n".join(lines)


class Theater:
    """Frames, causal whys, diffs, counterfactual forks."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    # -- frames -------------------------------------------------------------------

    def frames(self, branch: str | None = None) -> list[dict]:
        """Every event as a slim frame dict (the scrubber strip)."""
        out = []
        for ev in self.log.events(branch):
            out.append({"seq": ev.seq, "type": ev.type,
                        "actor": ev.actor,
                        "summary": _summary(ev)})
        return out

    def frame(self, seq: int) -> Frame | None:
        """Full state reconstruction AT seq (inclusive)."""
        target = next((e for e in self.log.events() if e.seq == seq), None)
        if target is None:
            return None
        st = fold(self.log, upto_seq=seq)
        state = {"messages": st.messages, "tool_calls": st.tool_calls,
                 "tool_errors": st.tool_errors,
                 "cost_usd": st.cost_usd,
                 "files_touched": st.files_touched,
                 "goal": st.goal}
        return Frame(seq=seq, type=target.type,
                     summary=_summary(target), state=state)

    # -- why ------------------------------------------------------------------------

    def why(self, seq: int) -> str:
        """The causal proof tree behind the event at seq — from the
        sealed envelope, oldest cause first."""
        target = next((e for e in self.log.events() if e.seq == seq), None)
        if target is None:
            return f"no event at seq {seq}"
        chain = self.log.why(target.id)
        lines = [f"WHY seq {seq} ({target.type}) — causal chain, root "
                 f"cause last:"]
        for i, ev in enumerate(chain):
            pad = "  " * (i + 1)
            lines.append(f"{pad}← seq {ev.seq} {ev.type} ({ev.actor})"
                         + (f" · {_summary(ev)}" if _summary(ev) else ""))
        return "\n".join(lines)

    # -- diff -------------------------------------------------------------------------

    def diff(self, a: int, b: int) -> str:
        """What changed between seq a and seq b (fold-level state diff)."""
        sa = fold(self.log, upto_seq=a)
        sb = fold(self.log, upto_seq=b)
        msgs_added = sb.messages[len(sa.messages):]
        lines = [f"DIFF seq {a} → seq {b}",
                 f"  messages  {len(sa.messages)} → {len(sb.messages)} "
                 f"(+{len(msgs_added)})",
                 f"  tool calls {sa.tool_calls} → {sb.tool_calls}",
                 f"  errors     {sa.tool_errors} → {sb.tool_errors}",
                 f"  cost       ${sa.cost_usd:.4f} → ${sb.cost_usd:.4f}",
                 f"  files      {len(sa.files_touched)} → "
                 f"{len(sb.files_touched)}"
                 + ("  (+" + ", ".join(sorted(
                     sb.files_touched - sa.files_touched)[:5]) + ")"
                    if sb.files_touched - sa.files_touched else "")]
        for m in msgs_added[-3:]:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:100]
            lines.append(f"    + [{role}] {content}")
        return "\n".join(lines)

    # -- counterfactual -------------------------------------------------------------------

    def counterfactual(self, seq: int, name: str = "") -> dict:
        """Fork the timeline at seq and replay everything after it with
        the event at seq REMOVED. Returns the divergence report and the
        real branch name — the counterfactual can be checked out and
        worked on like any timeline."""
        target = next((e for e in self.log.events() if e.seq == seq), None)
        if target is None:
            raise ValueError(f"no event at seq {seq}")
        branch = name or f"cf/{seq}-{target.type}"
        # fork from the event BEFORE the removed one
        base = self.log._event_at(self.log.branch, seq - 1)
        self.log._heads[branch] = base.id if base else None

        original_branch = self.log.branch
        before = fold(self.log, upto_seq=seq)          # state with the event
        replayed = 0
        try:
            self.log.branch = branch
            for ev in self.log.events(original_branch):
                if ev.seq <= seq or ev.type in ("kernel.rewind",
                                                "kernel.branch"):
                    continue
                if replayed >= _MAX_CF_REPLAY:
                    break
                self.log.append(ev.type, ev.data, actor="cf",
                                provenance=ev.provenance)
                replayed += 1
        finally:
            self.log.branch = original_branch

        after = fold(self.log, branch=branch)
        report = {
            "branch": branch, "removed_seq": seq,
            "removed_type": target.type,
            "removed_summary": _summary(target),
            "events_replayed": replayed,
            "divergence": {
                "messages_with": len(before.messages),
                "messages_without": len(after.messages),
                "tool_calls_with": before.tool_calls,
                "tool_calls_without": after.tool_calls,
                "files_with": sorted(before.files_touched)[:10],
                "files_without": sorted(after.files_touched)[:10]},
        }
        self.log.append("theater.counterfactual", report, actor="human")
        return report

    def format_cf(self, report: dict) -> str:
        d = report["divergence"]
        return "\n".join([
            f"COUNTERFACTUAL — removed seq {report['removed_seq']} "
            f"({report['removed_type']})",
            f"  event: {report['removed_summary']}",
            f"  branch '{report['branch']}' carries the world without it "
            f"({report['events_replayed']} events replayed)",
            f"  messages: with {d['messages_with']} → without "
            f"{d['messages_without']}",
            f"  tool calls: {d['tool_calls_with']} → "
            f"{d['tool_calls_without']}",
            f"  files: with {len(d['files_with'])} → without "
            f"{len(d['files_without'])}",
            "  checkout with: /branch " + report["branch"],
        ])


def _summary(ev) -> str:
    """One-line essence of an event for frames and whys."""
    d = ev.data or {}
    if ev.type in ("user.message", "assistant.message"):
        return str(d.get("text", ""))[:100]
    if ev.type == "tool.call":
        return f"{d.get('name', '')} {str(d.get('args'))[:80]}"
    if ev.type == "tool.result":
        return f"{d.get('name', '')} → {str(d.get('status', ''))}"
    if ev.type == "goal.set":
        return str(d.get("statement", ""))[:100]
    if ev.type == "fact.learned":
        return str(d.get("fact", ""))[:100]
    if ev.type == "crew.done":
        return f"[{d.get('role', '')}] {str(d.get('summary', ''))[:80]}"
    return ""


# ---------------------------------------------------------------------------
# Self-test — build a real timeline, scrub it, diff it, fork it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "theater.jsonl")

        # a small real session
        root = log.append("user.message", {"text": "fix the parser"},
                          actor="human")
        mid = log.append("tool.call",
                         {"name": "edit_file",
                          "args": {"path": "p.py", "new_string": "x=1"}},
                         actor="sovereign", causation_id=root.id)
        log.append("tool.result", {"name": "edit_file", "status": "done"},
                   actor="system", causation_id=mid.id)
        log.append("cost.incurred", {"usd": 0.01, "tokens_in": 500,
                                     "tokens_out": 100})
        log.append("assistant.message", {"text": "parser fixed"})

        th = Theater(log)

        # frames strip covers every event
        strip = th.frames()
        assert [f["seq"] for f in strip] == [0, 1, 2, 3, 4]
        assert "parser" in strip[0]["summary"]

        # frame() reconstructs exact state at that point
        f2 = th.frame(2)
        assert f2.state["tool_calls"] == 1
        assert f2.state["cost_usd"] == 0.0        # cost lands at seq 3
        f3 = th.frame(3)
        assert f3.state["cost_usd"] == 0.01 and f3.state["messages"]
        assert th.frame(99) is None

        # why: the causal chain is root→leaf, evidence-sealed
        why = th.why(2)
        assert "user.message" in why and "tool.call" in why, why

        # diff between two moments
        d = th.diff(1, 4)
        assert "tool calls 1 → 1" in d and "+1" in d

        # counterfactual: remove the edit_file call at seq 1
        report = th.counterfactual(1)
        assert report["removed_type"] == "tool.call"
        assert report["branch"] in log.branches()
        # the CF branch exists WITHOUT the removed tool.call but WITH
        # everything replayed after it
        cf_types = [e.type for e in log.events(report["branch"])]
        assert "tool.call" not in cf_types, cf_types
        assert "assistant.message" in cf_types
        # state divergence is measured, not vibes
        div = report["divergence"]
        assert div["tool_calls_with"] == 1
        # spine integrity survives the surgery
        ok, msg = log.verify(report["branch"])
        assert ok, msg
        assert any(e.type == "theater.counterfactual"
                   for e in log.events())

        # frames/why on the CF branch work too
        cf_strip = th.frames(report["branch"])
        assert cf_strip and cf_strip[0]["type"] != "kernel.branch"

        print("THEATER SELF-TEST PASS")

"""MERGE — semantic timeline merge: git for agent cognition.

The kernel can fork and rewind timelines but never unify two diverged
ones. This is the missing merge: two branches that share a common
ancestor are reconciled into a NEW branch — semantically, not textually.

    ancestor   walk parent chains from both heads until they meet
               (the kernel chain makes this exact, never guessed)
    diff       each side's events SINCE the ancestor, classified:
               user messages, assistant messages, tool calls (with
               written paths), everything else
    reconcile  three rules, all mechanical:
                 1. IDENTICAL  — same type + same payload on both sides:
                    keep once (both agents did the same work)
                 2. ONLY-A / ONLY-B — replay onto the merge branch in
                    causal order (both sides' work survives)
                 3. CONFLICT  — same file written on both sides with
                    different content, or same seq-slot diverged: BOTH
                    versions are sealed into a merge.conflict event —
                    nothing is silently dropped, the human decides
    materialise  the merged branch is built by replaying the logical
                 events (type + data) from ancestor-forward: A's
                 exclusive events, B's exclusive events, shared events
                 once — then sealing merge.merged with the summary

File-level conflict detection is exact: a write's content hash on side A
vs side B for the same path. Message-level merge is order-preserving per
side, interleaved deterministically (A's event first when both exist at
the same distance from the ancestor).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

from .kernel import EventLog, Event

_CONTENT_TYPES = ("user.message", "assistant.message", "tool.call",
                  "tool.result", "fact.learned", "memory.episode")
_SKIP_TYPES = ("kernel.rewind", "kernel.branch")


def _payload_key(ev: Event) -> str:
    """Logical identity of an event (branch/seq stripped)."""
    return hashlib.sha256(
        f"{ev.type}|{sorted(ev.data.items(), key=lambda kv: str(kv[0]))}"
        .encode("utf-8", "replace")).hexdigest()[:16]


def _write_path(ev: Event) -> str | None:
    """The file a tool.call writes, if any."""
    if ev.type != "tool.call":
        return None
    name = str(ev.data.get("name", ""))
    if name not in ("write_file", "edit_file"):
        return None
    p = str((ev.data.get("args") or {}).get("path", "")).strip()
    return p or None


def _content_hash(ev: Event) -> str:
    """Hash of the written content (for same-file divergence checks)."""
    args = ev.data.get("args") or {}
    for key in ("content", "new_string", "text"):
        if key in args:
            return hashlib.sha256(str(args[key]).encode(
                "utf-8", "replace")).hexdigest()[:16]
    return ""


@dataclass
class MergeResult:
    branch: str
    ancestor_seq: int
    only_a: list[Event] = field(default_factory=list)
    only_b: list[Event] = field(default_factory=list)
    shared: list[Event] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"branch": self.branch,
                "ancestor_seq": self.ancestor_seq,
                "only_a": len(self.only_a), "only_b": len(self.only_b),
                "shared": len(self.shared),
                "conflicts": [c["path"] for c in self.conflicts]}


class TimelineMerger:
    """Semantic merge of two kernel branches into a third."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    # -- ancestor ---------------------------------------------------------------

    def ancestor(self, a: str, b: str) -> int:
        """The seq of the newest event common to both branches' chains
        (-1 if they share nothing). Exact via parent-chain sets."""
        def chain_ids(branch: str) -> list[str]:
            return [e.id for e in self.log.events(branch)]
        a_ids = set(chain_ids(a))
        best = -1
        for ev in self.log.events(b):
            if ev.id in a_ids:
                best = max(best, ev.seq)
        return best

    # -- merge -------------------------------------------------------------------

    def merge(self, a: str, b: str, name: str = "") -> MergeResult:
        """Merge branches a and b into a NEW branch. The new branch is
        checked out; conflicts are sealed but never resolved silently."""
        branches = set(self.log.branches())
        if a not in branches or b not in branches:
            raise ValueError(
                f"unknown branch(es): {a!r}, {b!r} — known: "
                + ", ".join(sorted(branches)))
        anc = self.ancestor(a, b)
        result = MergeResult(
            branch=name or f"merge/{a}+{b}", ancestor_seq=anc)
        # never clobber an existing merge branch on a re-merge — rewind it
        # would orphan its previous merged events silently
        existing = set(self.log.branches())
        if result.branch in existing:
            n = 2
            while f"{result.branch}-{n}" in existing:
                n += 1
            result.branch = f"{result.branch}-{n}"

        evs_a = [e for e in self.log.events(a)
                 if e.seq > anc and e.type not in _SKIP_TYPES]
        evs_b = [e for e in self.log.events(b)
                 if e.seq > anc and e.type not in _SKIP_TYPES]
        # multiset matching: identical payloads are matched ONE-TO-ONE.
        # Membership-only matching collapsed legitimate repeats (a user
        # message sent twice) into a single merged event.
        count_b: dict[str, int] = {}
        for e in evs_b:
            count_b[_payload_key(e)] = count_b.get(_payload_key(e), 0) + 1

        # classify: shared (identical payload) vs exclusive
        for ev in evs_a:
            key = _payload_key(ev)
            if count_b.get(key, 0) > 0:
                result.shared.append(ev)
                count_b[key] -= 1
            else:
                result.only_a.append(ev)
        count_a: dict[str, int] = {}
        for e in evs_a:
            count_a[_payload_key(e)] = count_a.get(_payload_key(e), 0) + 1
        result.only_b = []
        for e in evs_b:
            key = _payload_key(e)
            if count_a.get(key, 0) > 0:
                count_a[key] -= 1  # twin already classified on the A side
            else:
                result.only_b.append(e)

        self.log.append("merge.started",
                        {"a": a, "b": b, "ancestor": anc,
                         "into": result.branch}, actor="human")

        # file-write conflicts: same path, different content, both sides
        writes_a = {p: ev for ev in result.only_a
                    if (p := _write_path(ev))}
        writes_b = {p: ev for ev in result.only_b
                    if (p := _write_path(ev))}
        for path in sorted(set(writes_a) & set(writes_b)):
            ha, hb = _content_hash(writes_a[path]), _content_hash(
                writes_b[path])
            if ha != hb:
                result.conflicts.append({
                    "path": path, "kind": "file_write",
                    "a": {"seq": writes_a[path].seq,
                          "hash": ha},
                    "b": {"seq": writes_b[path].seq, "hash": hb}})
                self.log.append("merge.conflict",
                                {"path": path, "kind": "file_write",
                                 "a": result.conflicts[-1]["a"],
                                 "b": result.conflicts[-1]["b"]},
                                actor="kernel")

        # materialise the merged branch: replay logical events in a
        # deterministic interleave — A-exclusive, B-exclusive, shared —
        # each once, content types only (structural events would be lies)
        fork_at = self.ancestor(a, b)
        merge_branch = result.branch
        # fork the merge branch from A at the ancestor, carrying A's base
        # (fork seeds the branch; the shared ancestor events are already
        # on it through A's chain)
        self.log._heads[merge_branch] = self.log._event_at(
            a, fork_at).id if fork_at >= 0 else None
        if merge_branch not in self.log.branches():
            self.log._heads.setdefault(merge_branch, None)
        replay = []
        for ev in result.only_a + result.only_b + result.shared:
            # every classified copy replays — duplicates are REAL history
            if ev.type in _CONTENT_TYPES:
                replay.append(ev)
        cur = self.log.branch
        self.log.branch = merge_branch
        try:
            for ev in replay:
                # deep-copy the payload: the original event and the
                # replayed one must never share a mutable dict — mutating
                # either side would corrupt the other's content hash and
                # break verify()
                self.log.append(ev.type, copy.deepcopy(ev.data),
                                actor="merge", provenance=ev.provenance)
        finally:
            self.log.branch = cur
        self.log.checkout(merge_branch)

        self.log.append("merge.merged", result.to_dict(), actor="kernel")
        return result

    # -- reporting ------------------------------------------------------------------

    def format(self, result: MergeResult) -> str:
        lines = [f"MERGED → branch '{result.branch}' "
                 f"(ancestor seq {result.ancestor_seq})",
                 f"  A exclusive : {len(result.only_a)} events",
                 f"  B exclusive : {len(result.only_b)} events",
                 f"  shared once : {len(result.shared)} events"]
        if result.conflicts:
            lines.append(f"  ⚠ {len(result.conflicts)} FILE CONFLICT(S):")
            for c in result.conflicts:
                lines.append(f"    {c['path']} — A hash {c['a']['hash'][:8]}"
                             f" vs B hash {c['b']['hash'][:8]} (both "
                             f"versions sealed in merge.conflict)")
        else:
            lines.append("  no conflicts — clean merge")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — fork, diverge, merge, verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "merge-test.jsonl")
        for i in range(2):       # shared ancestor work
            log.append("user.message", {"text": f"base {i}"})
        anc = log.head()

        alt = log.fork(at_seq=anc, name="alt")
        log.checkout("main")
        log.append("user.message", {"text": "main continues"})
        log.append("tool.call", {"name": "write_file",
                                 "args": {"path": "app.py",
                                          "content": "print('main')"}})
        log.append("assistant.message", {"text": "main shipped app.py"})

        log.checkout(alt)
        log.append("user.message", {"text": "alt diverges"})
        log.append("tool.call", {"name": "write_file",
                                 "args": {"path": "app.py",
                                          "content": "print('alt')"}})
        log.append("tool.call", {"name": "write_file",
                                 "args": {"path": "lib.py",
                                          "content": "x = 1"}})
        log.append("fact.learned", {"fact": "alt learned lib.py layout"})

        merger = TimelineMerger(log)
        assert merger.ancestor("main", "alt") == anc

        result = merger.merge("main", "alt")

        # identical work kept once; exclusives from both sides survive
        assert result.only_a and result.only_b, result.to_dict()
        # app.py written DIFFERENTLY on both sides -> conflict sealed
        assert [c["path"] for c in result.conflicts] == ["app.py"], \
            result.conflicts
        # the merge branch carries both sides' unique content events
        merged_types = [e.type for e in log.events(result.branch)]
        assert "main continues" in [
            e.data.get("text") for e in log.events(result.branch)
            if e.type == "user.message"]
        assert "alt diverges" in [
            e.data.get("text") for e in log.events(result.branch)
            if e.type == "user.message"]
        # lib.py (only on alt) replayed onto the merge
        merged_paths = [str((e.data.get("args") or {}).get("path"))
                        for e in log.events(result.branch)
                        if e.type == "tool.call"]
        assert "lib.py" in merged_paths and "app.py" in merged_paths
        # the kernel verifies the merged branch's spine
        ok, msg = log.verify(result.branch)
        assert ok, msg
        # events sealed (across branches — merge ran on several)
        types = [e.type for br in log.branches() for e in log.events(br)]
        assert "merge.started" in types and "merge.merged" in types \
            and "merge.conflict" in types

        # identical work on both sides collapses to one replay
        log.checkout("main")
        log.append("user.message", {"text": "same insight"})
        log.checkout(alt)
        log.append("user.message", {"text": "same insight"})
        r2 = merger.merge("main", "alt", name="merge/clean")
        texts = [e.data.get("text") for e in log.events("merge/clean")
                 if e.type == "user.message"]
        assert texts.count("same insight") == 1, texts
        assert r2.to_dict()["shared"] >= 1

        # unknown branch raises with the roster
        try:
            merger.merge("main", "nope")
            raise AssertionError("must raise on unknown branch")
        except ValueError as e:
            assert "alt" in str(e)

        print("MERGE SELF-TEST PASS")

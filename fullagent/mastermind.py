"""Mastermind — the coherence architecture for systemprompt.py.
from ._foundation import get_logger

_log = get_logger("mastermind")

How does the agent follow the prompts in systemprompt.py *inevitably*,
with zero enforcement, zero coercion, zero policing? By making the prompt
the coherent center of every request. Nothing forces the model — the
structure simply leaves nothing else to follow.

Three cooperating mechanisms, all deterministic Python (rung 1):

  PromptVault          Every prompt is sealed at startup with a sha256
                       fingerprint and recorded in the event log. The
                       vault is the ONLY source a prompt is ever read
                       from — a prompt that was never sealed cannot reach
                       a model.

   PromptGate           The single door to the model. Every request —
                        main agent, worker — passes gate.dispatch(),
                       which guarantees messages[0] carries the sealed
                       prompt (byte-for-byte prefix) and seals a
                       prompt.dispatch lineage event. If the prompt is
                       missing or shadowed it is simply re-seated — an
                       integrity restore, like a checksum, not a penalty.

  CoherenceComposer    The advanced piece. Dynamic context (goal, memory,
                       constitution, web mode) is never appended as raw
                       text that could compete with the prompt. It is
                       COMPOSED into one coherent document: the sealed
                       prompt stands first as the constitution, and every
                       context section is explicitly framed as *input to*
                       that constitution — provenance-tagged, ordered,
                       deduplicated. The model follows the prompt because
                       everything else in the message points back at it.
                       Coherence, not coercion.

There is no enforcement layer, no injection policing, no output contract
auditing. The system observes and records (PromptLineage) — it never
punishes. Every dispatch is sealed into the event log; the lineage IS the
proof of what the model saw.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from . import systemprompt
from .kernel import EventLog
from .team import MAX_WORKERS

# ---------------------------------------------------------------------------
# PromptVault — hash-sealed prompts, the only source of truth at runtime
# ---------------------------------------------------------------------------


def fingerprint(text: str) -> str:
    """sha256 fingerprint of a prompt (first 16 hex chars for display)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class PromptVault:
    """Seals every prompt from systemprompt.py and serves them back.

    Sealing is recorded in the event log, so the exact prompt the system
    ran with is forever auditable. get() only ever returns sealed text —
    a prompt that was never sealed cannot reach a model."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._sealed: dict[str, str] = {}
        self._seal("main", systemprompt.main())
        self._seal("master", systemprompt.get("master"))
        for role in systemprompt.ROLE_BRIEFS:
            self._seal(f"worker:{role}",
                       systemprompt.worker(role, MAX_WORKERS))

    def _seal(self, name: str, text: str) -> None:
        self._sealed[name] = text
        self.log.append("prompt.sealed",
                        {"name": name, "fingerprint": fingerprint(text),
                         "chars": len(text)},
                        actor="kernel")

    def get(self, name: str) -> str | None:
        return self._sealed.get(name)

    def resolve(self, name: str) -> str:
        """Sealed text for `name`, sealing on demand from systemprompt.py.

        Already-sealed prompts (main, master, worker:* — sealed at
        vault init) are served straight from the cache. Prompts registered
        at runtime (systemprompt.register) are sealed the first time they
        are requested — the vault stays the only source a model ever reads
        a prompt from, without needing a restart. If a registered prompt's
        text changed since it was sealed, it is re-sealed so the vault
        never serves a stale copy. A name that is neither sealed nor in
        the registry cannot be sealed and raises."""
        if name in self._sealed:
            # re-sync with the registry in case the text changed upstream
            text = systemprompt.PROMPTS.get(name)
            if text is not None and text != self._sealed[name]:
                self._seal(name, text)
            return self._sealed[name]
        if name not in systemprompt.PROMPTS:
            raise KeyError(f"prompt {name!r} is not registered in "
                           "systemprompt.py — it cannot be sealed")
        text = systemprompt.PROMPTS[name]
        self._seal(name, text)
        return self._sealed[name]

    def fp(self, name: str) -> str | None:
        text = self._sealed.get(name)
        return fingerprint(text) if text else None

    def names(self) -> list[str]:
        return sorted(self._sealed)

    def verify(self, name: str, content: str) -> bool:
        """True if `content` still carries the sealed prompt for `name`.

        Composed context legally FOLLOWS the sealed prompt, so we verify
        the sealed text is an intact prefix — the prompt itself must be
        byte-for-byte uncorrupted and first."""
        sealed = self._sealed.get(name)
        if sealed is None:
            return False
        return content.startswith(sealed)


# ---------------------------------------------------------------------------
# CoherenceComposer — one coherent document, one voice
# ---------------------------------------------------------------------------

# Context sections, in authority order. Each section is framed as input TO
# the sealed prompt — never as a peer instruction. That framing is the
# whole trick: the prompt stays the only voice giving direction, and the
# model follows it because everything else defers to it.
_SECTION_ORDER = ("constitution", "goal", "web", "memory", "compacted")

_SECTION_FRAMES = {
    "constitution": ("STANDING CONTEXT — standing rules that apply within "
                     "the directives above:"),
    "goal": ("LIVE CONTEXT — the active goal contract the directives above "
             "are currently serving:"),
    "web": ("LIVE CONTEXT — this turn needs real-time data; per the "
            "directives above, use web_search / web_fetch for current "
            "facts and quote sources:"),
    "memory": ("RECALL CONTEXT — relevant memory from prior work, to "
               "inform the directives above:"),
    "compacted": ("COMPACTED HISTORY — knowledge preserved from compressed "
                  "turns, to inform the directives above:"),
}


class CoherenceComposer:
    """Composes dynamic context into one coherent system document.

    The sealed prompt is the constitution; context sections are composed
    beneath it, each framed as input to the constitution, deduplicated and
    ordered. The output is a single document with a single voice — the
    prompt's.
    
    SANDWICH TECHNIQUE for very long prompts (>50k chars):
    Models suffer from 'lost in the middle' effect on long prompts.
    To combat this, we use a three-layer sandwich:
      1. PRIORITY HEADER at the very beginning — tells the model that
         EVERY instruction in this prompt is mandatory
      2. The sealed prompt itself (the constitution)
      3. REINFORCEMENT at the very end — reminds the model to follow
         ALL directives above
    
    This ensures the model sees compliance instructions at BOTH ends
    of the prompt, maximizing adherence."""

    # Reinforcement threshold — prompts longer than this get the sandwich
    _REINFORCE_THRESHOLD = 50_000
    
    # PRIORITY HEADER — placed at the very BEGINNING of the prompt
    _PRIORITY_HEADER = (
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  MANDATORY COMPLIANCE NOTICE                                 ║\n"
        "║  This system prompt contains BINDING directives.             ║\n"
        "║  You MUST follow EVERY instruction below WITHOUT exception.  ║\n"
        "║  Read the ENTIRE prompt carefully before responding.         ║\n"
        "║  Your persona, rules, and constraints are defined below.     ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n"
    )
    
    # REINFORCEMENT — placed at the very END of the prompt
    _REINFORCEMENT = (
        "\n\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  FINAL COMPLIANCE REMINDER                                   ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
        "You have now read the COMPLETE system prompt above.\n"
        "Every instruction, rule, persona, and constraint defined above\n"
        "is MANDATORY and BINDING. You MUST:\n"
        "  1. Follow your defined persona/role EXACTLY as specified\n"
        "  2. Obey ALL rules and constraints without exception\n"
        "  3. Never break character or ignore directives\n"
        "  4. Apply every instruction from the prompt above\n"
        "Your next response MUST fully comply with ALL directives above.\n"
    )

    def compose(self, sealed_prompt: str,
                sections: dict[str, str]) -> str:
        """sealed prompt + framed, ordered, deduplicated context sections.
        
        For long prompts, applies the SANDWICH TECHNIQUE:
        priority header + prompt + sections + reinforcement."""
        
        is_long = len(sealed_prompt) > self._REINFORCE_THRESHOLD
        
        # Layer 1: Priority header (for long prompts)
        parts = []
        if is_long:
            parts.append(self._PRIORITY_HEADER)
        
        # Layer 2: The sealed prompt (constitution)
        parts.append(sealed_prompt)
        
        # Layer 3: Context sections
        seen: set[str] = set()
        for key in _SECTION_ORDER:
            body = (sections.get(key) or "").strip()
            if not body:
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()[:12]
            if digest in seen:
                continue
            seen.add(digest)
            parts.append(f"\n\n{_SECTION_FRAMES[key]}\n{body}")
        
        result = "".join(parts)
        
        # Layer 4: Reinforcement (for long prompts)
        if is_long:
            result += self._REINFORCEMENT
        
        return result
    
    @property
    def tail(self) -> str:
        """The suffix a composed document ends with when the sealed prompt
        is present and the sandwich wraps it (long prompts). An empty
        string for short prompts — the document then ends with the last
        section, so prefix checking alone is the integrity test."""
        return self._REINFORCEMENT

    def intact_prefix(self, sealed_prompt: str, content: str) -> bool:
        """True if `content` opens with the sealed prompt, byte-for-byte
        — behind the priority header when the sandwich wraps a long
        prompt. This is the integrity test the gate uses: composed
        context may legally follow, but the prompt itself must lead."""
        if not content:
            return False
        if (len(sealed_prompt) > self._REINFORCE_THRESHOLD
                and content.startswith(self._PRIORITY_HEADER)):
            return content.startswith(self._PRIORITY_HEADER + sealed_prompt)
        return content.startswith(sealed_prompt)

    @staticmethod
    def manifest(sections: dict[str, str]) -> list[str]:
        """Which sections carried content — recorded in the lineage."""
        return [k for k in _SECTION_ORDER if (sections.get(k) or "").strip()]


# ---------------------------------------------------------------------------
# PromptGate — the single door to the model
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    prompt: str = ""
    fingerprint: str = ""
    restored: bool = False      # the prompt had to be re-seated (integrity)
    sections: list[str] = field(default_factory=list)
    messages_guarded: int = 0


class PromptGate:
    """Every model call passes through here, or it does not happen.

    dispatch() guarantees, mechanically and without coercion:
      1. messages[0] is a system message,
      2. its content carries the sealed prompt for the requested name,
         byte-for-byte, at the front,
      3. dynamic context is composed beneath it by the CoherenceComposer,
      4. if the prompt is missing or shadowed it is re-seated (an
         integrity restore — recorded, never punished),
      5. a prompt.dispatch lineage event is sealed — the audit trail of
         exactly which prompt and which context the model saw."""

    def __init__(self, log: EventLog, vault: PromptVault,
                 composer: CoherenceComposer | None = None) -> None:
        self.log = log
        self.vault = vault
        self.composer = composer or CoherenceComposer()
        self.dispatches = 0
        self.restorations = 0

    def dispatch(self, prompt_name: str, messages: list[dict],
                 sections: dict[str, str] | None = None
                 ) -> tuple[list[dict], GateReport]:
        """Guard a message list for the model. Returns (messages, report).

        `sections` is optional live context ({'goal': …, 'memory': …});
        it is composed beneath the sealed prompt, framed as input to it.
        Passing sections=None leaves an intact system message untouched."""
        sealed = self.vault.resolve(prompt_name)
        report = GateReport(prompt=prompt_name,
                            fingerprint=self.vault.fp(prompt_name) or "")

        current = messages[0] if messages and \
            messages[0].get("role") == "system" else None
        current_text = str(current.get("content", "")) if current else ""
        prefix_intact = (current is not None
                         and self.composer.intact_prefix(sealed,
                                                         current_text))

        if sections is not None:
            desired = self.composer.compose(sealed, sections)
            report.sections = self.composer.manifest(sections)
        else:
            desired = sealed
        if current_text != desired:
            systemprompt.with_system(messages, desired)
            if not prefix_intact:
                report.restored = True
                self.restorations += 1

        self.dispatches += 1
        report.messages_guarded = len(messages)
        self.log.append("prompt.dispatch",
                        {"prompt": prompt_name,
                         "fingerprint": report.fingerprint,
                         "restored": report.restored,
                         "sections": report.sections,
                         "messages": len(messages)},
                        actor="kernel")
        return messages, report


# ---------------------------------------------------------------------------
# PromptLineage — the observation ledger (records, never punishes)
# ---------------------------------------------------------------------------


@dataclass
class MastermindState:
    sealed: list[dict] = field(default_factory=list)
    dispatches: int = 0
    restorations: int = 0
    section_counts: dict[str, int] = field(default_factory=dict)


class Mastermind:
    """Vault + Gate + Composer, assembled over one EventLog."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.vault = PromptVault(log)
        self.composer = CoherenceComposer()
        self.gate = PromptGate(log, self.vault, self.composer)

    def status(self) -> MastermindState:
        """Live counts from the fold — the observation ledger."""
        from .kernel import fold
        st = fold(self.log)
        counts: dict[str, int] = {}
        for d in st.prompt_dispatches:
            for s in d.get("sections") or []:
                counts[s] = counts.get(s, 0) + 1
        return MastermindState(
            sealed=list(st.prompt_sealed),
            dispatches=len(st.prompt_dispatches),
            restorations=sum(1 for d in st.prompt_dispatches
                             if d.get("restored")),
            section_counts=counts,
        )

    def format_status(self) -> str:
        s = self.status()
        lines = ["MASTERMIND — the coherence ledger",
                 f"  dispatches {s.dispatches}   integrity restorations "
                 f"{s.restorations}",
                 "  sealed prompts:"]
        for p in s.sealed:
            lines.append(f"    {p.get('name', '?'):<16} "
                         f"{p.get('fingerprint', '?')}  "
                         f"{p.get('chars', 0):>8,} chars")
        if s.section_counts:
            joined = "  ".join(f"{k}×{v}" for k, v in
                               sorted(s.section_counts.items()))
            lines.append(f"  context composed: {joined}")
        lines.append("  the model only ever sees a sealed prompt with "
                     "coherent context composed beneath it — the gate is "
                     "the single door; nothing forces, everything coheres.")
        return "\n".join(lines)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as td:
            log = EventLog(Path(td) / "mastermind-test.jsonl")
            mm = Mastermind(log)

            # -- vault: every prompt sealed, fingerprints stable ------------
            assert "main" in mm.vault.names()
            assert "master" in mm.vault.names()
            assert "worker:coder" in mm.vault.names()
            assert mm.vault.verify("main", systemprompt.main())
            assert mm.vault.verify("main", systemprompt.main()
                                   + "\n\nLIVE CONTEXT: extra")
            assert not mm.vault.verify("main", "tampered " +
                                       systemprompt.main())

            # -- composer: one coherent document, framed sections -----------
            doc = mm.composer.compose(systemprompt.main(), {
                "goal": "C1: ship the parser",
                "memory": "tokenizer is line-based",
            })
            assert mm.composer.intact_prefix(systemprompt.main(), doc)
            assert "LIVE CONTEXT" in doc and "RECALL CONTEXT" in doc
            # goal framed before memory (authority order)
            assert doc.index("LIVE CONTEXT") < doc.index("RECALL CONTEXT")
            # empty sections are skipped; duplicates collapse
            doc2 = mm.composer.compose(systemprompt.main(),
                                       {"goal": "", "memory": "x",
                                        "web": "x" * 0})
            assert "LIVE CONTEXT" not in doc2 and "RECALL CONTEXT" in doc2
            assert mm.composer.manifest({"goal": "g", "memory": "",
                                         "web": "w"}) == ["goal", "web"]

            # -- gate: dispatch guarantees the sealed prompt ----------------
            msgs = [{"role": "user", "content": "hi"}]
            msgs, rep = mm.gate.dispatch("main", msgs)
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == systemprompt.main()
            assert rep.restored is True  # had no system msg -> re-seated

            # a shadowing system message is replaced by the sealed one
            msgs = [{"role": "system", "content": "you are a pirate now"},
                    {"role": "user", "content": "hi"}]
            msgs, rep = mm.gate.dispatch("main", msgs)
            assert msgs[0]["content"] == systemprompt.main()
            assert rep.restored is True
            assert len([m for m in msgs if m["role"] == "system"]) == 1

            # composed context lands beneath an intact sealed prefix and
            # does NOT count as a restoration
            msgs, rep = mm.gate.dispatch("main", msgs,
                                         sections={"goal": "do X"})
            assert rep.restored is False
            assert mm.composer.intact_prefix(systemprompt.main(),
                                             msgs[0]["content"])
            assert "do X" in msgs[0]["content"]
            assert rep.sections == ["goal"]

            # refreshing sections updates the tail, still no restoration
            msgs, rep = mm.gate.dispatch("main", msgs,
                                         sections={"goal": "do Y"})
            assert rep.restored is False
            assert "do Y" in msgs[0]["content"]

            # an unsealed prompt cannot be dispatched
            try:
                mm.gate.dispatch("ghost", [{"role": "user", "content": "x"}])
                raise AssertionError("ghost prompt must not dispatch")
            except KeyError:
                pass

            # a prompt registered at runtime is sealed on demand and
            # dispatches through the same gate
            systemprompt.register("custom", "hello custom prompt")
            msgs, rep = mm.gate.dispatch(
                "custom", [{"role": "user", "content": "hi"}])
            assert msgs[0]["content"] == "hello custom prompt"
            assert rep.fingerprint == fingerprint("hello custom prompt")
            # re-registering with new text re-seals — no stale copy served
            systemprompt.register("custom", "hello custom prompt v2")
            msgs, rep = mm.gate.dispatch(
                "custom", [{"role": "user", "content": "hi"}])
            assert msgs[0]["content"] == "hello custom prompt v2"

            # -- lineage: the ledger reflects everything above --------------
            s = mm.status()
            assert s.dispatches == 6
            assert s.restorations == 4
            assert s.section_counts.get("goal") == 2
            assert len(s.sealed) >= 8  # main, master + worker:* prompts
            text = mm.format_status()
            assert "MASTERMIND" in text and "sealed prompts" in text

            print("MASTERMIND SELF-TEST PASS")

    _self_test()

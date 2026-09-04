"""FullAgent — an advanced terminal AI agent.

Package layout:
    config.py       providers, models, effort levels, paths
    systemprompt.py the ONE home of every system prompt (single source)
    mastermind.py   prompt coherence: sealed vault, gate, composer, lineage
    tools.py        tool registry: files, shell, search, real-time web
    client.py       OpenAI-compatible streaming chat client
    agent.py        agent loop (LLM <-> tools), event-sourced on the kernel
    kernel.py       Temporal Kernel: append-only, content-addressed event log
    memory.py       Hippocampus: episodic memory + dead-end ledger
    goal.py         goal contracts with machine-checkable done-criteria
    judge.py        deterministic verification (no LLM judging)
    team.py         shared subagent substrate: roles, reports, write lock
    crew.py         persistent Codex-style subagents (spawn/send/wait/
                    close/resume) — SERIAL execution, one agent at a time
    supercomputer.py the /on machine: EIGHT cores running at the same
                    time on one mission — parallel recon, an escalating
                    plan relay (v1→v8), a world-wide deep dive, a fused
                    master plan, 8 parallel build workstreams, then
                    adversarial verify⇄repair rounds until the board is
                    clean. Event-sourced, bounded-memory, live board.
    workflows.py    saved multi-step pipelines — phased orchestration
    report.py       enterprise audit export (md/html) + forecasting
    autopilot.py    self-routing brain: auto-enables goal / web

    v3 advanced subsystems (all event-sourced on the same kernel):
    router.py       smart model routing — cheapest capable model per task
    semantic.py     semantic vector memory — meaning-based recall
    speculate.py    speculative execution — prefetch read-only tool calls
    dashboard.py    live observability — real-time ledger projection
    daemon.py       mission control — resumable long-running missions
    healer.py       self-healing — root-cause capture, fix, retry, lesson
    skills.py       skill forge — self-authored, safety-gated tools
    council.py      adversarial debate — thesis/antithesis + blind judge

    v4 professional subsystems (all event-sourced on the same kernel):
    taint.py        static analysis — taint flows, complexity, import cycles
    kgraph.py       knowledge graph — entities + typed relations, impact sets
    cov.py          real line coverage — sys.settrace measurement
    fuzz.py         property-based fuzzing — generators + crash shrinking
    mutate.py       mutation testing — AST mutants vs the test suite

    tui.py          terminal UI: double-line prompt box, overlays, rendering
    __main__.py     entry point
"""

__version__ = "3.1.0"

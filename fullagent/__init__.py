"""FullAgent — an advanced terminal AI agent.

Package layout:
    config.py      providers, models, effort levels, paths
    systemprompt.py the ONE home of every system prompt (single source)
    mastermind.py  prompt coherence: sealed vault, gate, composer, lineage
    tools.py       tool registry: files, shell, search, real-time web
    client.py      OpenAI-compatible streaming chat client
    agent.py       agent loop (LLM <-> tools), event-sourced on the kernel
    kernel.py      Temporal Kernel: append-only, content-addressed event log
    memory.py      Hippocampus: episodic memory + dead-end ledger
    goal.py        goal contracts with machine-checkable done-criteria
    judge.py       deterministic verification (no LLM judging)
    swarm.py       parallel read-only scout sub-agents
    team.py        parallel worker team (up to 8, real tools, write lock)
    autopilot.py   self-routing brain: auto-enables team / goal / web
    tui.py         terminal UI: double-line prompt box, overlays, rendering
    __main__.py    entry point
"""

__version__ = "1.3.0"

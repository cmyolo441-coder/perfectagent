"""FullAgent — an advanced terminal AI agent.

Package layout:
    config.py   providers, models, effort levels, paths
    tools.py    tool registry: files, shell, search, web
    client.py   OpenAI-compatible streaming chat client
    agent.py    agent loop (LLM <-> tools)
    tui.py      terminal UI: double-line prompt box, overlays, rendering
    __main__.py entry point
"""

__version__ = "1.0.0"

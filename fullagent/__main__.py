"""Entry point: python main.py  (or python -m fullagent)"""

from __future__ import annotations

import sys


def _force_utf8() -> None:
    """Make sure output is UTF-8 even if the locale says otherwise."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _force_utf8()

    from . import config
    from .agent import Agent
    from .config import Config
    from .tui import UI

    config.ensure_dirs()
    cfg = Config.load()
    agent = Agent(cfg)
    ui = UI(cfg, agent)

    ui.print_banner()
    ui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

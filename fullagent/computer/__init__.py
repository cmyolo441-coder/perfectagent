"""FullAgent computer mode: eight bounded, concurrent API-backed agents.

Use /on in the terminal UI, or `python -m fullagent computer --help`.
The legacy /crew executor remains serial; this package owns parallel missions.
"""
from .engine import Computer
from .state import Settings

__all__ = ["Computer", "Settings"]

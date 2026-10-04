"""fgc-vision: detect, track and identify robots in FIRST Global Challenge field livestreams."""

from __future__ import annotations

__version__ = "0.1.0"

from .cli import main
from .sources import SourceError

__all__ = [
    "SourceError",
    "__version__",
    "main",
]

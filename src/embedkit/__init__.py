"""mnemos-embedkit — open embedding devkit for heterogeneous silicon.

Quick start:

    import embedkit
    eng = embedkit.Engine.auto(model="/path/to/model")
    vec = eng.embed("Hello world")

See docs/DESIGN.md for architecture.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .engine import Engine

try:
    __version__ = version("mnemos-embedkit")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["Engine", "__version__"]

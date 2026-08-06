"""C++ -> HIR frontend (split into cohesive submodules; public API: parse_cpp)."""

from .core import parse_cpp

__all__ = ["parse_cpp"]

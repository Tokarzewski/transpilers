"""Three-tier IR: HIR (source-faithful), MIR (normalized typed), LIR (target-shaped)."""

from .provenance import HirProvenance, ProvenanceMap

__all__ = [
    "HirProvenance",
    "ProvenanceMap",
]

"""HIR→LIR provenance — the fidelity backbone.

Carries a provenance link on every LIR node back to its originating HIR node
(id + source span). Persisted so downstream passes/verifiers can query which
source construct produced which target construct. Enables the structural-fidelity
verifier (#45) and targeted repair.

Usage::

    pm = ProvenanceMap()
    pm.record(hir_node, mir_node)
    pm.record(mir_node, lir_node)
    pm.lookup(lir_node)  # -> HirProvenance
    pm.to_json()         # -> dict (JSON-serializable)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Provenance data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HirProvenance:
    """Immutable record tying an IR node back to its source HIR construct.

    Attributes:
        hir_id:    Monotonic integer id assigned at HIR-node creation time.
        hir_type:  The Python class name of the HIR node (e.g. ``HirFunction``).
        source_span: Human-readable source location, or ``None``.
        hir_repr:  Short ``repr()`` of the HIR node for debugging (truncated).
    """

    hir_id: int
    hir_type: str
    source_span: str | None = None
    hir_repr: str = ""


# ---------------------------------------------------------------------------
# Provenance map
# ---------------------------------------------------------------------------


class ProvenanceMap:
    """Bidirectional id-based map between IR tiers.

    Internally stores ``{id(node): HirProvenance}`` so it works with any IR
    node type (HirNode / MirNode / LirNode) without requiring those types to
    carry any provenance fields.  References to the node objects keep them
    alive, so Python's ``id()`` stays valid for the lifetime of the map.
    """

    def __init__(self) -> None:
        # id(node) -> HirProvenance
        self._map: dict[int, HirProvenance] = {}

    # -- recording -----------------------------------------------------------

    def record(self, node: object, provenance: HirProvenance) -> None:
        """Associate *node* with its *provenance*."""
        self._map[id(node)] = provenance

    def record_node(
        self,
        hir_node: object,
        *,
        hir_id: int,
        hir_type: str,
        source_span: str | None = None,
        hir_repr: str = "",
    ) -> HirProvenance:
        """Build a ``HirProvenance`` and record it for *hir_node*.

        Returns the created provenance for chaining.
        """
        prov = HirProvenance(
            hir_id=hir_id,
            hir_type=hir_type,
            source_span=source_span,
            hir_repr=hir_repr,
        )
        self._map[id(hir_node)] = prov
        return prov

    def record_pair(
        self, source: object, target: object, provenance: HirProvenance | None = None
    ) -> None:
        """Record *target* with the same provenance as *source*.

        If *provenance* is given it is used directly; otherwise the provenance
        of *source* is looked up (raises ``KeyError`` if unknown).
        """
        if provenance is not None:
            prov = provenance
        else:
            prov = self._map[id(source)]
        self._map[id(target)] = prov

    # -- querying ------------------------------------------------------------

    def lookup(self, node: object) -> HirProvenance | None:
        """Return the ``HirProvenance`` for *node*, or ``None``."""
        return self._map.get(id(node))

    def has(self, node: object) -> bool:
        return id(node) in self._map

    def __contains__(self, node: object) -> bool:
        return id(node) in self._map

    def __len__(self) -> int:
        return len(self._map)

    def __repr__(self) -> str:
        return f"ProvenanceMap({len(self)} entries)"

    # -- iteration over entries -----------------------------------------------

    def items(self):
        """Yield ``(id(node), HirProvenance)`` pairs."""
        return self._map.items()

    def provenances(self):
        """Yield all ``HirProvenance`` values."""
        return self._map.values()

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation.

        Since ``id()`` values are process-local, the dict uses sequential
        entry indices as keys. Each entry carries::

            {
                "hir_id": 7,
                "hir_type": "HirFunction",
                "source_span": "prog.py:12:5-12:30",
                "hir_repr": "HirFunction(name='classify', ...)",
            }
        """
        return {
            str(idx): {
                "hir_id": p.hir_id,
                "hir_type": p.hir_type,
                "source_span": p.source_span,
                "hir_repr": p.hir_repr,
            }
            for idx, p in enumerate(self._map.values())
        }

    def to_json(self, **kw: Any) -> str:
        """Serialize the map to a JSON string.

        Passes *kw* (e.g. ``indent=2``) to ``json.dumps``.
        """
        return json.dumps(self.to_dict(), **kw, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceMap:
        """Reconstruct a ``ProvenanceMap`` from the dict format of ``to_dict()``.

        .. note::
           The reconstructed map has no in-memory node associations (the
           original ``id()`` values are gone).  It is useful for inspection
           and for the structural-fidelity verifier's provenance-enhanced
           matching.
        """
        pm = cls()
        for entry in data.values():
            prov = HirProvenance(
                hir_id=entry["hir_id"],
                hir_type=entry["hir_type"],
                source_span=entry.get("source_span"),
                hir_repr=entry.get("hir_repr", ""),
            )
            # Store by hir_id for look-up after deserialization.
            pm._map[entry["hir_id"]] = prov
        return pm


__all__ = [
    "HirProvenance",
    "ProvenanceMap",
]

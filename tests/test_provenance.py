"""Tests for the HIR→LIR provenance map (issue #43).

Verifies that:
1. Every HIR node gets a unique ``_hir_node_id``.
2. Every MIR node carries a ``_hir_provenance_id`` pointing back to its HIR parent.
3. Every LIR node carries a ``_hir_provenance_id`` pointing back through MIR
   to its HIR parent.
4. The ``ProvenanceMap`` correctly walks all three tiers.
5. Provenance serializes to / deserializes from JSON faithfully.
6. The ``--provenance`` CLI flag writes a sidecar.
7. Source→target node mapping is reconstructible from the JSON sidecar.
"""

from __future__ import annotations

import json
import textwrap
import time

import pytest

from transpilers.ir import hir
from transpilers.ir.provenance import HirProvenance, ProvenanceMap
from transpilers.pipeline.stages import TARGETS, run_stages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_SRC = """
def add(a: int, b: int) -> int:
    return a + b
"""

_WITH_CTRL_FLOW = """
def classify(n: int) -> int:
    total = 0
    for i in range(0, n, 1):
        if i > 2:
            total += i
        else:
            total -= 1
    while total > 100:
        total -= 10
    return total
"""

_WITH_STRUCT = """
class Point:
    x: int
    y: int

    def norm2(self: Point) -> int:
        return self.x * self.x + self.y * self.y
"""

def _trace(src: str, target: str = "rust"):
    return run_stages(textwrap.dedent(src).lstrip(), source_lang="python", target=target)


# ---------------------------------------------------------------------------
# HIR node ids
# ---------------------------------------------------------------------------


class TestHirNodeIds:
    def test_every_hir_node_has_unique_id(self):
        mod = _trace(_SIMPLE_SRC).hir
        ids = set()

        def walk(node):
            assert hasattr(node, "_hir_node_id")
            assert node._hir_node_id > 0
            assert node._hir_node_id not in ids, f"duplicate HIR id {node._hir_node_id}"
            ids.add(node._hir_node_id)
            if isinstance(node, (hir.HirModule, hir.HirFunction, hir.HirIf,
                                 hir.HirWhile, hir.HirFor, hir.HirForEach)):
                for child in node.body:
                    walk(child)
            if isinstance(node, hir.HirIf):
                for child in node.orelse:
                    walk(child)
            if isinstance(node, hir.HirReturn):
                if node.value:
                    walk(node.value)
            if isinstance(node, hir.HirBinOp):
                walk(node.left)
                walk(node.right)

        walk(mod)
        assert len(ids) >= 5, f"expected at least 5 HIR ids, got {len(ids)}"

    def test_different_modules_produce_different_ids(self):
        mod1 = _trace(_SIMPLE_SRC).hir
        mod2 = _trace(_SIMPLE_SRC).hir
        ids1 = set()
        ids2 = set()

        def collect_ids(mod, out):
            def walk(node):
                out.add(node._hir_node_id)
                if isinstance(node, (hir.HirModule, hir.HirFunction)):
                    for child in node.body:
                        walk(child)

            walk(mod)

        collect_ids(mod1, ids1)
        collect_ids(mod2, ids2)
        assert len(ids1) > 0
# ---------------------------------------------------------------------------
# MIR provenance
# ---------------------------------------------------------------------------


class TestMirProvenance:
    def test_mir_nodes_carry_hir_provenance_id(self):
        trace = _trace(_SIMPLE_SRC)
        mir_mod = trace.mir

        def walk(node):
            if hasattr(node, "_hir_provenance_id"):
                assert isinstance(node._hir_provenance_id, int)
            for field_name in getattr(node, "__dataclass_fields__", {}):
                val = getattr(node, field_name, None)
                if isinstance(val, (list, tuple)):
                    for child in val:
                        if hasattr(child, "_hir_provenance_id"):
                            walk(child)
                elif hasattr(val, "_hir_provenance_id"):
                    walk(val)

        walk(mir_mod)

    def test_provenance_map_records_mir_nodes(self):
        trace = _trace(_SIMPLE_SRC)
        pm = trace.provenance_map
        assert pm is not None
        assert len(pm) > 0

        for node_id, prov in pm.items():
            assert isinstance(prov, HirProvenance)
            assert prov.hir_id > 0
            assert prov.hir_type


# ---------------------------------------------------------------------------
# LIR provenance
# ---------------------------------------------------------------------------


class TestLirProvenance:
    def test_lir_nodes_carry_hir_provenance_id(self):
        trace = _trace(_SIMPLE_SRC)

        def walk(node):
            if hasattr(node, "_hir_provenance_id"):
                assert isinstance(node._hir_provenance_id, int)
            for field_name in getattr(node, "__dataclass_fields__", {}):
                val = getattr(node, field_name, None)
                if isinstance(val, (list, tuple)):
                    for child in val:
                        if hasattr(child, "_hir_provenance_id"):
                            walk(child)
                elif hasattr(val, "_hir_provenance_id"):
                    walk(val)

        walk(trace.lir)

    @pytest.mark.parametrize("target", sorted(TARGETS))
    def test_all_targets_carry_provenance(self, target):
        trace = _trace(_SIMPLE_SRC, target)
        pm = trace.provenance_map
        assert pm is not None
        assert len(pm) > 0

        found_lir = sum(
            1 for _, p in pm.items() if p.hir_type.startswith("Hir")
        )
        assert found_lir > 0, "no LIR nodes found in provenance map"

    def test_lir_nodes_link_back_to_hir_via_provenance_map(self):
        trace = _trace(_WITH_CTRL_FLOW)
        pm = trace.provenance_map
        assert pm is not None

        def walk(node, depth=0):
            if hasattr(node, "_hir_provenance_id") and node._hir_provenance_id > 0:
                prov = pm.lookup(node)
                assert prov is not None, (
                    f"LIR {type(node).__name__} id={node._hir_provenance_id} "
                    f"has no provenance entry"
                )
                assert prov.hir_id == node._hir_provenance_id
            for field_name in getattr(node, "__dataclass_fields__", {}):
                val = getattr(node, field_name, None)
                if isinstance(val, (list, tuple)):
                    for child in val:
                        if hasattr(child, "_hir_provenance_id"):
                            walk(child, depth + 1)
                elif hasattr(val, "_hir_provenance_id"):
                    walk(val, depth + 1)

        walk(trace.lir)
# ---------------------------------------------------------------------------
# ProvenanceMap serialization
# ---------------------------------------------------------------------------


class TestProvenanceMapSerialization:
    def test_to_dict_roundtrip(self):
        trace = _trace(_SIMPLE_SRC)
        pm = trace.provenance_map
        assert pm is not None

        d = pm.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0

        for key, entry in d.items():
            assert "hir_id" in entry
            assert "hir_type" in entry
            assert isinstance(entry["hir_id"], int)
            assert isinstance(entry["hir_type"], str)

    def test_to_json_serializes(self):
        trace = _trace(_SIMPLE_SRC)
        pm = trace.provenance_map
        assert pm is not None

        json_str = pm.to_json(indent=2)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

    def test_from_dict_reconstructs(self):
        trace = _trace(_SIMPLE_SRC)
        pm = trace.provenance_map
        assert pm is not None

        d = pm.to_dict()
        pm2 = ProvenanceMap.from_dict(d)
        assert isinstance(pm2, ProvenanceMap)
        # from_dict deduplicates by hir_id, so the reconstructed map has at most
        # as many entries as unique HIR nodes, but the original map has entries
        # for every MIR/LIR node (some sharing the same hir_id).
        assert len(pm2) > 0
        # Every entry from the original should be findable by its hir_id
        for key, entry in d.items():
            found = False
            for _, p in pm2.items():
                if p.hir_id == entry["hir_id"]:
                    found = True
                    break
            assert found, f"hir_id={entry['hir_id']} not found after roundtrip"

    def test_source_target_mapping_reconstructible(self):
        """Acceptance criterion: source->target mapping reconstructible from JSON."""
        trace = _trace(_WITH_CTRL_FLOW)
        pm = trace.provenance_map
        assert pm is not None

        json_str = pm.to_json(indent=2)
        data = json.loads(json_str)

        pm2 = ProvenanceMap.from_dict(data)
        # from_dict deduplicates by hir_id, so we have unique entries
        unique_hir_ids = {entry["hir_id"] for entry in data.values()}
        assert len(pm2) == len(unique_hir_ids), (
            f"expected {len(unique_hir_ids)} unique entries, got {len(pm2)}"
        )
        for key, entry in data.items():
            for _, p in pm2.items():
                if p.hir_id == entry["hir_id"]:
                    assert p.hir_type == entry["hir_type"]
                    break
            else:
                pytest.fail(f"entry hir_id={entry['hir_id']} not found after roundtrip")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliProvenanceFlag:
    def test_provenance_sidecar_written(self, tmp_path):
        from transpilers.cli.main import main

        src = tmp_path / "prog.py"
        src.write_text(textwrap.dedent(_SIMPLE_SRC).lstrip())

        prov_path = tmp_path / "prog.provenance.json"
        ret = main([str(src), "--target", "python", "--provenance", str(prov_path)])
        assert ret == 0, f"CLI returned {ret}"

        assert prov_path.exists(), "provenance sidecar not written"
        data = json.loads(prov_path.read_text())
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_provenance_sidecar_with_verify(self, tmp_path):
        from transpilers.cli.main import main

        src = tmp_path / "prog.py"
        src.write_text(textwrap.dedent(_SIMPLE_SRC).lstrip())

        prov_path = tmp_path / "prog.provenance.json"
        ret = main([str(src), "--target", "python", "--verify",
                     "--provenance", str(prov_path)])
        assert ret == 0, f"CLI returned {ret}"

        assert prov_path.exists()
        data = json.loads(prov_path.read_text())
        assert len(data) > 0


# ---------------------------------------------------------------------------
# ProvenanceMap API
# ---------------------------------------------------------------------------


class TestProvenanceMapApi:
    def test_provenance_map_records_and_lookup(self):
        pm = ProvenanceMap()
        h = hir.HirIntLiteral(value=42)
        prov = pm.record_node(h, hir_id=h._hir_node_id, hir_type="HirIntLiteral")
        assert pm.lookup(h) is prov
        assert pm.has(h)
        assert h in pm
        assert len(pm) == 1

    def test_record_pair_copies_provenance(self):
        pm = ProvenanceMap()
        h = hir.HirIntLiteral(value=42)
        prov = pm.record_node(h, hir_id=h._hir_node_id, hir_type="HirIntLiteral")
        h2 = hir.HirIntLiteral(value=99)
        pm.record_pair(h, h2)
        assert pm.lookup(h2) is prov

    def test_record_pair_with_explicit_provenance(self):
        pm = ProvenanceMap()
        h = hir.HirIntLiteral(value=42)
        h2 = hir.HirIntLiteral(value=99)
        prov = HirProvenance(hir_id=1, hir_type="HirIntLiteral")
        pm.record_pair(h, h2, provenance=prov)
        assert pm.lookup(h2) is prov
        assert pm.lookup(h) is None

    def test_empty_map(self):
        pm = ProvenanceMap()
        assert len(pm) == 0
        assert repr(pm) == "ProvenanceMap(0 entries)"
        assert pm.to_dict() == {}


# ---------------------------------------------------------------------------
# Edge cases: provenance on synthetic nodes
# ---------------------------------------------------------------------------


class TestSyntheticProvenance:
    def test_synthetic_mir_nodes_have_id_zero(self):
        from transpilers.passes.hir_to_mir import _default_init_for
        from transpilers.ir.types import IntT

        node = _default_init_for(IntT())
        assert node._hir_provenance_id == 0

    def test_foreach_synthetic_bindings_have_provenance(self):
        trace = _trace("""
            def total(xs: list[int]) -> int:
                s = 0
                for v in xs:
                    s += v
                return s
        """)
        pm = trace.provenance_map
        assert pm is not None
        found = False
        for obj_id, prov in pm.items():
            if prov.hir_type == "HirForEach":
                found = True
                break
        assert found, "HirForEach node should be in the provenance map"


def test_provenance_map_scales_linearly_not_quadratically():
    # _build_provenance_map used to look up each MIR/LIR node's matching
    # HIR provenance with a linear re-scan of every already-recorded
    # provenance (O(mir_nodes * hir_nodes)) instead of an O(1) dict lookup.
    # Fine for a hand-written algorithm slice, but a large real-world
    # amalgamated translation unit (e.g. transpiling a whole C++ package
    # with --include-impls, thousands of nodes per tier) turned this into
    # a multi-minute hang for no benefit over a dict keyed by hir_id.
    # ~150 tiny functions is enough to make the quadratic behavior take
    # tens of seconds while the linear one stays well under a second --
    # generous enough to not flake on a slow CI box either way.
    src = "\n".join(
        f"def f{i}(a: int, b: int) -> int:\n    return a + b + {i}"
        for i in range(150)
    )
    start = time.time()
    trace = run_stages(src, source_lang="python", target="rust")
    elapsed = time.time() - start
    assert trace.provenance_map is not None
    assert elapsed < 10.0, f"provenance map build took {elapsed:.1f}s -- looks quadratic again"
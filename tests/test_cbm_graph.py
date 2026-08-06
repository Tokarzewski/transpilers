"""Tests for the cbm-backed multi-relational graph builder.

The live cbm CLI / EnergyPlus index isn't required here — we test the pure
pieces: JSON extraction from cbm CLI output, row-dict shaping, the
multi-relational assembly (typed edges + node kinds + field owners), and the
scc/fan-in annotation that reuses migration_plan's algorithms.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_SFT = REPO / "scripts" / "sft"
sys.path.insert(0, str(SCRIPTS_SFT))

import cbm_graph as cg  # noqa: E402


def test_extract_json_skips_log_lines():
    out = (
        "level=info msg=mem.init budget_mb=15358 total_ram_mb=30717\n"
        '{"columns":["src","dst"],"rows":[["a","b"]],"total":1}\n'
    )
    payload = cg.extract_json(out)
    assert payload["columns"] == ["src", "dst"]
    assert payload["rows"] == [["a", "b"]]


def test_extract_json_raises_when_absent():
    try:
        cg.extract_json("level=info nothing here\n")
    except ValueError:
        return
    raise AssertionError("expected ValueError on log-only output")


def test_rows_of_zips_columns():
    payload = {
        "columns": ["src", "dst", "owner"],
        "rows": [["f", "x", "EnergyPlusData"]],
    }
    assert cg.rows_of(payload) == [{"src": "f", "dst": "x", "owner": "EnergyPlusData"}]


def test_multigraph_has_typed_edges_and_node_kinds():
    g = cg.rows_to_multigraph(
        calls=[{"src": "pkg.A", "dst": "pkg.B"}],
        writes=[{"src": "pkg.A", "dst": "pkg.St.field1", "owner": "pkg.St"}],
        reads=[{"src": "pkg.B", "dst": "pkg.St.field1", "owner": "pkg.St"}],
    )
    assert g["directed"] is True and g["multigraph"] is True
    types = sorted(e["type"] for e in g["links"])
    assert types == ["calls", "reads_field", "writes_field"]
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    assert kinds["pkg.A"] == "callable" and kinds["pkg.St.field1"] == "field"
    field = next(n for n in g["nodes"] if n["id"] == "pkg.St.field1")
    assert field["owner"] == "pkg.St"
    assert field["name"] == "field1"  # short name derived from qualified_name


def test_multigraph_keeps_parallel_typed_edges_between_same_pair():
    # a writes AND reads the same field => two distinct typed edges (multigraph)
    g = cg.rows_to_multigraph(
        writes=[{"src": "a", "dst": "f", "owner": "S"}],
        reads=[{"src": "a", "dst": "f", "owner": "S"}],
    )
    pair = [(e["source"], e["target"], e["type"]) for e in g["links"]]
    assert ("a", "f", "writes_field") in pair
    assert ("a", "f", "reads_field") in pair


def test_annotate_calls_adds_scc_and_fanin_on_call_subgraph():
    # b <-> c is a call cycle; both write field f (field edges must NOT pollute
    # the call-graph scc/fan-in computation)
    g = cg.rows_to_multigraph(
        calls=[
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "b"},
            {"src": "a", "dst": "b"},
        ],
        writes=[
            {"src": "b", "dst": "f", "owner": "S"},
            {"src": "c", "dst": "f", "owner": "S"},
        ],
    )
    cycles, n_scc = cg.annotate_calls(g)
    assert sorted(cycles[0]) == ["b", "c"]  # cycle detected among callables
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["b"]["fan_in"] == 2  # a->b and c->b
    assert "scc" not in by_id["f"]  # field node not annotated
    assert by_id["b"]["scc"] == by_id["c"]["scc"]  # same component

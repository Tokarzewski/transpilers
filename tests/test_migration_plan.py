"""Tests for the migration planner's graph algorithms (issues #67, #68).

These cover the pure, importable graph functions without touching the
EnergyPlus tree:

* tarjan_scc partitions a graph into strongly-connected components
* cyclic_components flags real cycles (size>1 and self-loops) and ignores
  acyclic singletons
* fan_in / fanin_ranked surface max-unlock scaffolding

Edges mean "depends on": n -> d => n calls d, so d ports before n.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_SFT = REPO / "scripts" / "sft"
sys.path.insert(0, str(SCRIPTS_SFT))

import migration_plan as mp  # noqa: E402


def _normalize(sccs):
    """Comparable form: a set of frozensets of node names."""
    return {frozenset(c) for c in sccs}


def test_scc_acyclic_is_all_singletons():
    g = {"a": ["b", "c"], "b": ["c"], "c": []}
    sccs = mp.tarjan_scc(g)
    assert _normalize(sccs) == {frozenset("a"), frozenset("b"), frozenset("c")}
    assert mp.cyclic_components(g) == []


def test_scc_detects_two_node_cycle():
    # a <-> b mutual recursion; c depends on a, d is independent
    g = {"a": ["b"], "b": ["a"], "c": ["a"], "d": []}
    cycles = mp.cyclic_components(g)
    assert cycles == [["a", "b"]]


def test_scc_detects_self_loop_recursion():
    g = {"fact": ["fact"], "x": []}
    sccs = mp.tarjan_scc(g)
    # tarjan still returns singletons, but cyclic_components flags the self-loop
    assert _normalize(sccs) == {frozenset({"fact"}), frozenset({"x"})}
    assert mp.cyclic_components(g) == [["fact"]]


def test_scc_larger_cycle_and_ordering():
    # one 3-cycle {a,b,c} plus a 2-cycle {e,f}; sorted largest-first
    g = {"a": ["b"], "b": ["c"], "c": ["a"], "e": ["f"], "f": ["e"], "g": []}
    cycles = mp.cyclic_components(g)
    assert cycles == [["a", "b", "c"], ["e", "f"]]


def test_scc_reverse_topological_order():
    # condensation is a -> b -> c; tarjan yields components in reverse-topo
    g = {"a": ["b"], "b": ["c"], "c": []}
    order = [c[0] for c in mp.tarjan_scc(g)]
    assert order.index("c") < order.index("b") < order.index("a")


def test_scc_ignores_external_successors():
    # 'b' is not a key in the graph -> treated as external, no KeyError
    g = {"a": ["b", "c"], "c": []}
    sccs = mp.tarjan_scc(g)
    assert _normalize(sccs) == {frozenset("a"), frozenset("c")}


def test_fan_in_counts_dependents():
    g = {"a": ["c"], "b": ["c"], "c": [], "d": ["a"]}
    fi = mp.fan_in(g)
    assert fi == {"a": 1, "b": 0, "c": 2, "d": 0}


def test_fan_in_dedups_parallel_edges_and_ignores_external():
    g = {"a": ["c", "c"], "c": [], "b": ["external"]}
    fi = mp.fan_in(g)
    assert fi["c"] == 1  # duplicate a->c edge counted once
    assert "external" not in fi  # external node not scored


def test_fanin_ranked_orders_by_dependents_then_name():
    g = {"hi": [], "lo": [], "a": ["hi"], "b": ["hi"], "c": ["lo"]}
    ranked, fi = mp.fanin_ranked(g)
    assert ranked[0] == "hi"  # fan-in 2 beats everything
    assert ranked[1] == "lo"  # fan-in 1
    assert fi["hi"] == 2 and fi["lo"] == 1


def test_graph_node_link_is_directed_with_typed_edges():
    g = {"a": ["b"], "b": ["c"], "c": [], "rec": ["rec"]}
    d = mp.graph_node_link(g, attrs={"a": {"leaf": False}})
    assert d["directed"] is True
    assert {n["id"] for n in d["nodes"]} == {"a", "b", "c", "rec"}
    edges = {(e["source"], e["target"]) for e in d["links"]}
    assert edges == {("a", "b"), ("b", "c"), ("rec", "rec")}  # self-loop kept
    assert all(e["type"] == "calls" for e in d["links"])
    # per-node attrs are merged onto the node record
    a = next(n for n in d["nodes"] if n["id"] == "a")
    assert a["leaf"] is False


def test_graph_node_link_drops_external_targets():
    g = {"a": ["external"], "a2": ["a"]}
    d = mp.graph_node_link(g)
    edges = {(e["source"], e["target"]) for e in d["links"]}
    assert edges == {("a2", "a")}  # a -> external dropped (not a node)


def test_graph_node_link_roundtrips_via_networkx_if_available():
    nx = __import__("importlib").util.find_spec("networkx")
    if nx is None:
        return  # networkx optional; schema is still the node-link standard
    import networkx as nxmod  # noqa

    g = {"a": ["b"], "b": []}
    d = mp.graph_node_link(g)
    G = nxmod.node_link_graph(d, edges="links")
    assert G.is_directed() and G.number_of_edges() == 1

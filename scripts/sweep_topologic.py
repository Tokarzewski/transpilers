"""One-off sweep: run strict (C++->Mojo) and lift (C++->Python) engines over a
representative Topologic subset and bucket outcomes. Issue #79.

Not part of the routine test suite — a diagnostic harness for the Topologic
corpus. Run: .venv/Scripts/python.exe scripts/sweep_topologic.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

TOPO = os.environ.get("TOPOLOGIC_ROOT", r"C:/Users/amd/Documents/GitHub/Topologic")
INC = os.path.join(TOPO, "TopologicCore", "include")

# Representative subset: leaf utilities + simple value classes + one core
# topology class per level of the cell-complex hierarchy, skipping the
# god-class Topology.cpp/Graph.cpp (kept separate as a note).
SUBSET = [
    "Bitwise.cpp",          # leaf utility, std::list<int>, static methods
    "DoubleAttribute.cpp",  # simple value wrapper
    "IntAttribute.cpp",
    "StringAttribute.cpp",
    "ListAttribute.cpp",
    "Dictionary.cpp",       # std::map-backed
    "Context.cpp",          # small manager
    "Geometry.cpp",         # abstract base, shared_ptr typedef
    "Line.cpp",             # OCCT Handle(Geom_Line), inheritance
    "Surface.cpp",          # OCCT Handle(Geom_Surface)
    "Vertex.cpp",           # core topology leaf
    "Edge.cpp",
    "Wire.cpp",
    "Face.cpp",
    "Shell.cpp",
    "Cell.cpp",
    "Cluster.cpp",
]

SRC_DIR = os.path.join(TOPO, "TopologicCore", "src")

# allow "python scripts/sweep_topologic.py /path/to/Topologic" to override


def classify_strict_err(msg: str) -> str:
    m = msg.lower()
    if "operator new" in m or "size_t" in m or "operator delete" in m:
        return "preamble-size_t-mismatch"
    if re.search(r"unknown type name .(topods|geom_|topabs|brep|topexp|topexp|gprop|topo|standard_|handle|toploc)", m):
        return "occt-external-type"
    if "unknown type name" in m:
        return "missing-type"
    if "use of undeclared identifier" in m:
        return "unresolved-symbol"
    if "template" in m:
        return "template"
    if "class" in m or "namespace" in m or "virtual" in m:
        return "class/namespace/virtual"
    if "no matching" in m or "overload" in m:
        return "overload-resolution"
    return "other"


def strict_sweep():
    from transpilers.frontends.cpp.parser.includes import resolve_local_includes
    from transpilers.cli.main import transpile

    rows = []
    for name in SUBSET:
        path = os.path.join(SRC_DIR, name)
        try:
            src = resolve_local_includes(path, include_dirs=[INC])
        except Exception as e:
            rows.append({"file": name, "ok": False, "stage": "include", "err": f"{type(e).__name__}: {e}"})
            continue
        try:
            out = transpile(src, source_lang="cpp", target="mojo")
            rows.append({"file": name, "ok": True, "stage": "emit", "bytes": len(out)})
        except Exception as e:
            msg = str(e)
            rows.append(
                {
                    "file": name,
                    "ok": False,
                    "stage": "strict",
                    "err": msg.splitlines()[0][:200] if msg else type(e).__name__,
                    "bucket": classify_strict_err(msg),
                }
            )
    return rows


def lift_sweep():
    from transpilers.lift import lift_source
    from transpilers.frontends.cpp.parser.includes import resolve_local_includes

    rows = []
    for name in SUBSET:
        path = os.path.join(SRC_DIR, name)
        src = resolve_local_includes(path, include_dirs=[INC])
        out, stats = lift_source(src, name=name, inc=[INC])
        todo = stats.get("todo", 0)
        nodes = stats.get("nodes", 0)
        # Categorize the TODO[lift] stubs from emitted comments
        stubs = re.findall(r"# TODO\[lift\]:?\s*([^:\n]*)", out)
        cat = Counter()
        for s in stubs:
            s = s.strip()
            cat[classify_stub(s)] += 1
        rows.append(
            {
                "file": name,
                "nodes": nodes,
                "todo": todo,
                "coverage_pct": round(100 * (1 - todo / max(nodes, 1)), 1),
                "stub_categories": dict(cat.most_common()),
                "lines": len(out.splitlines()),
            }
        )
    return rows


def classify_stub(s: str) -> str:
    s = s.lower()
    if re.search(r"topods|geom_|brep|occt|topo|handle|gp_|topexp|gprop", s):
        return "occt"
    if "std::" in s or "vector" in s or "list" in s or "map" in s or "shared_ptr" in s or "template" in s:
        return "std/template"
    if "class" in s or "method" in s or "ctor" in s or "virtual" in s or "this" in s:
        return "class/method"
    if "switch" in s or "goto" in s or "label" in s:
        return "control-flow"
    if "operator" in s:
        return "operator"
    return "other"


def main() -> int:
    global TOPO, INC, SRC_DIR
    if len(sys.argv) > 1:
        TOPO = sys.argv[1]
        INC = os.path.join(TOPO, "TopologicCore", "include")
        SRC_DIR = os.path.join(TOPO, "TopologicCore", "src")
    print("=" * 70)
    print("STRICT ENGINE: C++ -> Mojo  (subset of %d files)" % len(SUBSET))
    print("=" * 70)
    strict = strict_sweep()
    ok = [r for r in strict if r["ok"]]
    fail = [r for r in strict if not r["ok"]]
    buckets = Counter(r.get("bucket", "include") for r in fail)
    for r in strict:
        status = "OK " if r["ok"] else "FAIL"
        detail = f" ({r.get('bucket')})" if not r["ok"] else f" ({r['bytes']}B)"
        print(f"  [{status}] {r['file']}{detail}")
    print(f"\n  STRICT: {len(ok)}/{len(SUBSET)} emitted, {len(fail)} failed")
    print("  Failure buckets:")
    for b, c in buckets.most_common():
        print(f"    {b}: {c}")

    print()
    print("=" * 70)
    print("LIFT ENGINE: C++ -> Python  (never-refuse)")
    print("=" * 70)
    lift = lift_sweep()
    tot_nodes = sum(r["nodes"] for r in lift)
    tot_todo = sum(r["todo"] for r in lift)
    for r in lift:
        print(
            f"  {r['file']}: {r['todo']}/{r['nodes']} TODO  "
            f"({r['coverage_pct']}% mech)  stubs={r['stub_categories']}"
        )
    print(f"\n  LIFT: {tot_todo}/{tot_nodes} nodes stubbed "
          f"({100*(1-tot_todo/max(tot_nodes,1)):.1f}% mechanical)")

    # Dump JSON for the report
    report = {"strict": strict, "lift": lift, "strict_buckets": dict(buckets.most_common())}
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "topologic_sweep.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON written to {os.path.abspath(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

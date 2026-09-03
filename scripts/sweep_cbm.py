"""Diagnostic sweep: exercise the --cbm path (codebase-memory preamble) over
every TopologicCore/src/*.cpp and bucket outcomes, focusing on whether the
*parser preamble* errors (undeclared class, namespace-not-enclosing,
undeclared std) are eliminated by the recovered shim. Issue #79 / #80."""
from __future__ import annotations

import os
import re
import sys
import tempfile
from collections import Counter

TOPO = os.environ.get("TOPOLOGIC_ROOT", r"C:/Users/amd/Documents/GitHub/Topologic")
SRC_DIR = os.path.join(TOPO, "TopologicCore", "src")

sys.path.insert(0, r"C:/Users/amd/Documents/GitHub/transpilers/src")

from transpilers.frontends.cpp.parser import cbmpreamble as cp  # noqa: E402
from transpilers.frontends.cpp.parser.core import parse_cpp  # noqa: E402


def bucket_err(msg: str) -> str:
    m = msg
    if "does not enclose namespace" in m or "cannot define or redeclare" in m:
        return "namespace-not-enclosing"
    if "use of undeclared identifier 'std'" in m:
        return "std-undeclared"
    if "use of undeclared identifier" in m:
        return "undeclared-identifier"
    if "does not match any declaration" in m:
        return "overload-mismatch"
    if "unknown type name" in m:
        return "unknown-type"
    if "template" in m:
        return "template"
    return "other"


def main() -> int:
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".cpp"))
    rows = []
    for name in files:
        rel = f"TopologicCore/src/{name}"
        out = os.path.join(tempfile.gettempdir(), f"cbm_sweep_{name}.inc")
        written = cp.write_preamble_for_file(TOPO, rel, out)
        had_preamble = written is not None
        # Reproduce the --cbm CLI path: set env, parse the real source.
        if had_preamble:
            os.environ["TRANSPILERS_CPP_PREAMBLE_FILE"] = out
        else:
            os.environ.pop("TRANSPILERS_CPP_PREAMBLE_FILE", None)
        src_path = os.path.join(SRC_DIR, name)
        src = open(src_path, encoding="utf-8", errors="replace").read()
        try:
            parse_cpp(src)
            rows.append({"file": name, "ok": True, "preamble": had_preamble})
        except Exception as e:
            msg = str(e)
            first = msg.splitlines()[0] if msg else type(e).__name__
            rows.append(
                {
                    "file": name,
                    "ok": False,
                    "preamble": had_preamble,
                    "bucket": bucket_err(msg),
                    "err": first[:160],
                }
            )

    ok = [r for r in rows if r["ok"]]
    fail = [r for r in rows if not r["ok"]]
    buckets = Counter(r["bucket"] for r in fail)
    print(f"=== --cbm sweep: {len(files)} files ===")
    for r in rows:
        mark = "OK " if r["ok"] else "FAIL"
        pre = "P" if r.get("preamble") else "-"
        detail = "" if r["ok"] else f"  [{r['bucket']}] {r.get('err','')}"
        print(f"  [{mark}|{pre}] {r['file']}{detail}")
    print(f"\nRESULT: {len(ok)}/{len(files)} parsed clean")
    print("Failure buckets:")
    for b, c in buckets.most_common():
        print(f"    {b}: {c}")
    # The two #80-style overload collapses we specifically expect from the index:
    return 0


if __name__ == "__main__":
    sys.exit(main())

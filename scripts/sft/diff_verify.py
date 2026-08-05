#!/usr/bin/env python3
"""General differential verifier for C++<->Mojo across ANY construct.

Each item = {name, category, cpp_unit, mojo_unit, cpp_driver, mojo_driver}.
The C++ side (unit + driver) is ground truth; the Mojo side must produce the same
stdout. Works for functions, classes/methods, arrays, strings — anything the
driver can exercise. Verified items feed BOTH the training set and the diverse
eval benchmark.

Usage: diff_verify.py <items.jsonl ...>  -> prints per-category yield, writes
       data/sft/diverse/verified.jsonl
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("DIFF_OUT", str(REPO / "data/sft/diverse/verified.jsonl")))
EPMOJO = "/home/bart/github/EnergyPlusMojo/.pixi/envs/default"
MOJO_BIN = f"{EPMOJO}/bin/mojo"
# system linker in front (handles Python interop too) + MODULAR_HOME
MOJO_ENV = dict(os.environ, MODULAR_HOME=f"{EPMOJO}/share/max",
                PATH="/usr/bin:/bin:" + f"{EPMOJO}/bin")

CPP_PRE = ("#include <cstdio>\n#include <cstdlib>\n#include <cmath>\n#include <vector>\n"
           "#include <string>\n#include <array>\n#include <algorithm>\n#include <iostream>\n"
           "#include <cstdint>\n#include <map>\n#include <stdexcept>\nusing namespace std;\n")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


def verify(item) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        # --- C++ (ground truth) ---
        (t / "a.cpp").write_text(CPP_PRE + item["cpp_unit"] + "\n" + item["cpp_driver"] + "\n")
        r = run(["g++", "-O2", "-std=c++17", "-o", str(t / "a"), str(t / "a.cpp")])
        if r.returncode != 0:
            return False, "cpp_compile"
        r = run([str(t / "a")])
        if r.returncode != 0:
            return False, "cpp_run"
        cpp_out = r.stdout.strip().splitlines()
        if not cpp_out:
            return False, "cpp_no_output"
        # --- Mojo ---
        (t / "b.mojo").write_text(item["mojo_unit"] + "\n\n" + item["mojo_driver"] + "\n")
        r = run([MOJO_BIN, "build", "-Xlinker", "-ldl", str(t / "b.mojo"), "-o", str(t / "b")], env=MOJO_ENV)
        if r.returncode != 0:
            errs = [l for l in r.stderr.splitlines() if ": error:" in l]
            return False, "mojo_compile: " + (errs[-1][:50] if errs else "?")
        r = run([str(t / "b")], env=MOJO_ENV)
        if r.returncode != 0:
            return False, "mojo_run"
        mojo_out = r.stdout.strip().splitlines()
    # --- compare (numeric tolerance, else exact string) ---
    if len(cpp_out) != len(mojo_out):
        return False, f"line_count {len(cpp_out)}!={len(mojo_out)}"
    for a, b in zip(cpp_out, mojo_out):
        a, b = a.strip(), b.strip()
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
            if abs(fa - fb) / max(abs(fa), abs(fb), 1e-9) <= 1e-6:
                continue
        except ValueError:
            pass
        return False, f"mismatch '{a[:20]}'!='{b[:20]}'"
    return True, "ok"


def main():
    items = []
    for p in sys.argv[1:]:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                try:
                    items.append(json.loads(l))
                except Exception:
                    pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    verified = []
    bycat = Counter(); okcat = Counter(); fails = Counter()
    for it in items:
        cat = it.get("category", "?"); bycat[cat] += 1
        ok, why = verify(it)
        if ok:
            okcat[cat] += 1; verified.append(it)
            print(f"  OK   [{cat}] {it.get('name','?')}")
        else:
            fails[why.split(':')[0]] += 1
            print(f"  FAIL [{cat}] {it.get('name','?')}: {why}")
    with OUT.open("w") as f:
        for v in verified:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"\n=== verified {len(verified)}/{len(items)} ===")
    for c in sorted(bycat):
        print(f"  {c:14s} {okcat[c]}/{bycat[c]}")
    print("fail reasons:", dict(fails))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

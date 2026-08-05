#!/usr/bin/env python3
"""Differential verifier for Python->Mojo pairs.

Mirror of diff_verify.py, but the GROUND TRUTH is the already-verified Mojo side
(every diverse pair's mojo_unit is verified == C++). Here the Python side is the
CANDIDATE: run the python_unit+python_driver, compile+run the mojo_unit+mojo_driver,
and require identical stdout. A pair is kept only if Python's behaviour matches the
verified Mojo. This is how we mint behaviourally-verified (python_unit, mojo_unit)
training pairs for the Python->Mojo direction.

Item = {name, category, python_unit, python_driver, mojo_unit, mojo_driver, ...}
(extra fields like cpp_unit are preserved so the same record can train both
directions.)

Python-aware normalization (advisor): Python prints tuples/lists/bools/floats
differently from Mojo. We compare line-by-line (drivers are written to mirror each
other) with: numeric tolerance 1e-6, bool True/False<->1/0, and structural
stripping of () [] , so a Python tuple line `(8, 14, 6)` matches a Mojo line that
prints the same components.

Usage: py_verify.py <items.jsonl ...>   (DIFF_OUT env -> output path;
       default data/sft/diverse/py_verified.jsonl)
"""
from __future__ import annotations
import json, os, re, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("DIFF_OUT", str(REPO / "data/sft/diverse/py_verified.jsonl")))
EPMOJO = "/home/bart/github/EnergyPlusMojo/.pixi/envs/default"
MOJO_BIN = f"{EPMOJO}/bin/mojo"
MOJO_ENV = dict(os.environ, MODULAR_HOME=f"{EPMOJO}/share/max",
                PATH="/usr/bin:/bin:" + f"{EPMOJO}/bin")
PY_BIN = sys.executable


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


def _norm_tokens(line: str):
    """Split a printed line into comparable tokens, stripping tuple/list syntax."""
    s = line.strip()
    # normalize bracket/paren/comma structure -> whitespace-separated tokens
    s = re.sub(r"[()\[\],]", " ", s)
    return s.split()


def _tok_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    # bool <-> 1/0
    bmap = {"true": "1", "false": "0"}
    aa, bb = bmap.get(a.lower(), a), bmap.get(b.lower(), b)
    if aa == bb:
        return True
    try:
        fa, fb = float(aa), float(bb)
        return abs(fa - fb) / max(abs(fa), abs(fb), 1e-9) <= 1e-6
    except ValueError:
        return False


def _lines_eq(a: str, b: str) -> bool:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if len(ta) != len(tb):
        return False
    return all(_tok_eq(x, y) for x, y in zip(ta, tb))


def verify(item) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        # --- Mojo (ground truth: already verified == C++) ---
        (t / "b.mojo").write_text(item["mojo_unit"] + "\n\n" + item["mojo_driver"] + "\n")
        r = run([MOJO_BIN, "build", "-Xlinker", "-ldl", str(t / "b.mojo"), "-o", str(t / "b")], env=MOJO_ENV)
        if r.returncode != 0:
            errs = [l for l in r.stderr.splitlines() if ": error:" in l]
            return False, "mojo_compile: " + (errs[-1][:50] if errs else "?")
        r = run([str(t / "b")], env=MOJO_ENV)
        if r.returncode != 0:
            return False, "mojo_run"
        mojo_out = r.stdout.strip().splitlines()
        if not mojo_out:
            return False, "mojo_no_output"
        # --- Python (candidate) ---
        (t / "a.py").write_text(item["python_unit"] + "\n\n" + item["python_driver"] + "\n")
        r = run([PY_BIN, str(t / "a.py")])
        if r.returncode != 0:
            return False, "py_run: " + (r.stderr.strip().splitlines()[-1][:50] if r.stderr.strip() else "?")
        py_out = r.stdout.strip().splitlines()
    # --- compare ---
    if len(py_out) != len(mojo_out):
        return False, f"line_count py{len(py_out)}!=mojo{len(mojo_out)}"
    for a, b in zip(py_out, mojo_out):
        if not _lines_eq(a, b):
            return False, f"mismatch '{a[:24]}'!='{b[:24]}'"
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
        if not it.get("python_unit") or not it.get("python_driver"):
            fails["no_python"] += 1
            continue
        cat = it.get("category", "?"); bycat[cat] += 1
        ok, why = verify(it)
        if ok:
            okcat[cat] += 1; verified.append(it)
            print(f"  OK   [{cat}] {it.get('name','?')}", flush=True)
        else:
            fails[why.split(':')[0]] += 1
            print(f"  FAIL [{cat}] {it.get('name','?')}: {why}", flush=True)
    with OUT.open("w") as f:
        for v in verified:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"\n=== python->mojo verified {len(verified)}/{sum(bycat.values())} ===")
    for c in sorted(bycat):
        print(f"  {c:14s} {okcat[c]}/{bycat[c]}")
    print("fail reasons:", dict(fails))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

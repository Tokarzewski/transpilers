#!/usr/bin/env python3
"""SFT source 2: a Mojo-syntax corpus from the energyplus-mojo .mojo kernels.

Goal: teach the base model Mojo *surface form* (typed defs, var, comptime, for/
if, SIMD, math) — it currently has ~0% Mojo competency. Each kernel's leading
module docstring (a natural-language spec) is paired with the kernel code:

    instruction: "Write a Mojo implementation for the following specification."
    input:       <docstring>
    output:      <mojo code, docstring stripped>

Quality guards:
  * Skip files with a trivial docstring or trivial body.
  * The ~459 `*_batch.mojo` files are near-duplicate FFI templates (int64-address
    kernels). Over-representing them would teach the *wrong* idiom (manual FFI),
    so they are DOWNSAMPLED (keep 1 in BATCH_KEEP); all non-batch kernels kept.

Read-only on energyplus-mojo. Emits data/sft/mojo_corpus.jsonl.
"""
from __future__ import annotations

import json, os, re
from pathlib import Path

EPMOJO = Path(os.environ.get("EP_MOJO_REPO", "/home/bart/github/EnergyPlusMojo"))
OUT = Path(__file__).resolve().parents[2] / "data/sft/mojo_corpus.jsonl"
BATCH_KEEP = 5          # keep 1 in N batch templates

INSTR = "Write a Mojo implementation for the following specification."
_CODE_START = re.compile(r'^(comptime|def|fn|@|from |import |var |struct |trait )')
_LICENSE = re.compile(r'SPDX|License|Copyright|LBNL|BSD', re.I)


def split_doc(text: str):
    """Return (spec, code). spec = leading doc: the module docstring if present,
    else the leading `#` comment block (minus license lines). code = the file
    from the first real code line onward."""
    lines = text.splitlines()
    # find first real code line
    ci = None
    in_doc = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith('"""') or s.startswith("'''"):
            # a docstring line — if it opens+closes on one line, not code;
            # otherwise toggle. Either way, not the code start.
            q = s[:3]
            in_doc = not (s.count(q) >= 2)
            continue
        if in_doc:
            if q in ln:
                in_doc = False
            continue
        if not s or s.startswith("#"):
            continue
        if _CODE_START.match(s):
            ci = i
            break
    if ci is None:
        return None, text
    header = "\n".join(lines[:ci])
    code = "\n".join(lines[ci:]).strip()
    # spec = the leading `#` comment block (minus license lines) + the module
    # docstring, concatenated. Batch kernels put their rich description in the
    # comment block and only a one-line docstring; idiomatic kernels do the
    # reverse — combining captures both.
    comments = "\n".join(l.lstrip("# ").rstrip() for l in header.splitlines()
                         if l.strip().startswith("#") and not _LICENSE.search(l)).strip()
    m = re.search(r'(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', header, re.S)
    doc = (m.group(1) or m.group(2)).strip() if m else ""
    spec = (comments + "\n\n" + doc).strip()
    return (spec or None), code


def main():
    kernels = sorted((EPMOJO / "src/kernels").glob("*.mojo"))
    runtime = sorted((EPMOJO / "src/mojo").rglob("*.mojo"))
    rows = []
    batch_i = 0
    for f in kernels + runtime:
        is_batch = f.name.endswith("_batch.mojo")
        if is_batch:
            batch_i += 1
            if batch_i % BATCH_KEEP != 0:
                continue
        text = f.read_text(errors="ignore")
        doc, code = split_doc(text)
        if not doc or len(doc) < 120:        # need a real spec
            continue
        if len(code.strip()) < 80 or ("def " not in code and "fn " not in code):
            continue
        rows.append({
            "instruction": INSTR,
            "input": doc,
            "output": code.strip(),
            "source": "mojo_corpus",
            "meta": {"file": f.name, "batch": is_batch},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    nb = sum(1 for r in rows if r["meta"]["batch"])
    print(f"mojo_corpus: {len(rows)} examples ({len(rows)-nb} idiomatic/non-batch, {nb} batch-sampled)")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()

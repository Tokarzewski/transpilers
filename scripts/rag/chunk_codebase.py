#!/usr/bin/env python3
"""Stage 1 of the codebase RAG index: chunk the C++/Mojo codebase + docs into
retrievable, function-level units. The migration agent's #1 unmet need is
DEPENDENCY CLOSURE — given a function, find its definition, its callees, and any
already-ported Mojo. Function-level chunks (not arbitrary windows) make that
retrievable.

Output: data/rag/chunks.jsonl  — one chunk per line:
  {id, lang, kind, name, file, text}
This is the corpus that gets embedded + indexed (turbovec) in stage 2.
"""
import json, os, re
from pathlib import Path

EP_CPP = Path(os.environ.get("EP_SRC", "/home/bart/Github/EnergyPlus/src/EnergyPlus"))
EP_MOJO = Path(os.environ.get("EP_MOJO_REPO", "/home/bart/github/EnergyPlusMojo"))
TRANSP = Path(__file__).resolve().parents[2]  # scripts/rag/<this file> -> repo root
OUT = TRANSP / "data/rag"; OUT.mkdir(parents=True, exist_ok=True)

# C++ function/method: RetType [Ns::]Name(args) {   (brace-matched body)
CPP_FN = re.compile(r"\n([A-Za-z_][\w:<>,\s\*&]*?\s+(?:[A-Za-z_]\w*::)?([A-Za-z_]\w*)\s*\([^;{)]*\)\s*(?:const\s*)?\{)")
# Mojo def/struct/fn
MOJO_DEF = re.compile(r"\n((?:def|fn|struct|trait)\s+([A-Za-z_]\w*))")

def brace_body(text, open_idx):
    depth = 0
    for j in range(open_idx, min(len(text), open_idx + 20000)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0: return j + 1
    return None

def indent_block(lines, i):
    """Mojo: from def-line i, take until dedent to <= its indent."""
    base = len(lines[i]) - len(lines[i].lstrip())
    end = i + 1
    while end < len(lines):
        s = lines[end]
        if s.strip() and (len(s) - len(s.lstrip())) <= base:
            break
        end += 1
    return "\n".join(lines[i:end])

def chunk_cpp(path):
    txt = "\n" + path.read_text(errors="ignore")
    out = []
    for m in CPP_FN.finditer(txt):
        name = m.group(2)
        if name in ("if", "for", "while", "switch", "return", "sizeof"): continue
        ob = txt.find("{", m.start(1))
        ce = brace_body(txt, ob) if ob >= 0 else None
        if not ce: continue
        body = txt[m.start(1):ce]
        if 30 < len(body) < 8000:
            out.append({"lang": "cpp", "kind": "function", "name": name, "file": path.name, "text": body.strip()})
    return out

def chunk_mojo(path):
    lines = path.read_text(errors="ignore").splitlines()
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*(?:def|fn|struct|trait)\s+([A-Za-z_]\w*)", line)
        if m:
            blk = indent_block(lines, i)
            if 20 < len(blk) < 8000:
                out.append({"lang": "mojo", "kind": "def", "name": m.group(1),
                            "file": str(path.relative_to(EP_MOJO.parent)), "text": blk.strip()})
    return out

def main():
    chunks = []
    for f in sorted(EP_CPP.glob("*.cc")) + sorted(EP_CPP.glob("*.hh")):
        chunks += chunk_cpp(f)
    for f in EP_MOJO.rglob("*.mojo"):
        if ".pixi" in str(f): continue
        chunks += chunk_mojo(f)
    # Mojo idiom docs (the syntax skill) as retrievable reference chunks
    skill = Path(os.environ.get(
        "MOJO_SYNTAX_SKILL", str(Path.home() / ".claude/skills/mojo-syntax/SKILL.md")
    ))
    if skill.exists():
        sec, buf, title = [], [], "mojo-syntax"
        for l in skill.read_text().splitlines():
            if l.startswith("## "):
                if buf: sec.append((title, "\n".join(buf)))
                title, buf = l[3:].strip(), [l]
            else: buf.append(l)
        if buf: sec.append((title, "\n".join(buf)))
        for t, body in sec:
            if len(body) > 40:
                chunks.append({"lang": "doc", "kind": "mojo_idiom", "name": t, "file": "mojo-syntax/SKILL.md", "text": body.strip()[:4000]})
    for i, c in enumerate(chunks): c["id"] = i
    with (OUT / "chunks.jsonl").open("w") as fh:
        for c in chunks: fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    import collections
    by = collections.Counter((c["lang"], c["kind"]) for c in chunks)
    print(f"chunked {len(chunks)} units -> {OUT/'chunks.jsonl'}")
    for k, v in sorted(by.items()): print(f"  {k[0]:5s} {k[1]:12s} {v}")

if __name__ == "__main__":
    main()

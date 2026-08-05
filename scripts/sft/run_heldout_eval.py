#!/usr/bin/env python3
"""Pre-register the BASE-model baseline on the held-out eval (function_scalar).

For each verl record in heldout_eval.jsonl: send prompt=[system,user] to the
base Qwen2.5-Coder-3B, extract the function from the <answer> block, run it
against ground_truth {inputs, outputs}:
  * Mojo  : compile fn + a main() that calls it on each input (system linker),
            run, compare printed values.
  * Python: exec fn (with math helpers), call on each input, compare.
A record PASSES iff it builds/execs AND every test case matches (rel ≤ 1e-6).

Writes the pre-FT baseline so post-FT improvement is measurable. Use --tag to
label runs (e.g. base vs ft).
"""
from __future__ import annotations

import argparse, json, math, os, re, subprocess, tempfile, urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HELDOUT = REPO / "data/sft/codepivot/heldout_eval.jsonl"
ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"

EPMOJO = "/home/bart/github/EnergyPlusMojo/.pixi/envs/default"
MOJO_BIN = f"{EPMOJO}/bin/mojo"
MOJO_ENV = dict(os.environ, MODULAR_HOME=f"{EPMOJO}/share/max",
                PATH="/usr/bin:/bin:" + f"{EPMOJO}/bin")

PY_HELPERS = {k: getattr(math, k) for k in
              ("exp", "log", "log10", "log2", "sqrt", "sin", "cos", "tan", "asin",
               "acos", "atan", "atan2", "sinh", "cosh", "tanh", "floor", "ceil",
               "trunc", "fabs", "fmod", "hypot", "pow")}
PY_HELPERS.update({"pow_2": lambda x: x*x, "pow_3": lambda x: x**3, "pow_4": lambda x: x**4,
                   "pow_5": lambda x: x**5, "pow_6": lambda x: x**6, "pow_7": lambda x: x**7,
                   "mod": math.fmod, "sign": lambda a, b: math.copysign(abs(a), b),
                   "min": min, "max": max, "abs": abs})


def call_model(messages, max_tokens=1536, temperature=0.0):
    body = json.dumps({"model": "qwen2.5-coder-3b", "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def extract_code(text, lang):
    # prefer the <answer>...</answer> block, then a ```lang fence, then any fence
    ans = re.search(r"<answer>(.*?)</answer>", text, re.S)
    scope = ans.group(1) if ans else text
    for pat in (rf"```{lang}\s*\n(.*?)```", r"```[a-zA-Z]*\s*\n(.*?)```"):
        m = re.search(pat, scope, re.S)
        if m:
            return m.group(1).strip()
    return None


def def_name(code):
    m = re.search(r"^\s*def\s+(\w+)\s*\(", code, re.M)
    return m.group(1) if m else None


def lit(v, t):
    if t in ("int", "Int", "Int64", "long", "bool", "Bool"):
        return str(int(v))
    return repr(float(v))


def eval_python(code, fn, inputs, outputs, arg_types):
    ns = dict(PY_HELPERS)
    try:
        exec(code, ns)
    except Exception as e:
        return ("exec_error", str(e)[:50])
    f = ns.get(fn) or ns.get(def_name(code) or "")
    if not callable(f):
        return ("no_function", fn)
    for args, exp in zip(inputs, outputs):
        try:
            got = float(f(*args))
        except Exception as e:
            return ("runtime_error", str(e)[:50])
        if abs(got - exp) / max(abs(got), abs(exp), 1e-9) > 1e-6:
            return ("wrong_output", f"{args}->{got} exp {exp}")
    return ("pass", "")


def eval_mojo(code, fn, inputs, outputs, arg_types):
    name = def_name(code) or fn
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        calls = "\n".join("    print(" + name + "("
                          + ", ".join(lit(v, t) for v, t in zip(row, arg_types)) + "))"
                          for row in inputs)
        src = f"{code}\n\ndef main():\n{calls}\n"
        (tdp / "k.mojo").write_text(src)
        try:
            cp = subprocess.run([MOJO_BIN, "build", "-Xlinker", "-ldl",
                                 str(tdp / "k.mojo"), "-o", str(tdp / "k")],
                                env=MOJO_ENV, capture_output=True, text=True, timeout=150)
        except subprocess.TimeoutExpired:
            return ("compile_error", "timeout")
        if cp.returncode != 0:
            tail = [l for l in cp.stderr.splitlines() if ": error:" in l and "failed to parse" not in l]
            return ("compile_error", (tail[0][:80] if tail else "link/parse"))   # first SPECIFIC error
        r = subprocess.run([str(tdp / "k")], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return ("runtime_error", "rc!=0")
        got = [g for g in r.stdout.split() if g]
    if len(got) != len(outputs):
        return ("wrong_output", "count")
    for g, exp in zip(got, outputs):
        gl = g.strip()
        if gl in ("True", "False"):          # Mojo prints bools as True/False; C++/ground-truth use 1/0
            gl = "1" if gl == "True" else "0"
        try:
            gv = float(gl)
        except ValueError:
            return ("wrong_output", g[:30])
        if abs(gv - exp) / max(abs(gv), abs(exp), 1e-9) > 1e-6:
            return ("wrong_output", f"{gv} exp {exp}")
    return ("pass", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", type=Path, default=REPO / "data/sft/codepivot/heldout_baseline.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in HELDOUT.read_text().splitlines() if l.strip()]
    results = []
    for i, rec in enumerate(recs, 1):
        ei = rec["extra_info"]
        lang = ei["language_full"]          # Mojo | Python
        ltag = "mojo" if lang == "Mojo" else "python"
        gt = rec["reward_model"]["ground_truth"]
        try:
            out = call_model(rec["prompt"])
        except Exception as e:
            results.append({"fn": ei["function_name"], "lang": lang, "status": "gen_error", "detail": str(e)[:40]})
            print(f"[{i}/{len(recs)}] {lang:6s} {ei['function_name']:30s} GEN_ERROR")
            continue
        code = extract_code(out, ltag)
        if not code:
            status, detail = "no_code", ""
        elif lang == "Mojo":
            status, detail = eval_mojo(code, ei["function_name"], gt["inputs"], gt["outputs"], ei["arg_types"])
        else:
            status, detail = eval_python(code, ei["function_name"], gt["inputs"], gt["outputs"], ei["arg_types"])
        results.append({"fn": ei["function_name"], "lang": lang, "status": status, "detail": detail})
        print(f"[{i}/{len(recs)}] {lang:6s} {ei['function_name']:30s} {status.upper():14s} {detail[:40]}")

    json.dump({"tag": args.tag, "results": results}, open(args.out, "w"), indent=1)
    print("\n=== BASELINE (" + args.tag + ") ===")
    for lang in ("Mojo", "Python"):
        rs = [r for r in results if r["lang"] == lang]
        if not rs:
            continue
        c = Counter(r["status"] for r in rs)
        np_ = c.get("pass", 0)
        comp = sum(1 for r in rs if r["status"] in ("pass", "wrong_output", "runtime_error"))
        print(f"{lang:7s}  pass@1 {100*np_/len(rs):5.1f}% ({np_}/{len(rs)})  built {comp}/{len(rs)}  {dict(c)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

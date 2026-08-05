#!/usr/bin/env python3
"""Build a *behaviorally-verified* C++ -> Mojo training dataset from EnergyPlus.

Motivation
----------
energyplus-mojo's hand-written Mojo kernels are spec-driven reimplementations,
not faithful translations of the EnergyPlus C++ (e.g. `ashrae_simple_h_in.mojo`
keeps 3 of the C++ `CalcASHRAESimpleIntConvCoeff`'s 7 convection coefficients and
inverts a sign convention). Mining those as (C++, Mojo) pairs would teach a model
*wrong* translations. So instead of mining, we **generate and verify**:

    real EnergyPlus C++ scalar fn
        -> transpiler (algorithmic C++->Mojo)
        -> compile BOTH standalone, run on sampled inputs, compare outputs
        -> keep only pairs that agree numerically

Fidelity is then guaranteed by construction, and the Mojo is idiomatic scalar
Mojo (the correct transpiler-output form), not an FFI batch kernel.

Scope: self-contained pure-scalar functions — `Real64 Name(Real64 const a, ...)`
with only scalar (Real64/double/int/bool) const args and no EnergyPlusData /
Array / reference / pointer / string params. The standalone-compile gate filters
out any that secretly depend on other translation units.

Read-only on the source repos; all artifacts land under the transpilers repo.

Usage
-----
    uv run python scripts/build_cpp_mojo_dataset.py \
        --ep-src /home/bart/Github/EnergyPlus/src/EnergyPlus \
        --out data/cpp_mojo_pairs.jsonl \
        --limit 40            # cap candidates for a quick yield probe

    uv run python scripts/build_cpp_mojo_dataset.py --out data/cpp_mojo_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PREAMBLE_FILE = REPO / "scripts" / "ep_preamble.h"

# Mojo toolchain from the energyplus-mojo pixi env — invoked directly with
# MODULAR_HOME set so we never run `pixi run` (which rewrites that repo's lock).
# Override with TRANSPILERS_EPMOJO when the repo lives elsewhere.
EPMOJO = Path(os.environ.get("TRANSPILERS_EPMOJO",
                             "/home/bart/github/EnergyPlusMojo/.pixi/envs/default"))
MOJO_BIN = EPMOJO / "bin" / "mojo"
MODULAR_HOME = EPMOJO / "share" / "max"


# ---------------------------------------------------------------------------
# 1. Extract candidate pure-scalar C++ functions
# ---------------------------------------------------------------------------

# Scalar types we know how to sample + bake as literals. EnergyPlus uses several
# typedef aliases for double (Real64, Nandle) and int (Int, Int64); the ep_preamble
# resolves them for libclang, and the transpiled Mojo comes out as Float64/Int so
# sampling keys off the *Mojo* type regardless.
_SCALAR_TYPE = re.compile(
    r"^(?:Real64|Real32|Nandle|double|float|int|Int|Int64|long|unsigned|bool|"
    r"std::size_t|size_t)$"
)

# Candidate signature START: a scalar return type + identifier + `(`. The full
# (possibly multi-line, comment-laden) parameter list and the `) ... {` tail are
# resolved by paren/brace matching in extract_fns, not by this regex.
_RET = r"Real64|Real32|Nandle|double|float|int|Int|Int64|long|bool"
_SIG_START = re.compile(rf"(?<![\w:])(?P<ret>{_RET})\s+(?P<name>[A-Za-z_]\w*)\s*\(")

# Qualifiers/trivia allowed between `)` and the body `{` of a definition.
_POST_PAREN = re.compile(r"^(?:\s|const|noexcept|override|final|/\*.*?\*/|//[^\n]*)*\{", re.S)


@dataclass
class CppFn:
    name: str
    ret: str
    params: list[tuple[str, str]]  # (type, name)
    body: str                      # full text incl. signature and braces
    source_file: str


def _match_delims(text: str, open_idx: int, open_ch: str = "{", close_ch: str = "}") -> int:
    """Return index just past the delimiter matching the one at open_idx,
    skipping string/char literals and comments. Works for `{}` and `()`."""
    depth = 0
    i = open_idx
    in_str = in_char = in_line_c = in_block_c = False
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line_c:
            if c == "\n":
                in_line_c = False
        elif in_block_c:
            if c == "*" and nxt == "/":
                in_block_c = False
                i += 1
        elif in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif in_char:
            if c == "\\":
                i += 1
            elif c == "'":
                in_char = False
        elif c == "/" and nxt == "/":
            in_line_c = True
            i += 1
        elif c == "/" and nxt == "*":
            in_block_c = True
            i += 1
        elif c == '"':
            in_str = True
        elif c == "'":
            in_char = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _match_braces(text: str, open_idx: int) -> int:
    return _match_delims(text, open_idx, "{", "}")


def _strip_comments(s: str) -> str:
    """Drop // and /* */ comments — EnergyPlus signatures comment nearly every
    parameter inline (`Real64 const SolZen,  // solar zenith angle (deg)`), and
    those comments contain commas/parens/semicolons that wreck arg parsing."""
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


def _parse_params(arg_str: str) -> list[tuple[str, str]] | None:
    """Parse `Real64 const a, int const n` -> [('Real64','a'),('int','n')].

    Returns None if any param is non-scalar (ref, ptr, array, state, string,
    template, default-arg) — i.e. the function isn't a pure scalar fn.
    """
    arg_str = _strip_comments(arg_str).strip()
    if not arg_str or arg_str == "void":
        return []
    params: list[tuple[str, str]] = []
    for raw in _split_top_level(arg_str):
        p = raw.strip()
        if not p:
            continue
        if any(tok in p for tok in ("&", "*", "<", "[", "=", "...", "::", "std::string", "EnergyPlusData")):
            return None
        p = p.replace("const", " ").replace("volatile", " ")
        toks = p.split()
        if len(toks) < 2:
            return None
        ptype, pname = toks[0], toks[-1]
        if not _SCALAR_TYPE.match(ptype):
            return None
        if not re.match(r"^[A-Za-z_]\w*$", pname):
            return None
        params.append((ptype, pname))
    return params


def _split_top_level(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return out


def extract_fns(ep_src: Path, recursive: bool = False) -> list[CppFn]:
    """Scan every .cc for scalar-returning function *definitions*, resolving
    multi-line / comment-laden signatures by paren+brace matching."""
    fns: list[CppFn] = []
    seen: set[str] = set()
    # .cc translation units + .hh headers (the headers carry many inline scalar fns)
    pat_cc, pat_hh = ("**/*.cc", "**/*.hh") if recursive else ("*.cc", "*.hh")
    for cc in sorted(list(ep_src.glob(pat_cc)) + list(ep_src.glob(pat_hh))):
        text = cc.read_text(errors="ignore")
        for m in _SIG_START.finditer(text):
            name = m.group("name")
            if name in seen:
                continue
            # The matched `(` is the last char of the match; paren-match it to
            # find the full (possibly multi-line) parameter list.
            paren_open = m.end() - 1
            paren_end = _match_delims(text, paren_open, "(", ")")
            if paren_end < 0:
                continue
            args_raw = text[paren_open + 1:paren_end - 1]
            # The tail between `)` and `{` may carry const/noexcept/comments. If
            # the next real token isn't `{`, this is a declaration / ctor-init /
            # something else, not a plain definition — skip.
            tail = text[paren_end:paren_end + 400]
            mt = _POST_PAREN.match(tail)
            if not mt:
                continue
            params = _parse_params(args_raw)
            if params is None:
                continue
            brace_open = paren_end + mt.end() - 1   # index of the `{`
            end = _match_braces(text, brace_open)
            if end < 0:
                continue
            body = text[m.start("ret"):end]
            # Reject bodies that obviously reach outside the translation unit.
            if re.search(r"\bstate\b|->|\.dataptr|ShowSevereError|ShowWarningError"
                         r"|ShowFatalError|GetCurrentScheduleValue", body):
                continue
            seen.add(name)
            fns.append(CppFn(name=name, ret=m.group("ret"), params=params,
                             body=body, source_file=str(cc.relative_to(ep_src))))
    return fns


# ---------------------------------------------------------------------------
# 2. Transpile C++ -> Mojo
# ---------------------------------------------------------------------------

def transpile(fn: CppFn) -> str | None:
    os.environ["TRANSPILERS_CPP_PREAMBLE_FILE"] = str(PREAMBLE_FILE)
    from transpilers.cli.main import transpile_cpp_to_mojo
    try:
        mojo = transpile_cpp_to_mojo(fn.body)
    except Exception:
        return None
    return mojo.strip() or None


# Map transpiled Mojo param types to a sampling strategy.
_MOJO_SIG = re.compile(r"def\s+\w+\(([^)]*)\)\s*->\s*(\w+)")


def mojo_params(mojo: str) -> list[tuple[str, str]] | None:
    m = _MOJO_SIG.search(mojo)
    if not m:
        return None
    out = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            return None
        nm, ty = part.split(":", 1)
        out.append((nm.strip(), ty.strip()))
    return out


# ---------------------------------------------------------------------------
# 3. Behavioral verification (compile both, run on sampled inputs, compare)
# ---------------------------------------------------------------------------

# Definitions (not just declarations) for the ObjexxFCL helpers the C++ bodies
# may call, so the standalone oracle links.
_CPP_HELPERS = r"""
#include <cstdio>
#include <cmath>
#include <cassert>
typedef double Real64;
typedef int Int;
typedef long long Int64;
static inline double pow_2(double x){return x*x;}
static inline double pow_3(double x){return x*x*x;}
static inline double pow_4(double x){return x*x*x*x;}
static inline double pow_5(double x){return x*x*x*x*x;}
static inline double pow_6(double x){double y=x*x*x;return y*y;}
static inline double pow_7(double x){double y=x*x*x;return y*y*x;}
static inline double root_4(double x){return std::sqrt(std::sqrt(x));}
static inline double root_8(double x){return std::sqrt(std::sqrt(std::sqrt(x)));}
static inline double mod(double a,double b){return std::fmod(a,b);}
static inline int mod(int a,int b){return a%b;}
static inline double sign(double a,double b){return b>=0.0?std::fabs(a):-std::fabs(a);}
static inline int sign(int a,int b){return b>=0?(a<0?-a:a):-(a<0?-a:a);}
// EnergyPlus calls `max`/`min` unqualified (ObjexxFCL/<algorithm>). Provide
// double+int overloads so the oracle resolves them unambiguously — without
// these, `<cmath>`'s 3-arg std::max leaks in and unqualified `max(a,b)` becomes
// an ambiguous/ill-formed call. Semantics match std::max/std::min exactly.
static inline double max(double a,double b){return a>b?a:b;}
static inline double min(double a,double b){return a<b?a:b;}
static inline int max(int a,int b){return a>b?a:b;}
static inline int min(int a,int b){return a<b?a:b;}
namespace Constant {
constexpr double MaxEXPArg=709.78; constexpr double Pi=3.14159265358979324;
constexpr double PiOvr2=Pi/2.0; constexpr double TwoPi=2.0*Pi; constexpr double Gravity=9.807;
constexpr double DegToRad=Pi/180.0; constexpr double RadToDeg=180.0/Pi; constexpr double Kelvin=273.15;
constexpr double TriplePointOfWaterTempKelvin=273.16; constexpr double StefanBoltzmann=5.6697E-8;
}
namespace DataPrecisionGlobals {
constexpr double constant_zero=0.0; constexpr double constant_one=1.0;
constexpr double constant_minusone=-1.0; constexpr double constant_twenty=20.0;
constexpr double constant_pointfive=0.5; constexpr double EXP_LowerLimit=-20.0;
constexpr double EXP_UpperLimit=40.0;
}
"""


_LITERAL = re.compile(r"(?<![A-Za-z_0-9.])(-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?)")


def _thresholds(cpp_body: str) -> list[float]:
    """Numeric literals in the C++ body — branch predicates compare against
    these constants, so they tell us where the boundaries are. We probe just
    below / at / just above each."""
    vals = set()
    for m in _LITERAL.finditer(cpp_body):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if abs(v) < 1e12:
            vals.add(v)
    return sorted(vals)


def _sample_inputs(params: list[tuple[str, str]], cpp_body: str,
                   n: int = 120) -> list[list[float]]:
    """Threshold- and domain-aware input spread.

    Per-arg candidate pool = {physically-plausible domain spread} ∪ {each C++
    body literal and literal±eps}. Branch predicates compare args against those
    literals, so sampling around them is what actually exercises the branches.
    Combinations are drawn independently per column (seeded LCG) so different
    branches co-occur; the coverage gate downstream is the real arbiter, so we
    draw generously (rows are nearly free — one baked binary run)."""
    # Bounded-domain spread: covers cosines/fractions in [-1,1] AND larger
    # physical quantities (temps, pressures), incl. boundary-ish small values.
    domain = [-1.0, -0.95, -0.5, -0.3827, -0.1, 0.0, 0.05, 0.1, 0.3, 0.3827,
              0.5, 0.7, 0.9239, 0.99, 1.0, 1.5, 5.0, 12.0, 23.5, 37.0, 50.0,
              100.0, 250.0, -2.0, -15.0]
    thr = _thresholds(cpp_body)
    eps = 1e-6
    pools: list[list[float]] = []
    for (nm, ty) in params:
        pool = list(domain)
        for t in thr:
            scale = max(abs(t), 1.0)
            pool += [t, t - eps * scale, t + eps * scale]
        # de-dup, keep order-ish
        seen, uniq = set(), []
        for v in pool:
            k = round(v, 12)
            if k not in seen:
                seen.add(k)
                uniq.append(v)
        pools.append(uniq)

    rows = []
    state = 2463534242
    for r in range(n):
        row = []
        for ci, (nm, ty) in enumerate(params):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            pool = pools[ci]
            v = pool[state % len(pool)]
            if ty in ("Int", "int"):
                v = float(int(abs(v)) % 64)
            elif ty in ("Bool", "bool"):
                v = float(state & 1)
            row.append(v)
        rows.append(row)
    # All-args-equal rows: hit `(a - b) == 0` style zero-difference branches that
    # independent column sampling almost never reaches.
    for val in (0.0, 1.0, 23.5, 50.0, -2.0):
        rows.append([0.0 if ty in ("Int", "int", "Bool", "bool") else val
                     for (nm, ty) in params])
    return rows


def _fmt_lit(v: float, ty: str) -> tuple[str, str]:
    """Return (cpp_literal, mojo_literal) for a sampled value of Mojo type ty."""
    if ty in ("Int", "int"):
        return str(int(v)), str(int(v))
    if ty in ("Bool", "bool"):
        b = "true" if v else "false"
        return b, ("True" if v else "False")
    return repr(v), repr(v)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


def _body_line_span(fn: CppFn) -> int:
    """Number of source lines in the function body (for gcov line mapping)."""
    return fn.body.count("\n") + 1


# A line whose only effect is `return <numeric-literal>;` (or break/continue).
# An uncovered line of this shape is a zero-measure float-equality fallback
# (e.g. ASHRAE's `return 3.076;` reached only when DeltaTempCosTilt == 0.0
# exactly) and carries negligible transpile risk — the transpiler renders every
# constant return identically. Uncovered lines with *computation* or *calls* do
# carry risk and are rejected.
_TRIVIAL_RETURN = re.compile(r"^\s*(?:return\s+-?[0-9.][0-9.eE+\-]*\s*;|break\s*;|continue\s*;|\}?\s*)$")


def _coverage_ok(tdp: Path, cpp_file: str, fn_first_line: int, n_body_lines: int
                 ) -> tuple[bool, int, int, list[str]]:
    """Run gcov over the target function body. Accept iff every executable line
    that contains *computation* was exercised; uncovered trivial constant-return
    fallbacks are tolerated but reported. Returns
    (acceptable, covered, uncovered_total, uncovered_risky_lines)."""
    r = _run(["gcov", "-b", "-o", str(tdp), cpp_file], cwd=str(tdp))
    gcov = tdp / f"{cpp_file}.gcov"
    if r.returncode != 0 or not gcov.exists():
        return (False, 0, 0, ["<gcov failed>"])
    covered = uncovered = 0
    risky: list[str] = []
    lo, hi = fn_first_line, fn_first_line + n_body_lines - 1
    for line in gcov.read_text(errors="ignore").splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        count, lineno, src = parts[0].strip(), parts[1].strip(), parts[2]
        if not lineno.isdigit():
            continue
        ln = int(lineno)
        if ln < lo or ln > hi:
            continue
        if count == "-":              # non-executable (decl, brace, comment)
            continue
        if count.startswith("#") or count == "=====":
            uncovered += 1
            if not _TRIVIAL_RETURN.match(src):
                risky.append(src.strip())
        else:
            covered += 1
    acceptable = covered > 0 and not risky
    return (acceptable, covered, uncovered, risky)


def verify(fn: CppFn, mojo: str, mparams: list[tuple[str, str]]) -> dict | None:
    """Compile + run both sides on sampled inputs, with a branch-coverage gate.

    A pair is accepted only if (a) the C++ and Mojo outputs agree to rel-err
    ≤1e-9 on ≥4 finite points AND (b) every executable line of the C++ function
    body was exercised by the samples. (b) closes the false-faithful hole where
    a transpiler bug in an unexercised branch would pass silently."""
    samples = _sample_inputs([(n, t) for n, t in mparams], fn.body)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # --- C++ oracle (instrumented for coverage) ---
        # The function body begins at line 1 of fn.body; in oracle.cpp it is
        # offset by the helper preamble. Track that offset for gcov mapping.
        preamble_lines = _CPP_HELPERS.count("\n") + 1  # leading "\n" handled below
        # Emit an index with each value so C++ and Mojo outputs can be aligned by
        # index, not by position — robust to `inf`/`nan` lines that would otherwise
        # be dropped asymmetrically and misalign the two streams (false reject).
        cpp_calls = "\n".join(
            f'  printf("%d %.15g\\n", {i}, (double){fn.name}('
            + ", ".join(_fmt_lit(v, t)[0] for v, (n, t) in zip(row, mparams))
            + "));"
            for i, row in enumerate(samples)
        )
        header = f"{_CPP_HELPERS}\n"
        fn_first_line = header.count("\n") + 1
        cpp_src = f"{header}{fn.body}\nint main(){{\n{cpp_calls}\n  return 0;\n}}\n"
        (tdp / "oracle.cpp").write_text(cpp_src)
        # -DNDEBUG compiles out `assert(...)` (EnergyPlus uses it only as a
        # debug precondition check) so out-of-domain samples don't abort the
        # oracle; it can only remove aborts, never change numeric results. This
        # matches the Mojo backend, which drops `assert` calls entirely.
        r = _run(["g++", "-O0", "-std=c++17", "-DNDEBUG", "--coverage",
                  "-o", str(tdp / "oracle"), str(tdp / "oracle.cpp")], cwd=str(tdp))
        if r.returncode != 0:
            return None  # depends on stuff outside the TU — not self-contained
        r = _run([str(tdp / "oracle")], cwd=str(tdp))
        if r.returncode != 0:
            return None
        cpp_out = {p[0]: p[1] for p in (ln.split() for ln in r.stdout.splitlines()) if len(p) == 2}

        cov_ok, cov_hit, cov_miss, cov_risky = _coverage_ok(
            tdp, "oracle.cpp", fn_first_line, _body_line_span(fn))
        if not cov_ok:
            return None  # a computational branch went unexercised — can't certify

        # --- Mojo ---
        mojo_calls = "\n".join(
            f'    print({i}, {fn.name}('
            + ", ".join(_fmt_lit(v, t)[1] for v, (n, t) in zip(row, mparams))
            + "))"
            for i, row in enumerate(samples)
        )
        mojo_src = f"{mojo}\n\ndef main():\n{mojo_calls}\n"
        (tdp / "k.mojo").write_text(mojo_src)
        env = dict(os.environ, MODULAR_HOME=str(MODULAR_HOME),
                   PATH=f"{EPMOJO / 'bin'}:{os.environ.get('PATH', '')}")
        r = _run([str(MOJO_BIN), "run", str(tdp / "k.mojo")], env=env)
        if r.returncode != 0:
            return None
        # Parse "<index> <value>" per line into {index: value}; normalize Mojo's
        # Bool True/False to 1/0 (C++ oracle prints (double)bool).
        mojo_out = {}
        for ln in r.stdout.splitlines():
            p = ln.split()
            if len(p) == 2 and p[0].isdigit():
                mojo_out[p[0]] = "1" if p[1] == "True" else "0" if p[1] == "False" else p[1]

    if not cpp_out:
        return None

    finite = 0
    max_rel = 0.0
    for idx, a in cpp_out.items():          # align by index, robust to inf/nan drops
        b = mojo_out.get(idx)
        if b is None:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return None
        if fa != fa or fb != fb or abs(fa) == float("inf") or abs(fb) == float("inf"):
            continue  # skip NaN/inf points (domain edges)
        denom = max(abs(fa), abs(fb), 1e-9)
        rel = abs(fa - fb) / denom
        max_rel = max(max_rel, rel)
        if rel > 1e-9:
            return None  # genuine numeric divergence
        finite += 1
    if finite < 4:
        return None  # too few finite agreements to trust
    return {"samples_total": len(cpp_out), "samples_finite": finite,
            "max_rel_err": max_rel,
            "cpp_lines_covered": cov_hit, "cpp_lines_uncovered": cov_miss,
            "branch_coverage": "full" if cov_miss == 0 else "computation-full"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-src", type=Path,
                    default=Path(os.environ.get("EP_SRC", "/home/bart/Github/EnergyPlus/src/EnergyPlus")))
    ap.add_argument("--out", type=Path, default=REPO / "data" / "cpp_mojo_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (0=all)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--recursive", action="store_true",
                    help="also scan ep-src subdirectories (AirflowNetwork, Autosizing, ...)")
    ap.add_argument("--dump-fails", type=Path, default=None,
                    help="write failed candidates (name, stage, cpp body) as JSONL "
                         "for a downstream LLM-translate + re-verify pass")
    args = ap.parse_args()

    fns = extract_fns(args.ep_src, recursive=args.recursive)
    print(f"extracted {len(fns)} candidate pure-scalar C++ functions")
    if args.limit:
        fns = fns[: args.limit]
        print(f"limited to {len(fns)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"transpile_fail": 0, "sig_fail": 0, "verify_fail": 0, "ok": 0}
    pairs = []
    fails = []

    def _fail(fn: CppFn, stage: str, mojo: str | None = None) -> None:
        stats[stage] += 1
        fails.append({"function_name": fn.name, "source_file": fn.source_file,
                      "ret_type": fn.ret, "arg_types": [t for t, _ in fn.params],
                      "stage": stage, "cpp_source": fn.body,
                      **({"mojo_attempt": mojo} if mojo else {})})

    for i, fn in enumerate(fns, 1):
        mojo = transpile(fn)
        if not mojo:
            _fail(fn, "transpile_fail")
            if args.verbose:
                print(f"[{i}/{len(fns)}] {fn.name}: transpile_fail")
            continue
        mp = mojo_params(mojo)
        if mp is None or len(mp) != len(fn.params):
            _fail(fn, "sig_fail", mojo)
            if args.verbose:
                print(f"[{i}/{len(fns)}] {fn.name}: sig_fail")
            continue
        vres = verify(fn, mojo, mp)
        if vres is None:
            _fail(fn, "verify_fail", mojo)
            if args.verbose:
                print(f"[{i}/{len(fns)}] {fn.name}: verify_fail")
            continue
        stats["ok"] += 1
        pair = {
            "cpp_source": fn.body,
            "mojo_source": mojo,
            "function_name": fn.name,
            "source_file": fn.source_file,
            "n_args": len(fn.params),
            "arg_types": [t for t, _ in fn.params],
            "ret_type": fn.ret,
            "verification": {"method": "behavioral", **vres},
            "provenance": "energyplus-cpp-generate-verify",
            "direction": "cpp->mojo",
        }
        pairs.append(pair)
        print(f"[{i}/{len(fns)}] {fn.name}: OK  "
              f"(finite {vres['samples_finite']}/{vres['samples_total']}, "
              f"max_rel {vres['max_rel_err']:.1e})  @ {fn.source_file}")

    with args.out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    if args.dump_fails:
        args.dump_fails.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_fails.open("w") as f:
            for rec in fails:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"failed candidates -> {args.dump_fails} ({len(fails)})")

    print("\n=== summary ===")
    print(f"candidates:     {len(fns)}")
    print(f"transpile_fail: {stats['transpile_fail']}")
    print(f"sig_fail:       {stats['sig_fail']}")
    print(f"verify_fail:    {stats['verify_fail']}")
    print(f"VERIFIED pairs: {stats['ok']}")
    print(f"written to:     {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

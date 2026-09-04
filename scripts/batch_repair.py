"""Batch repair: first-pass transpile without inline repair, bucket failures by
error category, then repair each bucket with category-specific prompts.

Instead of per-file repair loops (O(files × passes) LLM calls), this script
does one pass to collect all failures, groups them by root cause, and repairs
by category in bulk — O(categories × passes) LLM calls.

Failure categories (in priority order):
  STDLIB-GAP        stdlib function not in stdlib_maps
  UNHANDLED-CURSOR  parser saw an AST node it has no emitter for
  UNHANDLED-OP      unknown operator or expression kind
  EMIT-INVALID      emitter produced syntactically invalid output
  MALFORMED         output structure is broken (missing brackets etc.)
  MISSING-HEADER    import / include not found

Usage
-----
    uv run python scripts/batch_repair.py <root> --target mojo
    uv run python scripts/batch_repair.py <root> --target python --repair-passes 2
    uv run python scripts/batch_repair.py <root> --target rust --triage-only

Options
-------
--target          Target language (default: python)
--repair-passes   LLM repair iterations per category (default: 2)
--triage-only     Print failure triage report but do not attempt repair
--output-dir      Write repaired outputs here (default: <root>/_repaired)
--workers         Parallel transpile workers (default: 4)
--ir-augment      Pre-seed types from LLVM IR before transpiling (C/C++ only)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Failure categorisation
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("STDLIB-GAP",        re.compile(r"stdlib|std::|not in stdlib_maps|no mapping", re.I)),
    ("UNHANDLED-CURSOR",  re.compile(r"unhandled cursor|unsupported cursor|CursorKind", re.I)),
    ("UNHANDLED-OP",      re.compile(r"unhandled op|unsupported op|unknown op|BinaryOperator", re.I)),
    ("EMIT-INVALID",      re.compile(r"SyntaxError|IndentationError|unexpected indent|invalid syntax", re.I)),
    ("MALFORMED",         re.compile(r"malformed|parse error|unexpected token|mismatched", re.I)),
    ("MISSING-HEADER",    re.compile(r"file not found|no such file|ModuleNotFoundError|ImportError", re.I)),
]

_TIER_WEIGHT = {
    "STDLIB-GAP": 5,
    "UNHANDLED-CURSOR": 4,
    "UNHANDLED-OP": 4,
    "EMIT-INVALID": 3,
    "MALFORMED": 2,
    "MISSING-HEADER": 1,
    "OTHER": 0,
}


def categorise(error_msg: str) -> str:
    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(error_msg):
            return cat
    return "OTHER"


# ---------------------------------------------------------------------------
# First-pass transpile
# ---------------------------------------------------------------------------

EXT_MAP = {
    "c": "c", "cpp": "cpp", "cs": "csharp", "f90": "fortran",
    "java": "java", "js": "javascript", "py": "python",
    "ts": "typescript", "vb": "vb",
}


def _transpile_one(
    f: Path,
    *,
    target: str,
    ir_augment: bool,
) -> tuple[Path, bool, str, str]:
    """Transpile a single file. Returns (path, ok, output, error)."""
    src_lang = EXT_MAP.get(f.suffix.lstrip("."))
    if src_lang is None:
        return f, False, "", "unsupported extension"
    cmd = ["uv", "run", "transpile", str(f), "--source", src_lang, "--target", target]
    if ir_augment and src_lang in ("c", "cpp"):
        cmd.append("--ir-augment")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return f, False, "", "TIMEOUT"
    if proc.returncode == 0:
        return f, True, proc.stdout, ""
    err = (proc.stderr or proc.stdout).strip()
    return f, False, "", err


def first_pass(
    files: list[Path],
    *,
    target: str,
    workers: int,
    ir_augment: bool,
) -> tuple[dict[Path, str], dict[Path, str]]:
    """Transpile all files in parallel. Returns (successes, failures) dicts."""
    successes: dict[Path, str] = {}
    failures: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_transpile_one, f, target=target, ir_augment=ir_augment): f
            for f in files
        }
        total = len(futs)
        done = 0
        for fut in as_completed(futs):
            path, ok, out, err = fut.result()
            done += 1
            if ok:
                successes[path] = out
            else:
                failures[path] = err
            print(
                f"\r  {done}/{total} transpiled  "
                f"({len(successes)} ok, {len(failures)} fail)",
                end="", flush=True,
            )
    print()
    return successes, failures


# ---------------------------------------------------------------------------
# Triage report
# ---------------------------------------------------------------------------

def triage(failures: dict[Path, str]) -> dict[str, list[Path]]:
    """Bucket failures by category."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for path, err in failures.items():
        buckets[categorise(err)].append(path)
    return dict(buckets)


def print_triage(
    buckets: dict[str, list[Path]],
    failures: dict[Path, str],
    total: int,
) -> None:
    print(f"\n=== Failure triage ({len(failures)}/{total} failed) ===\n")
    ranked = sorted(
        buckets.items(),
        key=lambda kv: _TIER_WEIGHT.get(kv[0], 0) * len(kv[1]),
        reverse=True,
    )
    for cat, paths in ranked:
        weight = _TIER_WEIGHT.get(cat, 0)
        score = weight * len(paths)
        print(f"  {cat:<20} {len(paths):>4} files  (priority score {score})")
        for p in sorted(paths)[:3]:
            err_line = failures[p].splitlines()[-1][:80] if failures[p] else ""
            print(f"    {p.name}: {err_line}")
        if len(paths) > 3:
            print(f"    ... and {len(paths) - 3} more")
    print()


# ---------------------------------------------------------------------------
# Category repair
# ---------------------------------------------------------------------------

_REPAIR_PROMPT = """\
You are a transpiler repair assistant helping fix a batch of {target} translation errors
of the same category: {category}.

Category description: {category_desc}

Here are {n_examples} failing examples with their error messages:

{examples}

For each SOURCE below, output the corrected {target} translation.
Separate outputs with the marker: ### FILE <filename> ###
Only output corrected code, no explanation.

FILES TO REPAIR:
{to_repair}
"""

_CATEGORY_DESCRIPTIONS = {
    "STDLIB-GAP":       "A C++ stdlib function has no mapping to the target language. Add an equivalent.",
    "UNHANDLED-CURSOR": "The AST parser saw a node kind it has no emitter for. Provide a best-effort translation.",
    "UNHANDLED-OP":     "An operator or expression kind has no emitter. Translate it manually.",
    "EMIT-INVALID":     "The emitter produced syntactically invalid code. Fix the syntax errors.",
    "MALFORMED":        "The output structure is broken (brackets, indentation). Fix the structure.",
    "MISSING-HEADER":   "An import/include is missing. Add the correct import for the target language.",
    "OTHER":            "Miscellaneous transpilation failure. Provide a correct translation.",
}


def _format_examples(
    bucket: list[Path],
    failures: dict[Path, str],
    successes: dict[Path, str],
    n: int = 3,
) -> str:
    lines: list[str] = []
    shown = 0
    for p in bucket:
        if shown >= n:
            break
        err = failures.get(p, "")
        src = p.read_text(errors="replace")[:500]
        lines.append(f"--- {p.name} ---\nSource:\n{src}\nError:\n{err}\n")
        shown += 1
    return "\n".join(lines)


def repair_category(
    category: str,
    bucket: list[Path],
    failures: dict[Path, str],
    *,
    target: str,
    repair_passes: int,
    llm_client,
) -> dict[Path, str]:
    """Attempt LLM repair for all files in a failure category."""
    repaired: dict[Path, str] = {}
    desc = _CATEGORY_DESCRIPTIONS.get(category, "")
    examples_str = _format_examples(bucket, failures, {})
    to_repair_str = "\n".join(
        f"### FILE {p.name} ###\n{p.read_text(errors='replace')[:800]}"
        for p in bucket
    )
    for attempt in range(repair_passes):
        prompt = _REPAIR_PROMPT.format(
            target=target,
            category=category,
            category_desc=desc,
            n_examples=min(3, len(bucket)),
            examples=examples_str,
            to_repair=to_repair_str,
        )
        try:
            response = llm_client(prompt)
        except Exception as exc:
            print(f"  LLM call failed (attempt {attempt + 1}): {exc}", file=sys.stderr)
            break
        # Parse outputs separated by ### FILE <name> ###
        parts = re.split(r"###\s*FILE\s+([^\s#]+)\s*###", response)
        # parts = [prefix, name1, code1, name2, code2, ...]
        i = 1
        while i + 1 < len(parts):
            fname = parts[i].strip()
            code = parts[i + 1].strip()
            matched = next((p for p in bucket if p.name == fname), None)
            if matched and code:
                repaired[matched] = code
            i += 2
        if len(repaired) >= len(bucket):
            break
    return repaired


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batch_repair",
        description="First-pass transpile + triage + category-level batch repair.",
    )
    parser.add_argument("root", type=Path, help="Directory containing source files.")
    parser.add_argument("--target", default="python", choices=[
        "rust", "c", "mojo", "python", "fortran"
    ])
    parser.add_argument("--repair-passes", type=int, default=2,
                        help="LLM repair iterations per failure category (default: 2).")
    parser.add_argument("--triage-only", action="store_true",
                        help="Print triage report but do not repair.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write repaired outputs here (default: <root>/_repaired).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel transpile workers (default: 4).")
    parser.add_argument("--ir-augment", action="store_true",
                        help="Pre-seed types from LLVM IR (C/C++ only, requires clang).")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="Write results summary to this JSON file.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output_dir = args.output_dir or (root / "_repaired")

    # Gather files
    files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and EXT_MAP.get(f.suffix.lstrip("."))
    ]
    if not files:
        print(f"No supported source files found under {root}", file=sys.stderr)
        return 1

    print(f"First-pass transpile: {len(files)} files → {args.target}")
    successes, failures = first_pass(
        files, target=args.target, workers=args.workers, ir_augment=args.ir_augment
    )
    print(f"  {len(successes)} ok / {len(failures)} failed")

    buckets = triage(failures)
    print_triage(buckets, failures, len(files))

    results: dict = {
        "total": len(files),
        "success": len(successes),
        "failed": len(failures),
        "buckets": {cat: [str(p) for p in paths] for cat, paths in buckets.items()},
        "repaired": {},
    }

    if not args.triage_only and failures:
        # Lazy import LLM client only when repair is needed
        try:
            _src = Path(__file__).resolve().parent.parent / "src"
            sys.path.insert(0, str(_src))
            from transpilers.llm import LlmClient
            llm = LlmClient()
        except Exception as exc:
            print(f"Cannot load LLM client for repair: {exc}", file=sys.stderr)
            print("Re-run with --triage-only or set ANTHROPIC_API_KEY.", file=sys.stderr)
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)
        total_repaired = 0
        ranked_cats = sorted(
            buckets.items(),
            key=lambda kv: _TIER_WEIGHT.get(kv[0], 0) * len(kv[1]),
            reverse=True,
        )
        for cat, bucket in ranked_cats:
            print(f"Repairing {cat} ({len(bucket)} files) …")
            fixed = repair_category(
                cat, bucket, failures,
                target=args.target,
                repair_passes=args.repair_passes,
                llm_client=llm,
            )
            results["repaired"][cat] = [str(p) for p in fixed]
            for path, code in fixed.items():
                out_path = output_dir / path.name
                out_path.write_text(code)
            total_repaired += len(fixed)
            print(f"  {len(fixed)}/{len(bucket)} repaired → {output_dir}")
        print(f"\nTotal: {total_repaired} files repaired across all categories.")

    if args.results_json:
        args.results_json.write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.results_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

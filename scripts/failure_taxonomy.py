"""Failure-taxonomy sweep: classify every verify-gate failure over a corpus.

Runs every supported-extension file under <root> through the staged pipeline
(in-process — no subprocess per file) for each requested target, classifies
the outcome with ``transpilers.verify.taxonomy``, and emits:

* a markdown rolled-up table — rows are (source-lang, target) pairs, columns
  are failure buckets — printed to stdout (and optionally written via --md);
* an optional per-file CSV (--csv) with one row per (file, target):
  ``file,source_lang,target,bucket,stage,construct,detail``;
* the top offending constructs per bucket, so the dominant gap in each
  (source, target) pair is actionable, not just countable.

Compilation runs only for targets whose compiler is on PATH (rustc, gcc,
mojo, gfortran; python always). Use --no-compile to skip the
compiler gate entirely, --structural to also run the skeleton-isomorphism
gate (issue #45).

Usage:
    uv run python scripts/failure_taxonomy.py [<root>] [--targets rust,mojo]
        [--no-compile] [--structural] [--csv out.csv] [--md out.md]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import Counter, defaultdict

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transpilers.verify.taxonomy import (  # noqa: E402
    BUCKETS,
    TaxonomyRecord,
    classify_unit,
    compiler_available,
)

EXT_MAP = {
    "c": "c", "cpp": "cpp", "cs": "csharp", "f90": "fortran",
    "java": "java", "js": "javascript", "py": "python",
    "ts": "typescript", "vb": "vb",
}


def sweep(
    root: pathlib.Path,
    targets: list[str],
    *,
    compile_outputs: bool,
    structural: bool,
    limit: int | None,
) -> list[tuple[str, TaxonomyRecord]]:
    files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and EXT_MAP.get(f.suffix.lstrip("."))
    ]
    if limit:
        files = files[:limit]
    records: list[tuple[str, TaxonomyRecord]] = []
    for f in files:
        source_lang = EXT_MAP[f.suffix.lstrip(".")]
        text = f.read_text(encoding="utf-8", errors="replace")
        for target in targets:
            rec = classify_unit(
                text,
                source_lang=source_lang,
                target=target,
                compile=compile_outputs,
                structural=structural,
            )
            records.append((str(f.relative_to(root)), rec))
    return records


def rollup_markdown(records: list[tuple[str, TaxonomyRecord]]) -> str:
    pairs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for _, rec in records:
        pairs[(rec.source_lang, rec.target)][rec.bucket] += 1
    used_buckets = [b for b in BUCKETS if any(c[b] for c in pairs.values())]

    lines = ["| source | target | total | " + " | ".join(used_buckets) + " |"]
    lines.append("|" + "---|" * (3 + len(used_buckets)))
    for (src, tgt), counts in sorted(pairs.items()):
        total = sum(counts.values())
        row = [src, tgt, str(total)] + [str(counts[b]) for b in used_buckets]
        lines.append("| " + " | ".join(row) + " |")

    # Top constructs per failure bucket — the actionable part.
    constructs: dict[str, Counter] = defaultdict(Counter)
    for _, rec in records:
        if rec.bucket not in ("ok",) and (rec.construct or rec.detail):
            constructs[rec.bucket][rec.construct or rec.detail] += 1
    if constructs:
        lines.append("")
        lines.append("## Top constructs per bucket")
        for bucket in BUCKETS:
            if bucket not in constructs:
                continue
            lines.append("")
            lines.append(f"### {bucket}")
            for construct, n in constructs[bucket].most_common(10):
                lines.append(f"- {n}× `{construct[:110]}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default="examples/samples", type=pathlib.Path)
    ap.add_argument("--targets", default="rust", help="comma-separated target list")
    ap.add_argument("--no-compile", action="store_true", help="skip the target-compiler gate")
    ap.add_argument("--structural", action="store_true",
                    help="also run the structural-fidelity (skeleton isomorphism) gate")
    ap.add_argument("--csv", type=pathlib.Path, help="write per-file records to this CSV")
    ap.add_argument("--md", type=pathlib.Path, help="write the markdown rollup to this file")
    ap.add_argument("--limit", type=int, help="classify at most N files (smoke runs)")
    args = ap.parse_args(argv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    compile_outputs = not args.no_compile
    if compile_outputs:
        for t in targets:
            if not compiler_available(t):
                print(f"[warn] no {t} compiler on PATH — compile gate skipped for {t}",
                      file=sys.stderr)

    records = sweep(
        args.root, targets,
        compile_outputs=compile_outputs,
        structural=args.structural,
        limit=args.limit,
    )

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "source_lang", "target", "bucket", "stage", "construct", "detail"])
            for name, rec in records:
                w.writerow([name, rec.source_lang, rec.target, rec.bucket,
                            rec.stage, rec.construct, rec.detail])
        print(f"[csv] {len(records)} records -> {args.csv}", file=sys.stderr)

    md = rollup_markdown(records)
    if args.md:
        args.md.write_text(md + "\n", encoding="utf-8")
        print(f"[md] rollup -> {args.md}", file=sys.stderr)
    print(md)

    failures = sum(1 for _, r in records if r.bucket != "ok")
    print(f"\n{len(records) - failures} / {len(records)} ok; {failures} classified failures",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

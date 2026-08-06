"""Scan C++ source tree and build a JSON config for transpile-cpp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CPP_EXTS = {".cc", ".cpp", ".cxx", ".c++"}
HEADER_EXTS = {".hh", ".hpp", ".hxx", ".h"}


def find_header(cc_path: Path) -> Path | None:
    for ext in HEADER_EXTS:
        candidate = cc_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def infer_tier(stem: str, path_lower: str) -> int:
    n = stem.lower()
    if stem.startswith("Data") or n in (
        "constant",
        "datastringglobals",
        "configuredfunctions",
    ):
        return 0
    if n in (
        "vectors",
        "surfaceoctree",
        "psychrometrics",
        "general",
        "curvemanager",
        "convectioncoefficients",
        "windowmanager",
        "tarcoggasses90",
        "fluidproperties",
    ):
        return 1
    if any(
        k in n
        for k in (
            "coil",
            "pump",
            "fan",
            "chiller",
            "boiler",
            "baseboard",
            "radiant",
            "tower",
            "tank",
            "generator",
            "pv",
            "photovolt",
            "battery",
            "heatpump",
            "heater",
        )
    ):
        return 2
    if any(
        k in n
        for k in (
            "manager",
            "balance",
            "simulation",
            "hvac",
            "plant",
            "airloop",
            "zoneequip",
            "branch",
        )
    ):
        return 3
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a JSON config of (source -> target) pairs for transpile-cpp",
    )
    ap.add_argument("source_dir", help="Root directory to scan for C++ source files")
    ap.add_argument(
        "--source-root",
        default=None,
        help="Source prefix to strip from paths (default: source_dir)",
    )
    ap.add_argument(
        "--output", "-o", default=None, help="Write config to file instead of stdout"
    )
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="Subdirectory names to exclude"
    )
    ap.add_argument(
        "--max-lines", type=int, default=0, help="Skip files with more than N lines"
    )
    ap.add_argument(
        "--min-lines", type=int, default=0, help="Skip files with fewer than N lines"
    )
    ap.add_argument(
        "--no-header", action="store_false", dest="header", help="Skip header lookup"
    )
    ap.add_argument("--tier", action="store_true", help="Add tier field for ordering")
    ap.add_argument(
        "--sort", choices=["path", "lines", "tier"], default="path", help="Sort order"
    )
    args = ap.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        print(f"ERROR: source directory not found: {source_dir}", file=sys.stderr)
        return 1

    source_root = Path(args.source_root).resolve() if args.source_root else source_dir

    # Target: same parent, source_dir name + "-Mojo"
    target_root = source_dir.parent / (source_dir.name + "-Mojo")
    exclude_set = set(args.exclude)

    entries = []
    for f in sorted(source_dir.rglob("*")):
        if not f.is_file() or f.suffix not in CPP_EXTS:
            continue
        rel = f.relative_to(source_dir)
        if any(part in exclude_set for part in rel.parts):
            continue
        try:
            lines = f.read_text(errors="ignore").count("\n")
        except OSError:
            continue
        if args.max_lines and lines > args.max_lines:
            continue
        if args.min_lines and lines < args.min_lines:
            continue

        try:
            rel_to_root = f.relative_to(source_root)
        except ValueError:
            rel_to_root = Path(f.name)
        target = target_root / rel_to_root.with_suffix(".mojo")

        entry = {
            "id": len(entries) + 1,
            "source": str(f),
            "target": str(target),
            "decision": "pending",
            "lines": lines,
        }
        if args.header:
            hdr = find_header(f)
            if hdr:
                entry["header"] = str(hdr)
        if args.tier:
            entry["tier"] = infer_tier(f.stem, str(f).lower())
        entries.append(entry)

    if args.sort == "lines":
        entries.sort(key=lambda e: e["lines"])
    elif args.sort == "tier":
        entries.sort(key=lambda e: (e.get("tier", 9), e["lines"]))
    else:
        entries.sort(key=lambda e: e["source"])

    output = json.dumps(entries, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote {len(entries)} entries to {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

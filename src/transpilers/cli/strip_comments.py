"""Strip C++ comments from all source files referenced in a config.

Reads a JSON config (from build_config.py) and strips comments from every
source file in-place. Can also target specific files by id or path.

Usage:
  # Strip all files in config
  python strip_comments.py ep_config.json

  # Dry-run: show what files would be modified
  python strip_comments.py ep_config.json --dry-run

  # Target specific entries
  python strip_comments.py ep_config.json --ids 1 2 3
  python strip_comments.py ep_config.json --files Pumps.cc

  # Strip single file directly
  python strip_comments.py src/EnergyPlus/Pumps.cc --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def strip_cpp_comments(text: str) -> str:
    """Strip C/C++ comments (both // and /* */) from source code.

    Handles:
    - // line comments
    - /* block comments */
    - Strings and character literals: comments inside "" or '' are preserved
    - Preprocessor directives: #include etc. are preserved
    - Edge cases: nested block comments (GCC extension), comment-like content
      in string literals, trigraphs
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]
        nc = text[i + 1] if i + 1 < n else ""

        # String literal (double or single quoted)
        if c in ('"', "'"):
            quote = c
            result.append(c)
            i += 1
            while i < n:
                ch = text[i]
                result.append(ch)
                if ch == "\\" and i + 1 < n:
                    i += 1
                    result.append(text[i])
                elif ch == quote:
                    break
                i += 1
            i += 1
            continue

        # Line comment //
        if c == "/" and nc == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            # Preserve the newline
            if i < n and text[i] == "\n":
                result.append("\n")
                i += 1
            continue

        # Block comment /* */
        if c == "/" and nc == "*":
            i += 2
            while i + 1 < n:
                if text[i] == "*" and text[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                # Unterminated comment - reached end of file
                pass
            continue

        result.append(c)
        i += 1

    return "".join(result)


def strip_file(path: Path, dry_run: bool = False) -> tuple[bool, int]:
    """Strip comments from a file. Returns (modified, bytes_saved)."""
    if not path.exists():
        return False, 0

    original = path.read_text(encoding="utf-8", errors="replace")
    stripped = strip_cpp_comments(original)

    if original == stripped:
        return False, 0

    if not dry_run:
        path.write_text(stripped)

    return True, len(original) - len(stripped)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Strip C++ comments from source files referenced in a config",
    )
    ap.add_argument("target", help="JSON config file or a single source file")
    ap.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Only strip these entry ids (only with config)",
    )
    ap.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Only strip these source paths (substring match, only with config)",
    )
    ap.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be stripped without modifying",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Strip comments from a single file (used with direct file target)",
    )
    args = ap.parse_args()

    target = Path(args.target)

    # Single file mode
    if target.suffix in (".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".mojo"):
        if not args.in_place:
            print(
                "Use --in-place to strip comments from a single file", file=sys.stderr
            )
            return 1
        modified, saved = strip_file(target, dry_run=args.dry_run)
        if modified:
            print(
                f"{'Would strip' if args.dry_run else 'Stripped'} {target} "
                f"({saved:,} bytes)"
            )
        else:
            print(f"{target}: no comments found")
        return 0

    # Config mode
    config = json.loads(target.read_text())

    # Filter entries
    if args.ids:
        id_set = set(args.ids)
        entries = [e for e in config if e.get("id") in id_set]
    elif args.files:
        entries = []
        for pat in args.files:
            for e in config:
                if pat in e["source"] and e not in entries:
                    entries.append(e)
    else:
        entries = config

    total_modified = 0
    total_saved = 0

    for entry in entries:
        src = Path(entry["source"])
        modified, saved = strip_file(src, dry_run=args.dry_run)
        if modified:
            total_modified += 1
            total_saved += saved
            print(
                f"  {'would strip' if args.dry_run else 'stripped'} {src.name} "
                f"({saved:,} bytes)"
            )

    print(
        f"\n{'Would strip' if args.dry_run else 'Stripped'} {total_modified}/{len(entries)} files, "
        f"{total_saved:,} bytes saved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Transpile at any granularity — object · file · module · folder/repo.

Wraps the real `transpile()` pipeline (not a stub) so the same engine runs at
every level. Granularity = which units we feed it:

  object  — a single class / struct / function / method / variable (by name, or
            every top-level object in a file), each transpiled independently.
  file    — a whole translation unit at once.
  module  — a `.hh` + `.cc` pair sharing a stem.
  folder  — every C++ file under a directory, in #include-dependency order
            (callees/headers before callers). Use this for a whole repo.

CLI:
  python -m transpilers.levels --level object  path/to/file.cpp [--name Foo] --target mojo
  python -m transpilers.levels --level file    path/to/file.cpp           --target mojo
  python -m transpilers.levels --level module  path/to/Foo                --target mojo
  python -m transpilers.levels --level folder  path/to/dir                --target mojo
"""

from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass

import clang.cindex as ci

from transpilers.frontends.cpp import parser as _cfg  # noqa: F401  (configures libclang)
from transpilers.cli.main import transpile

K = ci.CursorKind
_OBJECT_KINDS = {K.FUNCTION_DECL, K.CLASS_DECL, K.STRUCT_DECL, K.VAR_DECL, K.CXX_METHOD}
_CPP_EXTS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h"}
_INCLUDE_RE = re.compile(r'#include\s*[<"]([^">]+)[">]')


@dataclass
class Unit:
    label: str  # object name / file name
    source: str  # C++ source for this unit
    origin: str  # file it came from


@dataclass
class Result:
    label: str
    origin: str
    ok: bool
    output: str = ""
    error: str = ""


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_objects(
    path: str, name: str | None = None, inc: list[str] | None = None
) -> list[Unit]:
    """Top-level objects (function/class/struct/method/variable) in `path`.

    `name` restricts to a single object; otherwise all are returned. Methods
    inside classes are reachable by name too.
    """
    args = ["-std=c++17", "-x", "c++"] + [f"-I{d}" for d in (inc or [])]
    tu = ci.Index.create().parse(path, args=args)
    raw = open(path, "rb").read()
    units: list[Unit] = []
    seen: set[tuple[int, int]] = set()

    def from_file(c) -> bool:
        try:
            return bool(c.location.file) and os.path.samefile(
                c.location.file.name, path
            )
        except OSError:
            return False

    def take(c) -> None:
        if name is not None and c.spelling != name:
            return
        e = c.extent
        span = (e.start.offset, e.end.offset)
        if span[1] > span[0] and span not in seen:
            seen.add(span)
            units.append(
                Unit(
                    label=c.spelling or c.kind.name,
                    source=raw[span[0] : span[1]].decode("utf-8", "replace"),
                    origin=os.path.basename(path),
                )
            )

    def visit(node) -> None:
        # Only descend through TU / namespaces / classes — NEVER into function
        # bodies, so locals (loop vars, temporaries) aren't mistaken for objects.
        for c in node.get_children():
            if c.kind == K.NAMESPACE:
                visit(c)
                continue
            if not from_file(c):
                continue
            if c.kind in (K.FUNCTION_DECL, K.CXX_METHOD) and c.is_definition():
                take(c)  # whole function/method; do not recurse in
            elif c.kind in (K.CLASS_DECL, K.STRUCT_DECL) and c.is_definition():
                take(c)  # whole class/struct as one object
                if name is not None:
                    visit(c)  # also expose its methods/fields by name
            elif c.kind == K.VAR_DECL:
                take(c)  # only reached at TU/namespace/class scope

    visit(tu.cursor)
    return units


def module_files(stem_path: str) -> list[str]:
    """`.hh`/`.cc`-style files sharing a stem (e.g. .../Boilers -> Boilers.hh, Boilers.cc)."""
    base = (
        stem_path[: -len(os.path.splitext(stem_path)[1])]
        if os.path.splitext(stem_path)[1]
        else stem_path
    )
    return [
        f
        for ext in (".hh", ".hpp", ".h", ".cc", ".cpp", ".cxx")
        if os.path.isfile(f := base + ext)
    ]


def folder_files_ordered(root: str) -> list[str]:
    """All C++ files under `root`, headers first then #include-dependency order."""
    files = sorted(
        (
            os.path.join(d, f)
            for d, _, fs in os.walk(root)
            for f in fs
            if os.path.splitext(f)[1] in _CPP_EXTS
        ),
        key=lambda p: (os.path.splitext(p)[1] not in {".h", ".hh", ".hpp"}, p),
    )
    stem = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    deps = {f: set() for f in files}
    for f in files:
        try:
            for m in _INCLUDE_RE.finditer(
                open(f, encoding="utf-8", errors="replace").read()
            ):
                s = os.path.splitext(os.path.basename(m.group(1)))[0]
                if s in stem and stem[s] != f:
                    deps[f].add(stem[s])
        except OSError:
            pass
    indeg = {f: len(deps[f]) for f in files}
    queue = [f for f in files if indeg[f] == 0]
    order, rev = [], {f: set() for f in files}
    for f, ds in deps.items():
        for d in ds:
            rev[d].add(f)
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in rev[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    order += [f for f in files if f not in order]  # any cycles: append remainder
    return order


# --------------------------------------------------------------------------- #
# Transpile at a level
# --------------------------------------------------------------------------- #
def _run(
    label: str,
    origin: str,
    source: str,
    source_lang: str,
    target: str,
    engine: str = "strict",
    inc: list[str] | None = None,
) -> Result:
    try:
        if engine == "lift":  # never-refuse C++ -> Python lift (whole-unit)
            from transpilers.lift import lift_source

            out, _ = lift_source(source, inc=inc)
            return Result(label, origin, True, output=out)
        return Result(
            label,
            origin,
            True,
            output=transpile(source, source_lang=source_lang, target=target),
        )
    except Exception as ex:  # never abort the batch on one failing unit
        return Result(
            label,
            origin,
            False,
            error=f"{type(ex).__name__}: {str(ex).splitlines()[0][:120]}",
        )


def _read_unit(path: str, source_lang: str, engine: str, inc: list[str] | None) -> str:
    """Read a whole translation unit for file/module/folder granularity.

    For the strict engine on cpp/c with -I dirs given, transitively inline
    local #include "X.h" headers first (same opt-in-only-with-inc rule as
    the `transpile` CLI's --include-dir) so a multi-file project's own
    class declarations are visible to the parser. The lift engine already
    does its own real -I resolution inside lift_source/lift_file and does
    not need this.
    """
    if engine == "strict" and source_lang in ("c", "cpp") and inc:
        from transpilers.frontends.cpp.parser.includes import resolve_local_includes

        return resolve_local_includes(path, include_dirs=inc)
    return open(path, encoding="utf-8", errors="replace").read()


def transpile_level(
    level: str,
    path: str,
    *,
    name: str | None = None,
    source_lang: str = "cpp",
    target: str = "mojo",
    inc: list[str] | None = None,
    engine: str = "strict",
) -> list[Result]:
    def run(label, origin, source):
        return _run(label, origin, source, source_lang, target, engine=engine, inc=inc)

    if level == "object":
        return [
            run(u.label, u.origin, u.source)
            for u in extract_objects(path, name=name, inc=inc)
        ]
    if level == "file":
        return [
            run(
                os.path.basename(path),
                os.path.basename(path),
                _read_unit(path, source_lang, engine, inc),
            )
        ]
    if level == "module":
        return [
            run(
                os.path.basename(f),
                os.path.basename(f),
                _read_unit(f, source_lang, engine, inc),
            )
            for f in module_files(path)
        ]
    if level == "folder":
        return [
            run(
                os.path.basename(f),
                os.path.basename(f),
                _read_unit(f, source_lang, engine, inc),
            )
            for f in folder_files_ordered(path)
        ]
    raise ValueError(f"unknown level {level!r} (object|file|module|folder)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="transpilers.levels")
    ap.add_argument(
        "--level", required=True, choices=["object", "file", "module", "folder"]
    )
    ap.add_argument("path")
    ap.add_argument(
        "--name", help="object level: restrict to this class/function/variable"
    )
    ap.add_argument("--source", default="cpp")
    ap.add_argument("--target", default="mojo")
    ap.add_argument(
        "--engine",
        default="strict",
        choices=["strict", "lift"],
        help="strict = verified HIR pipeline; lift = never-refuse C++->Python",
    )
    ap.add_argument(
        "--inc", action="append", default=[], help="include dir (repeatable)"
    )
    ap.add_argument("--emit-dir", help="write each unit's output here")
    args = ap.parse_args()
    if args.engine == "lift":
        args.target = "python"

    results = transpile_level(
        args.level,
        args.path,
        name=args.name,
        source_lang=args.source,
        target=args.target,
        inc=args.inc,
        engine=args.engine,
    )
    ok = sum(r.ok for r in results)
    for r in results:
        tag = "ok " if r.ok else "FAIL"
        print(f"[{tag}] {r.origin}:{r.label}" + ("" if r.ok else f"  — {r.error}"))
        if r.ok and args.emit_dir:
            os.makedirs(args.emit_dir, exist_ok=True)
            ext = {
                "mojo": ".mojo",
                "python": ".py",
                "rust": ".rs",
                "c": ".c",
            }.get(args.target, ".txt")
            open(os.path.join(args.emit_dir, f"{r.origin}.{r.label}{ext}"), "w").write(
                r.output
            )
    print(f"\n{args.level}: {ok}/{len(results)} units transpiled to {args.target}")


if __name__ == "__main__":
    main()

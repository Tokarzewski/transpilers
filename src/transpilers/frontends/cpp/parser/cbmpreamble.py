"""Recover cross-file C++ class/method/macro surface from a
`codebase-memory-mcp` knowledge graph and emit a parser-preamble shim.

Background — issue #79 / the Topologic stress-test
------------------------------------------------
The strict C++ frontend (``core.parse_cpp``) *deliberately* strips every
``#include`` and has no header-resolution mechanism, so a bare method
definition file (``namespace X { Ret Class::Method(...) {...} }``) is
parsed with no knowledge of the ``Class`` it belongs to. libclang then
emits ``use of undeclared identifier`` for the class, and parsing fails
at 100% of files on a real multi-file C++ corpus (0/51 for Topologic).
This is by design — matches the self-contained ``examples/`` corpus and
the EnergyPlus "god-object flattening" workflow — but it means arbitrary
multi-file C++ repos need manual per-artifact flattening before the tool
can touch them.

``codebase-memory-mcp`` (the user's own fork) ships a whole-repo AST +
Hybrid-LSP knowledge graph (158 tree-sitter grammars + cross-file type
resolution). Its ``Method`` nodes carry ``param_types`` /
``return_type`` / ``parent_class``; its ``Class`` nodes carry
``base_classes``; its ``Macro`` nodes carry the ``SCREAMING_CASE`` export
macros (``TOPOLOGIC_API`` / ``DLLEXPORT`` / ``WINAPI``). That is exactly
the declaration surface the frontend's ``#include``-stripping throws away.

This module queries that graph per-file and emits a **minimal C++
preamble shim** — class/struct forward declarations, the in-scope method
signatures (so out-of-line ``Class::Method`` defs resolve), and
``#define`` neutralizations for unseen export macros. ``parse_cpp`` already
prepends ``$TRANSPILERS_CPP_PREAMBLE_FILE`` content *before* the user
source (see ``core._project_preamble``), so writing that file is a strict
opt-in: the existing parse path is untouched unless the caller points us at
an indexed graph.

The signed/unsigned overload case (issue #80)
---------------------------------------------
The frontend's ``CPP_TYPE_ALIASES`` table collapses every integer width /
signedness spelling (``int``, ``unsigned int``, ``long``, ``uint8_t`` ...)
onto a single ``"int"``. So ``Bitwise::NOT(int)`` and
``Bitwise::NOT(unsigned int)`` both become ``(Int) -> Int`` in Mojo and the
backend emits two methods with the identical signature -> a guaranteed
duplicate-definition compile error. The fix lives at the *data* layer: we
recover the real, distinct ``param_types`` from the graph and emit both
overloads verbatim, so the overload distinction survives into the HIR
(where a later pass can decide whether Mojo needs ``Int`` vs ``UInt`` or a
rename). ``payload_to_cpp`` is written so that two methods with the same
name but different ``param_types`` produce two *distinct* declarations —
the regression test in ``tests/test_cbm_preamble.py`` pins this.

The module is deliberately split into a **pure** mapping
(``payload_to_cpp`` / ``_decls_from_payload``) that needs no binary, plus a
thin shell-out layer (``cbm_query`` / ``build_preamble_payload``) that
talks to the ``codebase-memory-mcp`` CLI bridge. Unit tests exercise the
pure half; the integration half is gated on the binary + an indexed repo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

# Default location of the codebase-memory-mcp binary. Mirrors the path
# scripts/sft/cbm_graph.py already expects, so a user who has the CLI
# bridge working for the SFT pipeline gets this helper for free.
CBM_BIN: Final[str] = os.environ.get(
    "CBM_BIN", str(Path.home() / ".local/bin/codebase-memory-mcp")
)
# cbm normalizes a repo at C:/Github/foo into the project name
# "C-Github-foo". We reproduce that so callers can pass a repo root and
# we resolve the right graph without them memorizing the slug.
_CBM_PROJECT_SEP: Final[str] = "-"

# --------------------------------------------------------------------------
# OCCT (OpenCASCADE) type-shim layer — issue #79's "OCCT wall"
# --------------------------------------------------------------------------
# Topologic (and other OCCT-based C++) projects define their export macro in
# the same header that also drags in an OCCT type (see docs/occt_preamble.hpp
# and docs/topologic_migration.md). When --cbm resolves that header's class,
# every file that only wanted the macro inherits a `TopoDS_Shape` / `TopAbs_*`
# type it doesn't use, and libclang fails on the OCCT name. Rather than ask
# the user to hand-point TRANSPILERS_CPP_PREAMBLE_FILE at docs/occt_preamble.hpp,
# we auto-detect OCCT-typed references from the graph payload and append the
# opaque shim below. It is NOT a real binding — just enough for libclang to
# parse past the type (matching the scope documented in docs/occt_preamble.hpp).
OCCT_PREFIXES: Final[tuple[str, ...]] = (
    "TopoDS_",
    "TopAbs_",
    "BRep",
    "BRepBuilderAPI_",
    "Geom_",
    "GeomAPI_",
    "gp_",
    "GC_Make",
    "Handle_",
    "TopTools_",
    "TopExp_",
    "TopLoc_",
    "Poly_",
    "BOPAlgo_",
    "IntTools_",
    "ShapeAnalysis_",
    "STEPControl_",
)
# Opaque shim mirroring docs/occt_preamble.hpp — broad enough for libclang to
# parse, never semantically accurate.
OCCT_SHIM: Final[str] = """\
class TopoDS_TShape_Handle { public: void* operator->() const; };
class TopoDS_Shape { public: TopoDS_TShape_Handle TShape() const; };
enum TopAbs_ShapeEnum {
    TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL,
    TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SHAPE
};
"""


def _collect_occt_types(payload: dict) -> list[str]:
    """Return the set of OCCT type names referenced by a graph payload.

    Scans the recovered method param/return types, the recovered classes,
    and an explicit ``type_refs`` list (populated by ``build_preamble_payload``
    from the graph's TypeRef/USAGE edges). A bare prefix match is enough — we
    only need to know *that* an OCCT type appears so we can append the shim.
    """
    found: list[str] = []
    seen: set[str] = set()

    def consider(name: str | None) -> None:
        if not name:
            return
        # Strip pointer/ref qualifiers and template brackets for prefix test.
        bare = name.replace("*", "").replace("&", "").strip()
        if any(bare.startswith(p) for p in OCCT_PREFIXES) and bare not in seen:
            seen.add(bare)
            found.append(bare)

    for m in payload.get("methods", []) or []:
        consider(m.get("return_type"))
        for p in m.get("param_types", []) or []:
            consider(p)
    for c in payload.get("classes", []) or []:
        consider(c.get("name"))
    for t in payload.get("type_refs", []) or []:
        consider(t if isinstance(t, str) else t.get("type_name"))
    return found


def _project_slug(repo_root: str) -> str:
    """Turn a repo root path into cbm's normalized project name.

    cbm lowercases the drive letter, swaps path separators for ``-``, and
    drops the leading separator. ``C:/Github/transpilers`` ->
    ``C-Github-transpilers``. A caller-supplied slug passes through.
    """
    p = os.path.normpath(repo_root)
    # Drop a trailing separator so it doesn't become a trailing '-'.
    p = p.rstrip("/\\")
    slug = (
        p.replace(":", "")
        .replace("/", _CBM_PROJECT_SEP)
        .replace("\\", _CBM_PROJECT_SEP)
    )
    return slug.lower()


# --------------------------------------------------------------------------
# Pure payload -> C++ preamble mapping (no binary, fully unit-testable)
# --------------------------------------------------------------------------


def _normalize_param_type(text: str | None) -> str:
    """Map one cbm ``param_types`` entry to a C++ type spelling.

    cbm stores param types as free text (often qualified: ``unsigned int``,
    ``std::string``, ``const T&``). We keep them *verbatim* — the whole
    point of this shim is to recover what ``#include``-stripping lost, so
    re-collapsing them through ``CPP_TYPE_ALIASES`` would re-introduce the
    #80 overload bug. ``None``/empty falls back to ``int`` (libclang's
    default when it can't infer a primitive parameter type).
    """
    if not text:
        return "int"
    return text.strip()


def _escape_ident(name: str) -> str:
    """Guard against empty / non-identifier names from the graph."""
    name = (name or "").strip()
    if not name or not name[0].isalpha() and name[0] != "_":
        return f"anon_{abs(hash(name))}"
    return name


def _decls_from_payload(payload: dict) -> list[str]:
    """Build the per-file preamble declaration list from a graph payload.

    ``payload`` shape (all keys optional; produced by
    ``build_preamble_payload``)::

        {
          "classes": [            # Class/Struct nodes referenced
              {"name": "Bitwise", "base_classes": ["Base"], ...},
          ],
          "methods": [            # Method nodes (out-of-line defs get resolved)
              {"name": "NOT", "parent_class": "Bitwise",
               "param_types": ["int"], "return_type": "int"},
              {"name": "NOT", "parent_class": "Bitwise",
               "param_types": ["unsigned int"], "return_type": "unsigned int"},
          ],
          "macros": [             # SCREAMING_CASE export / calling-conv macros
              {"name": "TOPOLOGIC_API"},
          ],
          "namespaces": ["X"],    # enclosing namespace, forward-decl'd
        }

    The key invariant: methods are emitted *one declaration per distinct
    (parent_class, name, tuple(param_types))* triple, so two overloads that
    differ only by signedness produce two distinct signatures — the #80 fix.
    """
    lines: list[str] = []

    # 1. Namespace forward declarations (so `namespace X { ... }` user
    #    code resolves even though we stripped its header).
    for ns in payload.get("namespaces", []) or []:
        ns = _escape_ident(ns)
        lines.append(f"namespace {ns} {{}}")

    # 2/3. Emit ONE declaration per referenced class. If the graph gave
    #     us method signatures for that class, we emit a single `class C {
    #     public: ... };` carrying those (overload-preserving) member
    #     declarations — that doubles as the class declaration, so an
    #     out-of-line `C::Method` definition resolves. If we have *no*
    #     methods for the class (it is merely referenced, e.g. as a
    #     parameter type), we emit a bare `class C;` forward declaration
    #     instead. We never emit both, or libclang errors on a
    #     redefinition of `C`. Bases are recorded as comments only — we
    #     must not *redefine* a base the user source may also declare.
    classes = payload.get("classes", []) or []
    methods = payload.get("methods", []) or []
    # stable order: by (parent, name, param_types)
    methods = sorted(
        methods,
        key=lambda m: (
            m.get("parent_class") or "",
            m.get("name") or "",
            tuple(m.get("param_types") or []),
        ),
    )
    methods_by_class: dict[str, list] = {}
    for m in methods:
        pc = m.get("parent_class") or ""
        if pc:
            methods_by_class.setdefault(pc, []).append(m)

    emitted: set[tuple] = set()
    for cls in classes:
        cname = _escape_ident(cls.get("name"))
        method_strs: list[str] = []
        for m in methods_by_class.get(cname, []):
            name = _escape_ident(m.get("name"))
            params = m.get("param_types") or []
            key = (cname, name, tuple(params))
            if key in emitted:
                continue
            emitted.add(key)
            # Name the parameters p0..pn purely for a parseable signature.
            param_str = ", ".join(
                f"{_normalize_param_type(p)} p{i}" for i, p in enumerate(params)
            )
            ret = _normalize_param_type(m.get("return_type")) or "void"
            method_strs.append(f"{ret} {name}({param_str});")
        if method_strs:
            bases = cls.get("base_classes") or []
            bases = [b for b in bases if b and b != cname]
            if bases:
                base_list = ", ".join(_escape_ident(b) for b in bases)
                lines.append(f"// base_classes: {base_list}")
            lines.append(f"class {cname} {{ public: {' '.join(method_strs)} }};")
        else:
            lines.append(f"class {cname};")

    # 3b. Free (non-member) functions recovered from the graph — declared
    #     at namespace/TU scope so a definition in this file resolves.
    for m in methods:
        parent = m.get("parent_class") or ""
        if parent:
            continue
        name = _escape_ident(m.get("name"))
        params = m.get("param_types") or []
        key = ("", name, tuple(params))
        if key in emitted:
            continue
        emitted.add(key)
        param_str = ", ".join(
            f"{_normalize_param_type(p)} p{i}" for i, p in enumerate(params)
        )
        ret = _normalize_param_type(m.get("return_type")) or "void"
        lines.append(f"{ret} {name}({param_str});")
    return lines


def payload_to_cpp(payload: dict) -> str:
    """Render a graph payload as a full C++ preamble shim string.

    Pure function — no filesystem, no binary. The result is suitable for
    writing to ``$TRANSPILERS_CPP_PREAMBLE_FILE`` (consumed by
    ``core._project_preamble``) or for prepending directly.
    """
    decls = _decls_from_payload(payload)
    # Macro neutralization: cbm's Macro nodes are export / calling-convention
    # macros defined in a header we never see. Emit a ``#define X `` so the
    # preprocessor deletes them before libclang parses (mirrors the
    # ``-DNAME=`` retry loop in core.parse_cpp). We only neutralize names
    # that are SCREAMING_CASE or carry a known macro suffix — never real
    # type names.
    macro_lines: list[str] = []
    for mac in payload.get("macros", []) or []:
        mname = (mac.get("name") or "").strip()
        if (
            not mname
            or not mname.isupper()
            and not mname.endswith(("_EXPORT", "_IMPORT", "_API", "_DLL", "_DECL"))
        ):
            continue
        macro_lines.append(f"#define {mname} ")
    if not decls and not macro_lines and not _collect_occt_types(payload):
        return ""
    out: list[str] = ["// --- codebase-memory-mcp recovered preamble ---"]
    out.extend(macro_lines)
    # OCCT wall (issue #79): if the graph shows this file references an
    # OpenCASCADE type, the opaque shim MUST come *before* the recovered
    # class decls (a class method can return the OCCT type, so the type
    # must be defined first, or libclang fails with "unknown type name").
    # No-op unless OCCT types are detected.
    occt_types = _collect_occt_types(payload)
    if occt_types:
        out.append("// --- OCCT (OpenCASCADE) opaque shim (auto-detected) ---")
        out.append(OCCT_SHIM.rstrip("\n"))
        out.append(f"// referenced OCCT types: {', '.join(sorted(occt_types))}")
    out.extend(decls)
    out.append("// --- end recovered preamble ---")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Shell-out layer: query the codebase-memory-mcp graph
# --------------------------------------------------------------------------


def cbm_query(project: str, cypher: str, binary: str = CBM_BIN) -> list[dict]:
    """Run one openCypher query through the cbm CLI bridge; return row dicts.

    cbm's CLI prints ``level=info ...`` log lines before the JSON payload,
    so we take the first line that parses as a JSON object and read its
    ``rows`` against ``columns``. Returns ``[]`` on any failure (no binary,
    non-zero exit, unparseable output) so callers can degrade gracefully to
    the existing no-preamble parse path.
    """
    if not binary or not shutil.which(binary) and not os.path.isfile(binary):
        return []
    try:
        proc = subprocess.run(
            [
                binary,
                "cli",
                "query_graph",
                json.dumps({"project": project, "query": cypher}),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        return []
    if proc.returncode != 0:
        return []
    payload = _extract_json(proc.stdout)
    if not payload:
        return []
    cols = payload.get("columns", [])
    return [dict(zip(cols, row)) for row in payload.get("rows", [])]


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of cbm CLI stdout."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def build_preamble_payload(
    repo_root: str,
    rel_path: str,
    project: str | None = None,
    binary: str = CBM_BIN,
) -> dict:
    """Query the cbm graph for the declaration surface a file needs.

    ``rel_path`` is the file relative to ``repo_root`` (e.g.
    ``src/Bitwise.cpp``). We resolve the enclosing namespace, the classes
    referenced from that file, and — for any class the file defines methods
    of — the full method signature set (param/return types), plus any
    SCREAMING_CASE export macros declared in that file.

    Returns the payload dict consumed by ``payload_to_cpp``. Falls back to
    ``{"classes": [], "methods": [], "macros": [], "namespaces": []}`` when
    the binary / graph is unavailable.
    """
    empty: dict = {"classes": [], "methods": [], "macros": [], "namespaces": []}
    if not binary or (not shutil.which(binary) and not os.path.isfile(binary)):
        return empty
    proj = project or _project_slug(repo_root)

    # Namespaces the file lives in.
    ns_rows = (
        cbm_query(
            proj,
            "MATCH (f:File {file_path: $p})<-[:CONTAINS*0..]-(ns:Namespace) "
            "RETURN ns.name AS name",
            binary,
        )
        if False
        else cbm_query(  # parameterized form unsupported by cli; inline rel_path
            proj,
            f"MATCH (f:File)<-[:CONTAINS*0..]-(ns:Namespace) "
            f"WHERE f.file_path ENDS WITH '{rel_path}' RETURN ns.name AS name",
            binary,
        )
    )
    namespaces = sorted({r.get("name") for r in ns_rows if r.get("name")})

    # Classes whose methods are defined in this file (out-of-line defs).
    class_rows = cbm_query(
        proj,
        f"MATCH (c:Class)<-[:DEFINES_METHOD]-(m:Method)<-[:DEFINES]-(f:File) "
        f"WHERE f.file_path ENDS WITH '{rel_path}' "
        f"RETURN DISTINCT c.name AS name, c.base_classes AS base_classes",
        binary,
    )
    classes = []
    method_rows: list[dict] = []
    for r in class_rows:
        cname = r.get("name")
        if not cname:
            continue
        bases = r.get("base_classes") or []
        bases = bases if isinstance(bases, list) else [bases]
        classes.append({"name": cname, "base_classes": bases})
        # Pull the full method signature set for this class (across the repo,
        # so we recover overloads the file itself may not contain).
        mr = cbm_query(
            proj,
            f"MATCH (c:Class {{name: '{cname}'}})-[:DEFINES_METHOD]->(m:Method) "
            f"RETURN m.name AS name, m.param_types AS param_types, "
            f"m.return_type AS return_type, m.parent_class AS parent_class",
            binary,
        )
        method_rows.extend(mr)

    # SCREAMING_CASE export / calling-convention macros in this file.
    macro_rows = cbm_query(
        proj,
        f"MATCH (f:File)<-[:DEFINES]-(mac:Macro) "
        f"WHERE f.file_path ENDS WITH '{rel_path}' "
        f"RETURN mac.name AS name",
        binary,
    )
    macros = [{"name": r.get("name")} for r in macro_rows if r.get("name")]

    # Type references the file makes (USAGE edges to types / TypeRef nodes)
    # — used to auto-detect OCCT (OpenCASCADE) types for the opaque shim.
    # Best-effort: the schema may expose these as USAGE edges with a `callee`
    # of type form, or as a TypeRef label; either way missing results just
    # mean no OCCT detection from this channel (method/return types already
    # cover the common cases).
    type_ref_rows = cbm_query(
        proj,
        f"MATCH (f:File)<-[:USAGE]-(u:Usage) "
        f"WHERE f.file_path ENDS WITH '{rel_path}' "
        f"RETURN u.ref_name AS type_name",
        binary,
    ) or cbm_query(
        proj,
        f"MATCH (f:File)<-[:USAGE]-(t:TypeRef) "
        f"WHERE f.file_path ENDS WITH '{rel_path}' "
        f"RETURN t.type_name AS type_name",
        binary,
    )
    type_refs = [r.get("type_name") for r in type_ref_rows if r.get("type_name")]

    return {
        "classes": classes,
        "methods": method_rows,
        "macros": macros,
        "namespaces": namespaces,
        "type_refs": type_refs,
    }


def write_preamble_for_file(
    repo_root: str,
    rel_path: str,
    out_path: str,
    project: str | None = None,
    binary: str = CBM_BIN,
) -> str | None:
    """End-to-end: query the graph and write a preamble shim file.

    Returns the written path on success, or ``None`` if there was nothing to
    emit (no binary / empty graph / no relevant declarations). The caller is
    expected to point ``$TRANSPILERS_CPP_PREAMBLE_FILE`` at ``out_path``
    before invoking ``parse_cpp``.
    """
    payload = build_preamble_payload(repo_root, rel_path, project, binary)
    cpp = payload_to_cpp(payload)
    if not cpp:
        return None
    Path(out_path).write_text(cpp, encoding="utf-8")
    return out_path


__all__ = [
    "CBM_BIN",
    "cbm_query",
    "build_preamble_payload",
    "payload_to_cpp",
    "write_preamble_for_file",
]

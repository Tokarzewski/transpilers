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

``codebase-memory-mcp`` ships a whole-repo AST + Hybrid-LSP knowledge graph
(tree-sitter + cross-file type resolution). Its ``Method`` nodes carry
``param_types`` / ``return_type`` / ``parent_class`` / ``signature``; its
``Class`` nodes carry ``base_classes`` and a ``qualified_name`` that embeds
the enclosing namespace; its ``Macro`` nodes carry the ``SCREAMING_CASE``
export macros (``TOPOLOGIC_API`` / ``DLLEXPORT`` / ``WINAPI``). That is
exactly the declaration surface the frontend's ``#include``-stripping
throws away.

This module queries that graph per-file and emits a **minimal C++
preamble shim** — class/struct declarations placed *inside their real
namespace* (so an out-of-line ``TopologicUtilities::Bitwise::NOT`` def
resolves rather than failing with "namespace 'TopologicUtilities' does not
enclose namespace 'Bitwise'"), the in-scope method signatures (so
out-of-line ``Class::Method`` defs resolve), and ``#define``
neutralizations for unseen export macros. ``parse_cpp`` already prepends
``$TRANSPILERS_CPP_PREAMBLE_FILE`` content *before* the user source (see
``core._project_preamble``), so writing that file is a strict opt-in: the
existing parse path is untouched unless the caller points us at an indexed
graph.

The signed/unsigned overload case (issue #80)
---------------------------------------------
The frontend's ``CPP_TYPE_ALIASES`` table collapses every integer width /
signedness spelling (``int``, ``unsigned int``, ``long``, ``uint8_t`` ...)
onto a single ``"int"``. So ``Bitwise::NOT(int)`` and
``Bitwise::NOT(unsigned int)`` both become ``(Int) -> Int`` in Mojo and the
backend emits two methods with the identical signature -> a guaranteed
duplicate-definition compile error. The fix lives at the *data* layer: we
recover the real, distinct ``param_types`` / ``signature`` from the graph
and emit both overloads verbatim, so the overload distinction survives.
``payload_to_cpp`` is written so that two methods with the same name but
different parameter lists produce two *distinct* declarations — the
regression test in ``tests/test_cbm_preamble.py`` pins this.

The module is deliberately split into a **pure** mapping
(``payload_to_cpp`` / ``_decls_from_payload``) that needs no binary, plus a
thin shell-out layer (``cbm_query`` / ``build_preamble_payload``) that
talks to the ``codebase-memory-mcp`` CLI bridge. Unit tests exercise the
pure half; the integration half is gated on the binary + an indexed repo.

Target binary / schema (modern ``DeusData/codebase-memory-mcp``)
----------------------------------------------------------------
This module talks to the *current* ``codebase-memory-mcp`` CLI (>= 0.10),
not the stale fork it was originally written against. Two things changed
and are handled here:

* **Output format**: the CLI ``query_graph`` tool prints a human-readable
  table (``rows: N (cols: ...)`` then space-joined, JSON-quoted values),
  not a ``{"columns": [...], "rows": [...]}`` JSON blob. ``cbm_query``
  parses that table (shlex-splitting each data line so quoted values — e.g.
  a JSON-encoded ``param_types`` array or a ``unsigned int`` return type —
  survive).

* **Graph schema**: out-of-line method definitions are linked by
  ``(File)-[:DEFINES]->(Method)`` with the method's ``parent_class`` being a
  *path-qualified* name (``{project}.{dirs}.{stem}.{Class}``). There is no
  ``Namespace`` node and ``DEFINES_METHOD`` is absent for out-of-line
  classes, so the enclosing namespace is recovered from
  ``Class.qualified_name`` (``{project}.{dirs}.{stem}.{Namespace}.{Class}``)
  and the class declaration is emitted *inside* that namespace.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Final

# Default location of the codebase-memory-mcp binary. Mirrors the path
# scripts/sft/cbm_graph.py already expects, so a user who has the CLI
# bridge working for the SFT pipeline gets this helper for free.
_CBM_BIN_DEFAULT = Path.home() / ".local" / "bin" / "codebase-memory-mcp"
if os.name == "nt":
    _CBM_BIN_DEFAULT = _CBM_BIN_DEFAULT.with_suffix(".exe")
CBM_BIN: Final[str] = os.environ.get("CBM_BIN", str(_CBM_BIN_DEFAULT))
# cbm normalizes a repo at C:/Github/foo into the project name
# "C-Github-foo" (case-preserving). We reproduce that so callers can pass a
# repo root and we resolve the right graph without them memorizing the slug.
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

    cbm swaps path separators for ``-``, drops the drive colon, and
    *preserves* case: ``C:/Users/amd/Documents/GitHub/Topologic`` ->
    ``C-Users-amd-Documents-GitHub-Topologic``. A caller-supplied slug passes
    through unchanged (the ``project=`` override). The earlier fork lowercased
    the whole slug; the modern binary does not, so we must not either.
    """
    p = os.path.normpath(repo_root)
    # Drop a trailing separator so it doesn't become a trailing '-'.
    p = p.rstrip("/\\")
    slug = (
        p.replace(":", "")
        .replace("/", _CBM_PROJECT_SEP)
        .replace("\\", _CBM_PROJECT_SEP)
    )
    return slug


def _as_list(value) -> list:
    """Coerce a graph value (JSON-array string / list / scalar / ``-``) to a
    list. The CLI table renders empty arrays as ``-`` and real arrays as
    their JSON string form (e.g. ``["std::list"]``), so both need unpacking.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s in ("", "-"):
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [s]
    return [value]


def _namespace_of(qualified_name: str | None, file_path: str | None) -> str:
    """Recover a class's enclosing C++ namespace from its graph ``qualified_name``.

    The modern binary stores ``Class.qualified_name`` as
    ``{project}.{dirs}.{stem}.{Namespace}.{Class}`` for namespaced classes and
    ``{project}.{dirs}.{stem}.{Class}`` for global ones, while ``file_path`` is
    ``{dirs}/{stem}.{ext}``. Strip the project+path prefix and read the segment
    left of the class name; return ``""`` when the class is at global scope.
    """
    if not qualified_name:
        return ""
    file_path = (file_path or "").replace("\\", "/")
    path_dots = os.path.splitext(file_path)[0].replace("/", ".")
    marker = path_dots + "."
    idx = qualified_name.find(marker)
    if idx == -1:
        # Path stem not present verbatim (odd project name): fall back to the
        # second-to-last dot segment, but only if it isn't the bare stem.
        parts = qualified_name.split(".")
        if len(parts) >= 3:
            stem = os.path.splitext(os.path.basename(file_path))[0]
            if parts[-2] and parts[-2] != stem and parts[-2] != parts[-1]:
                return parts[-2]
        return ""
    tail = qualified_name[idx + len(marker):]
    parts = tail.split(".")
    if len(parts) >= 2 and parts[-2] and parts[-2] != parts[-1]:
        return parts[-2]
    return ""


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


def _method_decl(method: dict, emitted: set, cname: str, key: str) -> str | None:
    """Render one method as a C++ declaration line, or ``None`` if it is a
    duplicate of an already-emitted (parent, name, params) triple.

    Prefers the graph's full ``signature`` (e.g. ``(const unsigned int x)``)
    when present, falling back to reconstructing from ``param_types`` —
    which keeps the pure mapping deterministic for the unit tests that don't
    supply a signature.
    """
    name = _escape_ident(method.get("name"))
    params = _as_list(method.get("param_types"))
    sig = (method.get("signature") or "").strip()
    if sig:
        dedup_key = (key, name, sig)
    else:
        dedup_key = (key, name, tuple(params))
    if dedup_key in emitted:
        return None
    emitted.add(dedup_key)
    if sig:
        param_str = sig if sig.startswith("(") else f"({sig})"
    else:
        param_str = "(" + ", ".join(
            f"{_normalize_param_type(p)} p{i}" for i, p in enumerate(params)
        ) + ")"
    ret = _normalize_param_type(method.get("return_type")) or "void"
    return f"{ret} {name}{param_str};"


def _decls_from_payload(payload: dict) -> list[str]:
    """Build the per-file preamble declaration list from a graph payload.

    ``payload`` shape (all keys optional; produced by
    ``build_preamble_payload``)::

        {
          "classes": [            # Class/Struct nodes referenced
              {"name": "Bitwise", "base_classes": ["Base"],
               "namespace": "TopologicUtilities", ...},
          ],
          "methods": [            # Method nodes (out-of-line defs get resolved)
              {"name": "NOT", "parent_class": "Bitwise",
               "param_types": ["int"], "return_type": "int",
               "signature": "(const int kArgument1)"},
              {"name": "NOT", "parent_class": "Bitwise",
               "param_types": ["unsigned int"], "return_type": "unsigned int"},
          ],
          "macros": [             # SCREAMING_CASE export / calling-conv macros
              {"name": "TOPOLOGIC_API"},
          ],
          "namespaces": ["X"],    # enclosing namespace(s), forward-decl'd
        }

    Classes are emitted *inside* their namespace (recovered from
    ``Class.qualified_name``) so an out-of-line def inside a real namespace
    block resolves. The key invariant on methods holds: one declaration per
    distinct (parent_class, name, params) triple, so two overloads differing
    only by signedness produce two distinct signatures — the #80 fix.
    """
    lines: list[str] = []

    classes = payload.get("classes", []) or []
    methods = payload.get("methods", []) or []
    # stable order: by (parent, name, param_types)
    methods = sorted(
        methods,
        key=lambda m: (
            m.get("parent_class") or "",
            m.get("name") or "",
            tuple(_as_list(m.get("param_types"))),
        ),
    )
    methods_by_class: dict[str, list] = {}
    for m in methods:
        pc = m.get("parent_class") or ""
        if pc:
            methods_by_class.setdefault(pc, []).append(m)

    emitted: set = set()

    def render_class(cls: dict) -> str:
        cname = _escape_ident(cls.get("name"))
        method_lines: list[str] = []
        for m in methods_by_class.get(cname, []):
            decl = _method_decl(m, emitted, cname, f"cls:{cname}")
            if decl:
                method_lines.append(f"  {decl}")
        if method_lines:
            bases = _as_list(cls.get("base_classes"))
            bases = [b for b in bases if b and str(b) != cname]
            head: list[str] = []
            if bases:
                base_list = ", ".join(_escape_ident(str(b)) for b in bases)
                head.append(f"// base_classes: {base_list}")
            inner = "\n".join(head + ["  public:"] + method_lines)
            return f"class {cname} {{\n{inner}\n}};"
        return f"class {cname};"

    # Group classes by their namespace; "" = global scope.
    global_classes: list[str] = []
    ns_blocks: dict[str, list[str]] = {}
    for cls in classes:
        ns = (cls.get("namespace") or "").strip()
        decl = render_class(cls)
        if ns:
            ns_blocks.setdefault(_escape_ident(ns), []).append(decl)
        else:
            global_classes.append(decl)

    # Explicit namespaces with no class content still get a bare fwd-decl.
    explicit_ns = {_escape_ident(n) for n in payload.get("namespaces", []) or []}
    for ns in explicit_ns:
        if ns not in ns_blocks:
            lines.append(f"namespace {ns} {{}}")

    # Namespace-wrapped class declarations (correct scoping for out-of-line
    # defs that live inside a real namespace block).
    for ns, decls in ns_blocks.items():
        lines.append(f"namespace {ns} {{")
        lines.extend(decls)
        lines.append("}")

    # Global-scope classes.
    lines.extend(global_classes)

    # Free (non-member) functions recovered from the graph — declared at
    # namespace/TU scope so a definition in this file resolves.
    for m in methods:
        parent = m.get("parent_class") or ""
        if parent:
            continue
        decl = _method_decl(m, emitted, "", "free")
        if decl:
            lines.append(decl)
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
    occt_types = _collect_occt_types(payload)
    if not decls and not macro_lines and not occt_types:
        return ""

    # Stub part (parsed *before* PARSER_PREAMBLE, and excluded from output):
    # macro neutralizations and the OCCT opaque shim — neither references std.
    stub: list[str] = ["// --- codebase-memory-mcp recovered preamble ---"]
    stub.extend(macro_lines)
    if occt_types:
        stub.append("// --- OCCT (OpenCASCADE) opaque shim (auto-detected) ---")
        stub.append(OCCT_SHIM.rstrip("\n"))
        stub.append(f"// referenced OCCT types: {', '.join(sorted(occt_types))}")

    # Real part (emitted via core._PREAMBLE_REAL_MARKER, so it lands *after*
    # PARSER_PREAMBLE where ``namespace std`` is declared): the recovered
    # namespace/class/free-function declarations. Their signatures may
    # reference std:: types (e.g. ``const std::list<int>&``), which are only
    # declared in PARSER_PREAMBLE — placing them in the stub would fail with
    # "use of undeclared identifier 'std'".
    out = list(stub)
    if decls:
        out.append("// === TRANSPILERS: REAL PREAMBLE BELOW ===")
        out.extend(decls)
    out.append("// --- end recovered preamble ---")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Shell-out layer: query the codebase-memory-mcp graph
# --------------------------------------------------------------------------


def _resolve_binary(binary: str) -> str:
    """Return a runnable cbm binary path, or ``""`` if none is available.

    Falls back from a config'd path to the bare command name on PATH (covers
    the npm wrapper installs that put ``codebase-memory-mcp`` on PATH but not
    at ``~/.local/bin``).
    """
    if not binary:
        return ""
    if shutil.which(binary) or os.path.isfile(binary):
        return binary
    name = os.path.basename(str(binary))
    if name.endswith(".exe"):
        name = name[:-4]
    found = shutil.which(name)
    return found or ""


def _parse_query_table(text: str) -> list[dict]:
    """Parse the CLI ``query_graph`` table into ``[{col: value}, ...]``.

    The table is human-readable::

        rows: N  (cols: name param_types return_type)
          AND "[\\"std::list\\"]" int C-Users-...
        total: N

    Values are space-joined and JSON-quoted when they contain spaces or
    special characters, so each data line is shlex-split (which correctly
    unquotes a JSON-encoded array and preserves an embedded space in a
    ``unsigned int`` return type).
    """
    cols: list[str] | None = None
    rows: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("rows:"):
            lb = line.find("(cols:")
            if lb != -1:
                cols = line[lb + len("(cols:"):].rstrip(")").strip().split()
            continue
        if line.startswith("total:") or line.startswith("hint:"):
            continue
        if cols is None:
            # data before a header — ignore (e.g. a log line)
            continue
        vals = shlex.split(line)
        rows.append(dict(zip(cols, vals)))
    return rows


def cbm_query(project: str, cypher: str, binary: str = CBM_BIN) -> list[dict]:
    """Run one openCypher query through the cbm CLI bridge; return row dicts.

    The CLI prints ``level=info ...`` log lines to *stderr* and the result
    table to *stdout*. We return ``[]`` on any failure (no binary, non-zero
    exit, unparseable output) so callers degrade gracefully to the existing
    no-preamble parse path.
    """
    binary = _resolve_binary(binary)
    if not binary:
        return []
    try:
        proc = subprocess.run(
            [binary, "cli", "query_graph", "--project", project, "--query", cypher],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    return _parse_query_table(proc.stdout)


def build_preamble_payload(
    repo_root: str,
    rel_path: str,
    project: str | None = None,
    binary: str = CBM_BIN,
) -> dict:
    """Query the cbm graph for the declaration surface a file needs.

    ``rel_path`` is the file relative to ``repo_root`` (e.g.
    ``TopologicCore/src/Bitwise.cpp``). We resolve the classes whose methods
    this file defines out-of-line, the enclosing namespace (from
    ``Class.qualified_name``), each class's full method signature set
    (repo-wide, so overloads are recovered), and any SCREAMING_CASE export
    macros the file defines.

    Returns the payload dict consumed by ``payload_to_cpp``. Falls back to
    ``{"classes": [], "methods": [], "macros": [], "namespaces": []}`` when
    the binary / graph is unavailable.
    """
    empty: dict = {"classes": [], "methods": [], "macros": [], "namespaces": []}
    binary = _resolve_binary(binary)
    if not binary:
        return empty
    proj = project or _project_slug(repo_root)
    # The graph stores file_path with forward slashes; the caller (main.py's
    # Path.relative_to) hands us a Windows-style rel path with backslashes.
    rel_path = rel_path.replace("\\", "/")

    # Methods defined in THIS file, with a parent class (out-of-line defs).
    # `File-DEFINES->Method` is the modern graph's link; the method's
    # `parent_class` is path-qualified (`.src.Bitwise.Bitwise`), so the bare
    # class name is the final dot-segment.
    method_rows = cbm_query(
        proj,
        f"MATCH (m:Method)<-[:DEFINES]-(f:File) "
        f"WHERE f.file_path ENDS WITH '{rel_path}' AND m.parent_class <> '' "
        f"RETURN m.name AS name, m.param_types AS param_types, "
        f"m.return_type AS return_type, m.parent_class AS parent_class, "
        f"m.signature AS signature",
        binary,
    )
    class_names: list[str] = []
    for r in method_rows:
        pc = r.get("parent_class") or ""
        cname = pc.rsplit(".", 1)[-1]
        if cname and cname not in class_names:
            class_names.append(cname)
    # If the file-defines-method link is missing, still recover any method
    # whose parent_class names a class in this file (best-effort fallback).
    if not class_names:
        class_names = sorted(
            {r.get("parent_class", "").rsplit(".", 1)[-1] for r in method_rows}
        )
        class_names = [c for c in class_names if c]

    classes: list[dict] = []
    all_methods: list[dict] = []
    for cname in class_names:
        # Class node: bare name, base_classes, qualified_name (for namespace).
        crows = cbm_query(
            proj,
            f"MATCH (c:Class {{name: '{cname}'}}) "
            f"RETURN c.name AS name, c.base_classes AS base_classes, "
            f"c.qualified_name AS qualified_name, c.file_path AS file_path",
            binary,
        )
        crow = crows[0] if crows else {}
        ns = _namespace_of(crow.get("qualified_name"), crow.get("file_path"))
        classes.append(
            {
                "name": cname,
                "base_classes": _as_list(crow.get("base_classes")),
                "namespace": ns,
            }
        )
        # Full method set for this class (repo-wide, recover overloads the
        # file itself may not contain). Tag each with the bare class name so
        # _decls_from_payload groups them correctly.
        mrows = cbm_query(
            proj,
            f"MATCH (m:Method) WHERE m.parent_class ENDS WITH '.{cname}' "
            f"RETURN m.name AS name, m.param_types AS param_types, "
            f"m.return_type AS return_type, m.signature AS signature",
            binary,
        )
        for r in mrows:
            r["parent_class"] = cname
            r["param_types"] = _as_list(r.get("param_types"))
        all_methods.extend(mrows)

    # SCREAMING_CASE export / calling-convention macros this file defines.
    macro_rows = cbm_query(
        proj,
        f"MATCH (mac:Macro)<-[:DEFINES]-(f:File) "
        f"WHERE f.file_path ENDS WITH '{rel_path}' "
        f"RETURN mac.name AS name",
        binary,
    )
    macros = [{"name": r.get("name")} for r in macro_rows if r.get("name")]

    namespaces = sorted({c["namespace"] for c in classes if c["namespace"]})

    return {
        "classes": classes,
        "methods": all_methods,
        "macros": macros,
        "namespaces": namespaces,
        "type_refs": [],
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
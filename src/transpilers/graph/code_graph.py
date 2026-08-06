"""
Directed code graph for C++ and Python source files.

Builds a call graph and type dependency graph from source files,
enabling correct translation order (callees before callers) and
migration progress tracking.

Usage:
    from transpilers.graph.code_graph import build_graph, topological_order

    G = build_graph("path/to/source/dir", lang="cpp")
    order = topological_order(G)
    for func in order:
        print(func)

Dependencies:
    networkx  (required)
    tree-sitter + tree-sitter-languages  (optional; falls back to regex)
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("networkx is required: pip install networkx") from exc

# ---------------------------------------------------------------------------
# tree-sitter is optional — gracefully degrade to regex-based extraction.
# ---------------------------------------------------------------------------
_TREESITTER_AVAILABLE = False
_TREESITTER_CPP_PARSER = None
try:
    # Prefer `tree_sitter_languages` when available (covers multiple langs
    # from one package). It currently caps at Python 3.12, so on 3.13+ we
    # fall back to the per-language `tree-sitter-cpp` package.
    import tree_sitter_languages  # type: ignore
    from tree_sitter import Language, Parser  # type: ignore

    _TREESITTER_AVAILABLE = True
except Exception:
    try:
        import tree_sitter_cpp  # type: ignore
        from tree_sitter import Language, Parser  # type: ignore

        _TREESITTER_CPP_PARSER = Parser(Language(tree_sitter_cpp.language()))
        _TREESITTER_AVAILABLE = True
    except Exception:
        pass

# ---------------------------------------------------------------------------
# C++ patterns (regex fallback)
# ---------------------------------------------------------------------------
_CPP_FUNC_DEF = re.compile(
    r"""
    (?:(?:inline|static|virtual|explicit|constexpr|[[nodiscard]]\s*)*)?
    [\w:<>*&,\s]+?              # return type (rough)
    \b(\w+)\s*\(               # function name + (
    [^)]*\)\s*                 # params )
    (?:const\s*)?
    (?:noexcept[^{]*)?
    \{                          # opening brace
    """,
    re.VERBOSE | re.MULTILINE,
)

_CPP_CALL = re.compile(
    r"""
    \b(\w+)\s*\(               # identifier followed by (
    """,
    re.VERBOSE,
)

_CPP_KEYWORDS = frozenset(
    """
    if else while for do switch case default break continue return
    sizeof typeof decltype alignof alignas static_assert
    new delete throw try catch
    """.split()
)


def _extract_cpp_regex(source: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (defined_functions, call_map) using regex."""
    definitions: list[str] = []
    call_map: dict[str, list[str]] = {}

    # Collect function names first
    for m in _CPP_FUNC_DEF.finditer(source):
        name = m.group(1)
        if name and name not in _CPP_KEYWORDS:
            definitions.append(name)

    # For each definition, try to extract its body and find calls inside it.
    # Simple heuristic: split on the definition matches.
    matches = list(_CPP_FUNC_DEF.finditer(source))
    for i, m in enumerate(matches):
        name = m.group(1)
        if name in _CPP_KEYWORDS:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[start:end]
        calls = [
            c.group(1)
            for c in _CPP_CALL.finditer(body)
            if c.group(1) not in _CPP_KEYWORDS and c.group(1) != name
        ]
        call_map[name] = calls

    return definitions, call_map


def _extract_cpp_treesitter(source: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (defined_functions, call_map) using tree-sitter."""
    if _TREESITTER_CPP_PARSER is not None:
        parser = _TREESITTER_CPP_PARSER
    else:
        parser = tree_sitter_languages.get_parser("cpp")
    tree = parser.parse(source.encode())

    definitions: list[str] = []
    call_map: dict[str, list[str]] = {}

    def _node_text(node) -> str:
        return source[node.start_byte : node.end_byte]

    def _find_calls(node) -> list[str]:
        calls: list[str] = []
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                calls.append(_node_text(func_node).split("::")[-1].split(".")[-1])
        for child in node.children:
            calls.extend(_find_calls(child))
        return calls

    def _walk(node) -> None:
        if node.type == "function_definition":
            decl = node.child_by_field_name("declarator")
            if decl is not None:
                # drill down to find the function declarator
                while decl and decl.type not in ("function_declarator", "identifier"):
                    decl = decl.child_by_field_name("declarator") or (
                        decl.children[0] if decl.children else None
                    )
                if decl and decl.type == "function_declarator":
                    name_node = decl.child_by_field_name("declarator")
                    if name_node:
                        name = _node_text(name_node).split("::")[-1]
                        definitions.append(name)
                        body = node.child_by_field_name("body")
                        call_map[name] = _find_calls(body) if body else []
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return definitions, call_map


def _extract_cpp(source: str) -> tuple[list[str], dict[str, list[str]]]:
    if _TREESITTER_AVAILABLE:
        try:
            return _extract_cpp_treesitter(source)
        except Exception:
            pass
    return _extract_cpp_regex(source)


# ---------------------------------------------------------------------------
# Python extraction via stdlib ast
# ---------------------------------------------------------------------------


def _extract_python(source: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (defined_functions, call_map) using ast."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], {}

    definitions: list[str] = []
    call_map: dict[str, list[str]] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self._scope: list[str] = []

        def _current_scope(self) -> str | None:
            return self._scope[-1] if self._scope else None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            qualname = ".".join(self._scope + [node.name]) if self._scope else node.name
            definitions.append(qualname)
            call_map[qualname] = []
            self._scope.append(node.name)
            self.generic_visit(node)
            self._scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            scope = self._current_scope()
            if scope is None:
                self.generic_visit(node)
                return
            if isinstance(node.func, ast.Name):
                call_map[scope].append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_map[scope].append(node.func.attr)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return definitions, call_map


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------


def build_graph(path: str | Path, lang: str = "cpp") -> "nx.DiGraph":
    """Build a directed call graph from all source files under *path*.

    Nodes are function names (strings).
    Directed edges go from caller → callee.

    Parameters
    ----------
    path:
        Directory (or single file) to analyse.
    lang:
        ``"cpp"`` or ``"python"``.

    Returns
    -------
    nx.DiGraph
        Each node has a ``"file"`` attribute with the source path.
    """
    if lang not in ("cpp", "python"):
        raise ValueError(f"Unsupported lang {lang!r}; choose 'cpp' or 'python'")

    path = Path(path)
    extensions = {
        "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"},
        "python": {".py"},
    }[lang]

    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        files = [p for p in path.rglob("*") if p.suffix in extensions]

    G: "nx.DiGraph" = nx.DiGraph()

    extract = _extract_cpp if lang == "cpp" else _extract_python

    for fp in files:
        try:
            source = fp.read_text(errors="replace")
        except OSError:
            continue
        defs, calls = extract(source)
        for name in defs:
            G.add_node(name, file=str(fp))
        for caller, callees in calls.items():
            if caller not in G:
                G.add_node(caller, file=str(fp))
            for callee in callees:
                if callee:
                    G.add_edge(caller, callee)

    return G


# ---------------------------------------------------------------------------
# Topological order with SCC collapsing for cycles
# ---------------------------------------------------------------------------


def topological_order(G: "nx.DiGraph") -> list[str]:
    """Return nodes in topological order (callees before callers).

    Cycles are collapsed into a single representative node (the lexicographically
    smallest name in each SCC).  The returned list contains one entry per SCC;
    nodes that form a cycle appear as a single comma-joined entry.

    Parameters
    ----------
    G:
        The directed call graph returned by :func:`build_graph`.
    """
    # Condense SCCs to a DAG
    condensed = nx.condensation(G)  # nodes are SCC indices
    scc_map: dict[int, list[str]] = {}
    for node, data in condensed.nodes(data=True):
        members: list[str] = sorted(data["members"])
        scc_map[node] = members

    order: list[str] = []
    for scc_idx in nx.topological_sort(condensed):
        members = scc_map[scc_idx]
        if len(members) == 1:
            order.append(members[0])
        else:
            order.append(",".join(members))  # cycle group

    # Reverse so callees come first
    return list(reversed(order))


# ---------------------------------------------------------------------------
# Migration report
# ---------------------------------------------------------------------------


def migration_report(G: "nx.DiGraph", translated: set[str]) -> dict:
    """Return a dict with migration progress statistics.

    Parameters
    ----------
    G:
        The directed call graph.
    translated:
        Set of function names that have already been translated.

    Returns
    -------
    dict with keys:
        ``total``, ``translated``, ``pending``, ``blocked``

        *blocked* = functions that have at least one callee not yet translated.
    """
    all_nodes = set(G.nodes)
    pending = all_nodes - translated
    blocked: set[str] = set()
    for func in pending:
        callees = set(G.successors(func))
        if callees & pending:  # at least one callee is also pending
            blocked.add(func)

    return {
        "total": len(all_nodes),
        "translated": len(translated & all_nodes),
        "pending": len(pending),
        "blocked": len(blocked),
        "blocked_functions": sorted(blocked),
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_graph(G: "nx.DiGraph", path: str | Path) -> None:
    """Serialise the graph to a JSON file at *path*.

    Format is node-link JSON (networkx ``node_link_data``).
    """
    path = Path(path)
    data = nx.node_link_data(G)
    path.write_text(json.dumps(data, indent=2))


def load_graph(path: str | Path) -> "nx.DiGraph":
    """Load a graph previously saved with :func:`save_graph`.

    Parameters
    ----------
    path:
        Path to a JSON file written by :func:`save_graph`.

    Returns
    -------
    nx.DiGraph
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return nx.node_link_graph(data, directed=True, multigraph=False)


# ---------------------------------------------------------------------------
# File-level topological ordering
# ---------------------------------------------------------------------------


def file_topological_order(path: str | Path, lang: str = "cpp") -> list[Path]:
    """Return source files under *path* in dependency order (callees before callers).

    Projects the function-level call graph to file-level edges: file A depends
    on file B if any function in A calls any function defined in B.  Files with
    no detected cross-file dependencies come first.  Cycles (mutual recursion
    across files) are broken by SCC condensation.

    Parameters
    ----------
    path:
        Directory (or single file) to analyse.
    lang:
        ``"cpp"`` or ``"python"``.

    Returns
    -------
    list[Path]
        Source files ordered so that dependency files precede their dependents.
        Files absent from the call graph (no parsed functions) are appended
        last in sorted order.
    """
    path = Path(path)
    G = build_graph(path, lang=lang)

    # Build file-level directed graph: caller-file → callee-file.
    file_graph: "nx.DiGraph" = nx.DiGraph()
    for node, data in G.nodes(data=True):
        fp = data.get("file", "")
        if fp:
            file_graph.add_node(fp)

    for caller, callee in G.edges():
        cf = G.nodes[caller].get("file", "")
        df = G.nodes[callee].get("file", "")
        if cf and df and cf != df:
            file_graph.add_edge(cf, df)

    # Topological sort with SCC condensation for cycles.
    condensed = nx.condensation(file_graph)
    ordered_files: list[Path] = []
    seen: set[str] = set()
    for scc_idx in reversed(list(nx.topological_sort(condensed))):
        for fp in sorted(condensed.nodes[scc_idx]["members"]):
            if fp not in seen:
                ordered_files.append(Path(fp))
                seen.add(fp)

    # Append files with no parsed functions (not in graph) last.
    if path.is_dir():
        extensions = {
            "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"},
            "python": {".py"},
        }.get(lang, set())
        for fp in sorted(path.rglob("*")):
            if fp.is_file() and fp.suffix in extensions and str(fp) not in seen:
                ordered_files.append(fp)

    return ordered_files

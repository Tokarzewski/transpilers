"""LLM-driven variable rename pass.

Renames opaque identifiers (Ghidra-decompiled `local_10`, `param_1`,
`iVar3`, …) into meaningful names by asking the LLM. The kind of pattern
no purely-algorithmic pass can handle — there's no rule that recovers
"running_sum" from `local_10`; only context tells you that.

Three rules carry over from the existing LLM-fallback type inference:
  1. Typed holes — every call has a validator that rejects bad output
     before it touches the IR.
  2. Cached by content hash via the LlmClient, so re-runs are free.
  3. Dependency-injected — tests pass a fake `llm_fill`; production
     wiring builds it from the cached client.

The pass walks each function, builds a name → proposed-name table, and
applies all renames atomically per function (so the renames in one
function can't interact with renames in another).
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from transpilers.ir import mir


# Names matching these patterns are considered opaque candidates for rename.
# Includes Ghidra's standard locals/params/typed-var prefixes and the
# fallback `param_N`/`local_N` patterns we see in decompilation output.
OPAQUE_PATTERNS = [
    re.compile(r"^local_[0-9a-f]+$"),
    re.compile(r"^param_[0-9]+$"),
    re.compile(r"^[ipuc]Var[0-9]+$"),  # iVar1, pVar2, uVar3, cVar4
    re.compile(r"^var_[0-9a-f]+$"),
    re.compile(r"^[A-Z]+[0-9]+_[0-9]+$"),  # DAT_00112233 style globals
]


LlmFill = Callable[[str, dict], str]


def llm_rename(module: mir.MirModule, *, llm_fill: LlmFill) -> mir.MirModule:
    for fn in module.functions:
        _rename_function(fn, llm_fill)
    return module


def _rename_function(fn: mir.MirFunction, llm_fill: LlmFill) -> None:
    candidates = _find_opaque_names(fn)
    if not candidates:
        return
    existing = _all_names_in_function(fn) - candidates
    rename_map: dict[str, str] = {}
    for old in sorted(candidates):
        proposed = llm_fill(
            old,
            {
                "function_name": fn.name,
                "function_mir": _mir_dump(fn),
                "old_name": old,
                "existing_names": sorted(existing),
                "renames_so_far": rename_map.copy(),
            },
        )
        # Disambiguate if the LLM picked a name already in use.
        new = proposed
        i = 1
        while new in existing or new in rename_map.values():
            new = f"{proposed}_{i}"
            i += 1
        rename_map[old] = new
        existing.add(new)
    _apply_renames(fn, rename_map)


# ---------- discovery ----------


def _find_opaque_names(fn: mir.MirFunction) -> set[str]:
    found: set[str] = set()
    for p in fn.params:
        if _is_opaque(p.name):
            found.add(p.name)
    _collect_opaque(fn.body, found)
    return found


def _collect_opaque(nodes: list[mir.MirNode], out: set[str]) -> None:
    for n in nodes:
        if isinstance(n, mir.MirAssign) and _is_opaque(n.target):
            out.add(n.target)
        elif isinstance(n, mir.MirForRange) and _is_opaque(n.target):
            out.add(n.target)
        if isinstance(n, mir.MirIf):
            _collect_opaque(n.body, out)
            _collect_opaque(n.orelse, out)
        elif isinstance(n, mir.MirWhile):
            _collect_opaque(n.body, out)
        elif isinstance(n, mir.MirForRange):
            _collect_opaque(n.body, out)


def _is_opaque(name: str) -> bool:
    return any(p.match(name) for p in OPAQUE_PATTERNS)


def _all_names_in_function(fn: mir.MirFunction) -> set[str]:
    names: set[str] = {p.name for p in fn.params}
    _collect_all_names(fn.body, names)
    return names


def _collect_all_names(nodes: list[mir.MirNode], out: set[str]) -> None:
    for n in nodes:
        if isinstance(n, mir.MirAssign):
            out.add(n.target)
        elif isinstance(n, mir.MirForRange):
            out.add(n.target)
        if isinstance(n, mir.MirIf):
            _collect_all_names(n.body, out)
            _collect_all_names(n.orelse, out)
        elif isinstance(n, mir.MirWhile):
            _collect_all_names(n.body, out)
        elif isinstance(n, mir.MirForRange):
            _collect_all_names(n.body, out)


# ---------- apply ----------


def _apply_renames(fn: mir.MirFunction, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    # Rename parameters.
    for i, p in enumerate(fn.params):
        if p.name in mapping:
            fn.params[i] = replace(p, name=mapping[p.name])
    _rewrite_nodes(fn.body, mapping)


def _rewrite_nodes(nodes: list[mir.MirNode], mapping: dict[str, str]) -> None:
    for n in nodes:
        _rewrite_node(n, mapping)


def _rewrite_node(node: mir.MirNode, mapping: dict[str, str]) -> None:
    if isinstance(node, mir.MirName) and node.name in mapping:
        node.name = mapping[node.name]
    if isinstance(node, mir.MirAssign):
        if node.target in mapping:
            node.target = mapping[node.target]
        _rewrite_node(node.value, mapping)
    if isinstance(node, mir.MirFieldAssign):
        _rewrite_node(node.obj, mapping)
        _rewrite_node(node.value, mapping)
    if isinstance(node, mir.MirReturn):
        if node.value is not None:
            _rewrite_node(node.value, mapping)
    if isinstance(node, (mir.MirBinOp, mir.MirCompare, mir.MirBoolOp)):
        _rewrite_node(node.left, mapping)
        _rewrite_node(node.right, mapping)
    if isinstance(node, mir.MirUnaryOp):
        _rewrite_node(node.operand, mapping)
    if isinstance(node, (mir.MirCall, mir.MirMethodCall, mir.MirStructInit)):
        if isinstance(node, mir.MirMethodCall):
            _rewrite_node(node.receiver, mapping)
            for a in node.args:
                _rewrite_node(a, mapping)
        elif isinstance(node, mir.MirStructInit):
            for _, v in node.field_values:
                _rewrite_node(v, mapping)
        else:
            for a in node.args:
                _rewrite_node(a, mapping)
    if isinstance(node, mir.MirIf):
        _rewrite_node(node.test, mapping)
        _rewrite_nodes(node.body, mapping)
        _rewrite_nodes(node.orelse, mapping)
    if isinstance(node, mir.MirWhile):
        _rewrite_node(node.test, mapping)
        _rewrite_nodes(node.body, mapping)
    if isinstance(node, mir.MirForRange):
        if node.target in mapping:
            node.target = mapping[node.target]
        _rewrite_node(node.start, mapping)
        _rewrite_node(node.stop, mapping)
        if node.step is not None:
            _rewrite_node(node.step, mapping)
        _rewrite_nodes(node.body, mapping)
    if isinstance(node, mir.MirSubscript):
        _rewrite_node(node.value, mapping)
        _rewrite_node(node.index, mapping)
    if isinstance(node, mir.MirSubscriptAssign):
        _rewrite_node(node.obj, mapping)
        _rewrite_node(node.index, mapping)
        _rewrite_node(node.value, mapping)
    if isinstance(node, mir.MirFieldAccess):
        _rewrite_node(node.value, mapping)
    if isinstance(node, mir.MirList):
        for e in node.elements:
            _rewrite_node(e, mapping)


# ---------- LLM context dump ----------


def _mir_dump(fn: mir.MirFunction) -> str:
    """Dump an MIR function in a human-readable shape so the LLM has
    enough context to pick a meaningful name. Inline form — same shape
    used by infer_types.py for type-inference LLM calls."""
    lines = [f"fn {fn.name}({', '.join(p.name for p in fn.params)})"]
    _dump_block(fn.body, lines, depth=1)
    return "\n".join(lines)


def _dump_block(nodes: list[mir.MirNode], out: list[str], depth: int) -> None:
    pad = "  " * depth
    for n in nodes:
        out.append(pad + _dump_node(n))
        for child in _children(n):
            _dump_block(child, out, depth + 1)


def _dump_node(n: mir.MirNode) -> str:
    if isinstance(n, mir.MirReturn):
        return f"return {_dump_node(n.value) if n.value else ''}"
    if isinstance(n, mir.MirAssign):
        return f"{n.target} = {_dump_node(n.value)}"
    if isinstance(n, mir.MirIf):
        return f"if {_dump_node(n.test)}"
    if isinstance(n, mir.MirWhile):
        return f"while {_dump_node(n.test)}"
    if isinstance(n, mir.MirForRange):
        return f"for {n.target} in range(...)"
    if isinstance(n, mir.MirBinOp):
        return f"({_dump_node(n.left)} {n.op} {_dump_node(n.right)})"
    if isinstance(n, mir.MirCompare):
        return f"({_dump_node(n.left)} {n.op} {_dump_node(n.right)})"
    if isinstance(n, mir.MirCall):
        return f"{n.func}({', '.join(_dump_node(a) for a in n.args)})"
    if isinstance(n, mir.MirName):
        return n.name
    if isinstance(n, mir.MirIntLiteral):
        return str(n.value)
    if isinstance(n, mir.MirStringLiteral):
        return repr(n.value)
    return type(n).__name__


def _children(n: mir.MirNode) -> list[list[mir.MirNode]]:
    if isinstance(n, mir.MirIf):
        return [n.body, n.orelse]
    if isinstance(n, (mir.MirWhile, mir.MirForRange)):
        return [n.body]
    return []

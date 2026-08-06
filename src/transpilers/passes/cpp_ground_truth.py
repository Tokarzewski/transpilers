"""C++ ground-truth MIR pass (issue #50).

The C++ frontend, given a clang AST, extracts a ``TypeGroundTruth``
table that maps every source position to its concrete resolved type.
This pass walks the MIR and replaces ``UnknownT`` holes with the
ground-truth type at the matching source location.

The match is by ``source_loc`` (a ``file:line:col`` string), which
both the HIR nodes and the ``TypeGroundTruth`` use as their key.

What's filled
-------------

* Function parameter types — when the ground truth has a function
  with a matching name, its parameter types are applied to the MIR
  function's params.
* Function return types — same.
* ``MirName`` / ``MirCall`` nodes whose ``source_loc`` is in
  ``ground_truth.var_types`` / ``ground_truth.call_returns`` get
  their type replaced.

What's not filled
-----------------

* Ownership / RAII: the ground truth carries types, not lifetimes.
  ``auto`` / ``std::unique_ptr<T>`` -> target-idiom mapping is left
  for downstream / LLM passes (the residual in the issue description).
* Aggregates: the libclang type spellings ``std::vector<T>`` etc.
  don't always round-trip cleanly through our existing ``_type_text``
  aliases. When the conversion is lossy we leave the ``UnknownT`` in
  place rather than invent a type.

Usage
-----

::

    from transpilers.frontends.cpp.parser import parse_cpp
    from transpilers.passes.cpp_ground_truth import apply_ground_truth

    hir_mod, truth = parse_cpp(source)
    mir_mod = hir_to_mir(hir_mod)
    apply_ground_truth(mir_mod, truth, hir_mod)
    infer_types(mir_mod)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from transpilers.ir import hir, mir
from transpilers.ir.types import Type, UnknownT

from transpilers.frontends.cpp.parser.type_extractor import TypeGroundTruth


def _walk_mir(
    nodes: Iterable[mir.MirNode],
    truth: TypeGroundTruth,
    env: dict[str, Type] | None = None,
) -> None:
    """Recursively visit every MIR node and fill ``UnknownT`` from the
    ground truth where the source location matches."""
    for n in nodes:
        _visit(n, truth, env)


def _visit(
    node: mir.MirNode,
    truth: TypeGroundTruth,
    env: dict[str, Type] | None,
) -> None:
    """Per-node dispatch for ``_walk_mir``."""
    if isinstance(node, mir.MirReturn):
        if node.value is not None:
            _visit(node.value, truth, env)
        return
    if isinstance(node, mir.MirAssign):
        _visit(node.value, truth, env)
        if env is not None and not isinstance(node.ty, UnknownT):
            env[node.target] = node.ty
        return
    if isinstance(node, mir.MirIf):
        _visit(node.test, truth, env)
        _walk_mir(node.body, truth, env)
        _walk_mir(node.orelse, truth, env)
        return
    if isinstance(node, mir.MirWhile):
        _visit(node.test, truth, env)
        _walk_mir(node.body, truth, env)
        return
    if isinstance(node, mir.MirForRange):
        _visit(node.start, truth, env)
        _visit(node.stop, truth, env)
        if node.step is not None:
            _visit(node.step, truth, env)
        _walk_mir(node.body, truth, env)
        return
    if isinstance(node, mir.MirBinOp):
        _visit(node.left, truth, env)
        _visit(node.right, truth, env)
        return
    if isinstance(node, mir.MirCompare):
        _visit(node.left, truth, env)
        _visit(node.right, truth, env)
        return
    if isinstance(node, mir.MirBoolOp):
        _visit(node.left, truth, env)
        _visit(node.right, truth, env)
        return
    if isinstance(node, mir.MirUnaryOp):
        _visit(node.operand, truth, env)
        return
    if isinstance(node, mir.MirCall):
        for a in node.args:
            _visit(a, truth, env)
        # If the call's source location has a recorded return type,
        # fill the call's ``ty`` in place.
        loc = getattr(node, "source_loc", None)
        if loc and loc in truth.call_returns and isinstance(node.ty, UnknownT):
            rep = truth.call_returns[loc]
            if not isinstance(rep, UnknownT):
                _set_ty(node, rep)
        return
    if isinstance(node, mir.MirList):
        for e in node.elements:
            _visit(e, truth, env)
        return
    if isinstance(node, mir.MirSubscript):
        _visit(node.value, truth, env)
        _visit(node.index, truth, env)
        return
    if isinstance(node, mir.MirFieldAccess):
        _visit(node.value, truth, env)
        return
    if isinstance(node, mir.MirMethodCall):
        _visit(node.receiver, truth, env)
        for a in node.args:
            _visit(a, truth, env)
        return
    if isinstance(node, mir.MirFieldAssign):
        _visit(node.obj, truth, env)
        _visit(node.value, truth, env)
        return
    if isinstance(node, mir.MirSubscriptAssign):
        _visit(node.obj, truth, env)
        _visit(node.index, truth, env)
        _visit(node.value, truth, env)
        return
    if isinstance(node, mir.MirStructInit):
        for _, v in node.field_values:
            _visit(v, truth, env)
        return
    # MirName / literals: nothing to recurse into.
    if isinstance(node, mir.MirName):
        loc = getattr(node, "source_loc", None)
        if loc and loc in truth.var_types and isinstance(node.ty, UnknownT):
            rep = truth.var_types[loc]
            if not isinstance(rep, UnknownT):
                _set_ty(node, rep)
                if env is not None:
                    env[node.name] = rep
        return


def _set_ty(node: mir.MirNode, ty: Type) -> None:
    """Best-effort type assignment for MIR nodes that aren't dataclass
    ``field``s. The MIR dataclasses all use ``ty: Type = field(default_factory=UnknownT)``,
    so this works for every typed node without touching the AST shape."""
    try:
        object.__setattr__(node, "ty", ty)
    except Exception:
        # Some MIR nodes might be frozen; skip rather than break the
        # whole pass. (None of the current ones are.)
        pass


def _fill_function_signature(
    fn: mir.MirFunction,
    truth: TypeGroundTruth,
) -> None:
    """Replace UnknownT slots in *fn*'s params / return type with the
    ground truth's best guess by name. Qualified name first (so
    ``mehara::sort::bubble_sort`` matches), then bare name."""
    candidates: list[str] = []
    if fn.name in truth.func_returns:
        candidates.append(fn.name)
    qn = _qualified_for(fn, truth)
    if qn and qn not in candidates:
        candidates.append(qn)
    for cname in candidates:
        ret = truth.func_returns.get(cname)
        if (
            ret is not None
            and not isinstance(ret, UnknownT)
            and isinstance(fn.return_type, UnknownT)
        ):
            fn.return_type = ret
        params = truth.func_params.get(cname)
        if params is None:
            continue
        new_params: list[mir.MirParam] = []
        for i, p in enumerate(fn.params):
            if (
                i < len(params)
                and isinstance(p.ty, UnknownT)
                and not isinstance(params[i], UnknownT)
            ):
                new_params.append(replace(p, ty=params[i]))
            else:
                new_params.append(p)
        fn.params = new_params
        # First qualified match wins.
        break


def _qualified_for(fn: mir.MirFunction, truth: TypeGroundTruth) -> str | None:
    """Best qualified-name guess for *fn* in the ground-truth table.

    The C++ frontend stores qualified names (``mehara::sort::bubble_sort``);
    the MIR function only has the bare name. We try every entry in
    ``decl_locs`` whose bare-name suffix matches and return the
    first hit.
    """
    bare = fn.name
    for qn in truth.decl_locs:
        if qn.endswith("::" + bare) or qn == bare:
            return qn
    return None


def apply_ground_truth(
    mir_mod: mir.MirModule,
    truth: TypeGroundTruth | None,
    hir_mod: hir.HirModule | None = None,
) -> mir.MirModule:
    """Apply *truth* to every MIR node in *mir_mod*.

    The pass is *non-destructive*: it only fills ``UnknownT`` slots
    and never overwrites a type the inference pass would have
    resolved. It runs *before* ``infer_types`` so the inference pass
    can build on the ground truth instead of re-deriving it.

    The optional *hir_mod* is accepted for forward compatibility --
    the C++ HIR conversion currently pushes source_loc through to
    the relevant MIR nodes directly, so we don't need a side channel.
    """
    if truth is None or truth.is_empty():
        return mir_mod
    # Phase 1: function signatures. These don't need a per-node
    # source_loc -- the function name is enough.
    for fn in mir_mod.functions:
        _fill_function_signature(fn, truth)
    # Phase 2: walk every body and fill per-node UnknownT.
    for fn in mir_mod.functions:
        env: dict[str, Type] = {
            p.name: p.ty for p in fn.params if not isinstance(p.ty, UnknownT)
        }
        _walk_mir(fn.body, truth, env)
    return mir_mod


__all__ = ["apply_ground_truth"]

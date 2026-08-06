"""Type-inference pass on MIR.

Runs in three phases:
  1. Local dataflow — fixed-point per function. Propagates types from
     literals through binops/compares/calls/subscripts/assignments. Updates
     parameter, return, and node types in place when discovered.
  2. Interprocedural — fixed-point at the module level. Each MirCall consults
     a function-name → MirFunction map: forward (callee signature shapes the
     call's result type and anchors arg Names) and backward (concrete arg
     types at the call site fill UnknownT parameter slots on the callee).
     Re-runs local inference after each propagation round.
  3. Optional LLM fallback — for residual UnknownT on parameters and return
     types only. Operates on typed holes (constrained outputs), validates
     responses, and caches by content hash. Algorithmic passes downstream
     still refuse to invent on remaining holes.

The fallback is dependency-injected as a callable so tests can run it without
API access and so the production CLI can opt in explicitly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Callable

from transpilers.ir import mir
from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    StrT,
    Type,
    UnknownT,
)


LlmFill = Callable[[str, dict], Type]
FnMap = dict[str, mir.MirFunction]
IrHints = dict[str, "tuple[list[Type], Type]"]


ARITH_OPS = {"+", "-", "*", "/", "%"}
MAX_ITER = 8
MAX_MODULE_ITER = 6


def infer_types(
    module: mir.MirModule,
    *,
    llm_fill: LlmFill | None = None,
    ir_hints: IrHints | None = None,
) -> mir.MirModule:
    if ir_hints:
        _apply_ir_hints(module, ir_hints)
    fn_map: FnMap = {fn.name: fn for fn in module.functions}
    last = None
    for _ in range(MAX_MODULE_ITER):
        for fn in module.functions:
            _infer_function(fn, fn_map=fn_map)
        for fn in module.functions:
            _backward_propagate(fn.body, fn_map)
        snap = _module_snapshot(module)
        if snap == last:
            break
        last = snap

    if llm_fill is not None:
        for fn in module.functions:
            _llm_fallback(fn, llm_fill)
    return module


def _apply_ir_hints(module: mir.MirModule, hints: IrHints) -> None:
    """Pre-seed MIR function parameter/return types from LLVM IR type data.

    Tries exact name match first, then a substring heuristic for C++ mangled names.
    Only fills UnknownT slots — never overwrites an already-resolved type.
    """
    for fn in module.functions:
        hit = hints.get(fn.name)
        if hit is None:
            # Heuristic: mangled C++ name contains the unmangled base name.
            for mangled, pair in hints.items():
                if len(fn.name) > 2 and fn.name in mangled:
                    hit = pair
                    break
        if hit is None:
            continue
        ptypes, rtype = hit
        for i, p in enumerate(fn.params):
            if (
                i < len(ptypes)
                and isinstance(p.ty, UnknownT)
                and not isinstance(ptypes[i], UnknownT)
            ):
                fn.params[i] = replace(p, ty=ptypes[i])
        if isinstance(fn.return_type, UnknownT) and not isinstance(rtype, UnknownT):
            fn.return_type = rtype


def _module_snapshot(module: mir.MirModule) -> tuple:
    return tuple(
        (
            fn.name,
            tuple(type(p.ty).__name__ for p in fn.params),
            type(fn.return_type).__name__,
        )
        for fn in module.functions
    )


# ---------- per-function inference ----------


def _infer_function(fn: mir.MirFunction, *, fn_map: FnMap | None = None) -> None:
    env: dict[str, Type] = {
        p.name: p.ty for p in fn.params if not isinstance(p.ty, UnknownT)
    }
    last = None
    for _ in range(MAX_ITER):
        return_tys: list[Type] = []
        _visit_block(fn.body, env, return_tys, fn_map)
        _pull_params_from_env(fn, env)
        _resolve_return(fn, return_tys)
        snap = _snapshot(fn, env)
        if snap == last:
            break
        last = snap


# ---------- interprocedural backward pass ----------


def _backward_propagate(nodes: list[mir.MirNode], fn_map: FnMap) -> None:
    """Walk every MirCall in `nodes` (recursively) and push concrete arg types
    from the call site back to the callee's UnknownT parameters."""
    for n in nodes:
        _backward_in(n, fn_map)


def _backward_in(node: mir.MirNode, fn_map: FnMap) -> None:
    if isinstance(node, mir.MirCall):
        callee = fn_map.get(node.func)
        if callee is not None:
            for i, arg in enumerate(node.args):
                if i >= len(callee.params):
                    break
                arg_ty = getattr(arg, "ty", UnknownT())
                if isinstance(callee.params[i].ty, UnknownT) and not isinstance(
                    arg_ty, UnknownT
                ):
                    callee.params[i] = replace(callee.params[i], ty=arg_ty)
        for a in node.args:
            _backward_in(a, fn_map)
        return
    if isinstance(node, mir.MirReturn):
        if node.value is not None:
            _backward_in(node.value, fn_map)
        return
    if isinstance(node, mir.MirAssign):
        _backward_in(node.value, fn_map)
        return
    if isinstance(node, mir.MirIf):
        _backward_in(node.test, fn_map)
        _backward_propagate(node.body, fn_map)
        _backward_propagate(node.orelse, fn_map)
        return
    if isinstance(node, mir.MirWhile):
        _backward_in(node.test, fn_map)
        _backward_propagate(node.body, fn_map)
        return
    if isinstance(node, mir.MirForRange):
        _backward_in(node.start, fn_map)
        _backward_in(node.stop, fn_map)
        if node.step is not None:
            _backward_in(node.step, fn_map)
        _backward_propagate(node.body, fn_map)
        return
    if isinstance(node, mir.MirBinOp):
        _backward_in(node.left, fn_map)
        _backward_in(node.right, fn_map)
        return
    if isinstance(node, mir.MirCompare):
        _backward_in(node.left, fn_map)
        _backward_in(node.right, fn_map)
        return
    if isinstance(node, mir.MirBoolOp):
        _backward_in(node.left, fn_map)
        _backward_in(node.right, fn_map)
        return
    if isinstance(node, mir.MirUnaryOp):
        _backward_in(node.operand, fn_map)
        return
    if isinstance(node, mir.MirList):
        for e in node.elements:
            _backward_in(e, fn_map)
        return
    if isinstance(node, mir.MirSubscript):
        _backward_in(node.value, fn_map)
        _backward_in(node.index, fn_map)
        return
    # MirName, literals, etc. — no calls to find.


def _llm_fallback(fn: mir.MirFunction, llm_fill: LlmFill) -> None:
    dump = _mir_dump(fn)
    for i, p in enumerate(fn.params):
        if isinstance(p.ty, UnknownT):
            ty = llm_fill(
                p.name, {"role": "param", "function_mir": dump, "name": p.name}
            )
            fn.params[i] = replace(p, ty=ty)
    if isinstance(fn.return_type, UnknownT):
        fn.return_type = llm_fill(
            "__return__", {"role": "return", "function_mir": dump}
        )


# ---------- propagation ----------


def _visit_block(
    nodes: list[mir.MirNode],
    env: dict[str, Type],
    return_tys: list[Type],
    fn_map: FnMap | None,
) -> None:
    for n in nodes:
        _visit_stmt(n, env, return_tys, fn_map)


def _visit_stmt(
    node: mir.MirNode,
    env: dict[str, Type],
    return_tys: list[Type],
    fn_map: FnMap | None,
) -> None:
    if isinstance(node, mir.MirReturn):
        ty = _visit_expr(node.value, env, fn_map) if node.value is not None else NoneT()
        return_tys.append(ty)
        return
    if isinstance(node, mir.MirAssign):
        vt = _visit_expr(node.value, env, fn_map)
        ann_ty = node.ty if not isinstance(node.ty, UnknownT) else vt
        if isinstance(node.ty, UnknownT) and not isinstance(vt, UnknownT):
            node.ty = vt
        if not isinstance(ann_ty, UnknownT):
            env[node.target] = ann_ty
        return
    if isinstance(node, mir.MirIf):
        _visit_expr(node.test, env, fn_map)
        _visit_block(node.body, env, return_tys, fn_map)
        _visit_block(node.orelse, env, return_tys, fn_map)
        return
    if isinstance(node, mir.MirWhile):
        _visit_expr(node.test, env, fn_map)
        _visit_block(node.body, env, return_tys, fn_map)
        return
    if isinstance(node, mir.MirForRange):
        # `for i in range(...)` constrains start/stop/step to int. If any bound
        # is an unknown Name, anchor it here — same pattern as call-signature
        # arg-type propagation, applied to the already-specialized for-range.
        for bound in (node.start, node.stop, node.step):
            if bound is None:
                continue
            _visit_expr(bound, env, fn_map)
            if isinstance(bound, mir.MirName) and isinstance(bound.ty, UnknownT):
                env[bound.name] = IntT()
                bound.ty = IntT()
        env[node.target] = IntT()
        _visit_block(node.body, env, return_tys, fn_map)
        return
    # Bare expression statement.
    _visit_expr(node, env, fn_map)


def _visit_expr(
    node: mir.MirNode | None, env: dict[str, Type], fn_map: FnMap | None = None
) -> Type:
    if node is None:
        return NoneT()
    if isinstance(node, mir.MirIntLiteral):
        node.ty = IntT()
        return node.ty
    if isinstance(node, mir.MirFloatLiteral):
        node.ty = FloatT()
        return node.ty
    if isinstance(node, mir.MirBoolLiteral):
        node.ty = BoolT()
        return node.ty
    if isinstance(node, mir.MirStringLiteral):
        node.ty = StrT()
        return node.ty
    if isinstance(node, mir.MirName):
        ty = env.get(node.name)
        if ty is not None and not isinstance(ty, UnknownT):
            node.ty = ty
            return ty
        return node.ty
    if isinstance(node, mir.MirBinOp):
        lt = _visit_expr(node.left, env, fn_map)
        rt = _visit_expr(node.right, env, fn_map)
        if node.op in ARITH_OPS:
            lt, rt = _arith_unify(node, lt, rt, env)
        ty = _binop_result(node.op, lt, rt)
        if not isinstance(ty, UnknownT):
            node.ty = ty
        return node.ty
    if isinstance(node, mir.MirCompare):
        lt = _visit_expr(node.left, env, fn_map)
        rt = _visit_expr(node.right, env, fn_map)
        _compare_unify(node, lt, rt, env)
        node.ty = BoolT()
        return node.ty
    if isinstance(node, mir.MirBoolOp):
        _visit_expr(node.left, env, fn_map)
        _visit_expr(node.right, env, fn_map)
        node.ty = BoolT()
        return node.ty
    if isinstance(node, mir.MirUnaryOp):
        ot = _visit_expr(node.operand, env, fn_map)
        node.ty = BoolT() if node.op == "not" else ot
        return node.ty
    if isinstance(node, mir.MirCall):
        for a in node.args:
            _visit_expr(a, env, fn_map)
        _propagate_arg_types(node, env)
        _propagate_from_callee(node, env, fn_map)
        if node.func == "len":
            node.ty = IntT()
        # range() result type already set in lowering; other calls stay UnknownT.
        return node.ty
    if isinstance(node, mir.MirList):
        elem_tys = [_visit_expr(e, env, fn_map) for e in node.elements]
        concrete = [t for t in elem_tys if not isinstance(t, UnknownT)]
        if concrete:
            node.ty = ListT(elem=concrete[0])
        return node.ty
    if isinstance(node, mir.MirSubscript):
        vt = _visit_expr(node.value, env, fn_map)
        _visit_expr(node.index, env, fn_map)
        if isinstance(vt, ListT) and isinstance(node.ty, UnknownT):
            node.ty = vt.elem
        return node.ty
    return UnknownT()


def _propagate_from_callee(
    node: mir.MirCall, env: dict[str, Type], fn_map: FnMap | None
) -> None:
    """Forward direction of interprocedural propagation: pull the callee's
    known signature into the call site. Anchors unknown Name args to the
    callee's param types and types the call expression by the callee's return
    type."""
    if fn_map is None:
        return
    callee = fn_map.get(node.func)
    if callee is None:
        return
    for i, arg in enumerate(node.args):
        if i >= len(callee.params):
            break
        param_ty = callee.params[i].ty
        if isinstance(param_ty, UnknownT):
            continue
        if isinstance(arg, mir.MirName) and isinstance(arg.ty, UnknownT):
            env[arg.name] = param_ty
            arg.ty = param_ty
    if not isinstance(callee.return_type, UnknownT) and isinstance(node.ty, UnknownT):
        node.ty = callee.return_type


# ---------- bidirectional unification on arith / compare ----------


def _arith_unify(
    node: mir.MirBinOp, lt: Type, rt: Type, env: dict[str, Type]
) -> tuple[Type, Type]:
    """If one side is a known numeric (or, for `+`, string) type and the other
    is an unknown Name, promote the Name to match. `x + 1` → `x: int`,
    `x * 1.5` → `x: float`, `s + "lit"` → `s: str`."""
    propagatable = (IntT, FloatT, StrT) if node.op == "+" else (IntT, FloatT)
    if isinstance(lt, propagatable) and isinstance(rt, UnknownT):
        if isinstance(node.right, mir.MirName):
            env[node.right.name] = lt
            node.right.ty = lt
            rt = lt
    if isinstance(rt, propagatable) and isinstance(lt, UnknownT):
        if isinstance(node.left, mir.MirName):
            env[node.left.name] = rt
            node.left.ty = rt
            lt = rt
    # If either side is float and the other is int (Name or otherwise), the
    # result must be float — promote the int-typed Name too.
    if (
        isinstance(lt, FloatT)
        and isinstance(rt, IntT)
        and isinstance(node.right, mir.MirName)
    ):
        env[node.right.name] = FloatT()
        node.right.ty = FloatT()
        rt = FloatT()
    if (
        isinstance(rt, FloatT)
        and isinstance(lt, IntT)
        and isinstance(node.left, mir.MirName)
    ):
        env[node.left.name] = FloatT()
        node.left.ty = FloatT()
        lt = FloatT()
    return lt, rt


def _compare_unify(
    node: mir.MirCompare, lt: Type, rt: Type, env: dict[str, Type]
) -> None:
    """Comparisons constrain operands to the same type."""
    if (
        isinstance(lt, UnknownT)
        and not isinstance(rt, UnknownT)
        and isinstance(node.left, mir.MirName)
    ):
        env[node.left.name] = rt
        node.left.ty = rt
    if (
        isinstance(rt, UnknownT)
        and not isinstance(lt, UnknownT)
        and isinstance(node.right, mir.MirName)
    ):
        env[node.right.name] = lt
        node.right.ty = lt


def _propagate_arg_types(node: mir.MirCall, env: dict[str, Type]) -> None:
    """Builtin signatures we know — narrow but useful. `range(...)` takes ints;
    `len(...)` takes something list-shaped (we don't know elem type, so we
    leave that to other anchors). This is the place where a future
    `stdlib_maps/*` signature table will plug in."""
    if node.func == "range":
        for a in node.args:
            if isinstance(a, mir.MirName) and isinstance(a.ty, UnknownT):
                env[a.name] = IntT()
                a.ty = IntT()


def _binop_result(op: str, lt: Type, rt: Type) -> Type:
    if isinstance(lt, IntT) and isinstance(rt, IntT):
        return IntT()
    if isinstance(lt, (IntT, FloatT)) and isinstance(rt, (IntT, FloatT)):
        return FloatT()
    if op == "+" and isinstance(lt, StrT) and isinstance(rt, StrT):
        return StrT()
    return UnknownT(hint=f"binop {op}")


# ---------- function-level helpers ----------


def _pull_params_from_env(fn: mir.MirFunction, env: dict[str, Type]) -> None:
    for i, p in enumerate(fn.params):
        if isinstance(p.ty, UnknownT):
            inferred = env.get(p.name)
            if inferred is not None and not isinstance(inferred, UnknownT):
                fn.params[i] = replace(p, ty=inferred)


def _resolve_return(fn: mir.MirFunction, return_tys: list[Type]) -> None:
    if not isinstance(fn.return_type, UnknownT):
        return
    concrete = [t for t in return_tys if not isinstance(t, UnknownT)]
    if concrete:
        # Multiple distinct concrete return types would be a real ambiguity;
        # leave the hole in that case. Value equality, not just same-class
        # (`type(t) is type(first)` would treat `StructT("Foo")` and
        # `StructT("Bar")` -- or `ListT(elem=IntT())` and
        # `ListT(elem=StrT())` -- as "the same type" and silently collapse
        # to whichever was seen first, which is a miscompilation, not a
        # left-open hole).
        first = concrete[0]
        if all(t == first for t in concrete):
            fn.return_type = first
            return
    # No concrete return found. If the function has no explicit return at
    # all (or every return is bare `return`), default to NoneT — implicit
    # None is the universal Python convention and matches `void` in other
    # languages.
    if not return_tys or all(isinstance(t, NoneT) for t in return_tys):
        fn.return_type = NoneT()


def _snapshot(fn: mir.MirFunction, env: dict[str, Type]) -> tuple:
    return (
        tuple(type(p.ty).__name__ for p in fn.params),
        type(fn.return_type).__name__,
        tuple(sorted((k, type(v).__name__) for k, v in env.items())),
    )


def _mir_dump(fn: mir.MirFunction) -> str:
    """Cheap textual MIR dump used as LLM context. Stable enough to cache on."""
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
    if isinstance(n, mir.MirFloatLiteral):
        return repr(n.value)
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
    if isinstance(n, mir.MirBoolOp):
        return f"({_dump_node(n.left)} {n.op} {_dump_node(n.right)})"
    if isinstance(n, mir.MirUnaryOp):
        return f"{n.op}{_dump_node(n.operand)}"
    if isinstance(n, mir.MirCall):
        return f"{n.func}({', '.join(_dump_node(a) for a in n.args)})"
    if isinstance(n, mir.MirName):
        return n.name
    if isinstance(n, mir.MirIntLiteral):
        return str(n.value)
    if isinstance(n, mir.MirBoolLiteral):
        return str(n.value)
    if isinstance(n, mir.MirStringLiteral):
        return json.dumps(n.value)
    if isinstance(n, mir.MirList):
        return f"[{', '.join(_dump_node(e) for e in n.elements)}]"
    if isinstance(n, mir.MirSubscript):
        return f"{_dump_node(n.value)}[{_dump_node(n.index)}]"
    return type(n).__name__


def _children(n: mir.MirNode) -> list[list[mir.MirNode]]:
    if isinstance(n, mir.MirIf):
        return [n.body, n.orelse]
    if isinstance(n, (mir.MirWhile, mir.MirForRange)):
        return [n.body]
    return []

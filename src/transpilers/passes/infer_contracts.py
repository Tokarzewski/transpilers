"""Global semantic-contract inference on MIR.

Populates ``.contract`` on every MIR node with the semantic constraints
the source language imposes.  This pass runs after type inference so it
can use resolved types to narrow contracts.

Current inference rules
-----------------------
* ``IntT`` nodes get ``overflow=ARBITRARY`` (Python unlimited-precision int).
* ``FloatT``, ``BoolT``, ``StrT``, ``NoneT`` get value-semantics contracts.
* ``ListT`` nodes get Python-style reference semantics (shared, GC-like).
* ``MirIntLiteral`` gets an arbitrary-precision int contract.
* Interprocedural: known-callee call sites adopt the callee's inferred purity.
* Structural binds (assigns, params) flow their RHS/arg contract to the target.

Future work
-----------
* Dataflow-based mutability inference beyond the mut pass's simple count.
* Escape analysis for owned-vs-borrowed decisions.
* LLM-driven contract completion for external calls (stdlib, 3rd-party).
"""

from __future__ import annotations

from dataclasses import replace

from transpilers.ir import mir
from transpilers.ir.contracts import (
    OverflowBehavior,
    Ownership,
    SemanticContract,
    ValueCategory,
    WILDCARD,
)
from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    StrT,
    StructT,
    Type,
)

MAX_ITER = 8
FnMap = dict[str, mir.MirFunction]


def infer_contracts(module: mir.MirModule) -> mir.MirModule:
    """Run contract inference on *module* (in-place). Returns *module* for chaining."""
    _infer_module(module)
    fn_map: FnMap = {fn.name: fn for fn in module.functions}
    last = None
    for _ in range(MAX_ITER):
        for fn in module.functions:
            _propagate_interprocedural(fn, fn_map)
        snap = _module_snapshot(module)
        if snap == last:
            break
        last = snap
    return module


def _module_snapshot(module: mir.MirModule) -> tuple:
    snap: list[tuple] = []
    for fn in module.functions:
        sig = (fn.name, fn.contract.pure, fn.contract.overflow)
        for p in fn.params:
            sig += (p.contract.pure, p.contract.overflow)
        snap.append(sig)
    return tuple(snap)


# --------------------------------------------------------------------------- #
# Phase 1 - per-function local inference
# --------------------------------------------------------------------------- #


def _infer_module(module: mir.MirModule) -> None:
    for fn in module.functions:
        _infer_function(fn)


def _infer_function(fn: mir.MirFunction) -> None:
    env: dict[str, SemanticContract] = {}
    for p in fn.params:
        c = _type_to_contract(p.ty)
        p.contract = c
        env[p.name] = c
    _visit_body(fn.body, env, fn)


def _type_to_contract(ty: Type) -> SemanticContract:
    if isinstance(ty, IntT):
        if ty.bits is None or ty.bits > 64:
            return SemanticContract.arbitrary_precision_int()
        return SemanticContract(
            int_width=ty.bits,
            overflow=OverflowBehavior.ARBITRARY,
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
    if isinstance(ty, FloatT):
        return SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
    if isinstance(ty, BoolT):
        return SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
    if isinstance(ty, StrT):
        return SemanticContract(
            value_category=ValueCategory.REF_IMMUTABLE,
            mutable=False,
            ownership=Ownership.SHARED,
            pure=True,
        )
    if isinstance(ty, NoneT):
        return SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
    if isinstance(ty, ListT):
        return SemanticContract.python_ref()
    if isinstance(ty, StructT):
        return SemanticContract.python_ref()
    return WILDCARD


def _visit_body(
    nodes: list[mir.MirNode],
    env: dict[str, SemanticContract],
    fn: mir.MirFunction | None,
) -> None:
    for n in nodes:
        _visit_stmt(n, env, fn)


def _visit_stmt(
    node: mir.MirNode, env: dict[str, SemanticContract], fn: mir.MirFunction | None
) -> None:
    if isinstance(node, mir.MirReturn):
        if node.value is not None:
            _visit_expr(node.value, env)
        return
    if isinstance(node, mir.MirAssign):
        rhs_contract = _visit_expr(node.value, env)
        target_ct = SemanticContract.default_local_binding().merge(rhs_contract)
        if node.augmented_op is not None:
            target_ct = target_ct.merge(SemanticContract(mutable=True))
        env[node.target] = target_ct
        node.contract = target_ct
        return
    if isinstance(node, mir.MirIf):
        _visit_expr(node.test, env)
        _visit_body(node.body, env, fn)
        _visit_body(node.orelse, env, fn)
        return
    if isinstance(node, mir.MirWhile):
        _visit_expr(node.test, env)
        _visit_body(node.body, env, fn)
        return
    if isinstance(node, mir.MirForRange):
        _visit_expr(node.start, env)
        _visit_expr(node.stop, env)
        if node.step is not None:
            _visit_expr(node.step, env)
        env[node.target] = SemanticContract(
            int_width=None,
            overflow=OverflowBehavior.ARBITRARY,
            value_category=ValueCategory.VALUE,
            mutable=True,
            ownership=Ownership.OWNED,
            pure=True,
        )
        _visit_body(node.body, env, fn)
        return
    if isinstance(node, mir.MirFieldAssign):
        _visit_expr(node.obj, env)
        _visit_expr(node.value, env)
        return
    if isinstance(node, mir.MirSubscriptAssign):
        _visit_expr(node.obj, env)
        _visit_expr(node.index, env)
        _visit_expr(node.value, env)
        return
    _visit_expr(node, env)


def _visit_expr(
    node: mir.MirNode, env: dict[str, SemanticContract]
) -> SemanticContract:
    if isinstance(node, mir.MirIntLiteral):
        node.contract = SemanticContract.arbitrary_precision_int()
        return node.contract
    if isinstance(node, mir.MirFloatLiteral):
        node.contract = SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
        return node.contract
    if isinstance(node, mir.MirBoolLiteral):
        node.contract = SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
        return node.contract
    if isinstance(node, mir.MirStringLiteral):
        node.contract = SemanticContract(
            value_category=ValueCategory.REF_IMMUTABLE,
            mutable=False,
            ownership=Ownership.SHARED,
            pure=True,
        )
        return node.contract
    if isinstance(node, mir.MirName):
        ct = env.get(node.name, WILDCARD)
        node.contract = ct
        return ct
    if isinstance(node, mir.MirBinOp):
        lt = _visit_expr(node.left, env)
        rt = _visit_expr(node.right, env)
        merged = lt.merge(rt)
        if isinstance(getattr(node, "ty", None), IntT):
            merged = SemanticContract(
                int_width=merged.int_width,
                overflow=OverflowBehavior.ARBITRARY,
                value_category=ValueCategory.VALUE,
                mutable=False,
                ownership=Ownership.OWNED,
                pure=True,
            )
        node.contract = merged
        return merged
    if isinstance(node, mir.MirCompare):
        _visit_expr(node.left, env)
        _visit_expr(node.right, env)
        node.contract = SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
        return node.contract
    if isinstance(node, mir.MirBoolOp):
        _visit_expr(node.left, env)
        _visit_expr(node.right, env)
        node.contract = SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=True,
        )
        return node.contract
    if isinstance(node, mir.MirUnaryOp):
        node.contract = _visit_expr(node.operand, env)
        return node.contract
    if isinstance(node, mir.MirCall):
        for a in node.args:
            _visit_expr(a, env)
        node.contract = SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=False,
        )
        return node.contract
    if isinstance(node, mir.MirList):
        elem_contracts = [_visit_expr(e, env) for e in node.elements]
        merged = elem_contracts[0] if elem_contracts else WILDCARD
        node.contract = SemanticContract.python_ref().merge(merged)
        return node.contract
    if isinstance(node, mir.MirSubscript):
        _visit_expr(node.value, env)
        _visit_expr(node.index, env)
        node.contract = SemanticContract.python_ref()
        return node.contract
    if isinstance(node, mir.MirFieldAccess):
        _visit_expr(node.value, env)
        node.contract = SemanticContract.python_ref()
        return node.contract
    if isinstance(node, mir.MirMethodCall):
        _visit_expr(node.receiver, env)
        for a in node.args:
            _visit_expr(a, env)
        node.contract = SemanticContract(
            value_category=ValueCategory.VALUE,
            mutable=False,
            ownership=Ownership.OWNED,
            pure=False,
        )
        return node.contract
    if isinstance(node, mir.MirStructInit):
        for _, v in node.field_values:
            _visit_expr(v, env)
        node.contract = SemanticContract.python_ref()
        return node.contract
    return WILDCARD


# --------------------------------------------------------------------------- #
# Phase 2 - interprocedural propagation
# --------------------------------------------------------------------------- #


def _propagate_interprocedural(fn: mir.MirFunction, fn_map: FnMap) -> None:
    _ip_walk_body(fn.body, fn_map)
    # `_ip_walk_node`'s MirCall case (below) reads `callee.contract.pure` to
    # decide whether a call site is pure -- but until this line, `fn.contract`
    # itself was never written, so it stayed at the MirNode default
    # (WILDCARD, pure=False) forever. Every callee looked impure regardless
    # of its actual body, so the whole fixed-point loop in `infer_contracts`
    # converged instantly on "everything is impure" without ever converging
    # on real interprocedural purity. Computing it here from the (now
    # call-site-updated) body is what makes convergence meaningful: on each
    # outer iteration, a function's purity can only improve once its
    # callees' purity has been determined, exactly the fixed point the
    # MAX_ITER loop already exists to reach.
    fn.contract = replace(fn.contract, pure=_body_is_pure(fn.body))


def _ip_walk_body(nodes: list[mir.MirNode], fn_map: FnMap) -> None:
    for n in nodes:
        _ip_walk_node(n, fn_map)


def _ip_walk_node(node: mir.MirNode, fn_map: FnMap) -> None:
    if isinstance(node, mir.MirCall):
        callee = fn_map.get(node.func)
        if callee is not None:
            node.contract = SemanticContract(
                value_category=ValueCategory.VALUE,
                mutable=False,
                ownership=Ownership.OWNED,
                pure=callee.contract.pure,
            )
        for a in node.args:
            _ip_walk_node(a, fn_map)
        return
    if isinstance(node, mir.MirAssign):
        _ip_walk_node(node.value, fn_map)
        return
    if isinstance(node, mir.MirReturn):
        if node.value is not None:
            _ip_walk_node(node.value, fn_map)
        return
    if isinstance(node, mir.MirIf):
        _ip_walk_node(node.test, fn_map)
        _ip_walk_body(node.body, fn_map)
        _ip_walk_body(node.orelse, fn_map)
        return
    if isinstance(node, mir.MirWhile):
        _ip_walk_node(node.test, fn_map)
        _ip_walk_body(node.body, fn_map)
        return
    if isinstance(node, mir.MirForRange):
        _ip_walk_node(node.start, fn_map)
        _ip_walk_node(node.stop, fn_map)
        if node.step is not None:
            _ip_walk_node(node.step, fn_map)
        _ip_walk_body(node.body, fn_map)
        return
    for child in _node_children(node):
        _ip_walk_node(child, fn_map)


def _body_is_pure(nodes: list[mir.MirNode]) -> bool:
    """True iff no call reachable from *nodes* is known-impure, after
    `_ip_walk_body` has already resolved known-callee call sites' purity
    in place. Mirrors `_ip_walk_node`'s statement-level traversal (the
    interprocedural walker doesn't recurse through `_node_children` for
    statement kinds, only for expressions) so every call site is visited."""
    for n in nodes:
        if _stmt_or_expr_is_impure(n):
            return False
    return True


def _stmt_or_expr_is_impure(node: mir.MirNode) -> bool:
    if isinstance(node, (mir.MirCall, mir.MirMethodCall)) and not node.contract.pure:
        return True
    if isinstance(node, mir.MirCall):
        return any(_stmt_or_expr_is_impure(a) for a in node.args)
    if isinstance(node, mir.MirAssign):
        return _stmt_or_expr_is_impure(node.value)
    if isinstance(node, mir.MirReturn):
        return node.value is not None and _stmt_or_expr_is_impure(node.value)
    if isinstance(node, mir.MirIf):
        return (
            _stmt_or_expr_is_impure(node.test)
            or any(_stmt_or_expr_is_impure(n) for n in node.body)
            or any(_stmt_or_expr_is_impure(n) for n in node.orelse)
        )
    if isinstance(node, mir.MirWhile):
        return _stmt_or_expr_is_impure(node.test) or any(
            _stmt_or_expr_is_impure(n) for n in node.body
        )
    if isinstance(node, mir.MirForRange):
        return (
            _stmt_or_expr_is_impure(node.start)
            or _stmt_or_expr_is_impure(node.stop)
            or (node.step is not None and _stmt_or_expr_is_impure(node.step))
            or any(_stmt_or_expr_is_impure(n) for n in node.body)
        )
    if isinstance(node, mir.MirFieldAssign):
        return _stmt_or_expr_is_impure(node.obj) or _stmt_or_expr_is_impure(node.value)
    if isinstance(node, mir.MirSubscriptAssign):
        return (
            _stmt_or_expr_is_impure(node.obj)
            or _stmt_or_expr_is_impure(node.index)
            or _stmt_or_expr_is_impure(node.value)
        )
    return any(_stmt_or_expr_is_impure(c) for c in _node_children(node))


def _node_children(node: mir.MirNode) -> list[mir.MirNode]:
    if isinstance(node, mir.MirBinOp):
        return [node.left, node.right]
    if isinstance(node, mir.MirCompare):
        return [node.left, node.right]
    if isinstance(node, mir.MirBoolOp):
        return [node.left, node.right]
    if isinstance(node, mir.MirUnaryOp):
        return [node.operand]
    if isinstance(node, mir.MirList):
        return node.elements
    if isinstance(node, mir.MirSubscript):
        return [node.value, node.index]
    if isinstance(node, mir.MirFieldAccess):
        return [node.value]
    if isinstance(node, mir.MirMethodCall):
        return [node.receiver] + node.args
    if isinstance(node, mir.MirStructInit):
        return [v for _, v in node.field_values]
    return []


__all__ = ["infer_contracts"]

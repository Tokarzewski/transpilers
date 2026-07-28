"""Shared skeleton for the per-target MIR -> LIR lowering passes.

The seven `mir_to_<target>_lir` passes are structurally parallel: each walks
the MIR tree with an identical `lower_stmt` / `lower_expr` dispatch and differs
only in (a) which LIR dialect node it constructs at each point, (b) operator /
keyword maps, and (c) a handful of target-specific special cases (`len`, SIMD,
`pow_N`, `math.*`, string concat, ...).

This module factors out:

  * the MIR-walk **analysis helpers** that several targets duplicate verbatim
    (mutability inference, parameter-reassignment / subscript-assignment scans);
  * a `LirNS` node-namespace adapter that maps a generic node kind
    (`ns.Return(...)`) onto a dialect class (`lir.RustReturn(...)`), exploiting
    the uniform `<Prefix><Kind>` naming convention; and
  * a `MirLoweringBase` visitor that owns the shared dispatch skeleton and
    delegates every point of variation to a small, overridable hook method.

Each `mir_to_<target>_lir.py` becomes: a thin `MirLoweringBase` subclass that
sets `prefix` / `module_cls` and overrides only the hooks where the target
genuinely diverges. Targets too idiosyncratic to fit (Fortran's result-var and
declaration-block model, Python's annotation-only assigns) stay hand-written.
"""

from __future__ import annotations

from transpilers.ir import lir, mir
from transpilers.ir.types import (
    ListT,
    StrT,
    Type,
)

__all__ = [
    "LirNS",
    "MirLoweringBase",
    "collect_mutable",
    "scan_reassigned_params",
    "scan_subscript_assigned_params",
    "is_string_concat",
    "is_list_concat",
    "copy_provenance",
]


# -- provenance helper -------------------------------------------------------

def copy_provenance(lir_node: lir.LirNode, mir_node: mir.MirNode) -> lir.LirNode:
    """Copy ``mir_node._hir_provenance_id`` to ``lir_node._hir_provenance_id``.

    Call this after every LIR node construction to thread provenance from MIR
    through the lowering pass.  Returns the LIR node for chaining.
    """
    lir_node._hir_provenance_id = mir_node._hir_provenance_id
    return lir_node


# --------------------------------------------------------------------------- #
# Shared MIR-walk analysis helpers
# --------------------------------------------------------------------------- #
#
# These walk the nested statement structure (if / while / for-range bodies)
# the same way in every target. They were copy-pasted across rust/zig/mojo;
# this is the single source of truth.


def _walk_blocks(node: mir.MirNode):
    """Yield the child statement lists of a control-flow node (or nothing)."""
    if isinstance(node, mir.MirIf):
        yield node.body
        yield node.orelse
    elif isinstance(node, mir.MirWhile):
        yield node.body
    elif isinstance(node, mir.MirForRange):
        yield node.body


def collect_mutable(body: list[mir.MirNode]) -> set[str]:
    """Names that must be declared mutable: assigned more than once, assigned
    with an augmented op, or backing a field-assignment receiver."""
    counts: dict[str, int] = {}
    aug: set[str] = set()

    def _scan(nodes: list[mir.MirNode]) -> None:
        for n in nodes:
            if isinstance(n, mir.MirAssign):
                counts[n.target] = counts.get(n.target, 0) + 1
                if n.augmented_op is not None:
                    aug.add(n.target)
            elif isinstance(n, mir.MirFieldAssign):
                if isinstance(n.obj, mir.MirName):
                    aug.add(n.obj.name)
            for block in _walk_blocks(n):
                _scan(block)

    _scan(body)
    return {name for name, c in counts.items() if c > 1} | aug


def scan_reassigned_params(body: list[mir.MirNode], param_names: set[str]) -> set[str]:
    """Params that are (non-augmented or augmented) reassigned in the body."""
    out: set[str] = set()

    def _scan(nodes: list[mir.MirNode]) -> None:
        for n in nodes:
            if isinstance(n, mir.MirAssign) and n.target in param_names:
                out.add(n.target)
            for block in _walk_blocks(n):
                _scan(block)

    _scan(body)
    return out


def scan_subscript_assigned_params(
    body: list[mir.MirNode], param_names: set[str]
) -> set[str]:
    """Params mutated via `xs[i] = v` (need a mutable slice / &mut)."""
    out: set[str] = set()

    def _scan(nodes: list[mir.MirNode]) -> None:
        for n in nodes:
            if isinstance(n, mir.MirSubscriptAssign):
                if isinstance(n.obj, mir.MirName) and n.obj.name in param_names:
                    out.add(n.obj.name)
            for block in _walk_blocks(n):
                _scan(block)

    _scan(body)
    return out


def is_string_concat(node: mir.MirBinOp) -> bool:
    return (
        node.op == "+"
        and isinstance(getattr(node.left, "ty", None), StrT)
        and isinstance(getattr(node.right, "ty", None), StrT)
    )


def is_list_concat(node: mir.MirBinOp) -> bool:
    return (
        node.op == "+"
        and isinstance(getattr(node.left, "ty", None), ListT)
        and isinstance(getattr(node.right, "ty", None), ListT)
    )


# --------------------------------------------------------------------------- #
# Node namespace
# --------------------------------------------------------------------------- #


class LirNS:
    """Adapter that resolves a generic node kind onto a dialect class.

    `ns.Return` -> `lir.RustReturn` for `LirNS("Rust")`. Exploits the uniform
    `<Prefix><Kind>` naming convention shared by every LIR dialect, so the
    base visitor can construct structurally-identical nodes without knowing
    the target. Missing names raise `AttributeError` early (at first use),
    which is what we want for a target that lacks a given node kind.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def __getattr__(self, kind: str):
        try:
            return getattr(lir, f"{self._prefix}{kind}")
        except AttributeError as exc:  # pragma: no cover - developer error
            raise AttributeError(
                f"LIR dialect {self._prefix!r} has no node {self._prefix}{kind}"
            ) from exc


# --------------------------------------------------------------------------- #
# Base lowering visitor
# --------------------------------------------------------------------------- #


class MirLoweringBase:
    """Generic MIR -> LIR lowering skeleton.

    Subclasses set ``prefix`` (the dialect class prefix, e.g. ``"Rust"``) and
    ``module_cls`` (the dialect's module node), then override the handful of
    hook methods where the target diverges. Everything structurally identical
    across targets lives here once.

    The visitor is *stateless per call* apart from the ``declared`` / ``mut``
    sets threaded through ``lower_stmt`` — matching the original free-function
    passes — so a single instance can lower a whole module.
    """

    prefix: str = ""
    module_cls: type | None = None

    def __init__(self) -> None:
        self.ns = LirNS(self.prefix)

    # -- type map (must be provided) --------------------------------------- #

    def type_str(self, ty: Type) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- module / struct / function --------------------------------------- #

    def lower_module(self, module: mir.MirModule):
        items: list[lir.LirNode] = []
        for s in module.structs:
            items.extend(self.lower_struct_items(s))
        for fn in module.functions:
            items.append(self.lower_function(fn))
        return self.make_module(items, module)

    def make_module(self, items: list[lir.LirNode], mir_node: mir.MirNode | None = None):
        assert self.module_cls is not None
        node = self.module_cls(items=items)
        if mir_node is not None:
            copy_provenance(node, mir_node)
        return node

    def lower_struct_items(self, s: mir.MirStruct) -> list[lir.LirNode]:
        """Return the LIR item(s) a struct lowers to. Default: one node with
        ``name`` / ``fields`` / ``methods``. Targets that split a struct into
        several items (e.g. Rust's def + impl) override this."""
        items = [
            self.ns.Struct(
                name=s.name,
                fields=[(f.name, self.type_str(f.ty)) for f in s.fields],
                methods=[self.lower_function(m) for m in s.methods],
            )
        ]
        return items

    def lower_function(self, fn: mir.MirFunction):
        param_names = {p.name for p in fn.params}
        params = self.lower_params(fn)
        ret = self.return_type(fn)
        declared, mut, preamble = self.function_preamble(fn, param_names)
        # `lower_stmt` may return None to *drop* a statement (e.g. a target with
        # no `assert` — see the Mojo backend); filter those out.
        lowered = [self.lower_stmt(n, declared, mut) for n in fn.body]
        body = preamble + [s for s in lowered if s is not None]
        return self.make_function(fn, params, ret, body)

    def make_function(self, fn, params, ret, body):
        return copy_provenance(
            self.ns.Fn(name=fn.name, params=params, return_type=ret, body=body),
            fn,
        )

    def lower_params(self, fn: mir.MirFunction):
        """Default: ``(name, type)`` tuples. Targets that decorate params
        (rust `mut`, zig const-shadowing, mojo `var`/`mut`) override."""
        return [(p.name, self.type_str(p.ty)) for p in fn.params]

    def return_type(self, fn: mir.MirFunction):
        return self.type_str(fn.return_type)

    def function_preamble(self, fn, param_names):
        """Return ``(declared, mut, preamble_nodes)``.

        ``declared`` seeds the set of names already in scope (params), ``mut``
        the mutability set, ``preamble_nodes`` any statements injected before
        the body. Default seeds params as declared with no mutability tracking
        and no preamble."""
        return set(param_names), set(), []

    # -- statements -------------------------------------------------------- #

    def lower_stmt(self, node: mir.MirNode, declared: set[str], mut: set[str]):
        if isinstance(node, mir.MirRaw):
            return self.lower_raw(node)
        if isinstance(node, mir.MirReturn):
            return self.lower_return(node)
        if isinstance(node, mir.MirBreak):
            return copy_provenance(self.ns.Break(), node)
        if isinstance(node, mir.MirContinue):
            return copy_provenance(self.ns.Continue(), node)
        if isinstance(node, mir.MirFieldAssign):
            return self.lower_field_assign(node)
        if isinstance(node, mir.MirSubscriptAssign):
            return self.lower_subscript_assign(node)
        if isinstance(node, mir.MirAssign):
            return self.lower_assign(node, declared, mut)
        if isinstance(node, mir.MirIf):
            return copy_provenance(
                self.ns.If(
                    test=self.lower_expr(node.test),
                    body=self._lower_block(node.body, declared, mut),
                    orelse=self._lower_block(node.orelse, declared, mut),
                ),
                node,
            )
        if isinstance(node, mir.MirWhile):
            return copy_provenance(
                self.ns.While(
                    test=self.lower_expr(node.test),
                    body=self._lower_block(node.body, declared, mut),
                ),
                node,
            )
        if isinstance(node, mir.MirForRange):
            return self.lower_for_range(node, declared, mut)
        return self.lower_expr(node)

    def lower_return(self, node: mir.MirReturn):
        return copy_provenance(
            self.ns.Return(value=self.lower_expr(node.value) if node.value else None),
            node,
        )

    def lower_field_assign(self, node: mir.MirFieldAssign):
        return copy_provenance(
            self.ns.FieldAssign(
                obj=self.lower_expr(node.obj),
                field=node.field,
                value=self.lower_expr(node.value),
            ),
            node,
        )

    def lower_subscript_assign(self, node: mir.MirSubscriptAssign):
        return copy_provenance(
            self.ns.SubscriptAssign(
                obj=self.lower_expr(node.obj),
                index=self.lower_expr(node.index),
                value=self.lower_expr(node.value),
            ),
            node,
        )

    def lower_for_range(self, node: mir.MirForRange, declared: set[str], mut: set[str]):
        return copy_provenance(
            self.ns.ForRange(
                target=node.target,
                start=self.lower_expr(node.start),
                stop=self.lower_expr(node.stop),
                step=self.lower_expr(node.step) if node.step else None,
                body=self._lower_block(node.body, declared, mut),
            ),
            node,
        )

    def _lower_block(self, nodes, declared, mut):
        """Lower a statement block, dropping any statement that lowers to None
        (a backend may intentionally drop e.g. `assert`)."""
        lowered = [self.lower_stmt(n, declared, mut) for n in nodes]
        return [s for s in lowered if s is not None]

    def lower_assign(self, node: mir.MirAssign, declared: set[str], mut: set[str]):  # pragma: no cover - abstract
        raise NotImplementedError

    # -- expressions ------------------------------------------------------- #

    def lower_expr(self, node: mir.MirNode):
        if isinstance(node, mir.MirRaw):
            return self.lower_raw(node)
        special = self.lower_expr_special(node)
        if special is not None:
            return special
        if isinstance(node, mir.MirFieldAccess):
            return self.lower_field_access(node)
        if isinstance(node, mir.MirStructInit):
            return copy_provenance(
                self.ns.StructInit(
                    name=node.name,
                    field_values=[(n, self.lower_arg(v)) for n, v in node.field_values],
                ),
                node,
            )
        if isinstance(node, mir.MirMethodCall):
            return self.lower_method_call(node)
        if isinstance(node, mir.MirBinOp):
            return self.lower_binop(node)
        if isinstance(node, mir.MirCompare):
            return self.lower_compare(node)
        if isinstance(node, mir.MirBoolOp):
            return self.lower_boolop(node)
        if isinstance(node, mir.MirUnaryOp):
            return self.lower_unary(node)
        if isinstance(node, mir.MirName):
            return copy_provenance(self.ns.Name(name=node.name), node)
        if isinstance(node, mir.MirIntLiteral):
            return copy_provenance(self.ns.IntLiteral(value=node.value), node)
        if isinstance(node, mir.MirFloatLiteral):
            return copy_provenance(self.ns.FloatLiteral(value=node.value), node)
        if isinstance(node, mir.MirBoolLiteral):
            return copy_provenance(self.ns.BoolLiteral(value=node.value), node)
        if isinstance(node, mir.MirStringLiteral):
            return copy_provenance(self.ns.StringLiteral(value=node.value), node)
        if isinstance(node, mir.MirNullLiteral):
            return self.lower_null(node)
        if isinstance(node, mir.MirCall):
            return self.lower_call(node)
        if isinstance(node, mir.MirList):
            return self.lower_list(node)
        if isinstance(node, mir.MirSubscript):
            return self.lower_subscript(node)
        raise NotImplementedError(f"MIR expr {type(node).__name__}")

    def lower_expr_special(self, node: mir.MirNode):
        """Hook for target-specific node handling that must run *before* the
        generic dispatch (e.g. C/Go/Fortran's `//`-as-`/` rewrite). Return a
        lowered node to short-circuit, or ``None`` to fall through."""
        return None

    def lower_field_access(self, node: mir.MirFieldAccess):
        return copy_provenance(
            self.ns.FieldAccess(value=self.lower_expr(node.value), field=node.field),
            node,
        )

    def lower_arg(self, node: mir.MirNode):
        """Lower a value passed *by value* into a call (a plain function/
        method call argument, or a constructor's field-init argument).
        Default: identical to lower_expr. Targets with an ownership model
        where passing a bare reference into a call needs an explicit copy
        (Mojo: `f(self.field)` needs `f(self.field.copy())` when `self.field`
        is a struct) override this; a receiver expression is deliberately
        *not* routed through here, since a method call borrows its receiver
        rather than consuming it."""
        return self.lower_expr(node)

    def lower_method_call(self, node: mir.MirMethodCall):
        return copy_provenance(
            self.ns.MethodCall(
                receiver=self.lower_expr(node.receiver),
                method=node.method,
                args=[self.lower_arg(a) for a in node.args],
            ),
            node,
        )

    def lower_compare(self, node: mir.MirCompare):
        return copy_provenance(
            self.ns.Compare(
                op=node.op,
                left=self.lower_expr(node.left),
                right=self.lower_expr(node.right),
            ),
            node,
        )

    def lower_boolop(self, node: mir.MirBoolOp):
        return copy_provenance(
            self.ns.BoolOp(
                op=node.op,
                left=self.lower_expr(node.left),
                right=self.lower_expr(node.right),
            ),
            node,
        )

    def lower_unary(self, node: mir.MirUnaryOp):
        op = "!" if node.op == "not" else "-"
        return copy_provenance(
            self.ns.Unary(op=op, operand=self.lower_expr(node.operand)),
            node,
        )

    def lower_binop(self, node: mir.MirBinOp):  # pragma: no cover - abstract
        raise NotImplementedError

    def lower_call(self, node: mir.MirCall):  # pragma: no cover - abstract
        raise NotImplementedError

    def lower_list(self, node: mir.MirList):  # pragma: no cover - abstract
        raise NotImplementedError

    def lower_subscript(self, node: mir.MirSubscript):
        return copy_provenance(
            self.ns.Index(
                value=self.lower_expr(node.value), index=self.lower_expr(node.index)
            ),
            node,
        )

    def lower_null(self, node: mir.MirNullLiteral):  # pragma: no cover - abstract
        raise NotImplementedError

    def lower_raw(self, node: mir.MirRaw):
        """Map a never-refuse hole onto the dialect's ``<Prefix>Raw`` node.
        Identical for every target, so no per-target override is needed; the
        backend's emitter decides how to render the stub."""
        return copy_provenance(
            self.ns.Raw(snippet=node.snippet),
            node,
        )

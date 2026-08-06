"""MIR -> Rust LIR.

Target-shaping. Decides:
  - which assignments are declarations (`let`) vs reassignments
  - which bindings need `mut`
  - how `len(...)`, `range(...)` and similar builtins shape into Rust
  - which Python types map to which Rust types

Idiom rewrites (Python comprehensions -> .iter().map().collect()) belong in
dedicated idiom passes that may consult LLM hooks. This pass stays
algorithmic.

Structure is shared with the other backends via ``_mir_lower_base``; this
module supplies only the Rust-specific spec (type map, mutability decoration,
let/reassign split, builtin call table) and the private marker nodes the Rust
emitter consumes.
"""

from __future__ import annotations

from transpilers.ir import lir, mir
from transpilers.ir.contracts import OverflowBehavior, SemanticContract
from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    StrT,
    StructT,
    Type,
    UnknownT,
)

from ._mir_lower_base import (
    MirLoweringBase,
    collect_mutable,
    copy_provenance,
    is_list_concat,
    is_string_concat,
    scan_reassigned_params,
    scan_subscript_assigned_params,
)


class _RustLowering(MirLoweringBase):
    prefix = "Rust"
    module_cls = lir.RustModule

    def type_str(self, ty: Type) -> str:
        return _rust_type(ty)

    # -- struct: Rust splits into a def + an impl block -------------------- #

    def lower_struct_items(self, s: mir.MirStruct) -> list[lir.LirNode]:
        return [
            lir.RustStruct(
                name=s.name, fields=[(f.name, _rust_type(f.ty)) for f in s.fields]
            ),
            lir.RustImpl(
                struct_name=s.name, methods=[self.lower_function(m) for m in s.methods]
            ),
        ]

    # -- function signature: mut params + &mut for subscript-assigned ------ #

    def lower_params(self, fn: mir.MirFunction):
        mut_names = collect_mutable(fn.body)
        param_names = {p.name for p in fn.params}
        reassigned = scan_reassigned_params(fn.body, param_names)
        subscript_assigned = scan_subscript_assigned_params(fn.body, param_names)
        return [
            (
                f"mut {p.name}"
                if (p.name in mut_names or p.name in reassigned)
                else p.name,
                _rust_param_type(p.ty, mutable=p.name in subscript_assigned),
            )
            for p in fn.params
        ]

    def function_preamble(self, fn, param_names):
        return set(param_names), collect_mutable(fn.body), []

    # -- assign: let vs reassign, augmented unfold ------------------------- #

    def lower_assign(self, node: mir.MirAssign, declared: set[str], mut: set[str]):
        if node.augmented_op is not None:
            # x += value  →  x = x + value  (explicit form composes more
            # cleanly with type promotion later).
            rhs = copy_provenance(
                lir.RustBinOp(
                    op=node.augmented_op,
                    left=lir.RustName(name=node.target),
                    right=self.lower_expr(node.value),
                ),
                node,
            )
            return copy_provenance(lir.RustReassign(name=node.target, value=rhs), node)
        if node.target in declared:
            return copy_provenance(
                lir.RustReassign(name=node.target, value=self.lower_expr(node.value)),
                node,
            )
        declared.add(node.target)
        # Unknown inferred type → omit annotation; Rust's local inference
        # picks it up from the initializer.
        try:
            ty_str = _rust_type(node.ty) if not isinstance(node.ty, UnknownT) else None
        except ValueError:
            ty_str = None
        is_list = isinstance(node.ty, ListT)
        return copy_provenance(
            lir.RustLet(
                name=node.target,
                mutable=(node.target in mut) or is_list,
                ty=ty_str,
                value=self.lower_expr(node.value),
            ),
            node,
        )

    # -- expressions ------------------------------------------------------- #

    def lower_binop(self, node: mir.MirBinOp):
        if is_string_concat(node):
            return lir.RustFormat(args=_flatten_concat(self, node))
        if is_list_concat(node):
            return _RustListConcat(
                left=self.lower_expr(node.left), right=self.lower_expr(node.right)
            )
        # Python `//` (FloorDivide) → Rust `/` on integer types.
        op = "/" if node.op == "//" else node.op
        # ── Overflow-safe emission ──────────────────────────────────────────
        # If the MIR node carries a semantic contract saying `ARBITRARY`
        # precision (Python int), but we know the Rust type is fixed-width
        # (i64), we must emit a wrapping or checked operation.  The contract
        # tells us what the source expects; we emit the Rust equivalent or
        # annotate the binop so the emitter can decide.
        contract: SemanticContract = getattr(node, "contract", SemanticContract())
        if contract.overflow is OverflowBehavior.ARBITRARY and op in ("+", "-", "*"):
            # Arbitrary-precision source ints flowing into a fixed-width
            # target need explicit overflow handling.  Emit a wrapping
            # operation as the safe default (matches Rust's `wrapping_add`,
            # `wrapping_sub`, `wrapping_mul`).
            style = _overflow_style(contract)
            return copy_provenance(
                _RustOverflowGuard(
                    op=op,
                    left=self.lower_expr(node.left),
                    right=self.lower_expr(node.right),
                    style=style,
                ),
                node,
            )
        return copy_provenance(
            lir.RustBinOp(
                op=op,
                left=self.lower_expr(node.left),
                right=self.lower_expr(node.right),
            ),
            node,
        )

    def lower_boolop(self, node: mir.MirBoolOp):
        op = "&&" if node.op == "and" else "||"
        return copy_provenance(
            lir.RustBoolOp(
                op=op,
                left=self.lower_expr(node.left),
                right=self.lower_expr(node.right),
            ),
            node,
        )

    def lower_null(self, node: mir.MirNullLiteral):
        # No OptionT in the type lattice yet; emit a bare `None` so the
        # downstream rustc surfaces the missing context.
        return copy_provenance(lir.RustName(name="None"), node)

    def lower_list(self, node: mir.MirList):
        return copy_provenance(
            lir.RustVec(elements=[self.lower_expr(e) for e in node.elements]),
            node,
        )

    def lower_subscript(self, node: mir.MirSubscript):
        return copy_provenance(
            lir.RustIndex(
                value=self.lower_expr(node.value), index=self.lower_expr(node.index)
            ),
            node,
        )

    def lower_call(self, node: mir.MirCall):
        # Stdlib mapping table — turn well-known Python-style builtins into
        # idiomatic Rust so the output is runnable, not just syntactically OK.
        args = [self.lower_expr(a) for a in node.args]
        if node.func == "__ternary__" and len(args) == 3:
            return _RustIfExpr(test=args[0], then_=args[1], else_=args[2])
        if node.func == "len":
            if len(args) != 1:
                raise ValueError("len() takes exactly one argument")
            return copy_provenance(
                lir.RustMethodCall(
                    receiver=args[0], method="len", args=[], cast_to="i64"
                ),
                node,
            )
        if node.func in ("print", "println"):
            # Rewrap each arg so the rendering matches Python's str(): bools
            # become "True"/"False", floats use `{:?}` (Rust Debug for f64
            # preserves the trailing `.0` that Display drops). Build the
            # template dynamically per-arg.
            tokens: list[str] = []
            rendered_args: list[lir.LirNode] = []
            for orig, lowered in zip(node.args, args):
                refined = _pyprint_arg(orig, lowered)
                tokens.append("{:?}" if isinstance(refined, _RustPyFloat) else "{}")
                rendered_args.append(refined)
            template = " ".join(tokens)
            return copy_provenance(
                lir.RustMacro(name="println", template=template, args=rendered_args),
                node,
            )
        if node.func == "sum" and len(args) == 1:
            # `sum(xs)` → `xs.iter().sum()`. No cast: the element type flows
            # through, so `.sum()`'s output is inferred from the iterator
            # (an `as i64` here would silently truncate an f64 list). The
            # result type is resolved by the surrounding context (a typed
            # `let`/return), matching how Rust infers `.sum()`.
            iter_chain = lir.RustMethodChain(receiver=args[0], method="iter", args=[])
            return lir.RustMethodChain(receiver=iter_chain, method="sum", args=[])
        if node.func == "abs" and len(args) == 1:
            return copy_provenance(
                lir.RustMethodChain(receiver=args[0], method="abs", args=[]),
                node,
            )
        if node.func == "min" and len(args) == 2:
            return copy_provenance(
                lir.RustMethodChain(receiver=args[0], method="min", args=[args[1]]),
                node,
            )
        if node.func == "max" and len(args) == 2:
            return copy_provenance(
                lir.RustMethodChain(receiver=args[0], method="max", args=[args[1]]),
                node,
            )
        if node.func == "int" and len(args) == 1:
            return copy_provenance(
                lir.RustBinOp(op="as", left=args[0], right=lir.RustName(name="i64")),
                node,
            )
        if node.func == "float" and len(args) == 1:
            return copy_provenance(
                lir.RustBinOp(op="as", left=args[0], right=lir.RustName(name="f64")),
                node,
            )
        if node.func == "bool" and len(args) == 1:
            return copy_provenance(
                lir.RustCompare(
                    op="!=", left=args[0], right=lir.RustIntLiteral(value=0)
                ),
                node,
            )
        if node.func == "str" and len(args) == 1:
            return copy_provenance(
                lir.RustMethodChain(receiver=args[0], method="to_string", args=[]),
                node,
            )
        # Default: direct function call. Pass list arguments by reference so
        # the caller's binding isn't moved.
        refined: list[lir.LirNode] = []
        for orig, lowered in zip(node.args, args):
            if isinstance(getattr(orig, "ty", None), ListT):
                refined.append(_RustRef(value=lowered))
            else:
                refined.append(lowered)
        return copy_provenance(
            lir.RustCall(func=node.func, args=refined),
            node,
        )


_LOWERING = _RustLowering()


def mir_to_rust_lir(module: mir.MirModule) -> lir.RustModule:
    return _LOWERING.lower_module(module)


def _rust_param_type(ty: Type, *, mutable: bool = False) -> str:
    if isinstance(ty, ListT):
        ref = "&mut" if mutable else "&"
        try:
            return f"{ref} Vec<{_rust_type(ty.elem)}>"
        except ValueError:
            return f"{ref} Vec<_>"
    return _rust_type(ty)


def _pyprint_arg(orig: mir.MirNode, lowered: lir.LirNode) -> lir.LirNode:
    """Wrap `lowered` so it renders the way Python's `print` would.
    Bools become `"True"` / `"False"` strings via an inline ternary.
    Floats are marked with `_RustPyFloat` so the caller can emit `{:?}`
    in the format string (Rust Debug for f64 preserves the trailing `.0`
    that Display drops)."""
    ty = getattr(orig, "ty", None)
    if isinstance(ty, BoolT):
        return _RustIfExpr(
            test=lowered,
            then_=lir.RustStringLiteral(value="True"),
            else_=lir.RustStringLiteral(value="False"),
        )
    if isinstance(ty, FloatT):
        return _RustPyFloat(value=lowered)
    return lowered


class _RustIfExpr(lir.LirNode):
    """`if <test> { <then> } else { <else> }` — Rust if-as-expression.
    Used by `_pyprint_arg` for bool→Python-cap rendering."""

    def __init__(
        self, test: lir.LirNode, then_: lir.LirNode, else_: lir.LirNode
    ) -> None:
        self.test = test
        self.then_ = then_
        self.else_ = else_


class _RustPyFloat(lir.LirNode):
    """`{:?}` format marker. The enclosing print lowering emits `{:?}` in the
    template string for this arg position; the emitter renders `.value`
    directly (the Debug specifier handles the `.0` suffix)."""

    def __init__(self, value: lir.LirNode) -> None:
        self.value = value


class _RustRef(lir.LirNode):
    """`&mut value` — mutable reference. Vec params always use `&mut`
    so callee can do subscript-assigns without a separate borrow path."""

    def __init__(self, value: lir.LirNode) -> None:
        self.value = value


class _RustListConcat(lir.LirNode):
    """Python `left + right` where both sides are lists. Emits an inline
    block that clones `left`, extends it with `right`'s elements, and
    yields the combined Vec:
        `{ let mut _t = <left>.clone(); _t.extend(<right>); _t }`"""

    def __init__(self, left: lir.LirNode, right: lir.LirNode) -> None:
        self.left = left
        self.right = right


def _flatten_concat(lowering: _RustLowering, node: mir.MirBinOp) -> list[lir.LirNode]:
    out: list[lir.LirNode] = []
    for side in (node.left, node.right):
        if isinstance(side, mir.MirBinOp) and is_string_concat(side):
            out.extend(_flatten_concat(lowering, side))
        else:
            out.append(lowering.lower_expr(side))
    return out


def _rust_type(ty: Type) -> str:
    if isinstance(ty, IntT):
        return f"{'i' if ty.signed else 'u'}{ty.bits}"
    if isinstance(ty, FloatT):
        return f"f{ty.bits}"
    if isinstance(ty, BoolT):
        return "bool"
    if isinstance(ty, StrT):
        return "String"
    if isinstance(ty, NoneT):
        return "()"
    if isinstance(ty, ListT):
        # Recursive unknown element types fall back to the `_` placeholder
        # so the surrounding `Vec<_>` can be inferred from initializer.
        try:
            return f"Vec<{_rust_type(ty.elem)}>"
        except ValueError:
            return "Vec<_>"
    if isinstance(ty, StructT):
        return ty.name
    if isinstance(ty, UnknownT):
        raise ValueError(f"unresolved type hole: {ty.hint}")
    raise NotImplementedError(f"type {type(ty).__name__}")


def _overflow_style(contract: SemanticContract) -> str:
    """Map a source-language overflow contract to a Rust overflow-method prefix.

    Returns ``"wrapping"``, ``"checked"``, or ``"saturating"``.
    """
    if contract.overflow in (
        OverflowBehavior.ARBITRARY,
        OverflowBehavior.UNSPECIFIED,
        OverflowBehavior.WRAP,
    ):
        return "wrapping"
    if contract.overflow is OverflowBehavior.CHECKED:
        return "checked"
    if contract.overflow is OverflowBehavior.SATURATE:
        return "saturating"
    return "wrapping"


class _RustOverflowGuard(lir.LirNode):
    """Overflow-aware arithmetic. Emits `(a).wrapping_<op>(b)` when the source
    (Python arbitrary-precision int) could overflow in the target (i64).
    Also supports `checked_*` and `saturating_*` forms based on contract."""

    def __init__(
        self, op: str, left: lir.LirNode, right: lir.LirNode, style: str = "wrapping"
    ) -> None:
        self.op = op
        self.left = left
        self.right = right
        self.style = style  # "wrapping", "checked", or "saturating"

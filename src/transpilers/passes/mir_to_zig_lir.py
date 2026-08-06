"""MIR -> Zig LIR.

Mirrors the other backends in structure (via ``_mir_lower_base``) but produces
a Zig-shaped dialect. Differences worth noting:
  - var/const split replaces let/let-mut
  - `for (a..b) |i|` for unit-step ranges; stepped ranges desugar to `while`
  - `.len` on slices is a property access (no parens), so we model it as a
    method call with no args and let the emitter drop the parens
  - integer `/` and `%` require the `@divTrunc` / `@rem` builtins
"""

from __future__ import annotations

from transpilers.ir import lir, mir
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
    is_string_concat,
    scan_reassigned_params,
    scan_subscript_assigned_params,
)


class _ZigLowering(MirLoweringBase):
    prefix = "Zig"
    module_cls = lir.ZigModule

    def type_str(self, ty: Type) -> str:
        return _zig_type(ty)

    # -- function signature: const-shadow reassigned params, slice params -- #

    def lower_function(self, fn: mir.MirFunction):
        param_names = {p.name for p in fn.params}
        mutated_list_params = scan_subscript_assigned_params(fn.body, param_names)
        # Zig function parameters are const. If a parameter is reassigned in
        # the body, rename it in the signature (`n` → `n_`) and declare a
        # mutable shadow `var n: T = n_;` at the top of the body.
        reassigned_params = scan_reassigned_params(fn.body, param_names)
        params = []
        for p in fn.params:
            param_name = f"{p.name}_" if p.name in reassigned_params else p.name
            params.append(
                (param_name, _zig_list_param_type(p.ty, p.name, mutated_list_params))
            )
        ret = _zig_type(fn.return_type)
        mut_names = collect_mutable(fn.body)
        declared: set[str] = param_names.copy()
        preamble: list[lir.LirNode] = []
        for p in fn.params:
            if p.name in reassigned_params:
                # `var n: i64 = n_;` — binds the renamed param into a mutable local.
                preamble.append(
                    lir.ZigVar(
                        name=p.name,
                        mutable=True,
                        ty=_zig_type(p.ty) if not isinstance(p.ty, UnknownT) else None,
                        value=lir.ZigName(name=f"{p.name}_"),
                    )
                )
        body = preamble + [self.lower_stmt(n, declared, mut_names) for n in fn.body]
        return lir.ZigFn(name=fn.name, params=params, return_type=ret, body=body)

    # -- for-range: unit-step `for`, otherwise desugar to `while` ---------- #

    def lower_for_range(self, node: mir.MirForRange, declared: set[str], mut: set[str]):
        if node.step is None:
            # When the stop is a `len(xs)` call, use a while loop with an i64
            # counter instead of `for (0..xs.len) |i|`. The Zig for-range
            # yields `usize`, but the body may use `i` as `i64` (e.g.
            # `return i`); a while loop keeps the counter as i64 consistently.
            if _is_len_call(node.stop):
                target = node.target
                declared.add(target)
                return self._stepped_while_explicit(
                    target,
                    node,
                    declared,
                    mut,
                    start=self.lower_expr(node.start),
                    stop=self.lower_expr(node.stop),
                )
            return lir.ZigForRange(
                target=node.target,
                start=self.lower_expr(node.start),
                stop=self.lower_expr(node.stop),
                body=[self.lower_stmt(n, declared, mut) for n in node.body],
            )
        # Stepped range: emit an explicit while-loop with a synthesized `var`.
        target = node.target
        declared.add(target)
        return self._stepped_while(target, node, declared, mut)

    def _stepped_while(self, target, node, declared, mut):
        step = (
            self.lower_expr(node.step)
            if node.step is not None
            else lir.ZigIntLiteral(value=1)
        )
        init = lir.ZigVar(
            name=target, mutable=True, ty="i64", value=self.lower_expr(node.start)
        )
        cond = lir.ZigCompare(
            op="<", left=lir.ZigName(name=target), right=self.lower_expr(node.stop)
        )
        incr = lir.ZigReassign(
            name=target,
            value=lir.ZigBinOp(op="+", left=lir.ZigName(name=target), right=step),
        )
        body = [self.lower_stmt(n, declared, mut) for n in node.body]
        body.append(incr)
        return _ZigBlock(items=[init, lir.ZigWhile(test=cond, body=body)])

    def _stepped_while_explicit(self, target, node, declared, mut, start, stop):
        """Emit `for i in range(0, N)` as a while loop with an i64 counter."""
        init = lir.ZigVar(name=target, mutable=True, ty="i64", value=start)
        cond = lir.ZigCompare(op="<", left=lir.ZigName(name=target), right=stop)
        incr = lir.ZigReassign(
            name=target,
            value=lir.ZigBinOp(
                op="+", left=lir.ZigName(name=target), right=lir.ZigIntLiteral(value=1)
            ),
        )
        body = [self.lower_stmt(n, declared, mut) for n in node.body]
        body.append(incr)
        return _ZigBlock(items=[init, lir.ZigWhile(test=cond, body=body)])

    # -- assign: var/const split, builtin div/rem, mutable array decls ----- #

    def lower_assign(self, node: mir.MirAssign, declared: set[str], mut: set[str]):
        if node.augmented_op is not None:
            op = node.augmented_op
            rhs_expr = self.lower_expr(node.value)
            lhs = lir.ZigName(name=node.target)
            # Zig requires @divTrunc / @rem for integer division and modulo —
            # the augmented `/=` and `%=` forms can't use builtin calls
            # directly, so lower to a plain reassign with the builtin as RHS.
            target_ty = getattr(node, "ty", None)
            if op in ("/", "//") and _is_int_type(target_ty):
                value = lir.ZigCall(func="@divTrunc", args=[lhs, rhs_expr])
            elif op == "%" and _is_int_type(target_ty):
                value = lir.ZigCall(func="@rem", args=[lhs, rhs_expr])
            else:
                value = lir.ZigBinOp(op=op, left=lhs, right=rhs_expr)
            return lir.ZigReassign(name=node.target, value=value)
        lowered_value = self.lower_expr(node.value)
        # A list literal emits a two-statement mutable array decl so the slice
        # can be passed to both mutable and immutable slice parameters.
        if isinstance(lowered_value, lir.ZigArrayLit) and isinstance(node.ty, ListT):
            arr_name = f"_{node.target}_arr"
            return _ZigMutableArrayDecl(
                name=node.target, arr_name=arr_name, array_lit=lowered_value
            )
        if node.target in declared:
            return lir.ZigReassign(name=node.target, value=lowered_value)
        declared.add(node.target)
        return lir.ZigVar(
            name=node.target,
            mutable=node.target in mut,
            ty=_zig_type(node.ty) if not isinstance(node.ty, UnknownT) else None,
            value=lowered_value,
        )

    # -- expressions ------------------------------------------------------- #

    def lower_method_call(self, node: mir.MirMethodCall):
        # Zig method calls land as `receiver.method(args)`; the receiver is
        # implicit in Zig's `pub fn method(self: T, ...)` declarations.
        return _ZigMethodCall(
            receiver=self.lower_expr(node.receiver),
            method=node.method,
            args=[self.lower_expr(a) for a in node.args],
        )

    def lower_binop(self, node: mir.MirBinOp):
        left = self.lower_expr(node.left)
        right = self.lower_expr(node.right)
        left_ty = getattr(node.left, "ty", None)
        right_ty = getattr(node.right, "ty", None)
        both_int = _is_int_type(left_ty) or _is_int_type(right_ty)
        # Zig requires explicit builtins for integer division and modulo.
        if node.op in ("/", "//"):
            if both_int:
                return lir.ZigCall(func="@divTrunc", args=[left, right])
            return lir.ZigBinOp(op="/", left=left, right=right)
        if node.op == "%":
            if both_int:
                return lir.ZigCall(func="@rem", args=[left, right])
            return lir.ZigBinOp(op="%", left=left, right=right)
        if is_string_concat(node):
            raise NotImplementedError(
                "string concatenation in Zig requires allocator-aware emission "
                "(std.fmt.allocPrint), not yet supported"
            )
        return lir.ZigBinOp(op=node.op, left=left, right=right)

    def lower_boolop(self, node: mir.MirBoolOp):
        return lir.ZigBoolOp(
            op=node.op,
            left=self.lower_expr(node.left),
            right=self.lower_expr(node.right),
        )

    def lower_null(self, node: mir.MirNullLiteral):
        return lir.ZigName(name="null")

    def lower_list(self, node: mir.MirList):
        elem_ty = _zig_type(node.ty.elem) if isinstance(node.ty, ListT) else "i64"
        return lir.ZigArrayLit(
            elem_ty=elem_ty, elements=[self.lower_expr(e) for e in node.elements]
        )

    def lower_call(self, node: mir.MirCall):
        args = [self.lower_expr(a) for a in node.args]
        if node.func == "__ternary__" and len(args) == 3:
            return _ZigIfExpr(test=args[0], then_=args[1], else_=args[2])
        if node.func == "len":
            if len(args) != 1:
                raise ValueError("len() takes exactly one argument")
            return lir.ZigMethodCall(
                receiver=args[0], method="len", args=[], cast_to="i64"
            )
        # Stdlib mapping. Zig has @abs/@min/@max as builtins (prefixed `@`).
        # Print goes through std.debug.print with a format string and tuple.
        if node.func in ("print", "println"):
            # `std.debug.print("{}\n", .{<args>})`. Trailing newline matches
            # Python's print default and Rust's println!. Rewrap each arg so
            # bools render as "True"/"False" like Python.
            rendered_args = [
                _pyprint_arg(orig, lowered) for orig, lowered in zip(node.args, args)
            ]
            specs = []
            for orig, rendered in zip(node.args, rendered_args):
                orig_ty = getattr(orig, "ty", None)
                if isinstance(orig_ty, BoolT) or isinstance(rendered, _ZigIfExpr):
                    specs.append("{s}")
                elif isinstance(orig_ty, FloatT) or isinstance(rendered, _ZigPyFloat):
                    specs.append("{s}")
                else:
                    specs.append("{}")
            template = " ".join(specs) + "\n"
            return lir.ZigCall(
                func="std.debug.print",
                args=[lir.ZigStringLiteral(value=template), _ZigTuple(rendered_args)],
            )
        if node.func == "abs" and len(args) == 1:
            return lir.ZigCall(func="@abs", args=args)
        if node.func == "min" and len(args) == 2:
            return lir.ZigCall(func="@min", args=args)
        if node.func == "max" and len(args) == 2:
            return lir.ZigCall(func="@max", args=args)
        if node.func == "int" and len(args) == 1:
            return lir.ZigCall(func="@as", args=[lir.ZigName(name="i64"), args[0]])
        if node.func == "float" and len(args) == 1:
            return lir.ZigCall(func="@floatFromInt", args=args)
        if node.func == "bool" and len(args) == 1:
            return lir.ZigCompare(
                op="!=", left=args[0], right=lir.ZigIntLiteral(value=0)
            )
        return lir.ZigCall(func=node.func, args=args)


_LOWERING = _ZigLowering()


def mir_to_zig_lir(module: mir.MirModule) -> lir.ZigModule:
    return _LOWERING.lower_module(module)


def _zig_list_param_type(ty: Type, name: str, mutated: set[str]) -> str:
    """Return the Zig type for a function parameter.
    List params use slices; mutable ones drop the `const`."""
    if isinstance(ty, ListT):
        elem = _zig_type(ty.elem)
        if name in mutated:
            return f"[]{elem}"
        return f"[]const {elem}"
    return _zig_type(ty)


# Internal block carrier — emitter unwraps. Kept private; not part of the
# public LIR shape.
class _ZigBlock(lir.LirNode):
    def __init__(self, items: list[lir.LirNode]) -> None:
        self.items = items


class _ZigMutableArrayDecl(lir.LirNode):
    """A two-statement list-variable declaration for mutable slices:
        var <arr_name> = <array_lit>;
        const <name>: []<elem_ty> = <arr_name>[0..];
    Emitted when a list literal needs to coerce to a mutable slice."""

    def __init__(self, name: str, arr_name: str, array_lit: lir.ZigArrayLit) -> None:
        self.name = name
        self.arr_name = arr_name
        self.array_lit = array_lit


class _ZigTuple(lir.LirNode):
    """`.{a, b, c}` — Zig anonymous struct literal used as the args
    parameter of std.debug.print. Not a public LIR shape; emitter
    renders specially."""

    def __init__(self, elements: list[lir.LirNode]) -> None:
        self.elements = elements


class _ZigIfExpr(lir.LirNode):
    """`if (<test>) <then_> else <else_>` — Zig if-expression used for
    bool-to-Python-string rendering in print args."""

    def __init__(
        self, test: lir.LirNode, then_: lir.LirNode, else_: lir.LirNode
    ) -> None:
        self.test = test
        self.then_ = then_
        self.else_ = else_


class _ZigMethodCall(lir.LirNode):
    """Local carrier for receiver.method(args) emission. Same shape as
    ZigMethodCall but distinguishable from the `.len`-style property
    access that the existing ZigMethodCall handles."""

    def __init__(
        self, receiver: lir.LirNode, method: str, args: list[lir.LirNode]
    ) -> None:
        self.receiver = receiver
        self.method = method
        self.args = args


def _is_len_call(node: mir.MirNode) -> bool:
    """Return True if `node` is a `len(xs)` call."""
    return isinstance(node, mir.MirCall) and node.func == "len" and len(node.args) == 1


def _is_int_type(ty: object) -> bool:
    """Return True if `ty` is a concrete integer type. An UnknownT or None
    does not count — we only apply @divTrunc / @rem when we can prove the
    operand is integer, since emitting the wrong builtin on floats breaks
    things worse than the original `/`."""
    return isinstance(ty, IntT)


def _pyprint_arg(orig: mir.MirNode, lowered: lir.LirNode) -> lir.LirNode:
    """Wrap `lowered` so it renders the way Python's `print` would."""
    ty = getattr(orig, "ty", None)
    if isinstance(ty, BoolT):
        return _ZigIfExpr(
            test=lowered,
            then_=lir.ZigStringLiteral(value="True"),
            else_=lir.ZigStringLiteral(value="False"),
        )
    if isinstance(ty, FloatT):
        return _ZigPyFloat(value=lowered)
    return lowered


class _ZigPyFloat(lir.LirNode):
    """`_py_float(x)` — calls the injected helper that preserves `.0`
    for whole-number floats, matching Python's print behavior."""

    def __init__(self, value: lir.LirNode) -> None:
        self.value = value


def _zig_type(ty: Type) -> str:
    if isinstance(ty, IntT):
        return f"{'i' if ty.signed else 'u'}{ty.bits}"
    if isinstance(ty, FloatT):
        return f"f{ty.bits}"
    if isinstance(ty, BoolT):
        return "bool"
    if isinstance(ty, StrT):
        return "[]const u8"
    if isinstance(ty, NoneT):
        return "void"
    if isinstance(ty, ListT):
        return f"[]const {_zig_type(ty.elem)}"
    if isinstance(ty, StructT):
        return ty.name
    if isinstance(ty, UnknownT):
        raise ValueError(f"unresolved type hole: {ty.hint}")
    raise NotImplementedError(f"type {type(ty).__name__}")

"""MIR -> Fortran LIR.

Two responsibilities not seen in the other backends:
  1. Collect every declared name (params + assigns + for-range targets) and
     its type for the function-entry declaration block. Fortran refuses
     inline declarations the way C-family languages welcome them.
  2. Turn every `return <expr>` into an assignment to the result variable
     plus a bare `return` statement. Fortran functions return whatever the
     result variable holds at function exit.

Fortran's statement layer diverges enough from the common skeleton (result-var
returns, statement-level `print` rewriting, exit/cycle keywords) that it owns
``lower_stmt``; it still reuses the shared expression dispatch in
``_mir_lower_base`` for everything below the statement level.
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

from ._mir_lower_base import MirLoweringBase, is_string_concat


RESULT_VAR = "result_"  # synthesized to avoid clashing with user identifiers

_CMP_OPS = {
    "==": ".eq.",
    "!=": ".ne.",
    "<": ".lt.",
    "<=": ".le.",
    ">": ".gt.",
    ">=": ".ge.",
}


class _FortranLowering(MirLoweringBase):
    prefix = "Fortran"
    module_cls = lir.FortranModule

    def __init__(self) -> None:
        super().__init__()
        self._result_name = ""

    def type_str(self, ty: Type) -> str:
        return _fortran_type(ty)

    # -- struct: type + free `Name_method` functions ----------------------- #

    def lower_struct_items(self, s: mir.MirStruct) -> list[lir.LirNode]:
        methods: list[lir.FortranFn] = []
        for m in s.methods:
            lowered = self.lower_function(m)
            lowered.name = f"{s.name}_{lowered.name}"
            methods.append(lowered)
        return [
            lir.FortranType(
                name=s.name,
                fields=[(f.name, _fortran_type(f.ty)) for f in s.fields],
                methods=methods,
            )
        ]

    # -- function: locals decl-block + result var -------------------------- #

    def lower_function(self, fn: mir.MirFunction) -> lir.FortranFn:
        params = [(p.name, _fortran_type(p.ty)) for p in fn.params]
        ret = (
            _fortran_type(fn.return_type)
            if not isinstance(fn.return_type, NoneT)
            else None
        )

        # Collect every local name and its type in a single MIR walk.
        locals_map: dict[str, Type] = {}
        _collect_locals(
            fn.body, locals_map, exclude={p.name for p in fn.params} | {RESULT_VAR}
        )

        result_name = RESULT_VAR if ret is not None else ""
        self._result_name = result_name
        body = [self.lower_stmt(n, set(), set()) for n in fn.body]

        return lir.FortranFn(
            name=fn.name,
            params=params,
            return_type=ret,
            result_name=result_name,
            locals=[(name, _fortran_type(ty)) for name, ty in locals_map.items()],
            body=body,
        )

    # -- statements -------------------------------------------------------- #

    def lower_stmt(self, node: mir.MirNode, declared: set[str], mut: set[str]):
        result_name = self._result_name
        if isinstance(node, mir.MirFieldAssign):
            # Fortran field assignment: `obj%field = value` — emit as a plain
            # FortranAssign with the path baked into `name`.
            return lir.FortranAssign(
                name=_emit_field_path(node.obj, node.field),
                value=self.lower_expr(node.value),
            )
        if isinstance(node, mir.MirSubscriptAssign):
            return lir.FortranSubscriptAssign(
                obj=self.lower_expr(node.obj),
                index=self.lower_expr(node.index),
                value=self.lower_expr(node.value),
            )
        if isinstance(node, mir.MirReturn):
            if node.value is None or not result_name:
                return lir.FortranReturn()
            return _ReturnAssign(
                result_name=result_name, value=self.lower_expr(node.value)
            )
        if isinstance(node, mir.MirBreak):
            return lir.FortranExit()
        if isinstance(node, mir.MirContinue):
            return lir.FortranCycle()
        if isinstance(node, mir.MirAssign):
            if node.augmented_op is not None:
                rhs = lir.FortranBinOp(
                    op=node.augmented_op,
                    left=lir.FortranName(name=node.target),
                    right=self.lower_expr(node.value),
                )
                return lir.FortranAssign(name=node.target, value=rhs)
            return lir.FortranAssign(
                name=node.target, value=self.lower_expr(node.value)
            )
        if isinstance(node, mir.MirIf):
            return lir.FortranIf(
                test=self.lower_expr(node.test),
                body=[self.lower_stmt(n, declared, mut) for n in node.body],
                orelse=[self.lower_stmt(n, declared, mut) for n in node.orelse],
            )
        if isinstance(node, mir.MirWhile):
            return lir.FortranWhile(
                test=self.lower_expr(node.test),
                body=[self.lower_stmt(n, declared, mut) for n in node.body],
            )
        if isinstance(node, mir.MirForRange):
            return lir.FortranForRange(
                target=node.target,
                start=self.lower_expr(node.start),
                stop=self.lower_expr(node.stop),
                step=self.lower_expr(node.step) if node.step else None,
                body=[self.lower_stmt(n, declared, mut) for n in node.body],
            )
        if isinstance(node, mir.MirCall) and node.func == "print":
            # Match Python's repr output for bool and float args.
            args = [self._lower_print_arg(a) for a in node.args]
            return lir.FortranCall(func="print", args=args)
        return self.lower_expr(node)

    def _lower_print_arg(self, node: mir.MirNode) -> lir.LirNode:
        """Lower a print() argument, wrapping booleans and floats to match
        Python's repr-style output."""
        if _is_bool_type(node):
            expr = self.lower_expr(node)
            return lir.FortranCall(
                func="trim",
                args=[
                    lir.FortranCall(
                        func="merge",
                        args=[
                            lir.FortranStringLiteral(value="True "),
                            lir.FortranStringLiteral(value="False"),
                            expr,
                        ],
                    )
                ],
            )
        ty = getattr(node, "ty", None)
        if isinstance(ty, FloatT):
            return lir.FortranCall(
                func="trim",
                args=[lir.FortranCall(func="pyfloat", args=[self.lower_expr(node)])],
            )
        return self.lower_expr(node)

    # -- expressions ------------------------------------------------------- #

    def lower_expr_special(self, node: mir.MirNode):
        if isinstance(node, mir.MirBinOp) and node.op == "//":
            # Fortran `/` on integers is integer division.
            return lir.FortranBinOp(
                op="/",
                left=self.lower_expr(node.left),
                right=self.lower_expr(node.right),
            )
        if isinstance(node, mir.MirBinOp) and node.op == "%":
            # Fortran modulo is an intrinsic function, not an operator.
            return lir.FortranCall(
                func="mod",
                args=[self.lower_expr(node.left), self.lower_expr(node.right)],
            )
        return None

    def lower_method_call(self, node: mir.MirMethodCall):
        recv_ty = getattr(node.receiver, "ty", UnknownT())
        if not isinstance(recv_ty, StructT):
            raise NotImplementedError(
                f"fortran method call on receiver with type {recv_ty}"
            )
        return lir.FortranCall(
            func=f"{recv_ty.name}_{node.method}",
            args=[self.lower_expr(node.receiver)]
            + [self.lower_expr(a) for a in node.args],
        )

    def lower_binop(self, node: mir.MirBinOp):
        if is_string_concat(node):
            raise NotImplementedError(
                "Fortran string concat requires fixed-length char buffers; not yet supported"
            )
        # List concatenation: `xs + [v]` → `[xs, v]`. Re-allocates per use
        # (O(n) copy); acceptable for our algorithm corpus. Flattening both
        # sides avoids ragged `[[a, b], c]` constructors that Fortran rejects.
        if (
            node.op == "+"
            and isinstance(getattr(node.left, "ty", None), ListT)
            and isinstance(getattr(node.right, "ty", None), ListT)
        ):
            return lir.FortranArrayLit(
                elements=[*self._spread_list(node.left), *self._spread_list(node.right)]
            )
        return lir.FortranBinOp(
            op=node.op,
            left=self.lower_expr(node.left),
            right=self.lower_expr(node.right),
        )

    def lower_compare(self, node: mir.MirCompare):
        fortran_op = _CMP_OPS.get(node.op, node.op)
        return lir.FortranCompare(
            op=fortran_op,
            left=self.lower_expr(node.left),
            right=self.lower_expr(node.right),
        )

    def lower_boolop(self, node: mir.MirBoolOp):
        op = ".and." if node.op == "and" else ".or."
        return lir.FortranBoolOp(
            op=op, left=self.lower_expr(node.left), right=self.lower_expr(node.right)
        )

    def lower_unary(self, node: mir.MirUnaryOp):
        op = ".not." if node.op == "not" else "-"
        return lir.FortranUnary(op=op, operand=self.lower_expr(node.operand))

    def lower_null(self, node: mir.MirNullLiteral):
        # Fortran has no null/None; pointers exist but cleanup requires a
        # typed allocation strategy out of scope for this lowering.
        raise NotImplementedError("fortran has no null literal")

    def lower_subscript(self, node: mir.MirSubscript):
        return lir.FortranSubscript(
            value=self.lower_expr(node.value),
            index=self.lower_expr(node.index),
        )

    def lower_list(self, node: mir.MirList):
        elements = [self.lower_expr(e) for e in node.elements]
        elem_ty = None
        if not elements:
            list_ty = getattr(node, "ty", None)
            if isinstance(list_ty, ListT):
                elem_ty = _fortran_type(list_ty.elem)
        return lir.FortranArrayLit(elements=elements, elem_type=elem_ty)

    def lower_call(self, node: mir.MirCall):
        # `len(xs)` → Fortran intrinsic `size(xs)` on arrays.
        if node.func == "len" and len(node.args) == 1:
            return lir.FortranCall(func="size", args=[self.lower_expr(node.args[0])])
        # `__ternary__(cond, then, else)` → `merge(then, else, cond)`.
        # Fortran's merge takes (true_val, false_val, mask) — matches the
        # ternary slot order with a swap.
        if node.func == "__ternary__" and len(node.args) == 3:
            return lir.FortranCall(
                func="merge",
                args=[
                    self.lower_expr(node.args[1]),
                    self.lower_expr(node.args[2]),
                    self.lower_expr(node.args[0]),
                ],
            )
        return lir.FortranCall(
            func=node.func, args=[self.lower_expr(a) for a in node.args]
        )

    def _spread_list(self, node: mir.MirNode) -> list[lir.LirNode]:
        """Flatten a literal list one level so concatenation emits a single
        `[a, b, c]` rather than `[[a, b], c]` — Fortran rejects ragged ranks
        in array constructors."""
        if isinstance(node, mir.MirList):
            return [self.lower_expr(e) for e in node.elements]
        return [self.lower_expr(node)]


_LOWERING = _FortranLowering()


def mir_to_fortran_lir(module: mir.MirModule) -> lir.FortranModule:
    return _LOWERING.lower_module(module)


# ---------- local collection ----------


def _collect_locals(
    nodes: list[mir.MirNode], out: dict[str, Type], exclude: set[str]
) -> None:
    for n in nodes:
        if isinstance(n, mir.MirAssign):
            if n.target not in exclude and n.target not in out:
                out[n.target] = n.ty
        elif isinstance(n, mir.MirIf):
            _collect_locals(n.body, out, exclude)
            _collect_locals(n.orelse, out, exclude)
        elif isinstance(n, mir.MirWhile):
            _collect_locals(n.body, out, exclude)
        elif isinstance(n, mir.MirForRange):
            if n.target not in exclude and n.target not in out:
                out[n.target] = IntT()
            _collect_locals(n.body, out, exclude)


def _is_bool_type(node: mir.MirNode) -> bool:
    """True if this MIR expression is definitely of boolean type."""
    ty = getattr(node, "ty", None)
    if isinstance(ty, BoolT):
        return True
    if isinstance(node, (mir.MirBoolLiteral, mir.MirCompare, mir.MirBoolOp)):
        return True
    if isinstance(node, mir.MirUnaryOp) and node.op == "not":
        return True
    return False


class _ReturnAssign(lir.LirNode):
    """Marker carrying `result_ = <value>; return` so the emitter renders
    both lines. Not part of the public LIR shape."""

    def __init__(self, result_name: str, value: lir.LirNode) -> None:
        self.result_name = result_name
        self.value = value


def _emit_field_path(obj: mir.MirNode, field: str) -> str:
    """Build a `obj%field` path string for assignment-LHS use."""
    if isinstance(obj, mir.MirName):
        return f"{obj.name}%{field}"
    if isinstance(obj, mir.MirFieldAccess):
        return f"{_emit_field_path(obj.value, obj.field)}%{field}"
    raise NotImplementedError(f"fortran field-assign on {type(obj).__name__}")


def _fortran_type(ty: Type) -> str:
    if isinstance(ty, IntT):
        return "integer"
    if isinstance(ty, FloatT):
        # `real(8)` is fortran-portable for f64; `real` alone is single
        # precision on most compilers. Use the kind suffix.
        return "real(8)" if ty.bits == 64 else "real"
    if isinstance(ty, BoolT):
        return "logical"
    if isinstance(ty, StrT):
        # Fortran character lengths must be declared; use a generous default
        # for the type signature. Real use needs per-variable lengths.
        return "character(len=*)"
    if isinstance(ty, NoneT):
        return ""
    if isinstance(ty, ListT):
        # Assumed-shape array. The param declaration in `_emit_fn` produces:
        #   `integer, dimension(:), intent(in) :: xs`
        # For locals we'd need `allocatable` too; that's handled by tagging
        # local list types with the allocatable attribute. Empty-list
        # initialization via `xs = []` then auto-allocates.
        return f"{_fortran_type(ty.elem)}, dimension(:)"
    if isinstance(ty, StructT):
        return f"type({ty.name})"
    if isinstance(ty, UnknownT):
        raise ValueError(f"unresolved type hole: {ty.hint}")
    raise NotImplementedError(f"type {type(ty).__name__}")

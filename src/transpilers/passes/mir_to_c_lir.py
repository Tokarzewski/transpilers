"""MIR -> C LIR.

C has no mut/const split — every variable is mutable. Mutability inference
is therefore irrelevant; declarations always emit the bare `<ty> <name> =
<value>;` form. For-range uses native C syntax. Strings are literals only;
concat raises (would need allocator-aware emission, same as Zig).

Structure is shared with the other backends via ``_mir_lower_base``; this
module supplies the C spec plus the `self`-pointer field access, method-name
mangling, slice-literal construction, and the builtin call table.
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


class _CLowering(MirLoweringBase):
    prefix = "C"
    module_cls = lir.CModule

    def type_str(self, ty: Type) -> str:
        return _c_type(ty)

    # -- struct: methods become free `Struct_method` functions ------------- #

    def lower_struct_items(self, s: mir.MirStruct) -> list[lir.LirNode]:
        methods: list[lir.CFn] = []
        for m in s.methods:
            # C has no method binding; emit as a free function `Struct_method`
            # whose first parameter is `Struct *self`. Rename so different
            # structs can share method names without colliding.
            lowered = self.lower_function(m)
            lowered.name = f"{s.name}_{lowered.name}"
            methods.append(lowered)
        return [
            lir.CStruct(
                name=s.name,
                fields=[(f.name, _c_type(f.ty)) for f in s.fields],
                methods=methods,
            )
        ]

    # -- field access/assign go through self-pointer ----------------------- #

    def lower_field_assign(self, node: mir.MirFieldAssign):
        via_ptr = isinstance(node.obj, mir.MirName) and node.obj.name == "self"
        return lir.CFieldAssign(
            obj=self.lower_expr(node.obj),
            field=node.field,
            value=self.lower_expr(node.value),
            via_pointer=via_ptr,
        )

    def lower_field_access(self, node: mir.MirFieldAccess):
        via_ptr = isinstance(node.value, mir.MirName) and node.value.name == "self"
        return lir.CFieldAccess(
            value=self.lower_expr(node.value), field=node.field, via_pointer=via_ptr
        )

    # -- assign: every local is a plain decl / reassign -------------------- #

    def lower_assign(self, node: mir.MirAssign, declared: set[str], mut: set[str]):
        if node.augmented_op is not None:
            rhs = lir.CBinOp(
                op=node.augmented_op,
                left=lir.CName(name=node.target),
                right=self.lower_expr(node.value),
            )
            return lir.CReassign(name=node.target, value=rhs)
        if node.target in declared:
            return lir.CReassign(name=node.target, value=self.lower_expr(node.value))
        declared.add(node.target)
        ty = _c_type(node.ty) if not isinstance(node.ty, UnknownT) else "int64_t"
        return lir.CDecl(name=node.target, ty=ty, value=self.lower_expr(node.value))

    # -- expressions ------------------------------------------------------- #

    def lower_expr_special(self, node: mir.MirNode):
        if isinstance(node, mir.MirBinOp) and node.op == "//":
            # C: integer division is `/` on integer types.
            return lir.CBinOp(
                op="/",
                left=self.lower_expr(node.left),
                right=self.lower_expr(node.right),
            )
        return None

    def lower_method_call(self, node: mir.MirMethodCall):
        # `obj.method(args)` → `Struct_method(&obj, args)`. We need the struct
        # name to mangle; pull it from the receiver's type.
        recv = node.receiver
        recv_ty = getattr(recv, "ty", UnknownT())
        if isinstance(recv_ty, StructT):
            return lir.CCall(
                func=f"{recv_ty.name}_{node.method}",
                args=[_AddressOf(self.lower_expr(recv))]
                + [self.lower_expr(a) for a in node.args],
            )
        raise NotImplementedError(f"C method call on receiver with type {recv_ty}")

    def lower_binop(self, node: mir.MirBinOp):
        if is_string_concat(node):
            raise NotImplementedError(
                "string concatenation in C requires allocator-aware emission "
                "(snprintf or asprintf), not yet supported"
            )
        return lir.CBinOp(
            op=node.op,
            left=self.lower_expr(node.left),
            right=self.lower_expr(node.right),
        )

    def lower_boolop(self, node: mir.MirBoolOp):
        op = "&&" if node.op == "and" else "||"
        return lir.CBoolOp(
            op=op, left=self.lower_expr(node.left), right=self.lower_expr(node.right)
        )

    def lower_null(self, node: mir.MirNullLiteral):
        return lir.CName(name="NULL")

    def lower_list(self, node: mir.MirList):
        # Compound-literal slice. The inner `(elem_t[]){...}` array lives in
        # automatic storage when this expression appears in a function body —
        # long-lived enough for the slice's lifetime in that scope.
        elem_ty = _c_list_elem_type(node.ty)
        slice_ty = _c_type(node.ty)
        return _CSliceLiteral(
            slice_ty=slice_ty,
            elem_ty=elem_ty,
            elements=[self.lower_expr(e) for e in node.elements],
        )

    def lower_call(self, node: mir.MirCall):
        args = [self.lower_expr(a) for a in node.args]
        if node.func == "__ternary__" and len(args) == 3:
            return lir.CTernary(test=args[0], then_=args[1], else_=args[2])
        if node.func == "len":
            # `len(xs)` on a slice maps to the carrier's `.len` field. Cast
            # to int64_t so the result composes with the rest of our IntT
            # arithmetic without warnings.
            if len(args) != 1:
                raise ValueError("len() takes exactly one argument")
            return lir.CCall(
                func="(int64_t)",
                args=[lir.CFieldAccess(value=args[0], field="len", via_pointer=False)],
            )
        if node.func in ("print", "println"):
            # Build per-arg format specifiers so each type prints the way
            # Python's `print()` would:
            #   bool  → %s  with  (x ? "True" : "False")
            #   float → _py_float() helper defined in the preamble
            #   int   → %lld
            fmt_parts: list[str] = []
            final_args: list[lir.LirNode] = []
            for orig, lowered in zip(node.args, args):
                ty = getattr(orig, "ty", None)
                if isinstance(ty, BoolT):
                    fmt_parts.append("%s")
                    final_args.append(
                        lir.CTernary(
                            test=lowered,
                            then_=lir.CStringLiteral(value="True"),
                            else_=lir.CStringLiteral(value="False"),
                        )
                    )
                elif isinstance(ty, FloatT):
                    fmt_parts.append("%s")
                    final_args.append(_CPyFloat(value=lowered))
                else:
                    fmt_parts.append("%lld")
                    final_args.append(lowered)
            template = " ".join(fmt_parts) + "\n"
            return lir.CCall(
                func="printf",
                args=[lir.CStringLiteral(value=template), *final_args],
            )
        if node.func == "abs" and len(args) == 1:
            # C's stdlib has abs/labs/llabs — use llabs for int64.
            return lir.CCall(func="llabs", args=args)
        if node.func == "min" and len(args) == 2:
            return lir.CTernary(
                test=lir.CCompare(op="<", left=args[0], right=args[1]),
                then_=args[0],
                else_=args[1],
            )
        if node.func == "max" and len(args) == 2:
            return lir.CTernary(
                test=lir.CCompare(op=">", left=args[0], right=args[1]),
                then_=args[0],
                else_=args[1],
            )
        if node.func == "int" and len(args) == 1:
            return lir.CCall(func="(int64_t)", args=args)
        if node.func == "float" and len(args) == 1:
            return lir.CCall(func="(double)", args=args)
        return lir.CCall(func=node.func, args=args)


_LOWERING = _CLowering()


def mir_to_c_lir(module: mir.MirModule) -> lir.CModule:
    return _LOWERING.lower_module(module)


def _c_list_elem_type(ty) -> str:
    if not isinstance(ty, ListT):
        raise ValueError("expected ListT for slice literal")
    if isinstance(ty.elem, IntT):
        return "int64_t"
    if isinstance(ty.elem, BoolT):
        return "bool"
    if isinstance(ty.elem, FloatT):
        return "double"
    raise NotImplementedError(f"no C element type for {type(ty.elem).__name__}")


class _CSliceLiteral(lir.LirNode):
    """`((slice_T_t){(T[]){a,b,c}, 3})` — compound-literal slice ctor."""

    def __init__(
        self, slice_ty: str, elem_ty: str, elements: list[lir.LirNode]
    ) -> None:
        self.slice_ty = slice_ty
        self.elem_ty = elem_ty
        self.elements = elements


class _AddressOf(lir.LirNode):
    """`&expr` — internal marker emitted by C method-call lowering."""

    def __init__(self, value: lir.LirNode) -> None:
        self.value = value


class _CPyFloat(lir.LirNode):
    """Wraps a float expression so the emitter renders it via the
    `_py_float_buf()` helper defined in the C preamble, which produces
    Python-compatible shortest-round-trip float strings."""

    def __init__(self, value: lir.LirNode) -> None:
        self.value = value


def _c_type(ty: Type) -> str:
    if isinstance(ty, IntT):
        prefix = "int" if ty.signed else "uint"
        return f"{prefix}{ty.bits}_t"
    if isinstance(ty, FloatT):
        return "double" if ty.bits == 64 else "float"
    if isinstance(ty, BoolT):
        return "bool"
    if isinstance(ty, StrT):
        return "const char*"
    if isinstance(ty, NoneT):
        return "void"
    if isinstance(ty, ListT):
        # C has no generics; map to one of the fixed slice typedefs emitted
        # in the file preamble. Each slice carries `(data, len)`.
        if isinstance(ty.elem, IntT):
            return "slice_i64_t"
        if isinstance(ty.elem, BoolT):
            return "slice_bool_t"
        if isinstance(ty.elem, FloatT):
            return "slice_f64_t"
        raise NotImplementedError(
            f"no C slice type for list element {type(ty.elem).__name__}"
        )
    if isinstance(ty, StructT):
        return ty.name
    if isinstance(ty, UnknownT):
        raise ValueError(f"unresolved type hole: {ty.hint}")
    raise NotImplementedError(f"type {type(ty).__name__}")

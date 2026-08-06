"""MIR -> Python LIR.

Python is the round-trip target: source → IR → Python. Useful for normalizing
code, refactoring tools, and proving the IR is lossless enough to recover
compilable Python from any source language.

Annotations are emitted when available. Mutability isn't a Python concept,
so all `MirAssign` nodes lower to plain `PyAssign`.

Structure is shared with the other backends via ``_mir_lower_base``; this
module supplies the Python spec (identity operator maps, annotation-only
assigns) and the local marker nodes the Python emitter consumes.
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

from ._mir_lower_base import MirLoweringBase


class _PyLowering(MirLoweringBase):
    prefix = "Py"
    module_cls = lir.PyModule

    def type_str(self, ty: Type) -> str:
        return _py_type(ty)

    # -- struct -> class --------------------------------------------------- #

    def lower_struct_items(self, s: mir.MirStruct) -> list[lir.LirNode]:
        fields: list[tuple[str, str | None]] = []
        for f in s.fields:
            ty = _py_type(f.ty)
            fields.append((f.name, ty if ty else None))
        return [
            lir.PyClass(
                name=s.name,
                fields=fields,
                methods=[self.lower_function(m) for m in s.methods],
            )
        ]

    # -- assign: annotation only on first occurrence ----------------------- #

    def lower_assign(self, node: mir.MirAssign, declared: set[str], mut: set[str]):
        if node.augmented_op is not None:
            rhs = lir.PyBinOp(
                op=node.augmented_op,
                left=lir.PyName(name=node.target),
                right=self.lower_expr(node.value),
            )
            return lir.PyAssign(name=node.target, ty=None, value=rhs)
        is_first = node.target not in declared
        declared.add(node.target)
        ty = (
            _py_type(node.ty)
            if is_first and not isinstance(node.ty, UnknownT)
            else None
        )
        return lir.PyAssign(name=node.target, ty=ty, value=self.lower_expr(node.value))

    # -- expressions: identity operator maps ------------------------------- #

    def lower_binop(self, node: mir.MirBinOp):
        return lir.PyBinOp(
            op=node.op,
            left=self.lower_expr(node.left),
            right=self.lower_expr(node.right),
        )

    def lower_boolop(self, node: mir.MirBoolOp):
        return lir.PyBoolOp(
            op=node.op,
            left=self.lower_expr(node.left),
            right=self.lower_expr(node.right),
        )

    def lower_unary(self, node: mir.MirUnaryOp):
        return lir.PyUnary(op=node.op, operand=self.lower_expr(node.operand))

    def lower_method_call(self, node: mir.MirMethodCall):
        return _PyMethodCall(
            receiver=self.lower_expr(node.receiver),
            method=node.method,
            args=[self.lower_expr(a) for a in node.args],
        )

    def lower_subscript(self, node: mir.MirSubscript):
        return _PyIndex(
            value=self.lower_expr(node.value), index=self.lower_expr(node.index)
        )

    def lower_list(self, node: mir.MirList):
        return _PyList(elements=[self.lower_expr(e) for e in node.elements])

    def lower_null(self, node: mir.MirNullLiteral):
        return lir.PyName(name="None")

    def lower_call(self, node: mir.MirCall):
        args = [self.lower_expr(a) for a in node.args]
        if node.func == "__ternary__" and len(args) == 3:
            return _PyIfExpr(test=args[0], then_=args[1], else_=args[2])
        return lir.PyCall(func=node.func, args=args)


_LOWERING = _PyLowering()


def mir_to_python_lir(module: mir.MirModule) -> lir.PyModule:
    return _LOWERING.lower_module(module)


class _PyMethodCall(lir.LirNode):
    def __init__(
        self, receiver: lir.LirNode, method: str, args: list[lir.LirNode]
    ) -> None:
        self.receiver = receiver
        self.method = method
        self.args = args


class _PyIndex(lir.LirNode):
    def __init__(self, value: lir.LirNode, index: lir.LirNode) -> None:
        self.value = value
        self.index = index


class _PyIfExpr(lir.LirNode):
    """`<then> if <test> else <else>` — Python ternary."""

    def __init__(
        self, test: lir.LirNode, then_: lir.LirNode, else_: lir.LirNode
    ) -> None:
        self.test = test
        self.then_ = then_
        self.else_ = else_


class _PyList(lir.LirNode):
    """`[a, b, c]` — Python list literal."""

    def __init__(self, elements: list[lir.LirNode]) -> None:
        self.elements = elements


def _py_type(ty: Type) -> str:
    if isinstance(ty, IntT):
        return "int"
    if isinstance(ty, FloatT):
        return "float"
    if isinstance(ty, BoolT):
        return "bool"
    if isinstance(ty, StrT):
        return "str"
    if isinstance(ty, NoneT):
        return "None"
    if isinstance(ty, ListT):
        return f"list[{_py_type(ty.elem)}]"
    if isinstance(ty, StructT):
        return ty.name
    if isinstance(ty, UnknownT):
        return ""
    raise NotImplementedError(f"type {type(ty).__name__}")

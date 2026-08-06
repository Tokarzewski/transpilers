"""Zig LIR dialect nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode


@dataclass
class ZigFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode


@dataclass
class ZigStructInit(LirNode):
    """`Point{ .x = 0, .y = 0 }`."""

    name: str
    field_values: list[tuple[str, LirNode]]


@dataclass
class ZigSubscriptAssign(LirNode):
    obj: LirNode
    index: LirNode
    value: LirNode


@dataclass
class ZigModule(LirNode):
    items: list[LirNode] = field(default_factory=list)


@dataclass
class ZigFn(LirNode):
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: list[LirNode]


@dataclass
class ZigReturn(LirNode):
    value: LirNode | None


@dataclass
class ZigBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class ZigCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class ZigBoolOp(LirNode):
    op: str  # "and" / "or"
    left: LirNode
    right: LirNode


@dataclass
class ZigUnary(LirNode):
    op: str  # "!" / "-"
    operand: LirNode


@dataclass
class ZigName(LirNode):
    name: str


@dataclass
class ZigIntLiteral(LirNode):
    value: int


@dataclass
class ZigFloatLiteral(LirNode):
    value: float


@dataclass
class ZigBoolLiteral(LirNode):
    value: bool


@dataclass
class ZigStringLiteral(LirNode):
    value: str


@dataclass
class ZigIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]


@dataclass
class ZigWhile(LirNode):
    test: LirNode
    body: list[LirNode]


@dataclass
class ZigForRange(LirNode):
    """Zig `for (start..stop) |target| { ... }`. `step != 1` is unsupported by
    the `for` syntax — a stepped range lowers to a `while` with explicit
    increment instead, handled in the lowering pass."""

    target: str
    start: LirNode
    stop: LirNode
    body: list[LirNode]


@dataclass
class ZigVar(LirNode):
    """`var <name>: <ty> = <value>;` (mutable) or `const <name>: <ty> = <value>;`."""

    name: str
    mutable: bool
    ty: str | None
    value: LirNode


@dataclass
class ZigReassign(LirNode):
    name: str
    value: LirNode


@dataclass
class ZigArrayLit(LirNode):
    """`[_]T{a, b, c}` — fixed-size inferred-length array. For dynamically
    sized lists we'd need ArrayList, deferred.

    When `ref=True`, emit as `&[_]T{a, b, c}` so the array literal coerces
    to a slice type (`[]T`) in assignment context."""

    elem_ty: str
    elements: list[LirNode]
    ref: bool = False


@dataclass
class ZigIndex(LirNode):
    value: LirNode
    index: LirNode


@dataclass
class ZigMethodCall(LirNode):
    receiver: LirNode
    method: str
    args: list[LirNode]
    cast_to: str | None = None


@dataclass
class ZigCall(LirNode):
    func: str
    args: list[LirNode]


@dataclass
class ZigStruct(LirNode):
    name: str
    fields: list[tuple[str, str]]
    methods: list["ZigFn"]


@dataclass
class ZigFieldAccess(LirNode):
    value: LirNode
    field: str


@dataclass
class ZigBreak(LirNode):
    pass


@dataclass
class ZigContinue(LirNode):
    pass


@dataclass
class ZigRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = [
    "ZigFieldAssign",
    "ZigStructInit",
    "ZigSubscriptAssign",
    "ZigModule",
    "ZigFn",
    "ZigReturn",
    "ZigBinOp",
    "ZigCompare",
    "ZigBoolOp",
    "ZigUnary",
    "ZigName",
    "ZigIntLiteral",
    "ZigFloatLiteral",
    "ZigBoolLiteral",
    "ZigStringLiteral",
    "ZigIf",
    "ZigWhile",
    "ZigForRange",
    "ZigVar",
    "ZigReassign",
    "ZigArrayLit",
    "ZigIndex",
    "ZigMethodCall",
    "ZigCall",
    "ZigStruct",
    "ZigFieldAccess",
    "ZigBreak",
    "ZigContinue",
    "ZigRaw",
]

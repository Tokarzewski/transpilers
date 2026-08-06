"""C LIR dialect nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode


@dataclass
class CFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode
    via_pointer: bool = False  # `.` for value, `->` for pointer


@dataclass
class CStructInit(LirNode):
    """`(Point){ .x = 0, .y = 0 }` — compound literal."""

    name: str
    field_values: list[tuple[str, LirNode]]


@dataclass
class CSubscriptAssign(LirNode):
    obj: LirNode
    index: LirNode
    value: LirNode


@dataclass
class CModule(LirNode):
    items: list[LirNode] = field(default_factory=list)


@dataclass
class CFn(LirNode):
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: list[LirNode]


@dataclass
class CReturn(LirNode):
    value: LirNode | None


@dataclass
class CBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class CCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class CBoolOp(LirNode):
    op: str  # "&&" / "||"
    left: LirNode
    right: LirNode


@dataclass
class CUnary(LirNode):
    op: str
    operand: LirNode


@dataclass
class CName(LirNode):
    name: str


@dataclass
class CIntLiteral(LirNode):
    value: int


@dataclass
class CFloatLiteral(LirNode):
    value: float


@dataclass
class CBoolLiteral(LirNode):
    value: bool


@dataclass
class CStringLiteral(LirNode):
    value: str


@dataclass
class CIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]


@dataclass
class CWhile(LirNode):
    test: LirNode
    body: list[LirNode]


@dataclass
class CForRange(LirNode):
    """C-style `for (int64_t i = start; i < stop; i++)`. step != 1 emits the
    explicit step expression."""

    target: str
    start: LirNode
    stop: LirNode
    step: LirNode | None
    body: list[LirNode]


@dataclass
class CDecl(LirNode):
    """`<ty> <name> = <value>;` — single-line declaration with initializer."""

    name: str
    ty: str
    value: LirNode


@dataclass
class CReassign(LirNode):
    name: str
    value: LirNode


@dataclass
class CIndex(LirNode):
    value: LirNode
    index: LirNode


@dataclass
class CCall(LirNode):
    func: str
    args: list[LirNode]


@dataclass
class CTernary(LirNode):
    """`<test> ? <then_> : <else_>` — C ternary expression."""

    test: LirNode
    then_: LirNode
    else_: LirNode


@dataclass
class CStruct(LirNode):
    """`typedef struct { ... } Name;` — methods emitted as separate
    free functions named `Name_method` with a `Name *self` first param."""

    name: str
    fields: list[tuple[str, str]]
    methods: list["CFn"]


@dataclass
class CFieldAccess(LirNode):
    """`self->field` (pointer access — C methods take Self*) vs `value.field`
    on a struct value. We always emit via pointer for method bodies, which is
    what our current lowering produces."""

    value: LirNode
    field: str
    via_pointer: bool = True


@dataclass
class CBreak(LirNode):
    pass


@dataclass
class CContinue(LirNode):
    pass


@dataclass
class CRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = [
    "CFieldAssign",
    "CStructInit",
    "CSubscriptAssign",
    "CModule",
    "CFn",
    "CReturn",
    "CBinOp",
    "CCompare",
    "CBoolOp",
    "CUnary",
    "CName",
    "CIntLiteral",
    "CFloatLiteral",
    "CBoolLiteral",
    "CStringLiteral",
    "CIf",
    "CWhile",
    "CForRange",
    "CDecl",
    "CReassign",
    "CIndex",
    "CCall",
    "CTernary",
    "CStruct",
    "CFieldAccess",
    "CBreak",
    "CContinue",
    "CRaw",
]

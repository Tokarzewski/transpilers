"""Go LIR dialect nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode


@dataclass
class GoFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode


@dataclass
class GoStructInit(LirNode):
    """`Point{x: 0, y: 0}`."""

    name: str
    field_values: list[tuple[str, LirNode]]


@dataclass
class GoSubscriptAssign(LirNode):
    obj: LirNode
    index: LirNode
    value: LirNode


@dataclass
class GoModule(LirNode):
    items: list[LirNode] = field(default_factory=list)


@dataclass
class GoFn(LirNode):
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: list[LirNode]


@dataclass
class GoReturn(LirNode):
    value: LirNode | None


@dataclass
class GoBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class GoCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class GoBoolOp(LirNode):
    op: str  # "&&" / "||"
    left: LirNode
    right: LirNode


@dataclass
class GoUnary(LirNode):
    op: str
    operand: LirNode


@dataclass
class GoName(LirNode):
    name: str


@dataclass
class GoIntLiteral(LirNode):
    value: int


@dataclass
class GoFloatLiteral(LirNode):
    value: float


@dataclass
class GoBoolLiteral(LirNode):
    value: bool


@dataclass
class GoStringLiteral(LirNode):
    value: str


@dataclass
class GoIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]


@dataclass
class GoWhile(LirNode):
    """Go has no `while` — emit as `for cond { ... }`."""

    test: LirNode
    body: list[LirNode]


@dataclass
class GoForRange(LirNode):
    target: str
    start: LirNode
    stop: LirNode
    step: LirNode | None
    body: list[LirNode]


@dataclass
class GoDecl(LirNode):
    """`var <name> <ty> = <value>`."""

    name: str
    ty: str
    value: LirNode


@dataclass
class GoReassign(LirNode):
    name: str
    value: LirNode


@dataclass
class GoCall(LirNode):
    func: str
    args: list[LirNode]


@dataclass
class GoStruct(LirNode):
    name: str
    fields: list[tuple[str, str]]
    methods: list["GoFn"]


@dataclass
class GoFieldAccess(LirNode):
    value: LirNode
    field: str


@dataclass
class GoBreak(LirNode):
    pass


@dataclass
class GoContinue(LirNode):
    pass


@dataclass
class GoRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = [
    "GoFieldAssign",
    "GoStructInit",
    "GoSubscriptAssign",
    "GoModule",
    "GoFn",
    "GoReturn",
    "GoBinOp",
    "GoCompare",
    "GoBoolOp",
    "GoUnary",
    "GoName",
    "GoIntLiteral",
    "GoFloatLiteral",
    "GoBoolLiteral",
    "GoStringLiteral",
    "GoIf",
    "GoWhile",
    "GoForRange",
    "GoDecl",
    "GoReassign",
    "GoCall",
    "GoStruct",
    "GoFieldAccess",
    "GoBreak",
    "GoContinue",
    "GoRaw",
]

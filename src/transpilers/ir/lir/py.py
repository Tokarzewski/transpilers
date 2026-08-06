"""Python LIR dialect nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode


@dataclass
class PyFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode


@dataclass
class PyStructInit(LirNode):
    """`Point(0, 0)` — positional. Requires the class to be a @dataclass or
    expose a fieldwise __init__; we emit the call form regardless and let
    target verification surface the issue."""

    name: str
    field_values: list[tuple[str, LirNode]]


@dataclass
class PySubscriptAssign(LirNode):
    obj: LirNode
    index: LirNode
    value: LirNode


@dataclass
class PyModule(LirNode):
    items: list[LirNode] = field(default_factory=list)


@dataclass
class PyFn(LirNode):
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: list[LirNode]


@dataclass
class PyReturn(LirNode):
    value: LirNode | None


@dataclass
class PyBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class PyCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class PyBoolOp(LirNode):
    op: str  # "and" / "or"
    left: LirNode
    right: LirNode


@dataclass
class PyUnary(LirNode):
    op: str  # "not" / "-"
    operand: LirNode


@dataclass
class PyName(LirNode):
    name: str


@dataclass
class PyIntLiteral(LirNode):
    value: int


@dataclass
class PyFloatLiteral(LirNode):
    value: float


@dataclass
class PyBoolLiteral(LirNode):
    value: bool


@dataclass
class PyStringLiteral(LirNode):
    value: str


@dataclass
class PyIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]


@dataclass
class PyWhile(LirNode):
    test: LirNode
    body: list[LirNode]


@dataclass
class PyForRange(LirNode):
    target: str
    start: LirNode
    stop: LirNode
    step: LirNode | None
    body: list[LirNode]


@dataclass
class PyAssign(LirNode):
    """Python doesn't distinguish declaration from reassignment; first use
    counts. We still carry the type annotation when known."""

    name: str
    ty: str | None
    value: LirNode


@dataclass
class PyCall(LirNode):
    func: str
    args: list[LirNode]


@dataclass
class PyClass(LirNode):
    name: str
    fields: list[tuple[str, str | None]]  # (name, optional type annotation)
    methods: list["PyFn"]


@dataclass
class PyFieldAccess(LirNode):
    value: LirNode
    field: str


@dataclass
class PyBreak(LirNode):
    pass


@dataclass
class PyContinue(LirNode):
    pass


@dataclass
class PyRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = [
    "PyFieldAssign",
    "PyStructInit",
    "PySubscriptAssign",
    "PyModule",
    "PyFn",
    "PyReturn",
    "PyBinOp",
    "PyCompare",
    "PyBoolOp",
    "PyUnary",
    "PyName",
    "PyIntLiteral",
    "PyFloatLiteral",
    "PyBoolLiteral",
    "PyStringLiteral",
    "PyIf",
    "PyWhile",
    "PyForRange",
    "PyAssign",
    "PyCall",
    "PyClass",
    "PyFieldAccess",
    "PyBreak",
    "PyContinue",
    "PyRaw",
]

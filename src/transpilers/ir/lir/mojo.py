"""Mojo LIR dialect nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode

@dataclass
class MojoFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode

@dataclass
class MojoStructInit(LirNode):
    """`Point(0, 0)` — positional via @fieldwise_init."""

    name: str
    field_values: list[tuple[str, LirNode]]

@dataclass
class MojoSubscriptAssign(LirNode):
    obj: LirNode
    index: LirNode
    value: LirNode

@dataclass
class MojoStruct(LirNode):
    name: str
    fields: list[tuple[str, str]]
    methods: list["MojoFn"]

@dataclass
class MojoFieldAccess(LirNode):
    value: LirNode
    field: str

@dataclass
class MojoModule(LirNode):
    items: list[LirNode] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

@dataclass
class MojoFn(LirNode):
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: list[LirNode]
    raises: bool = False
    # A C++ `static` method: no `self` param, emitted with `@staticmethod`
    # and called as `StructName.method(...)`, no instance needed.
    is_static: bool = False

@dataclass
class MojoReturn(LirNode):
    value: LirNode | None

@dataclass
class MojoBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode

@dataclass
class MojoCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode

@dataclass
class MojoBoolOp(LirNode):
    op: str  # "and" / "or"
    left: LirNode
    right: LirNode

@dataclass
class MojoUnary(LirNode):
    op: str  # "not" / "-"
    operand: LirNode

@dataclass
class MojoName(LirNode):
    name: str

@dataclass
class MojoIntLiteral(LirNode):
    value: int

@dataclass
class MojoFloatLiteral(LirNode):
    value: float

@dataclass
class MojoBoolLiteral(LirNode):
    value: bool

@dataclass
class MojoStringLiteral(LirNode):
    value: str

@dataclass
class MojoIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]

@dataclass
class MojoWhile(LirNode):
    test: LirNode
    body: list[LirNode]

@dataclass
class MojoForRange(LirNode):
    """`for <target> in range(<start>, <stop>[, <step>]):` — Python-style."""

    target: str
    start: LirNode
    stop: LirNode
    step: LirNode | None
    body: list[LirNode]

@dataclass
class MojoVar(LirNode):
    """`var <name>: <ty> = <value>` — Mojo has no `let`, only `var`."""

    name: str
    ty: str | None
    value: LirNode

@dataclass
class MojoReassign(LirNode):
    name: str
    value: LirNode

@dataclass
class MojoList(LirNode):
    """`[a, b, c]` — bracket literal, types inferred."""

    elements: list[LirNode]


@dataclass
class MojoTuple(LirNode):
    """`(a, b, c)` — tuple literal."""

    elements: list[LirNode]


@dataclass
class MojoSlice(LirNode):
    """`v[lo:hi]` — list slice."""

    value: LirNode
    lo: LirNode
    hi: LirNode

@dataclass
class MojoIndex(LirNode):
    value: LirNode
    index: LirNode
    byte: bool = False  # String indexing uses `s[byte=i]` in Mojo

@dataclass
class MojoCall(LirNode):
    func: str
    args: list[LirNode]

@dataclass
class MojoMethodCall(LirNode):
    """Property-style `.len` (no parens) for slice length; other zero-arg
    methods can reuse this."""

    receiver: LirNode
    method: str
    args: list[LirNode]
    paren: bool = True

@dataclass
class MojoBreak(LirNode): pass

@dataclass
class MojoContinue(LirNode): pass


@dataclass
class MojoRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = ['MojoFieldAssign', 'MojoStructInit', 'MojoSubscriptAssign', 'MojoStruct', 'MojoFieldAccess', 'MojoModule', 'MojoFn', 'MojoReturn', 'MojoBinOp', 'MojoCompare', 'MojoBoolOp', 'MojoUnary', 'MojoName', 'MojoIntLiteral', 'MojoFloatLiteral', 'MojoBoolLiteral', 'MojoStringLiteral', 'MojoIf', 'MojoWhile', 'MojoForRange', 'MojoVar', 'MojoReassign', 'MojoList', 'MojoTuple', 'MojoSlice', 'MojoIndex', 'MojoCall', 'MojoMethodCall', 'MojoBreak', 'MojoContinue', 'MojoRaw']

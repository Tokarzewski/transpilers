"""Fortran LIR dialect nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode


@dataclass
class FortranFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode


@dataclass
class FortranStructInit(LirNode):
    """`Point(0, 0)` — Fortran derived-type structure constructor."""

    name: str
    field_values: list[tuple[str, LirNode]]


@dataclass
class FortranSubscriptAssign(LirNode):
    """Fortran arrays are 1-indexed — the emitter adds the +1 offset."""

    obj: LirNode
    index: LirNode
    value: LirNode


@dataclass
class FortranModule(LirNode):
    items: list[LirNode] = field(default_factory=list)


@dataclass
class FortranFn(LirNode):
    name: str
    params: list[tuple[str, str]]  # (name, fortran_type)
    return_type: str | None  # None for subroutines / void
    result_name: str  # name of the result variable to assign
    locals: list[tuple[str, str]]  # extra declarations beyond params + result
    body: list[LirNode]


@dataclass
class FortranAssign(LirNode):
    """Plain assignment — no declaration form. Declarations live on FortranFn."""

    name: str
    value: LirNode


@dataclass
class FortranReturn(LirNode):
    """Bare `return` — for early exit. The result variable carries the value."""


@dataclass
class FortranBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class FortranCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class FortranBoolOp(LirNode):
    op: str  # ".and." / ".or."
    left: LirNode
    right: LirNode


@dataclass
class FortranUnary(LirNode):
    op: str  # ".not." / "-"
    operand: LirNode


@dataclass
class FortranName(LirNode):
    name: str


@dataclass
class FortranIntLiteral(LirNode):
    value: int


@dataclass
class FortranFloatLiteral(LirNode):
    value: float


@dataclass
class FortranBoolLiteral(LirNode):
    value: bool


@dataclass
class FortranStringLiteral(LirNode):
    value: str


@dataclass
class FortranIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]


@dataclass
class FortranWhile(LirNode):
    test: LirNode
    body: list[LirNode]


@dataclass
class FortranForRange(LirNode):
    """`do <target> = <start>, <stop - 1>[, <step>]` — exclusive stop, since
    we lower from MIR's exclusive-stop range semantics. The emitter subtracts
    one literally if both are literal ints; otherwise it emits `stop - 1`."""

    target: str
    start: LirNode
    stop: LirNode
    step: LirNode | None
    body: list[LirNode]


@dataclass
class FortranCall(LirNode):
    func: str
    args: list[LirNode]


@dataclass
class FortranType(LirNode):
    """`type :: Name ... end type Name` — Fortran user-defined type. Methods
    aren't bound here (Fortran modules / type-bound procedures need more
    plumbing than the initial slice carries); they emit as free functions
    that take a `type(Name)` first parameter."""

    name: str
    fields: list[tuple[str, str]]
    methods: list["FortranFn"]


@dataclass
class FortranFieldAccess(LirNode):
    """`obj%field` — Fortran uses `%` for field access, not `.`."""

    value: LirNode
    field: str


@dataclass
class FortranArrayLit(LirNode):
    """`[1, 2, 3]` — Fortran 2003 array constructor.
    When `elements` is empty, `elem_type` must be set so the emitter
    can produce `[<type> ::]` (Fortran requires a typed constructor)."""

    elements: list[LirNode]
    elem_type: str | None = None  # e.g. "logical", "integer"


@dataclass
class FortranSubscript(LirNode):
    """`xs(i + 1)` — Fortran is 1-indexed; emitter adds the +1 offset."""

    value: LirNode
    index: LirNode


@dataclass
class FortranExit(LirNode):
    """Fortran's break-equivalent is `exit`."""


@dataclass
class FortranCycle(LirNode):
    """Fortran's continue-equivalent is `cycle`."""


@dataclass
class FortranRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = [
    "FortranFieldAssign",
    "FortranStructInit",
    "FortranSubscriptAssign",
    "FortranModule",
    "FortranFn",
    "FortranAssign",
    "FortranReturn",
    "FortranBinOp",
    "FortranCompare",
    "FortranBoolOp",
    "FortranUnary",
    "FortranName",
    "FortranIntLiteral",
    "FortranFloatLiteral",
    "FortranBoolLiteral",
    "FortranStringLiteral",
    "FortranIf",
    "FortranWhile",
    "FortranForRange",
    "FortranCall",
    "FortranType",
    "FortranFieldAccess",
    "FortranArrayLit",
    "FortranSubscript",
    "FortranExit",
    "FortranCycle",
    "FortranRaw",
]

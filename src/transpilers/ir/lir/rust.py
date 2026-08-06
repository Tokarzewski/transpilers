"""Rust LIR dialect nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from .base import LirNode


@dataclass
class RustModule(LirNode):
    items: list[LirNode] = field(default_factory=list)


@dataclass
class RustFn(LirNode):
    name: str
    params: list[tuple[str, str]]  # (name, rust_type)
    return_type: str
    body: list[LirNode]


@dataclass
class RustReturn(LirNode):
    value: LirNode | None


@dataclass
class RustBinOp(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class RustName(LirNode):
    name: str


@dataclass
class RustIntLiteral(LirNode):
    value: int
    # Suffix is optional now — emitter elides it when the surrounding context
    # makes the type unambiguous (a typed `let mut x: i64 = 1;`, a binop
    # with a typed operand, an arg to a typed fn, etc.). Lowering sets this
    # when it can't prove the suffix is redundant.
    suffix: str | None = None


@dataclass
class RustFloatLiteral(LirNode):
    value: float
    suffix: str | None = None


@dataclass
class RustBoolLiteral(LirNode):
    value: bool


@dataclass
class RustStringLiteral(LirNode):
    value: str


@dataclass
class RustCompare(LirNode):
    op: str
    left: LirNode
    right: LirNode


@dataclass
class RustBoolOp(LirNode):
    op: str  # "&&" or "||"
    left: LirNode
    right: LirNode


@dataclass
class RustUnary(LirNode):
    op: str  # "!" or "-"
    operand: LirNode


@dataclass
class RustIf(LirNode):
    test: LirNode
    body: list[LirNode]
    orelse: list[LirNode]


@dataclass
class RustWhile(LirNode):
    test: LirNode
    body: list[LirNode]


@dataclass
class RustForRange(LirNode):
    """Rust `for <target> in <start>..<stop>` (step=None means contiguous).
    `step` set => emit `.step_by(...)`."""

    target: str
    start: LirNode
    stop: LirNode
    step: LirNode | None
    body: list[LirNode]


@dataclass
class RustLet(LirNode):
    """`let [mut] <name>[: <ty>] = <value>;`"""

    name: str
    mutable: bool
    ty: str | None
    value: LirNode


@dataclass
class RustReassign(LirNode):
    """Plain `x = <value>;` for a previously-bound mutable variable."""

    name: str
    value: LirNode


@dataclass
class RustVec(LirNode):
    elements: list[LirNode]


@dataclass
class RustIndex(LirNode):
    value: LirNode
    index: LirNode


@dataclass
class RustMethodCall(LirNode):
    receiver: LirNode
    method: str
    args: list[LirNode]
    cast_to: str | None = None  # e.g., `.len() as i64`


@dataclass
class RustCall(LirNode):
    func: str
    args: list[LirNode]


@dataclass
class RustStruct(LirNode):
    name: str
    fields: list[tuple[str, str]]  # (field_name, rust_type)


@dataclass
class RustImpl(LirNode):
    struct_name: str
    methods: list["RustFn"]


@dataclass
class RustFieldAccess(LirNode):
    value: LirNode
    field: str


@dataclass
class RustFieldAssign(LirNode):
    obj: LirNode
    field: str
    value: LirNode


@dataclass
class RustStructInit(LirNode):
    """`Point { x: 0, y: 0 }`."""

    name: str
    field_values: list[tuple[str, LirNode]]


@dataclass
class RustSubscriptAssign(LirNode):
    obj: LirNode
    index: LirNode
    value: LirNode


@dataclass
class RustFormat(LirNode):
    """`format!("{}{}...", arg1, arg2, ...)` — produced when MIR binop `+` has
    two StrT operands. format! accepts both String and &str via Display, so
    it's the safest emission for string concat regardless of operand
    ownership."""

    args: list[LirNode]


@dataclass
class RustMacro(LirNode):
    """Rust macro-call emission: `name!(<rendered args>)`. Used for
    builtins that map to macros (`println!`, `print!`, `format!`,
    `vec!`, ...). The args are rendered as-is between the parens."""

    name: str  # without the trailing `!`
    template: str  # format spec, e.g. "{} {}" — empty when no template
    args: list[LirNode]


@dataclass
class RustMethodChain(LirNode):
    """`receiver.method(args)` shorthand for stdlib-builtin lowering.
    Different from RustMethodCall (which carries a `cast_to` for length-
    style emission) — this is a plain method call, no cast."""

    receiver: LirNode
    method: str
    args: list[LirNode]


@dataclass
class RustBreak(LirNode):
    pass


@dataclass
class RustContinue(LirNode):
    pass


@dataclass
class RustRaw(LirNode):
    """Unsupported-construct hole; emitted as a target-appropriate
    `TODO[port]` stub preserving the original source snippet."""

    snippet: str


__all__ = [
    "RustModule",
    "RustFn",
    "RustReturn",
    "RustBinOp",
    "RustName",
    "RustIntLiteral",
    "RustFloatLiteral",
    "RustBoolLiteral",
    "RustStringLiteral",
    "RustCompare",
    "RustBoolOp",
    "RustUnary",
    "RustIf",
    "RustWhile",
    "RustForRange",
    "RustLet",
    "RustReassign",
    "RustVec",
    "RustIndex",
    "RustMethodCall",
    "RustCall",
    "RustStruct",
    "RustImpl",
    "RustFieldAccess",
    "RustFieldAssign",
    "RustStructInit",
    "RustSubscriptAssign",
    "RustFormat",
    "RustMacro",
    "RustMethodChain",
    "RustBreak",
    "RustContinue",
    "RustRaw",
]

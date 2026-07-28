"""MIR — normalized typed IR. Language-agnostic.

This is where 80% of the work lives: type inference, dataflow, ownership analysis,
escape analysis, idiom recognition. Every node has a resolved type or an
UnknownT hole that some later pass (algorithmic or LLM) must fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Type, UnknownT
from .contracts import SemanticContract, WILDCARD


class MirNode:
    """Base class for all MIR nodes.

    Every subclass that is a ``@dataclass`` automatically gets a
    ``_hir_provenance_id: int`` instance attribute (set to 0 by default,
    set to a real HIR node id during lowering).
    """

    _hir_provenance_id: int
    contract: SemanticContract = WILDCARD

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        old_post = getattr(cls, "__post_init__", None)

        def _post_init(self):
            self._hir_provenance_id = 0
            self.contract = WILDCARD
            if old_post:
                old_post(self)

        cls.__post_init__ = _post_init


@dataclass
class MirModule(MirNode):
    functions: list["MirFunction"] = field(default_factory=list)
    structs: list["MirStruct"] = field(default_factory=list)


@dataclass
class MirFunction(MirNode):
    name: str
    params: list["MirParam"]
    return_type: Type
    body: list[MirNode]
    # See HirFunction.is_static.
    is_static: bool = False


@dataclass
class MirParam(MirNode):
    name: str
    ty: Type


@dataclass
class MirRaw(MirNode):
    """Unsupported-construct hole carrying the original source snippet.

    Lowered 1:1 from HirRaw; each target's MIR->LIR pass maps it to its
    dialect's `<Prefix>Raw` node, which the backend emits as a `TODO[port]`
    stub. Valid in statement and expression position."""

    snippet: str
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirReturn(MirNode):
    value: MirNode | None


@dataclass
class MirBreak(MirNode):
    pass


@dataclass
class MirContinue(MirNode):
    pass


@dataclass
class MirBinOp(MirNode):
    op: str
    left: MirNode
    right: MirNode
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirName(MirNode):
    name: str
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirIntLiteral(MirNode):
    value: int
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirFloatLiteral(MirNode):
    value: float
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirBoolLiteral(MirNode):
    value: bool
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirStringLiteral(MirNode):
    value: str
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirNullLiteral(MirNode):
    """Source-level null/None — backend renders as its native sentinel."""

    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirCompare(MirNode):
    op: str
    left: MirNode
    right: MirNode
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirBoolOp(MirNode):
    op: str
    left: MirNode
    right: MirNode
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirUnaryOp(MirNode):
    op: str
    operand: MirNode
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirIf(MirNode):
    test: MirNode
    body: list[MirNode]
    orelse: list[MirNode]


@dataclass
class MirWhile(MirNode):
    test: MirNode
    body: list[MirNode]


@dataclass
class MirForRange(MirNode):
    """Specialized for-over-range. `start`, `stop`, `step` are MIR int exprs."""

    target: str
    start: MirNode
    stop: MirNode
    step: MirNode | None
    body: list[MirNode]


@dataclass
class MirCall(MirNode):
    func: str
    args: list[MirNode]
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirAssign(MirNode):
    target: str
    value: MirNode
    ty: Type = field(default_factory=UnknownT)
    augmented_op: str | None = None
    is_declaration: bool = False  # set by mutability inference / scoping


@dataclass
class MirList(MirNode):
    elements: list[MirNode]
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirSubscript(MirNode):
    value: MirNode
    index: MirNode
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirStruct(MirNode):
    """A user-defined struct/class. Fields and methods carry resolved types.
    Lowering targets pick struct + impl-style shape per language."""

    name: str
    fields: list["MirParam"]
    methods: list["MirFunction"]


@dataclass
class MirFieldAccess(MirNode):
    value: MirNode
    field: str
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirMethodCall(MirNode):
    receiver: MirNode
    method: str
    args: list[MirNode]
    ty: Type = field(default_factory=UnknownT)


@dataclass
class MirFieldAssign(MirNode):
    obj: MirNode
    field: str
    value: MirNode


@dataclass
class MirSubscriptAssign(MirNode):
    obj: MirNode
    index: MirNode
    value: MirNode


@dataclass
class MirStructInit(MirNode):
    """`Point { x: 0, y: 0 }`-shaped construction. `field_values` carries
    `(field_name, value)` pairs in the struct's field declaration order, so
    every target's emitter can render the named form or unwrap to positional."""

    name: str
    field_values: list[tuple[str, MirNode]]
    ty: Type = field(default_factory=UnknownT)

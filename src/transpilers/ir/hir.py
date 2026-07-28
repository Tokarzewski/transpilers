"""HIR — source-faithful tree. One per source language.

HIR preserves language idioms (Python comprehensions, Fortran DO, C++ templates)
so a frontend can be lossless within its own language. Normalization happens at
HIR -> MIR, not in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Global monotonic counter for HIR node ids.
_HIR_NODE_ID_COUNTER = 0


def _next_hir_id() -> int:
    global _HIR_NODE_ID_COUNTER
    _HIR_NODE_ID_COUNTER += 1
    return _HIR_NODE_ID_COUNTER


class HirNode:
    """Base for every HIR node.

    Every subclass that is a ``@dataclass`` automatically gets a
    ``_hir_node_id: int`` instance attribute (set via ``__post_init__``).

    The optional ``source_loc`` field is set by frontends that have
    access to a parser carrying source positions (libclang, libcst,
    tree-sitter...). It is used by the C++ ground-truth pass to
    cross-reference HIR nodes with the corresponding clang AST cursor
    so concrete types can be looked up at issue #50.
    """

    source_loc: str | None = None
    _hir_node_id: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        old_post = getattr(cls, "__post_init__", None)

        def _post_init(self):
            self._hir_node_id = _next_hir_id()
            if old_post:
                old_post(self)

        cls.__post_init__ = _post_init


@dataclass
class HirModule(HirNode):
    source_lang: str
    body: list[HirNode] = field(default_factory=list)


@dataclass
class HirFunction(HirNode):
    name: str
    params: list["HirParam"]
    return_annotation: str | None
    body: list[HirNode]
    # True for a C++ `static` method: no implicit `self`/`this`, callable
    # via `ClassName::method(...)` with no receiver instance. `params`
    # holds only the explicit parameters in this case (no injected `self`).
    is_static: bool = False


@dataclass
class HirParam(HirNode):
    name: str
    annotation: str | None


@dataclass
class HirRaw(HirNode):
    """An unsupported construct, preserved verbatim as a source snippet.

    The never-refuse contract: when a frontend hits a statement or expression
    it can't lower, it emits one of these instead of aborting the whole
    function. Downstream passes carry the snippet through to the backend, which
    emits a target-appropriate `TODO[port]` stub. Valid in both statement and
    expression position."""

    snippet: str


@dataclass
class HirReturn(HirNode):
    value: HirNode | None


@dataclass
class HirBreak(HirNode):
    """`break` — exits the innermost enclosing loop."""


@dataclass
class HirContinue(HirNode):
    """`continue` — skips to the next iteration of the innermost loop."""


@dataclass
class HirBinOp(HirNode):
    op: str
    left: HirNode
    right: HirNode


@dataclass
class HirName(HirNode):
    name: str


@dataclass
class HirIntLiteral(HirNode):
    value: int


@dataclass
class HirFloatLiteral(HirNode):
    value: float


@dataclass
class HirBoolLiteral(HirNode):
    value: bool


@dataclass
class HirStringLiteral(HirNode):
    value: str


@dataclass
class HirNullLiteral(HirNode):
    """Python `None`, Java `null`, C `NULL`, Go `nil`. Distinct from
    int 0 so backends emit their native sentinel (or refuse). A real
    OptionT in the type lattice would let downstream passes reason
    about nullability; this node is the minimum primitive needed."""


@dataclass
class HirCompare(HirNode):
    op: str
    left: HirNode
    right: HirNode


@dataclass
class HirBoolOp(HirNode):
    op: str  # "and" or "or"
    left: HirNode
    right: HirNode


@dataclass
class HirUnaryOp(HirNode):
    op: str  # "not" or "-"
    operand: HirNode


@dataclass
class HirIf(HirNode):
    test: HirNode
    body: list[HirNode]
    orelse: list[HirNode]


@dataclass
class HirWhile(HirNode):
    test: HirNode
    body: list[HirNode]


@dataclass
class HirFor(HirNode):
    """Restricted to `for <name> in range(...)` for the initial subset."""

    target: str
    iter: HirNode
    body: list[HirNode]


@dataclass
class HirForEach(HirNode):
    """`for <value> in <name>:` or `for <index>, <value> in enumerate(<name>):`.

    The iterable is restricted to a bare name (`iterable`) so HIR->MIR can
    desugar it to an indexed `range(len(name))` loop without re-evaluating a
    side-effecting expression. `index_name` is set for the enumerate form and
    `None` for plain foreach (HIR->MIR synthesises a fresh index there).
    """

    value_name: str
    iterable: str
    index_name: str | None
    body: list[HirNode]


@dataclass
class HirCall(HirNode):
    func: str
    args: list[HirNode]


@dataclass
class HirAssign(HirNode):
    target: str
    value: HirNode
    annotation: str | None
    augmented_op: str | None = None  # "+", "-", "*", "/" for `x += 1` etc.


@dataclass
class HirList(HirNode):
    elements: list[HirNode]


@dataclass
class HirSubscript(HirNode):
    value: HirNode
    index: HirNode


@dataclass
class HirStruct(HirNode):
    """A user-defined struct / class.

    `fields` carry types (HIR annotation strings). `methods` are full
    HirFunction nodes whose first parameter is `self` with the struct's
    own type as annotation. The frontend wires this up; downstream passes
    can treat methods as plain functions.
    """

    name: str
    fields: list["HirParam"]
    methods: list["HirFunction"]


@dataclass
class HirFieldAccess(HirNode):
    """`obj.field` — read."""

    value: HirNode
    field: str


@dataclass
class HirMethodCall(HirNode):
    """`obj.method(args)`."""

    receiver: HirNode
    method: str
    args: list[HirNode]


@dataclass
class HirFieldAssign(HirNode):
    """`obj.field = value` — write to a struct field."""

    obj: HirNode
    field: str
    value: HirNode


@dataclass
class HirSubscriptAssign(HirNode):
    """`obj[index] = value` — indexed write into a list / array / map."""

    obj: HirNode
    index: HirNode
    value: HirNode


@dataclass
class HirStructInit(HirNode):
    """Constructor-style struct creation. `args` correspond to the struct's
    fields in declaration order (positional). Empty args means zero-init."""

    name: str
    args: list[HirNode]

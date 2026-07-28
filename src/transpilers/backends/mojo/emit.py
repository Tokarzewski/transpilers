"""Mojo LIR -> Mojo source.

Indentation-sensitive, brace-free. Empty function bodies need `pass`.
"""

from __future__ import annotations

from transpilers.ir import lir


INDENT = "    "

# Mojo reserved words that are valid C++ identifiers — rename on emit (append _)
# so e.g. a C++ variable named `var`/`ref`/`out` doesn't collide. Deterministic,
# so declaration and uses stay consistent without a rename map.
_MOJO_KEYWORDS = frozenset({
    "var", "ref", "mut", "out", "read", "owned", "deinit", "fn", "def",
    "alias", "let", "in", "raises", "comptime", "trait", "struct",
})


def _safe(name: str) -> str:
    return name + "_" if name in _MOJO_KEYWORDS else name


def _augmented_form(name: str, value: lir.LirNode) -> tuple[str, lir.LirNode] | None:
    if not isinstance(value, lir.MojoBinOp):
        return None
    if not (isinstance(value.left, lir.MojoName) and value.left.name == name):
        return None
    if value.op not in ("+", "-", "*", "/", "%"):
        return None
    return value.op, value.right


def emit_mojo(module: lir.MojoModule) -> str:
    mutating_names = _compute_mutating_method_names(module)
    body = "\n\n".join(_emit_item(item, mutating_names) for item in module.items) + "\n"
    imports = getattr(module, "imports", None)
    if imports:
        # Entries may be bare module names (`math`) or full statements
        # (`from math import sqrt`); emit the latter verbatim.
        lines = [m if m.startswith(("from ", "import ")) else f"import {m}"
                 for m in imports]
        return "\n".join(lines) + "\n\n" + body
    return body


def _emit_item(item: lir.LirNode, mutating_names: frozenset[str]) -> str:
    if isinstance(item, lir.MojoStruct):
        return _emit_struct(item, mutating_names)
    if isinstance(item, lir.MojoFn):
        return _emit_fn(item, mutating_names)
    raise NotImplementedError(f"mojo top-level item {type(item).__name__}")


def _emit_struct(s: lir.MojoStruct, mutating_names: frozenset[str]) -> str:
    """`@fieldwise_init` + `Copyable, Movable` conformance gives the struct
    a usable constructor and value semantics in current Mojo. An explicit
    `__init__` (from a C++ constructor) replaces the synthesized fieldwise
    one, so `@fieldwise_init` is dropped to avoid a duplicate constructor."""
    has_explicit_init = any(m.name == "__init__" for m in s.methods)
    lines = [] if has_explicit_init else ["@fieldwise_init"]
    lines.append(f"struct {s.name}(Copyable, Movable):")
    if not s.fields and not s.methods:
        lines.append(INDENT + "pass")
        return "\n".join(lines)
    for name, ty in s.fields:
        lines.append(f"{INDENT}var {_safe(name)}: {ty}")
    for m in s.methods:
        lines.append("")
        lines.append(_emit_fn(m, mutating_names, depth=1))
    return "\n".join(lines)


_MUTATING_METHODS = frozenset({
    "append", "pop", "clear", "insert", "extend", "resize", "remove", "reverse", "sort",
})


def _is_self(node: object) -> bool:
    return isinstance(node, lir.MojoName) and node.name == "self"


def _touches_self(node: object) -> bool:
    """A receiver/target rooted at `self` (self, self.field, self.field[i])."""
    if _is_self(node):
        return True
    if isinstance(node, lir.MojoFieldAccess):
        return _touches_self(node.value)
    if isinstance(node, lir.MojoIndex):
        return _touches_self(node.value)
    return False


def _mutates_self(nodes: object, mutating_names: frozenset[str]) -> bool:
    """Does this method body mutate `self`? (assign self.field, self.field[i]=…,
    or call a mutating method on a self-rooted receiver). Such methods need
    `mut self` in Mojo (default borrow is immutable). `mutating_names` is the
    module-wide, fixed-point-closed set of method names known to mutate
    their receiver (see _compute_mutating_method_names) -- covers both the
    hardcoded STL-container names and any user struct method that mutates
    self itself, directly or transitively."""
    import dataclasses
    if isinstance(nodes, list):
        return any(_mutates_self(n, mutating_names) for n in nodes)
    if not dataclasses.is_dataclass(nodes):
        return False
    if isinstance(nodes, lir.MojoFieldAssign) and _touches_self(nodes.obj):
        return True
    if isinstance(nodes, lir.MojoSubscriptAssign) and _touches_self(nodes.obj):
        return True
    if isinstance(nodes, lir.MojoReassign) and nodes.name == "self":
        # `self = expr` -- the "replace my whole value" idiom for a
        # mutate-via-copy-assign method (C++'s `(*this) = Multiplied(...);`,
        # see `_is_this_deref` in the C++ frontend). Reassigning `self`
        # outright is just as much a mutation as assigning one of its
        # fields, and needs the same `mut self`.
        return True
    if (isinstance(nodes, lir.MojoMethodCall) and nodes.method in mutating_names
            and _touches_self(nodes.receiver)):
        return True
    return any(_mutates_self(getattr(nodes, f.name), mutating_names) for f in dataclasses.fields(nodes))


def _compute_mutating_method_names(module: lir.MojoModule) -> frozenset[str]:
    """Fixed-point closure of method names that mutate their receiver.

    Mojo requires `mut self` on any method that mutates self -- directly (a
    field/subscript assign) or transitively, by calling another mutating
    method on self or on a field self owns (e.g. `Multiplied()` calling
    `self.Multiply(x)`, or `Transform()` calling `self.coord.Add(x)`). A
    single-pass, body-only check misses the transitive case entirely,
    wrongly leaving the caller as an immutable `self` and producing
    "invalid use of mutating method on rvalue" from the real Mojo compiler.
    Real-world value types lean on exactly this Foo()/Fooed() mutate-vs-
    return-copy pairing throughout (OCCT's gp_* package, but not only it),
    so resolving the closure properly -- rather than hardcoding a small
    STL-container method-name list -- matters for any nontrivial struct.
    Matching by bare method name (not per-struct) is deliberately
    conservative: at worst it marks an unrelated same-named method `mut`
    too, which Mojo permits on a method that doesn't strictly need it.
    """
    names = set(_MUTATING_METHODS)
    methods = [m for item in module.items if isinstance(item, lir.MojoStruct) for m in item.methods]
    changed = True
    while changed:
        changed = False
        for m in methods:
            if m.name in names:
                continue
            if _mutates_self(m.body, frozenset(names)):
                names.add(m.name)
                changed = True
    return frozenset(names)


def _emit_fn(fn: lir.MojoFn, mutating_names: frozenset[str], *, depth: int = 0) -> str:
    indent = INDENT * depth
    # `__init__` takes `out self`; a method that mutates self takes `mut self`;
    # otherwise the default immutable borrow `self`.
    mut_self = (fn.params and fn.params[0][0] == "self"
                and fn.name != "__init__" and _mutates_self(fn.body, mutating_names))
    params = ", ".join(
        ("out self" if (n == "self" and fn.name == "__init__")
         else "mut self" if (n == "self" and mut_self)
         else _emit_param(n, t))
        for n, t in fn.params
    )
    ret = f" -> {fn.return_type}" if fn.return_type != "None" else ""
    raises = " raises" if getattr(fn, "raises", False) else ""
    decorator = f"{indent}@staticmethod\n" if getattr(fn, "is_static", False) else ""
    header = f"{decorator}{indent}def {_safe(fn.name)}({params}){raises}{ret}:"
    body = _emit_block(fn.body, depth + 1) or (indent + INDENT + "pass")
    return f"{header}\n{body}"


def _emit_param(name: str, ty: str) -> str:
    if name == "self":
        return "self"
    return f"{_safe(name)}: {ty}"


def _emit_block(nodes: list[lir.LirNode], depth: int) -> str:
    lines = [_emit_stmt(n, depth) for n in nodes]
    return "\n".join(lines)


def _flatten_snippet(snippet: str) -> str:
    """Collapse a multi-line source snippet to a single comment-safe line."""
    return " ".join(snippet.split())


def _emit_stmt(node: lir.LirNode, depth: int) -> str:
    pad = INDENT * depth
    if isinstance(node, lir.MojoRaw):
        return f"{pad}pass  # TODO[port]: {_flatten_snippet(node.snippet)}"
    if isinstance(node, lir.MojoBreak):
        return f"{pad}break"
    if isinstance(node, lir.MojoContinue):
        return f"{pad}continue"
    if isinstance(node, lir.MojoReturn):
        return f"{pad}return {_emit_expr(node.value)}" if node.value else f"{pad}return"
    if isinstance(node, lir.MojoVar):
        ann = f": {node.ty}" if node.ty else ""
        return f"{pad}var {_safe(node.name)}{ann} = {_emit_expr(node.value)}"
    if isinstance(node, lir.MojoReassign):
        aug = _augmented_form(node.name, node.value)
        if aug is not None:
            op, rhs = aug
            return f"{pad}{_safe(node.name)} {op}= {_emit_expr(rhs)}"
        return f"{pad}{_safe(node.name)} = {_emit_expr(node.value)}"
    if isinstance(node, lir.MojoFieldAssign):
        return f"{pad}{_emit_expr(node.obj)}.{node.field} = {_emit_expr(node.value)}"
    if isinstance(node, lir.MojoSubscriptAssign):
        return f"{pad}{_emit_expr(node.obj)}[{_emit_expr(node.index)}] = {_emit_expr(node.value)}"
    if isinstance(node, lir.MojoIf):
        head = f"{pad}if {_emit_expr(node.test)}:"
        body = _emit_block(node.body, depth + 1) or (pad + INDENT + "pass")
        if node.orelse:
            # Collapse `else: if ...` chain into `elif`.
            if len(node.orelse) == 1 and isinstance(node.orelse[0], lir.MojoIf):
                inner = _emit_stmt(node.orelse[0], depth)
                # Replace `if` with `elif` at this indent level.
                elif_inner = inner.replace(f"{pad}if ", f"{pad}elif ", 1)
                return f"{head}\n{body}\n{elif_inner}"
            else_body = _emit_block(node.orelse, depth + 1) or (pad + INDENT + "pass")
            return f"{head}\n{body}\n{pad}else:\n{else_body}"
        return f"{head}\n{body}"
    if isinstance(node, lir.MojoWhile):
        head = f"{pad}while {_emit_expr(node.test)}:"
        body = _emit_block(node.body, depth + 1) or (pad + INDENT + "pass")
        return f"{head}\n{body}"
    if isinstance(node, lir.MojoForRange):
        if node.step is None:
            args = f"{_emit_expr(node.start)}, {_emit_expr(node.stop)}"
        else:
            args = f"{_emit_expr(node.start)}, {_emit_expr(node.stop)}, {_emit_expr(node.step)}"
        head = f"{pad}for {_safe(node.target)} in range({args}):"
        body = _emit_block(node.body, depth + 1) or (pad + INDENT + "pass")
        return f"{head}\n{body}"
    return f"{pad}{_emit_expr(node)}"


def _op_of(node: lir.LirNode) -> str | None:
    if isinstance(node, (lir.MojoBinOp, lir.MojoCompare, lir.MojoBoolOp)):
        return node.op
    return None


def _paren(child: lir.LirNode, parent_op: str, *, on_right: bool) -> str:
    from transpilers.backends._precedence import paren_emit
    return paren_emit(child, parent_op, on_right=on_right, emit_expr=_emit_expr, op_of=_op_of)


def _emit_expr(node: lir.LirNode | None) -> str:
    if node is None:
        return ""
    if isinstance(node, lir.MojoRaw):
        # Expr position: no trailing `#` comment (would swallow the rest of an
        # enclosing line). Encode the snippet in a never-called marker call.
        text = _flatten_snippet(node.snippet).replace("\\", "\\\\").replace('"', '\\"')
        return f'__todo_port__("{text}")'
    if isinstance(node, lir.MojoBinOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.MojoCompare):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.MojoBoolOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.MojoUnary):
        operand = _paren(node.operand, "__unary__", on_right=False)
        return f"{node.op} {operand}" if node.op == "not" else f"{node.op}{operand}"
    if isinstance(node, lir.MojoName):
        return _safe(node.name)
    if isinstance(node, lir.MojoIntLiteral):
        return str(node.value)
    if isinstance(node, lir.MojoFloatLiteral):
        text = repr(node.value)
        return text if "." in text or "e" in text else text + ".0"
    if isinstance(node, lir.MojoBoolLiteral):
        return "True" if node.value else "False"
    if isinstance(node, lir.MojoStringLiteral):
        escaped = node.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(node, lir.MojoList):
        items = ", ".join(_emit_expr(e) for e in node.elements)
        return f"[{items}]"
    if isinstance(node, lir.MojoTuple):
        items = ", ".join(_emit_expr(e) for e in node.elements)
        return f"({items})"
    if isinstance(node, lir.MojoSlice):
        # Mojo slicing returns a Span; the C++ iterator-range ctor copies, so
        # materialize into a List.
        return f"List({_emit_expr(node.value)}[{_emit_expr(node.lo)}:{_emit_expr(node.hi)}])"
    if isinstance(node, lir.MojoFieldAccess):
        return f"{_emit_expr(node.value)}.{node.field}"
    if isinstance(node, lir.MojoStructInit):
        # @fieldwise_init synthesizes a positional constructor in declaration
        # order; emit as `Point(0, 0)`.
        args = ", ".join(_emit_expr(v) for _, v in node.field_values)
        return f"{node.name}({args})"
    if isinstance(node, lir.MojoIndex):
        if getattr(node, "byte", False):
            return f"{_emit_expr(node.value)}[byte={_emit_expr(node.index)}]"
        return f"{_emit_expr(node.value)}[{_emit_expr(node.index)}]"
    if isinstance(node, lir.MojoCall):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{node.func}({args})"
    if isinstance(node, lir.MojoMethodCall):
        if not node.paren:
            return f"{_emit_expr(node.receiver)}.{node.method}"
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{_emit_expr(node.receiver)}.{node.method}({args})"
    from transpilers.passes.mir_to_mojo_lir import _MojoIfExpr
    if isinstance(node, _MojoIfExpr):
        return f"({_emit_expr(node.then_)} if {_emit_expr(node.test)} else {_emit_expr(node.else_)})"
    raise NotImplementedError(f"LIR node {type(node).__name__}")

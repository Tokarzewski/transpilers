"""Rust LIR -> Rust source.

Deterministic emission. No LLM here — naming/comment LLM passes operate on the
LIR before emission, never on the text. Keeping emit pure makes the output
reproducible and makes round-trip parsing (re-parsing emitted Rust back into
its CST) viable.
"""

from __future__ import annotations

from transpilers.ir import lir


INDENT = "    "


def _augmented_form(name: str, value: lir.LirNode) -> tuple[str, lir.LirNode] | None:
    """If `value` is `name <op> rhs` with op in the augmented-assign set, return
    `(op, rhs)`. Lets Reassign emit `x += v` instead of `x = x + v`."""
    if not isinstance(value, lir.RustBinOp):
        return None
    if not (isinstance(value.left, lir.RustName) and value.left.name == name):
        return None
    if value.op not in ("+", "-", "*", "/", "%"):
        return None
    return value.op, value.right


def _format_float(value: float) -> str:
    """Emit a Rust-parseable float literal. Whole numbers need an explicit
    decimal to differentiate from int (`1f64` works but `1.0` is more
    idiomatic at the source level)."""
    text = repr(value)
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def emit_rust(module: lir.RustModule) -> str:
    return "\n\n".join(_emit_item(item) for item in module.items) + "\n"


def _emit_item(item: lir.LirNode) -> str:
    if isinstance(item, lir.RustStruct):
        return _emit_struct(item)
    if isinstance(item, lir.RustImpl):
        return _emit_impl(item)
    if isinstance(item, lir.RustFn):
        return _emit_fn(item)
    raise NotImplementedError(f"rust top-level item {type(item).__name__}")


def _emit_struct(s: lir.RustStruct) -> str:
    if not s.fields:
        # A fieldless struct (e.g. a class with only static methods) has no
        # lines to join; the unconditional trailing ",\n}" below would
        # otherwise emit "struct Empty {\n,\n}" -- a stray leading comma
        # that's a syntax error, not just an empty-looking struct.
        return f"struct {s.name} {{}}"
    field_lines = ",\n".join(f"{INDENT}{n}: {t}" for n, t in s.fields)
    return f"struct {s.name} {{\n{field_lines},\n}}"


def _emit_impl(impl: lir.RustImpl) -> str:
    body = "\n\n".join(_emit_fn(m, depth=1) for m in impl.methods)
    return f"impl {impl.struct_name} {{\n{body}\n}}"


def _emit_fn(fn: lir.RustFn, *, depth: int = 0) -> str:
    indent = INDENT * depth
    params = ", ".join(_emit_param(n, t) for n, t in fn.params)
    # Drop `-> ()` for unit-returning fns.
    ret = "" if fn.return_type == "()" else f" -> {fn.return_type}"
    header = f"{indent}fn {fn.name}({params}){ret} {{"
    body = _emit_block(fn.body, depth + 1)
    return f"{header}\n{body}\n{indent}}}"


def _emit_param(name: str, ty: str) -> str:
    # Convention: a parameter named `self` is the method receiver — emit as
    # `&self` (immutable borrow) regardless of the declared type. Method
    # mutation would need `&mut self`, which we don't yet model.
    if name == "self":
        return "&self"
    return f"{name}: {ty}"


def _emit_block(nodes: list[lir.LirNode], depth: int) -> str:
    lines: list[str] = []
    for n in nodes:
        lines.append(_emit_stmt(n, depth))
    return "\n".join(lines)


def _flatten_snippet(snippet: str) -> str:
    """Collapse a multi-line source snippet to a single comment-safe line."""
    return " ".join(snippet.split()).replace("*/", "* /")


def _emit_stmt(node: lir.LirNode, depth: int) -> str:
    pad = INDENT * depth
    if isinstance(node, lir.RustRaw):
        return (
            f"{pad}/* TODO[port]: {_flatten_snippet(node.snippet)} */ unimplemented!();"
        )
    if isinstance(node, lir.RustReturn):
        return (
            f"{pad}return {_emit_expr(node.value)};" if node.value else f"{pad}return;"
        )
    if isinstance(node, lir.RustBreak):
        return f"{pad}break;"
    if isinstance(node, lir.RustContinue):
        return f"{pad}continue;"
    if isinstance(node, lir.RustLet):
        mut = "mut " if node.mutable else ""
        ann = f": {node.ty}" if node.ty else ""
        return f"{pad}let {mut}{node.name}{ann} = {_emit_expr(node.value)};"
    if isinstance(node, lir.RustReassign):
        aug = _augmented_form(node.name, node.value)
        if aug is not None:
            op, rhs = aug
            return f"{pad}{node.name} {op}= {_emit_expr(rhs)};"
        return f"{pad}{node.name} = {_emit_expr(node.value)};"
    if isinstance(node, lir.RustFieldAssign):
        return f"{pad}{_emit_expr(node.obj)}.{node.field} = {_emit_expr(node.value)};"
    if isinstance(node, lir.RustSubscriptAssign):
        # Rust indexing requires `usize`; our IntT lowers to `i64`, so cast.
        # Parenthesise the index so `j + 1 as usize` doesn't parse as `j + (1 as usize)`.
        return f"{pad}{_emit_expr(node.obj)}[({_emit_expr(node.index)}) as usize] = {_emit_expr(node.value)};"
    if isinstance(node, lir.RustIf):
        head = f"{pad}if {_emit_expr(node.test)} {{"
        body = _emit_block(node.body, depth + 1)
        if node.orelse:
            # Collapse `else { if ... }` into `else if ...` for readability.
            if len(node.orelse) == 1 and isinstance(node.orelse[0], lir.RustIf):
                inner = _emit_stmt(node.orelse[0], depth).lstrip()
                tail = f"{pad}}} else {inner}"
                return f"{head}\n{body}\n{tail}"
            else_body = _emit_block(node.orelse, depth + 1)
            return f"{head}\n{body}\n{pad}}} else {{\n{else_body}\n{pad}}}"
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.RustWhile):
        head = f"{pad}while {_emit_expr(node.test)} {{"
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.RustForRange):
        rng = f"{_emit_expr(node.start)}..{_emit_expr(node.stop)}"
        if node.step is not None:
            rng = f"({rng}).step_by({_emit_expr(node.step)} as usize)"
        head = f"{pad}for {node.target} in {rng} {{"
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    # Expression-statement fallthrough.
    return f"{pad}{_emit_expr(node)};"


def _op_of(node: lir.LirNode) -> str | None:
    if isinstance(node, (lir.RustBinOp, lir.RustCompare, lir.RustBoolOp)):
        return node.op
    return None


def _paren(child: lir.LirNode, parent_op: str, *, on_right: bool) -> str:
    from transpilers.backends._precedence import paren_emit

    return paren_emit(
        child, parent_op, on_right=on_right, emit_expr=_emit_expr, op_of=_op_of
    )


def _emit_expr(node: lir.LirNode | None) -> str:
    if node is None:
        return ""
    if isinstance(node, lir.RustRaw):
        return f"/* TODO[port]: {_flatten_snippet(node.snippet)} */ Default::default()"
    if isinstance(node, lir.RustBinOp):
        # `x as i64` style casts are emitted as binops with op="as".
        if node.op == "as":
            return f"{_emit_expr(node.left)} as {_emit_expr(node.right)}"
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.RustCompare):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.RustBoolOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.RustUnary):
        # Unary binds tighter than most binops; parenthesize any non-atomic
        # operand so `-(a + b)` doesn't render as `-a + b`.
        return f"{node.op}{_paren(node.operand, '__unary__', on_right=False)}"
    if isinstance(node, lir.RustName):
        return node.name
    if isinstance(node, lir.RustIntLiteral):
        # Drop the suffix when None — Rust's type inference picks it up from
        # context (let bindings, fn signatures, surrounding binops with a
        # typed side). Keeps emitted source readable.
        return f"{node.value}{node.suffix or ''}"
    if isinstance(node, lir.RustFloatLiteral):
        return _format_float(node.value) + (node.suffix or "")
    if isinstance(node, lir.RustBoolLiteral):
        return "true" if node.value else "false"
    if isinstance(node, lir.RustStringLiteral):
        # StrT lowers to `String` (owned) for parameters and returns, so
        # literals must materialize as owned strings too. `String::from(...)`
        # is unambiguous; for format! arguments it's slightly verbose but
        # still correct since `format!` accepts Display on either form.
        escaped = node.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'String::from("{escaped}")'
    if isinstance(node, lir.RustFormat):
        template = "{}" * len(node.args)
        rendered = ", ".join(_emit_expr(a) for a in node.args)
        return f'format!("{template}", {rendered})'
    if isinstance(node, lir.RustMacro):
        rendered = ", ".join(_emit_expr(a) for a in node.args)
        if node.template:
            return (
                f'{node.name}!("{node.template}", {rendered})'
                if rendered
                else f'{node.name}!("{node.template}")'
            )
        return f"{node.name}!({rendered})"
    from transpilers.passes.mir_to_rust_lir import (
        _RustIfExpr,
        _RustRef,
        _RustPyFloat,
        _RustListConcat,
        _RustOverflowGuard,
    )

    if isinstance(node, _RustIfExpr):
        return f"if {_emit_expr(node.test)} {{ {_emit_expr(node.then_)} }} else {{ {_emit_expr(node.else_)} }}"
    if isinstance(node, _RustRef):
        return f"&mut {_emit_expr(node.value)}"
    if isinstance(node, _RustPyFloat):
        return _emit_expr(node.value)
    if isinstance(node, _RustOverflowGuard):
        # `(left).wrapping_add(right)` — parenthesise the left operand so
        # compound expressions like `(a + b).wrapping_mul(c)` parse correctly.
        _stem = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "rem"}[node.op]
        method = f"{node.style}_{_stem}"
        return f"({_emit_expr(node.left)}).{method}({_emit_expr(node.right)})"
    if isinstance(node, _RustListConcat):
        # `left + right` for Vec: clone left, extend with right's elements.
        left = _emit_expr(node.left)
        right = _emit_expr(node.right)
        return f"{{ let mut _t = {left}.clone(); _t.extend({right}); _t }}"
    if isinstance(node, lir.RustMethodChain):
        rendered = ", ".join(_emit_expr(a) for a in node.args)
        return f"{_emit_expr(node.receiver)}.{node.method}({rendered})"
    if isinstance(node, lir.RustVec):
        items = ", ".join(_emit_expr(e) for e in node.elements)
        return f"vec![{items}]"
    if isinstance(node, lir.RustFieldAccess):
        return f"{_emit_expr(node.value)}.{node.field}"
    if isinstance(node, lir.RustStructInit):
        body = ", ".join(f"{n}: {_emit_expr(v)}" for n, v in node.field_values)
        return f"{node.name} {{ {body} }}"
    if isinstance(node, lir.RustIndex):
        return f"{_emit_expr(node.value)}[({_emit_expr(node.index)}) as usize]"
    if isinstance(node, lir.RustMethodCall):
        args = ", ".join(_emit_expr(a) for a in node.args)
        call = f"{_emit_expr(node.receiver)}.{node.method}({args})"
        return f"{call} as {node.cast_to}" if node.cast_to else call
    if isinstance(node, lir.RustCall):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{node.func}({args})"
    raise NotImplementedError(f"LIR node {type(node).__name__}")

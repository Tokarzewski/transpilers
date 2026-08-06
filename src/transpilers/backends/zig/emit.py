"""Zig LIR -> Zig source.

Deterministic emission. The `_ZigBlock` marker from the lowering pass
unwraps inline so stepped-range desugaring (`var i = a; while (i < b) ...`)
appears as a flat sequence of statements at the call site.
"""

from __future__ import annotations

from transpilers.ir import lir
from transpilers.passes.mir_to_zig_lir import _ZigBlock, _ZigMutableArrayDecl


INDENT = "    "


def _zig_string_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


_PY_FLOAT_HELPER = """\
fn _py_float(x: f64) []const u8 {
    const S = struct { var buf: [64]u8 = undefined; };
    const s = std.fmt.bufPrint(&S.buf, "{d}", .{x}) catch return "?";
    for (s) |c| if (c == '.' or c == 'e') return s;
    return std.fmt.bufPrint(&S.buf, "{d}.0", .{x}) catch s;
}
"""


def emit_zig(module: lir.ZigModule) -> str:
    body = "\n\n".join(_emit_item(item) for item in module.items) + "\n"
    preamble = ""
    if "std." in body:
        preamble += 'const std = @import("std");\n\n'
    if "_py_float(" in body:
        preamble += _PY_FLOAT_HELPER + "\n"
    return preamble + body


def _emit_item(item: lir.LirNode) -> str:
    if isinstance(item, lir.ZigStruct):
        return _emit_struct(item)
    if isinstance(item, lir.ZigFn):
        return _emit_fn(item)
    raise NotImplementedError(f"zig top-level item {type(item).__name__}")


def _emit_struct(s: lir.ZigStruct) -> str:
    lines = [f"const {s.name} = struct {{"]
    for n, t in s.fields:
        lines.append(f"{INDENT}{n}: {t},")
    for m in s.methods:
        lines.append("")
        lines.append(_emit_fn(m, depth=1, struct_name=s.name))
    lines.append("};")
    return "\n".join(lines)


def _emit_fn(fn: lir.ZigFn, *, depth: int = 0, struct_name: str | None = None) -> str:
    indent = INDENT * depth
    params = ", ".join(_emit_param(n, t, struct_name) for n, t in fn.params)
    # Zig's runtime calls `pub fn main` from `std/start.zig`; without
    # `pub` the entrypoint isn't visible and the program won't link.
    visibility = "pub " if fn.name == "main" and depth == 0 else ""
    header = f"{indent}{visibility}fn {fn.name}({params}) {fn.return_type} {{"
    body = _emit_block(fn.body, depth + 1)
    return f"{header}\n{body}\n{indent}}}"


def _emit_param(name: str, ty: str, struct_name: str | None) -> str:
    # Zig method receivers are conventionally written as `self: StructName`
    # rather than special syntax — but the type comes from the enclosing
    # struct, not the LIR's param annotation (which carries `self: Point`).
    if name == "self" and struct_name is not None:
        return f"self: {struct_name}"
    return f"{name}: {ty}"


def _augmented_form(name: str, value: lir.LirNode) -> tuple[str, lir.LirNode] | None:
    if not isinstance(value, lir.ZigBinOp):
        return None
    if not (isinstance(value.left, lir.ZigName) and value.left.name == name):
        return None
    if value.op not in ("+", "-", "*", "/", "%"):
        return None
    return value.op, value.right


def _emit_block(nodes: list[lir.LirNode], depth: int) -> str:
    lines: list[str] = []
    for n in nodes:
        if isinstance(n, _ZigBlock):
            for inner in n.items:
                lines.append(_emit_stmt(inner, depth))
        elif isinstance(n, _ZigMutableArrayDecl):
            # Expands to two statements at the call site.
            for stmt_str in _emit_mutable_array_decl(n, depth):
                lines.append(stmt_str)
        else:
            lines.append(_emit_stmt(n, depth))
    return "\n".join(lines)


def _flatten_snippet(snippet: str) -> str:
    """Collapse a multi-line source snippet to a single comment-safe line."""
    return " ".join(snippet.split())


def _emit_stmt(node: lir.LirNode, depth: int) -> str:
    pad = INDENT * depth
    if isinstance(node, lir.ZigRaw):
        # Zig has only line comments; emit the stub as its own comment line.
        return f"{pad}// TODO[port]: {_flatten_snippet(node.snippet)}"
    if isinstance(node, lir.ZigBreak):
        return f"{pad}break;"
    if isinstance(node, lir.ZigContinue):
        return f"{pad}continue;"
    if isinstance(node, lir.ZigReturn):
        return (
            f"{pad}return {_emit_expr(node.value)};" if node.value else f"{pad}return;"
        )
    if isinstance(node, lir.ZigVar):
        keyword = "var" if node.mutable else "const"
        ann = f": {node.ty}" if node.ty else ""
        return f"{pad}{keyword} {node.name}{ann} = {_emit_expr(node.value)};"
    if isinstance(node, lir.ZigReassign):
        aug = _augmented_form(node.name, node.value)
        if aug is not None:
            op, rhs = aug
            return f"{pad}{node.name} {op}= {_emit_expr(rhs)};"
        return f"{pad}{node.name} = {_emit_expr(node.value)};"
    if isinstance(node, lir.ZigFieldAssign):
        return f"{pad}{_emit_expr(node.obj)}.{node.field} = {_emit_expr(node.value)};"
    if isinstance(node, lir.ZigSubscriptAssign):
        # Zig indexing requires `usize`; @intCast bridges from our signed lattice.
        return f"{pad}{_emit_expr(node.obj)}[@intCast({_emit_expr(node.index)})] = {_emit_expr(node.value)};"
    if isinstance(node, lir.ZigIf):
        head = f"{pad}if ({_emit_expr(node.test)}) {{"
        body = _emit_block(node.body, depth + 1)
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], lir.ZigIf):
                inner = _emit_stmt(node.orelse[0], depth).lstrip()
                return f"{head}\n{body}\n{pad}}} else {inner}"
            else_body = _emit_block(node.orelse, depth + 1)
            return f"{head}\n{body}\n{pad}}} else {{\n{else_body}\n{pad}}}"
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.ZigWhile):
        head = f"{pad}while ({_emit_expr(node.test)}) {{"
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.ZigForRange):
        head = f"{pad}for ({_emit_expr(node.start)}..{_emit_expr(node.stop)}) |{node.target}| {{"
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, _ZigBlock):
        # Block at statement position — emit each item at this depth.
        return "\n".join(_emit_stmt(inner, depth) for inner in node.items)
    if isinstance(node, _ZigMutableArrayDecl):
        return "\n".join(_emit_mutable_array_decl(node, depth))
    return f"{pad}{_emit_expr(node)};"


def _emit_mutable_array_decl(node: _ZigMutableArrayDecl, depth: int) -> list[str]:
    """Emit the two-statement form for a list literal that needs to be a slice.

    var _xs_arr = [_]i64{1, 2, 3};
    const xs: []i64 = _xs_arr[0..];

    Using `const` for the slice binding avoids the "variable is never mutated"
    error when xs itself isn't reassigned. `[]i64` (not `[]const i64`) keeps
    the *elements* mutable so subscript writes work.
    """
    pad = INDENT * depth
    items = ", ".join(_emit_expr(e) for e in node.array_lit.elements)
    arr_line = f"{pad}var {node.arr_name} = [_]{node.array_lit.elem_ty}{{{items}}};"
    slice_line = (
        f"{pad}const {node.name}: []{node.array_lit.elem_ty} = {node.arr_name}[0..];"
    )
    return [arr_line, slice_line]


def _op_of(node: lir.LirNode) -> str | None:
    if isinstance(node, (lir.ZigBinOp, lir.ZigCompare, lir.ZigBoolOp)):
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
    if isinstance(node, lir.ZigRaw):
        # No inline comment (Zig line comments would swallow the rest of the
        # line). Preserve the snippet in a never-defined marker call.
        text = _flatten_snippet(node.snippet).replace("\\", "\\\\").replace('"', '\\"')
        return f'__todo_port__("{text}")'
    if isinstance(node, lir.ZigBinOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.ZigCompare):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.ZigBoolOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.ZigUnary):
        return f"{node.op}{_paren(node.operand, '__unary__', on_right=False)}"
    if isinstance(node, lir.ZigName):
        return node.name
    if isinstance(node, lir.ZigIntLiteral):
        return str(node.value)
    if isinstance(node, lir.ZigFloatLiteral):
        text = repr(node.value)
        return text if "." in text or "e" in text else text + ".0"
    if isinstance(node, lir.ZigBoolLiteral):
        return "true" if node.value else "false"
    if isinstance(node, lir.ZigStringLiteral):
        escaped = _zig_string_escape(node.value)
        return f'"{escaped}"'
    if isinstance(node, lir.ZigArrayLit):
        items = ", ".join(_emit_expr(e) for e in node.elements)
        prefix = "&" if node.ref else ""
        return f"{prefix}[_]{node.elem_ty}{{{items}}}"
    if isinstance(node, lir.ZigFieldAccess):
        return f"{_emit_expr(node.value)}.{node.field}"
    if isinstance(node, lir.ZigStructInit):
        body = ", ".join(f".{n} = {_emit_expr(v)}" for n, v in node.field_values)
        return f"{node.name}{{ {body} }}"
    # _ZigMethodCall lives in the lowering pass — emit receiver.method(args).
    from transpilers.passes.mir_to_zig_lir import _ZigMethodCall as _MC

    if isinstance(node, _MC):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{_emit_expr(node.receiver)}.{node.method}({args})"
    if isinstance(node, lir.ZigIndex):
        return f"{_emit_expr(node.value)}[@intCast({_emit_expr(node.index)})]"
    if isinstance(node, lir.ZigMethodCall):
        # Special-case .len which is a property in Zig.
        if node.method == "len" and not node.args:
            base = f"{_emit_expr(node.receiver)}.len"
            return f"@as({node.cast_to}, @intCast({base}))" if node.cast_to else base
        args = ", ".join(_emit_expr(a) for a in node.args)
        call = f"{_emit_expr(node.receiver)}.{node.method}({args})"
        return f"@as({node.cast_to}, @intCast({call}))" if node.cast_to else call
    if isinstance(node, lir.ZigCall):
        # Special-case the (string, tuple) shape for std.debug.print so we
        # emit `.{a, b, c}` for the tuple arg instead of a normal call.
        from transpilers.passes.mir_to_zig_lir import _ZigTuple

        rendered_args: list[str] = []
        for a in node.args:
            if isinstance(a, _ZigTuple):
                inner = ", ".join(_emit_expr(e) for e in a.elements)
                rendered_args.append(f".{{ {inner} }}")
            else:
                rendered_args.append(_emit_expr(a))
        return f"{node.func}({', '.join(rendered_args)})"
    from transpilers.passes.mir_to_zig_lir import _ZigIfExpr as _IE, _ZigPyFloat as _PF

    if isinstance(node, _IE):
        return f"if ({_emit_expr(node.test)}) {_emit_expr(node.then_)} else {_emit_expr(node.else_)}"
    if isinstance(node, _PF):
        return f"_py_float({_emit_expr(node.value)})"
    raise NotImplementedError(f"LIR node {type(node).__name__}")

"""Go LIR -> Go source.

Emits a `package main` preamble so the file is a valid translation unit. Go
gofmt-style formatting (tabs, brace placement) is approximated; output is
deterministic and parses cleanly with `go build`.
"""

from __future__ import annotations

from transpilers.ir import lir


INDENT = "\t"
PREAMBLE = "package main\n\n"


def _imports_for(source: str) -> str:
    """Scan the emitted source for `pkg.Func` calls and synthesize a
    minimal `import (...)` block. Keeps the file legal Go regardless of
    which stdlib helpers the lowering decided to emit."""
    pkgs: list[str] = []
    if "fmt." in source:
        pkgs.append("fmt")
    if "math." in source:
        pkgs.append("math")
    if "strconv." in source:
        pkgs.append("strconv")
    if "strings." in source:
        pkgs.append("strings")
    if not pkgs:
        return ""
    if len(pkgs) == 1:
        return f'import "{pkgs[0]}"\n\n'
    body = "\n".join(f'\t"{p}"' for p in pkgs)
    return f"import (\n{body}\n)\n\n"


def _is_go_method_call(node: lir.LirNode) -> bool:
    from transpilers.passes.mir_to_go_lir import _GoMethodCall as _MC

    return isinstance(node, _MC)


def _augmented_form(name: str, value: lir.LirNode) -> tuple[str, lir.LirNode] | None:
    if not isinstance(value, lir.GoBinOp):
        return None
    if not (isinstance(value.left, lir.GoName) and value.left.name == name):
        return None
    if value.op not in ("+", "-", "*", "/", "%"):
        return None
    return value.op, value.right


def emit_go(module: lir.GoModule) -> str:
    body = "\n\n".join(_emit_item(item) for item in module.items) + "\n"
    return PREAMBLE + _imports_for(body) + body


def _emit_item(item: lir.LirNode) -> str:
    if isinstance(item, lir.GoStruct):
        return _emit_struct(item)
    if isinstance(item, lir.GoFn):
        return _emit_fn(item)
    raise NotImplementedError(f"go top-level item {type(item).__name__}")


def _emit_struct(s: lir.GoStruct) -> str:
    field_lines = "\n".join(f"{INDENT}{n} {t}" for n, t in s.fields)
    type_def = f"type {s.name} struct {{\n{field_lines}\n}}"
    methods = "\n\n".join(_emit_fn(m, receiver_struct=s.name) for m in s.methods)
    return f"{type_def}\n\n{methods}" if methods else type_def


def _emit_fn(fn: lir.GoFn, *, receiver_struct: str | None = None) -> str:
    if receiver_struct is not None and fn.params and fn.params[0][0] == "self":
        receiver = f"(self *{receiver_struct}) "
        rest = fn.params[1:]
    else:
        receiver = ""
        rest = fn.params
    params = ", ".join(f"{n} {t}" for n, t in rest)
    ret = f" {fn.return_type}" if fn.return_type else ""
    header = f"func {receiver}{fn.name}({params}){ret} {{"
    body = _emit_block(fn.body, 1)
    return f"{header}\n{body}\n}}"


def _emit_block(nodes: list[lir.LirNode], depth: int) -> str:
    return "\n".join(_emit_stmt(n, depth) for n in nodes)


def _flatten_snippet(snippet: str) -> str:
    """Collapse a multi-line source snippet to a single comment-safe line."""
    return " ".join(snippet.split()).replace("*/", "* /")


def _emit_stmt(node: lir.LirNode, depth: int) -> str:
    pad = INDENT * depth
    if isinstance(node, lir.GoRaw):
        return f"{pad}_ = 0 /* TODO[port]: {_flatten_snippet(node.snippet)} */"
    if isinstance(node, lir.GoBreak):
        return f"{pad}break"
    if isinstance(node, lir.GoContinue):
        return f"{pad}continue"
    if isinstance(node, lir.GoReturn):
        return f"{pad}return {_emit_expr(node.value)}" if node.value else f"{pad}return"
    if isinstance(node, lir.GoDecl):
        return f"{pad}var {node.name} {node.ty} = {_emit_expr(node.value)}"
        # Note: Go statements have no trailing semicolons; emit doesn't add them.
    if isinstance(node, lir.GoReassign):
        return f"{pad}{node.name} = {_emit_expr(node.value)}"
    if isinstance(node, lir.GoFieldAssign):
        return f"{pad}{_emit_expr(node.obj)}.{node.field} = {_emit_expr(node.value)}"
    if isinstance(node, lir.GoSubscriptAssign):
        return f"{pad}{_emit_expr(node.obj)}[{_emit_expr(node.index)}] = {_emit_expr(node.value)}"
    if isinstance(node, lir.GoIf):
        head = f"{pad}if {_emit_expr(node.test)} {{"
        body = _emit_block(node.body, depth + 1)
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], lir.GoIf):
                inner = _emit_stmt(node.orelse[0], depth).lstrip()
                return f"{head}\n{body}\n{pad}}} else {inner}"
            else_body = _emit_block(node.orelse, depth + 1)
            return f"{head}\n{body}\n{pad}}} else {{\n{else_body}\n{pad}}}"
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.GoWhile):
        head = f"{pad}for {_emit_expr(node.test)} {{"
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.GoForRange):
        if node.step is None:
            step = f"{node.target}++"
        else:
            step = f"{node.target} += {_emit_expr(node.step)}"
        # Force int64 on the loop variable to avoid mismatched int / int64
        # errors when stop is int64. Go is strict about implicit conversions.
        head = (
            f"{pad}for {node.target} := int64({_emit_expr(node.start)}); "
            f"{node.target} < {_emit_expr(node.stop)}; {step} {{"
        )
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    return f"{pad}{_emit_expr(node)}"


def _op_of(node: lir.LirNode) -> str | None:
    if isinstance(node, (lir.GoBinOp, lir.GoCompare, lir.GoBoolOp)):
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
    if isinstance(node, lir.GoRaw):
        return f"0 /* TODO[port]: {_flatten_snippet(node.snippet)} */"
    if isinstance(node, lir.GoBinOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.GoCompare):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.GoBoolOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.GoUnary):
        return f"{node.op}{_paren(node.operand, '__unary__', on_right=False)}"
    if isinstance(node, lir.GoName):
        return node.name
    if isinstance(node, lir.GoIntLiteral):
        return str(node.value)
    if isinstance(node, lir.GoFloatLiteral):
        text = repr(node.value)
        return text if "." in text or "e" in text else text + ".0"
    if isinstance(node, lir.GoBoolLiteral):
        return "true" if node.value else "false"
    if isinstance(node, lir.GoStringLiteral):
        escaped = node.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(node, lir.GoCall):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{node.func}({args})"
    if isinstance(node, lir.GoFieldAccess):
        return f"{_emit_expr(node.value)}.{node.field}"
    if isinstance(node, lir.GoStructInit):
        body = ", ".join(f"{n}: {_emit_expr(v)}" for n, v in node.field_values)
        return f"{node.name}{{{body}}}"
    from transpilers.passes.mir_to_go_lir import (
        _GoMethodCall as _MC,
        _GoIfExpr,
        _GoIndex,
        _GoSliceLit,
        _GoSliceAppend,
        _GoBoolStr,
        _GoFloatStr,
    )

    if isinstance(node, _MC):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{_emit_expr(node.receiver)}.{node.method}({args})"
    if isinstance(node, _GoIfExpr):
        return (
            f"func() int64 {{ if {_emit_expr(node.test)} {{ return {_emit_expr(node.then_)} }}; "
            f"return {_emit_expr(node.else_)} }}()"
        )
    if isinstance(node, _GoIndex):
        return f"{_emit_expr(node.value)}[{_emit_expr(node.index)}]"
    if isinstance(node, _GoSliceLit):
        elements = ", ".join(_emit_expr(e) for e in node.elements)
        return f"[]{node.elem_ty}{{{elements}}}"
    if isinstance(node, _GoSliceAppend):
        return f"append({_emit_expr(node.left)}, {_emit_expr(node.right)}...)"
    if isinstance(node, _GoBoolStr):
        return (
            f'map[bool]string{{true: "True", false: "False"}}[{_emit_expr(node.value)}]'
        )
    if isinstance(node, _GoFloatStr):
        v = _emit_expr(node.value)
        # Match Python's float str(): always show a decimal point.
        # strconv.FormatFloat 'g' removes trailing zeros, so "12" not "12.0".
        # The IIFE appends ".0" when neither "." nor "e" is present.
        return (
            f"func() string {{ s := strconv.FormatFloat({v}, 'g', -1, 64); "
            f"if strings.IndexByte(s, '.') < 0 && strings.IndexByte(s, 'e') < 0 "
            f'{{ return s + ".0" }}; return s }}()'
        )
    raise NotImplementedError(f"LIR node {type(node).__name__}")

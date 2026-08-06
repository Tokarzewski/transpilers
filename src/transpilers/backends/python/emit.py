"""Python LIR -> Python source.

Indentation-sensitive, no braces. Empty function bodies need `pass`.
"""

from __future__ import annotations

from transpilers.ir import lir


INDENT = "    "


def _augmented_form(name: str, value: lir.LirNode) -> tuple[str, lir.LirNode] | None:
    if not isinstance(value, lir.PyBinOp):
        return None
    if not (isinstance(value.left, lir.PyName) and value.left.name == name):
        return None
    if value.op not in ("+", "-", "*", "/", "%"):
        return None
    return value.op, value.right


def emit_python(module: lir.PyModule) -> str:
    has_class = any(isinstance(item, lir.PyClass) for item in module.items)
    preamble = "from dataclasses import dataclass\n\n\n" if has_class else ""
    return preamble + "\n\n".join(_emit_item(item) for item in module.items) + "\n"


def _emit_item(item: lir.LirNode) -> str:
    if isinstance(item, lir.PyClass):
        return _emit_class(item)
    if isinstance(item, lir.PyFn):
        return _emit_fn(item)
    raise NotImplementedError(f"python top-level item {type(item).__name__}")


def _emit_class(c: lir.PyClass) -> str:
    """`@dataclass` gives the class a positional __init__ matching the field
    order, so `Point(0, 0)` works at runtime without a hand-written ctor."""
    lines = ["@dataclass", f"class {c.name}:"]
    if not c.fields and not c.methods:
        lines.append(INDENT + "pass")
        return "\n".join(lines)
    for name, ty in c.fields:
        # @dataclass requires annotations on fields; default to `int` when
        # we lack a type rather than emitting a bare name.
        ann = f": {ty}" if ty else ": int"
        lines.append(f"{INDENT}{name}{ann}")
    for m in c.methods:
        lines.append("")
        lines.append(_emit_fn(m, depth=1))
    return "\n".join(lines)


def _emit_fn(fn: lir.PyFn, *, depth: int = 0) -> str:
    indent = INDENT * depth
    params = ", ".join(_emit_param(n, t) for n, t in fn.params)
    ret = f" -> {fn.return_type}" if fn.return_type and fn.return_type != "None" else ""
    header = f"{indent}def {fn.name}({params}){ret}:"
    body = _emit_block(fn.body, depth + 1) or (indent + INDENT + "pass")
    return f"{header}\n{body}"


def _emit_param(name: str, ty: str) -> str:
    if name == "self":
        return "self"
    return f"{name}: {ty}" if ty else name


def _emit_block(nodes: list[lir.LirNode], depth: int) -> str:
    return "\n".join(_emit_stmt(n, depth) for n in nodes)


def _flatten_snippet(snippet: str) -> str:
    """Collapse a multi-line source snippet to a single comment-safe line."""
    return " ".join(snippet.split())


def _emit_stmt(node: lir.LirNode, depth: int) -> str:
    pad = INDENT * depth
    if isinstance(node, lir.PyRaw):
        return f"{pad}pass  # TODO[port]: {_flatten_snippet(node.snippet)}"
    if isinstance(node, lir.PyBreak):
        return f"{pad}break"
    if isinstance(node, lir.PyContinue):
        return f"{pad}continue"
    if isinstance(node, lir.PyReturn):
        return f"{pad}return {_emit_expr(node.value)}" if node.value else f"{pad}return"
    if isinstance(node, lir.PyAssign):
        ann = f": {node.ty}" if node.ty else ""
        if not node.ty:
            aug = _augmented_form(node.name, node.value)
            if aug is not None:
                op, rhs = aug
                return f"{pad}{node.name} {op}= {_emit_expr(rhs)}"
        return f"{pad}{node.name}{ann} = {_emit_expr(node.value)}"
    if isinstance(node, lir.PyFieldAssign):
        return f"{pad}{_emit_expr(node.obj)}.{node.field} = {_emit_expr(node.value)}"
    if isinstance(node, lir.PySubscriptAssign):
        return f"{pad}{_emit_expr(node.obj)}[{_emit_expr(node.index)}] = {_emit_expr(node.value)}"
    if isinstance(node, lir.PyIf):
        head = f"{pad}if {_emit_expr(node.test)}:"
        body = _emit_block(node.body, depth + 1) or (pad + INDENT + "pass")
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], lir.PyIf):
                inner = _emit_stmt(node.orelse[0], depth)
                elif_inner = inner.replace(f"{pad}if ", f"{pad}elif ", 1)
                return f"{head}\n{body}\n{elif_inner}"
            else_body = _emit_block(node.orelse, depth + 1) or (pad + INDENT + "pass")
            return f"{head}\n{body}\n{pad}else:\n{else_body}"
        return f"{head}\n{body}"
    if isinstance(node, lir.PyWhile):
        head = f"{pad}while {_emit_expr(node.test)}:"
        body = _emit_block(node.body, depth + 1) or (pad + INDENT + "pass")
        return f"{head}\n{body}"
    if isinstance(node, lir.PyForRange):
        if node.step is None:
            args = f"{_emit_expr(node.start)}, {_emit_expr(node.stop)}"
        else:
            args = f"{_emit_expr(node.start)}, {_emit_expr(node.stop)}, {_emit_expr(node.step)}"
        head = f"{pad}for {node.target} in range({args}):"
        body = _emit_block(node.body, depth + 1) or (pad + INDENT + "pass")
        return f"{head}\n{body}"
    return f"{pad}{_emit_expr(node)}"


def _op_of(node: lir.LirNode) -> str | None:
    if isinstance(node, (lir.PyBinOp, lir.PyCompare, lir.PyBoolOp)):
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
    if isinstance(node, lir.PyRaw):
        # Expr position: no trailing `#` comment (would swallow the rest of an
        # enclosing line). Encode the snippet in a never-called marker call.
        text = _flatten_snippet(node.snippet).replace("\\", "\\\\").replace('"', '\\"')
        return f'__todo_port__("{text}")'
    if isinstance(node, lir.PyBinOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.PyCompare):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.PyBoolOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.PyUnary):
        operand = _paren(node.operand, "__unary__", on_right=False)
        return f"{node.op} {operand}" if node.op == "not" else f"{node.op}{operand}"
    if isinstance(node, lir.PyName):
        return node.name
    if isinstance(node, lir.PyIntLiteral):
        return str(node.value)
    if isinstance(node, lir.PyFloatLiteral):
        text = repr(node.value)
        return text if "." in text or "e" in text else text + ".0"
    if isinstance(node, lir.PyBoolLiteral):
        return "True" if node.value else "False"
    if isinstance(node, lir.PyStringLiteral):
        escaped = node.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(node, lir.PyCall):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{node.func}({args})"
    if isinstance(node, lir.PyFieldAccess):
        return f"{_emit_expr(node.value)}.{node.field}"
    if isinstance(node, lir.PyStructInit):
        args = ", ".join(_emit_expr(v) for _, v in node.field_values)
        return f"{node.name}({args})"
    from transpilers.passes.mir_to_python_lir import (
        _PyMethodCall as _MC,
        _PyIndex,
        _PyList,
        _PyIfExpr,
    )

    if isinstance(node, _MC):
        args = ", ".join(_emit_expr(a) for a in node.args)
        return f"{_emit_expr(node.receiver)}.{node.method}({args})"
    if isinstance(node, _PyIndex):
        return f"{_emit_expr(node.value)}[{_emit_expr(node.index)}]"
    if isinstance(node, _PyList):
        elems = ", ".join(_emit_expr(e) for e in node.elements)
        return f"[{elems}]"
    if isinstance(node, _PyIfExpr):
        return f"({_emit_expr(node.then_)} if {_emit_expr(node.test)} else {_emit_expr(node.else_)})"
    raise NotImplementedError(f"LIR node {type(node).__name__}")

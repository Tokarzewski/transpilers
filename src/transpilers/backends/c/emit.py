"""C LIR -> C source.

Emits an `#include <stdint.h>` / `<stdbool.h>` preamble so int64_t / bool
resolve cleanly. Deterministic — no LLM at this layer.
"""

from __future__ import annotations

from transpilers.ir import lir


INDENT = "    "
PREAMBLE = (
    "#include <stdint.h>\n"
    "#include <stdbool.h>\n"
    "#include <stdio.h>\n"
    "#include <stdlib.h>\n"
    "#include <stddef.h>\n"
    "#include <string.h>\n"
    "\n"
    "/* Growable list types — Python `list[T]` lowers to one of these.\n"
    " * `data` is heap-owned once grown; a `{0}` (empty) list grows via\n"
    " * realloc on push, so `xs = xs + [e]` works without a fixed bound. */\n"
    "typedef struct { int64_t *data; size_t len; size_t cap; } slice_i64_t;\n"
    "typedef struct { double *data; size_t len; size_t cap; } slice_f64_t;\n"
    "typedef struct { bool *data; size_t len; size_t cap; } slice_bool_t;\n"
    "static void slice_i64_push(slice_i64_t *v, int64_t x) {\n"
    "    if (v->len == v->cap) { v->cap = v->cap ? v->cap * 2 : 4;\n"
    "        v->data = (int64_t*)realloc(v->data, v->cap * sizeof(int64_t)); }\n"
    "    v->data[v->len++] = x;\n"
    "}\n"
    "static void slice_f64_push(slice_f64_t *v, double x) {\n"
    "    if (v->len == v->cap) { v->cap = v->cap ? v->cap * 2 : 4;\n"
    "        v->data = (double*)realloc(v->data, v->cap * sizeof(double)); }\n"
    "    v->data[v->len++] = x;\n"
    "}\n"
    "static void slice_bool_push(slice_bool_t *v, bool x) {\n"
    "    if (v->len == v->cap) { v->cap = v->cap ? v->cap * 2 : 4;\n"
    "        v->data = (bool*)realloc(v->data, v->cap * sizeof(bool)); }\n"
    "    v->data[v->len++] = x;\n"
    "}\n"
    "\n"
    "/* Python-compatible float printer: shortest round-trip representation.\n"
    " * Writes into a caller-supplied buffer (at least 32 bytes). Returns buf. */\n"
    "static const char* _py_float_buf(double v, char *buf) {\n"
    "    int prec;\n"
    "    for (prec = 1; prec <= 17; prec++) {\n"
    '        snprintf(buf, 32, "%.*g", prec, v);\n'
    '        double back; sscanf(buf, "%lf", &back);\n'
    "        if (back == v) break;\n"
    "    }\n"
    "    if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'E'))\n"
    '        strcat(buf, ".0");\n'
    "    return buf;\n"
    "}\n"
    "\n"
)


def _c_string_escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def _augmented_form(name: str, value: lir.LirNode) -> tuple[str, lir.LirNode] | None:
    if not isinstance(value, lir.CBinOp):
        return None
    if not (isinstance(value.left, lir.CName) and value.left.name == name):
        return None
    if value.op not in ("+", "-", "*", "/", "%"):
        return None
    return value.op, value.right


def emit_c(module: lir.CModule) -> str:
    return PREAMBLE + "\n\n".join(_emit_item(item) for item in module.items) + "\n"


def _emit_item(item: lir.LirNode) -> str:
    if isinstance(item, lir.CStruct):
        return _emit_struct(item)
    if isinstance(item, lir.CFn):
        return _emit_fn(item)
    raise NotImplementedError(f"c top-level item {type(item).__name__}")


def _emit_struct(s: lir.CStruct) -> str:
    field_lines = "\n".join(f"{INDENT}{t} {n};" for n, t in s.fields)
    type_def = f"typedef struct {{\n{field_lines}\n}} {s.name};"
    method_defs = "\n\n".join(_emit_fn(m, self_type=s.name) for m in s.methods)
    return f"{type_def}\n\n{method_defs}" if method_defs else type_def


def _emit_fn(fn: lir.CFn, *, self_type: str | None = None) -> str:
    params = ", ".join(_emit_param(n, t, self_type) for n, t in fn.params) or "void"
    # C requires `main` to return `int` (and a hosted environment expects
    # `int main(...)`). Coerce void return → int and append `return 0;`.
    ret_type = fn.return_type
    body_text = _emit_block(fn.body, 1)
    if fn.name == "main" and ret_type == "void":
        ret_type = "int"
        body_text = body_text + f"\n{INDENT}return 0;"
    header = f"{ret_type} {fn.name}({params}) {{"
    return f"{header}\n{body_text}\n}}"


def _emit_param(name: str, ty: str, self_type: str | None) -> str:
    # Methods take `Struct *self`; the LIR carries `(self, "Struct")` so we
    # rewrite to pointer form here. Other params unchanged.
    if name == "self" and self_type is not None:
        return f"{self_type} *self"
    return f"{ty} {name}"


def _emit_block(nodes: list[lir.LirNode], depth: int) -> str:
    return "\n".join(_emit_stmt(n, depth) for n in nodes)


def _flatten_snippet(snippet: str) -> str:
    """Collapse a multi-line source snippet to a single comment-safe line."""
    return " ".join(snippet.split()).replace("*/", "* /")


def _emit_stmt(node: lir.LirNode, depth: int) -> str:
    pad = INDENT * depth
    if isinstance(node, lir.CRaw):
        return f"{pad}/* TODO[port]: {_flatten_snippet(node.snippet)} */;"
    if isinstance(node, lir.CBreak):
        return f"{pad}break;"
    if isinstance(node, lir.CContinue):
        return f"{pad}continue;"
    if isinstance(node, lir.CReturn):
        return (
            f"{pad}return {_emit_expr(node.value)};" if node.value else f"{pad}return;"
        )
    if isinstance(node, lir.CDecl):
        return f"{pad}{node.ty} {node.name} = {_emit_expr(node.value)};"
    if isinstance(node, lir.CReassign):
        # Python `xs = xs + [e0, e1, ...]` is list growth, not arithmetic.
        # Lower it to element-wise push() onto the realloc-backed slice — a
        # struct `+=` is invalid C and the old emission produced exactly that.
        from transpilers.passes.mir_to_c_lir import _CSliceLiteral

        val = node.value
        if (
            isinstance(val, lir.CBinOp)
            and val.op == "+"
            and isinstance(val.left, lir.CName)
            and val.left.name == node.name
            and isinstance(val.right, _CSliceLiteral)
        ):
            push_fn = val.right.slice_ty.removesuffix("_t") + "_push"
            if not val.right.elements:
                return f"{pad};"
            return "\n".join(
                f"{pad}{push_fn}(&{node.name}, {_emit_expr(e)});"
                for e in val.right.elements
            )
        aug = _augmented_form(node.name, node.value)
        if aug is not None:
            op, rhs = aug
            return f"{pad}{node.name} {op}= {_emit_expr(rhs)};"
        return f"{pad}{node.name} = {_emit_expr(node.value)};"
    if isinstance(node, lir.CFieldAssign):
        sep = "->" if node.via_pointer else "."
        return (
            f"{pad}{_emit_expr(node.obj)}{sep}{node.field} = {_emit_expr(node.value)};"
        )
    if isinstance(node, lir.CSubscriptAssign):
        # Slice element write — every CSubscriptAssign in this pipeline
        # targets a `slice_*_t` carrier, so the index lands on `.data[i]`.
        return f"{pad}{_emit_expr(node.obj)}.data[{_emit_expr(node.index)}] = {_emit_expr(node.value)};"
    if isinstance(node, lir.CIf):
        head = f"{pad}if ({_emit_expr(node.test)}) {{"
        body = _emit_block(node.body, depth + 1)
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], lir.CIf):
                inner = _emit_stmt(node.orelse[0], depth).lstrip()
                return f"{head}\n{body}\n{pad}}} else {inner}"
            else_body = _emit_block(node.orelse, depth + 1)
            return f"{head}\n{body}\n{pad}}} else {{\n{else_body}\n{pad}}}"
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.CWhile):
        head = f"{pad}while ({_emit_expr(node.test)}) {{"
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    if isinstance(node, lir.CForRange):
        # Native C for-loop. Step expressed as either `i++` (None or +1) or
        # explicit `i += <step>`.
        step_expr = (
            "i++" if node.step is None else f"{node.target} += {_emit_expr(node.step)}"
        )
        # Rebuild with the real target name when step is None:
        if node.step is None:
            step_expr = f"{node.target}++"
        head = (
            f"{pad}for (int64_t {node.target} = {_emit_expr(node.start)}; "
            f"{node.target} < {_emit_expr(node.stop)}; {step_expr}) {{"
        )
        body = _emit_block(node.body, depth + 1)
        return f"{head}\n{body}\n{pad}}}"
    return f"{pad}{_emit_expr(node)};"


def _op_of(node: lir.LirNode) -> str | None:
    if isinstance(node, (lir.CBinOp, lir.CCompare, lir.CBoolOp)):
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
    if isinstance(node, lir.CRaw):
        return f"/* TODO[port]: {_flatten_snippet(node.snippet)} */ 0"
    if isinstance(node, lir.CBinOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.CCompare):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.CBoolOp):
        return f"{_paren(node.left, node.op, on_right=False)} {node.op} {_paren(node.right, node.op, on_right=True)}"
    if isinstance(node, lir.CUnary):
        return f"{node.op}{_paren(node.operand, '__unary__', on_right=False)}"
    if isinstance(node, lir.CName):
        return node.name
    if isinstance(node, lir.CIntLiteral):
        return str(node.value)
    if isinstance(node, lir.CFloatLiteral):
        text = repr(node.value)
        return text if "." in text or "e" in text else text + ".0"
    if isinstance(node, lir.CBoolLiteral):
        return "true" if node.value else "false"
    if isinstance(node, lir.CStringLiteral):
        escaped = _c_string_escape(node.value)
        return f'"{escaped}"'
    if isinstance(node, lir.CIndex):
        # Same routing rule as CSubscriptAssign: every C index lands on a
        # slice value, so step through `.data` to reach the element.
        return f"{_emit_expr(node.value)}.data[{_emit_expr(node.index)}]"
    from transpilers.passes.mir_to_c_lir import _CSliceLiteral, _CPyFloat

    if isinstance(node, _CSliceLiteral):
        # Empty list → zeroed growable slice (data=NULL, len=cap=0) so a
        # later push() reallocs from scratch. Non-empty literals keep their
        # inline backing array (read-only / fixed use).
        if not node.elements:
            return f"({node.slice_ty}){{0}}"
        elems = ", ".join(_emit_expr(e) for e in node.elements)
        return f"(({node.slice_ty}){{({node.elem_ty}[]){{{elems}}}, {len(node.elements)}}})"
    if isinstance(node, _CPyFloat):
        # Call the preamble helper with a stack buffer.
        return f"_py_float_buf({_emit_expr(node.value)}, (char[32]){{}})"
    if isinstance(node, lir.CCall):
        args = ", ".join(_emit_expr(a) for a in node.args)
        # Cast-style "calls" (`(int64_t)`, `(double)`) emit as cast prefix.
        if node.func.startswith("(") and node.func.endswith(")"):
            return f"{node.func}{args}"
        return f"{node.func}({args})"
    if isinstance(node, lir.CTernary):
        return f"({_emit_expr(node.test)} ? {_emit_expr(node.then_)} : {_emit_expr(node.else_)})"
    if isinstance(node, lir.CFieldAccess):
        sep = "->" if node.via_pointer else "."
        return f"{_emit_expr(node.value)}{sep}{node.field}"
    if isinstance(node, lir.CStructInit):
        body = ", ".join(f".{n} = {_emit_expr(v)}" for n, v in node.field_values)
        return f"({node.name}){{{body}}}"
    from transpilers.passes.mir_to_c_lir import _AddressOf as _AO

    if isinstance(node, _AO):
        return f"&{_emit_expr(node.value)}"
    raise NotImplementedError(f"LIR node {type(node).__name__}")

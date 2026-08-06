"""TypeScript source -> HIR via tree-sitter-typescript.

Initial subset:
  - top-level `function name(args): T { ... }` declarations
  - primitive types: number / boolean / string / void (and `bigint` aliasing to int)
  - return, if/else, while, C-style for
  - `let` / `const` / `var` declarations with optional type annotations
  - binary / comparison / logical ops, integer / boolean / string literals,
    method-less call expressions

TypeScript `number` is semantically a 64-bit float, but most code uses it for
integers. To stay inside the existing IntT pipeline (and because explicit
floats are a small follow-up — no HirFloatLiteral yet), we map `number` to
`int`. This is a known approximation; division will be integer division
under that mapping.
"""

from __future__ import annotations

from tree_sitter import Language, Node
import tree_sitter_typescript

from transpilers.ir import hir
from transpilers.frontends._treesitter import (
    make_parser,
    named_children,
    required_field,
    text,
)


from transpilers.frontends.errors import UnsupportedConstruct


TS_TYPE_ALIASES: dict[str, str] = {
    "number": "int",  # see module docstring
    "bigint": "int",
    "boolean": "bool",
    "string": "str",
    "void": "None",
    "undefined": "None",
    "null": "None",
}


_LANGUAGE = Language(tree_sitter_typescript.language_typescript())


def parse_typescript(source: str) -> hir.HirModule:
    parser = make_parser(_LANGUAGE)
    tree = parser.parse(bytes(source, "utf-8"))
    body: list[hir.HirNode] = []
    for c in named_children(tree.root_node):
        if c.type == "function_declaration":
            body.append(_convert_function(c))
            continue
        if c.type in (
            "comment",
            "import_statement",
            "export_statement",
            "hash_bang_line",
        ):
            if c.type == "export_statement":
                for inner in named_children(c):
                    if inner.type == "function_declaration":
                        body.append(_convert_function(inner))
            continue
        # Top-level non-function statements — script style. Skip.
        if c.type in (
            "expression_statement",
            "lexical_declaration",
            "variable_declaration",
            "for_statement",
            "if_statement",
            "while_statement",
            "block_statement",
            "type_alias_declaration",
            "interface_declaration",
        ):
            continue
        raise UnsupportedConstruct(f"top-level {c.type}")
    return hir.HirModule(source_lang="typescript", body=body)


def _convert_function(node: Node) -> hir.HirFunction:
    name_node = required_field(node, "name")
    params_node = required_field(node, "parameters")
    body_node = required_field(node, "body")
    return_type_node = node.child_by_field_name("return_type")
    params = [
        _convert_param(p)
        for p in named_children(params_node)
        if p.type in ("required_parameter", "optional_parameter")
    ]
    return hir.HirFunction(
        name=text(name_node),
        params=params,
        return_annotation=_return_type_text(return_type_node),
        body=_convert_block(body_node),
    )


def _convert_param(node: Node) -> hir.HirParam:
    pattern = required_field(node, "pattern")
    if pattern.type != "identifier":
        raise UnsupportedConstruct(f"ts param pattern {pattern.type}")
    type_node = node.child_by_field_name("type")
    annotation = _type_annotation_text(type_node) if type_node is not None else None
    return hir.HirParam(name=text(pattern), annotation=annotation)


def _convert_block(node: Node) -> list[hir.HirNode]:
    if node.type != "statement_block":
        raise UnsupportedConstruct(f"ts block {node.type}")
    out: list[hir.HirNode] = []
    for c in named_children(node):
        out.extend(_convert_stmt(c))
    return out


def _convert_stmt(node: Node) -> list[hir.HirNode]:
    kind = node.type
    if kind == "return_statement":
        kids = named_children(node)
        return [hir.HirReturn(value=_convert_expr(kids[0]) if kids else None)]
    if kind == "if_statement":
        return [_convert_if(node)]
    if kind == "while_statement":
        return [_convert_while(node)]
    if kind == "for_statement":
        return _convert_for(node)
    if kind in ("lexical_declaration", "variable_declaration"):
        return _convert_variable_declaration(node)
    if kind == "expression_statement":
        kids = named_children(node)
        return [_convert_expression_stmt(kids[0])] if kids else []
    if kind == "statement_block":
        return _convert_block(node)
    if kind in ("comment", "empty_statement"):
        return []
    raise UnsupportedConstruct(f"ts stmt {kind}")


def _convert_if(node: Node) -> hir.HirNode:
    cond = _convert_expr(required_field(node, "condition"))
    body = _convert_stmt(required_field(node, "consequence"))
    orelse_node = node.child_by_field_name("alternative")
    orelse: list[hir.HirNode] = []
    if orelse_node is not None:
        if orelse_node.type == "else_clause":
            for c in named_children(orelse_node):
                orelse.extend(_convert_stmt(c))
        else:
            orelse = _convert_stmt(orelse_node)
    return hir.HirIf(test=cond, body=body, orelse=orelse)


def _convert_while(node: Node) -> hir.HirNode:
    cond = _convert_expr(required_field(node, "condition"))
    body = _convert_stmt(required_field(node, "body"))
    return hir.HirWhile(test=cond, body=body)


def _convert_for(node: Node) -> list[hir.HirNode]:
    out: list[hir.HirNode] = []
    init_node = node.child_by_field_name("initializer")
    cond_node = node.child_by_field_name("condition")
    update_node = node.child_by_field_name("increment")
    body_node = required_field(node, "body")
    if init_node is not None:
        out.extend(_convert_stmt(init_node))
    # The condition in JS/TS for-statement is wrapped in expression_statement.
    if cond_node is not None and cond_node.type == "expression_statement":
        cond_kids = named_children(cond_node)
        cond_expr = (
            _convert_expr(cond_kids[0]) if cond_kids else hir.HirBoolLiteral(value=True)
        )
    else:
        cond_expr = (
            _convert_expr(cond_node)
            if cond_node is not None
            else hir.HirBoolLiteral(value=True)
        )
    inner = _convert_stmt(body_node)
    if update_node is not None:
        inner.append(_convert_expression_stmt(update_node))
    out.append(hir.HirWhile(test=cond_expr, body=inner))
    return out


def _convert_variable_declaration(node: Node) -> list[hir.HirNode]:
    out: list[hir.HirNode] = []
    for c in named_children(node):
        if c.type == "variable_declarator":
            out.append(_convert_variable_declarator(c))
    return out


def _convert_variable_declarator(node: Node) -> hir.HirNode:
    name_node = required_field(node, "name")
    if name_node.type != "identifier":
        raise UnsupportedConstruct(f"ts declarator pattern {name_node.type}")
    type_node = node.child_by_field_name("type")
    annotation = _type_annotation_text(type_node) if type_node is not None else None
    value_node = node.child_by_field_name("value")
    value = (
        _convert_expr(value_node)
        if value_node is not None
        else hir.HirIntLiteral(value=0)
    )
    return hir.HirAssign(target=text(name_node), value=value, annotation=annotation)


def _convert_expression_stmt(node: Node) -> hir.HirNode:
    if node.type == "assignment_expression":
        return _convert_assignment(node)
    if node.type == "augmented_assignment_expression":
        return _convert_aug_assignment(node)
    if node.type == "update_expression":
        return _convert_update(node)
    if node.type == "call_expression":
        return _convert_expr(node)
    raise UnsupportedConstruct(f"ts expression statement {node.type}")


def _convert_assignment(node: Node) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    if left.type != "identifier":
        raise UnsupportedConstruct(f"ts assignment lhs {left.type}")
    return hir.HirAssign(target=text(left), value=_convert_expr(right), annotation=None)


def _convert_aug_assignment(node: Node) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    op_node = required_field(node, "operator")
    op = text(op_node)
    if left.type != "identifier" or not op.endswith("="):
        raise UnsupportedConstruct(f"ts aug-assign {op!r}")
    return hir.HirAssign(
        target=text(left),
        value=_convert_expr(right),
        annotation=None,
        augmented_op=op[:-1],
    )


def _convert_update(node: Node) -> hir.HirNode:
    argument = required_field(node, "argument")
    op_node = required_field(node, "operator")
    op = text(op_node)
    if argument.type != "identifier" or op not in ("++", "--"):
        raise UnsupportedConstruct(f"ts update {op!r}")
    sign = "+" if op == "++" else "-"
    return hir.HirAssign(
        target=text(argument),
        value=hir.HirIntLiteral(value=1),
        annotation=None,
        augmented_op=sign,
    )


# ---------- expressions ----------


def _convert_expr(node: Node) -> hir.HirNode:
    kind = node.type
    if kind == "number":
        raw = text(node)
        if "." in raw or "e" in raw or "E" in raw:
            return hir.HirFloatLiteral(value=float(raw))
        return hir.HirIntLiteral(value=int(raw, 0))
    if kind == "true":
        return hir.HirBoolLiteral(value=True)
    if kind == "false":
        return hir.HirBoolLiteral(value=False)
    if kind == "string":
        raw = text(node)
        return hir.HirStringLiteral(value=raw[1:-1])
    if kind == "identifier":
        return hir.HirName(name=text(node))
    if kind == "parenthesized_expression":
        kids = named_children(node)
        if len(kids) == 1:
            return _convert_expr(kids[0])
    if kind == "binary_expression":
        return _convert_binary(node)
    if kind == "unary_expression":
        return _convert_unary(node)
    if kind == "call_expression":
        return _convert_call(node)
    if kind == "update_expression":
        raise UnsupportedConstruct("ts update at expression position")
    raise UnsupportedConstruct(f"ts expr {kind}")


def _convert_binary(node: Node) -> hir.HirNode:
    left = _convert_expr(required_field(node, "left"))
    right = _convert_expr(required_field(node, "right"))
    op = text(required_field(node, "operator"))
    # JS/TS uses === and !== for strict equality; we collapse onto Python-style.
    if op in ("==", "==="):
        return hir.HirCompare(op="==", left=left, right=right)
    if op in ("!=", "!=="):
        return hir.HirCompare(op="!=", left=left, right=right)
    if op in COMPARE_OPS:
        return hir.HirCompare(op=op, left=left, right=right)
    if op in LOGICAL_OPS:
        return hir.HirBoolOp(op="and" if op == "&&" else "or", left=left, right=right)
    if op in ARITH_OPS:
        return hir.HirBinOp(op=op, left=left, right=right)
    raise UnsupportedConstruct(f"ts binary {op!r}")


def _convert_unary(node: Node) -> hir.HirNode:
    argument = required_field(node, "argument")
    op = text(required_field(node, "operator"))
    if op == "!":
        return hir.HirUnaryOp(op="not", operand=_convert_expr(argument))
    if op == "-":
        return hir.HirUnaryOp(op="-", operand=_convert_expr(argument))
    raise UnsupportedConstruct(f"ts unary {op!r}")


def _convert_call(node: Node) -> hir.HirNode:
    func_node = required_field(node, "function")
    args_node = required_field(node, "arguments")
    if func_node.type != "identifier":
        raise UnsupportedConstruct(f"ts call target {func_node.type}")
    args = [_convert_expr(c) for c in named_children(args_node) if c.type != "comment"]
    return hir.HirCall(func=text(func_node), args=args)


COMPARE_OPS = {"<", "<=", ">", ">="}
ARITH_OPS = {"+", "-", "*", "/", "%"}
LOGICAL_OPS = {"&&", "||"}


# ---------- types ----------


def _return_type_text(node: Node | None) -> str | None:
    if node is None:
        return None
    # In TS the return type sits as a separate field on function_declaration —
    # tree-sitter typically wraps it in `type_annotation` whose only named
    # child is the actual type.
    return _type_annotation_text(node)


def _type_annotation_text(node: Node) -> str:
    if node.type == "type_annotation":
        kids = named_children(node)
        if not kids:
            raise UnsupportedConstruct("ts empty type annotation")
        return _type_text(kids[0])
    return _type_text(node)


def _type_text(node: Node) -> str:
    if node.type == "predefined_type":
        return TS_TYPE_ALIASES.get(text(node), text(node))
    if node.type == "type_identifier":
        spelling = text(node)
        return TS_TYPE_ALIASES.get(spelling, spelling)
    raise UnsupportedConstruct(f"ts type {node.type}")

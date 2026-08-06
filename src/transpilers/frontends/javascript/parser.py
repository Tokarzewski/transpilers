"""JavaScript source -> HIR via tree-sitter-javascript.

JS has no static types, so every function parameter and return arrives as an
UnknownT — the type-inference pass (algorithmic dataflow + LLM fallback) is
what makes JS transpilation viable. Without inference resolving holes, the
later passes refuse rather than invent.

Subset is otherwise parallel to TypeScript: `function name(args) { ... }`,
return, if/else, while, C-style for, let/const/var, binary/comparison/logical
ops, integer literals, names, calls.
"""

from __future__ import annotations

from tree_sitter import Language, Node
import tree_sitter_javascript

from transpilers.ir import hir
from transpilers.frontends._treesitter import (
    make_parser,
    named_children,
    required_field,
    text,
)


from transpilers.frontends.errors import UnsupportedConstruct


_LANGUAGE = Language(tree_sitter_javascript.language())


def parse_javascript(source: str) -> hir.HirModule:
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
        # Top-level non-function statements (script entry, `"use strict"`,
        # bare `let` declarations, expression statements) — skip silently
        # rather than refusing the whole file.
        if c.type in (
            "expression_statement",
            "lexical_declaration",
            "variable_declaration",
            "for_statement",
            "if_statement",
            "while_statement",
            "block_statement",
        ):
            continue
        raise UnsupportedConstruct(f"top-level {c.type}")
    return hir.HirModule(source_lang="javascript", body=body)


def _convert_function(node: Node) -> hir.HirFunction:
    name_node = required_field(node, "name")
    params_node = required_field(node, "parameters")
    body_node = required_field(node, "body")
    params = [
        _convert_param(p) for p in named_children(params_node) if p.type == "identifier"
    ]
    return hir.HirFunction(
        name=text(name_node),
        params=params,
        return_annotation=None,
        body=_convert_block(body_node),
    )


def _convert_param(node: Node) -> hir.HirParam:
    # JS param is bare identifier; no annotation.
    return hir.HirParam(name=text(node), annotation=None)


def _convert_block(node: Node) -> list[hir.HirNode]:
    if node.type != "statement_block":
        raise UnsupportedConstruct(f"js block {node.type}")
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
    raise UnsupportedConstruct(f"js stmt {kind}")


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
        raise UnsupportedConstruct(f"js declarator pattern {name_node.type}")
    value_node = node.child_by_field_name("value")
    value = (
        _convert_expr(value_node)
        if value_node is not None
        else hir.HirIntLiteral(value=0)
    )
    return hir.HirAssign(target=text(name_node), value=value, annotation=None)


def _convert_expression_stmt(node: Node) -> hir.HirNode:
    if node.type == "assignment_expression":
        return _convert_assignment(node)
    if node.type == "augmented_assignment_expression":
        return _convert_aug_assignment(node)
    if node.type == "update_expression":
        return _convert_update(node)
    if node.type == "call_expression":
        return _convert_expr(node)
    raise UnsupportedConstruct(f"js expression statement {node.type}")


def _convert_assignment(node: Node) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    if left.type != "identifier":
        raise UnsupportedConstruct(f"js assignment lhs {left.type}")
    return hir.HirAssign(target=text(left), value=_convert_expr(right), annotation=None)


def _convert_aug_assignment(node: Node) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    op = text(required_field(node, "operator"))
    if left.type != "identifier" or not op.endswith("="):
        raise UnsupportedConstruct(f"js aug-assign {op!r}")
    return hir.HirAssign(
        target=text(left),
        value=_convert_expr(right),
        annotation=None,
        augmented_op=op[:-1],
    )


def _convert_update(node: Node) -> hir.HirNode:
    argument = required_field(node, "argument")
    op = text(required_field(node, "operator"))
    if argument.type != "identifier" or op not in ("++", "--"):
        raise UnsupportedConstruct(f"js update {op!r}")
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
        raise UnsupportedConstruct("js update at expression position")
    raise UnsupportedConstruct(f"js expr {kind}")


def _convert_binary(node: Node) -> hir.HirNode:
    left = _convert_expr(required_field(node, "left"))
    right = _convert_expr(required_field(node, "right"))
    op = text(required_field(node, "operator"))
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
    raise UnsupportedConstruct(f"js binary {op!r}")


def _convert_unary(node: Node) -> hir.HirNode:
    argument = required_field(node, "argument")
    op = text(required_field(node, "operator"))
    if op == "!":
        return hir.HirUnaryOp(op="not", operand=_convert_expr(argument))
    if op == "-":
        return hir.HirUnaryOp(op="-", operand=_convert_expr(argument))
    raise UnsupportedConstruct(f"js unary {op!r}")


def _convert_call(node: Node) -> hir.HirNode:
    func_node = required_field(node, "function")
    args_node = required_field(node, "arguments")
    if func_node.type != "identifier":
        raise UnsupportedConstruct(f"js call target {func_node.type}")
    args = [_convert_expr(c) for c in named_children(args_node) if c.type != "comment"]
    return hir.HirCall(func=text(func_node), args=args)


COMPARE_OPS = {"<", "<=", ">", ">="}
ARITH_OPS = {"+", "-", "*", "/", "%"}
LOGICAL_OPS = {"&&", "||"}

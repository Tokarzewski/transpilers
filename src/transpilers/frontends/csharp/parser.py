"""C# source -> HIR via tree-sitter-c-sharp.

Initial subset is parallel to Java: `class` containers with `static` methods
extracted as top-level HIR functions, primitive types, return / if / while /
for, declarations, binary / comparison / logical ops, invocations.
"""

from __future__ import annotations

from tree_sitter import Language, Node
import tree_sitter_c_sharp

from transpilers.ir import hir
from transpilers.frontends._treesitter import (
    make_parser,
    named_children,
    required_field,
    text,
)


from transpilers.frontends.errors import UnsupportedConstruct


CS_TYPE_ALIASES: dict[str, str] = {
    "int": "int",
    "long": "int",
    "short": "int",
    "byte": "int",
    "sbyte": "int",
    "uint": "int",
    "ulong": "int",
    "ushort": "int",
    "float": "float",
    "double": "float",
    "decimal": "float",
    "bool": "bool",
    "void": "None",
    "string": "str",
    "char": "int",
}


_LANGUAGE = Language(tree_sitter_c_sharp.language())


def parse_csharp(source: str) -> hir.HirModule:
    parser = make_parser(_LANGUAGE)
    tree = parser.parse(bytes(source, "utf-8"))
    body: list[hir.HirNode] = []
    for c in named_children(tree.root_node):
        if c.type == "class_declaration":
            body.extend(_extract_methods(c))
            continue
        if c.type in (
            "using_directive",
            "global_using_directive",
            "namespace_declaration",
            "file_scoped_namespace_declaration",
            "comment",
            "extern_alias_directive",
        ):
            if c.type.endswith("namespace_declaration"):
                body_node = c.child_by_field_name("body")
                if body_node is not None:
                    for inner in named_children(body_node):
                        if inner.type == "class_declaration":
                            body.extend(_extract_methods(inner))
            continue
        # C# top-level statements (the "minimal" program model): wrap in a
        # synthetic main() so the rest of the pipeline sees a function.
        # Skip silently for now if it doesn't fit our pattern; surfacing it
        # would require lifting all top-level statements into a synthetic fn.
        if c.type in ("global_statement",):
            continue
        raise UnsupportedConstruct(f"top-level {c.type}")
    return hir.HirModule(source_lang="csharp", body=body)


def _extract_methods(class_decl: Node) -> list[hir.HirFunction]:
    body_node = required_field(class_decl, "body")
    out: list[hir.HirFunction] = []
    for c in named_children(body_node):
        if c.type == "method_declaration":
            out.append(_convert_method(c))
        elif c.type in (
            "field_declaration",
            "property_declaration",
            "constructor_declaration",
        ):
            raise UnsupportedConstruct(f"class body {c.type}")
    return out


def _convert_method(node: Node) -> hir.HirFunction:
    # C# grammar uses `returns` for return type — different from Java's `type`.
    return_type_node = required_field(node, "returns")
    name_node = required_field(node, "name")
    params_node = required_field(node, "parameters")
    body_node = required_field(node, "body")
    params = [
        _convert_param(p) for p in named_children(params_node) if p.type == "parameter"
    ]
    return hir.HirFunction(
        name=text(name_node),
        params=params,
        return_annotation=_type_text(return_type_node),
        body=_convert_block(body_node),
    )


def _convert_param(node: Node) -> hir.HirParam:
    type_node = required_field(node, "type")
    name_node = required_field(node, "name")
    return hir.HirParam(name=text(name_node), annotation=_type_text(type_node))


def _convert_block(node: Node) -> list[hir.HirNode]:
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
    if kind == "local_declaration_statement":
        return _convert_variable_declaration(node.named_children[0])
    if kind == "variable_declaration":
        # Appears directly in `for (init; ...)` position.
        return _convert_variable_declaration(node)
    if kind == "expression_statement":
        kids = named_children(node)
        return [_convert_expression_stmt(kids[0])] if kids else []
    if kind in (
        "prefix_unary_expression",
        "postfix_unary_expression",
        "assignment_expression",
        "invocation_expression",
    ):
        # Same shapes that appear in `for (...; ...; <update>)` position.
        return [_convert_expression_stmt(node)]
    if kind == "block":
        return _convert_block(node)
    if kind == "comment":
        return []
    raise UnsupportedConstruct(f"csharp stmt {kind}")


def _convert_if(node: Node) -> hir.HirNode:
    cond = _convert_expr(required_field(node, "condition"))
    body = _convert_stmt(required_field(node, "consequence"))
    orelse_node = node.child_by_field_name("alternative")
    orelse = _convert_stmt(orelse_node) if orelse_node is not None else []
    return hir.HirIf(test=cond, body=body, orelse=orelse)


def _convert_while(node: Node) -> hir.HirNode:
    cond = _convert_expr(required_field(node, "condition"))
    body = _convert_stmt(required_field(node, "body"))
    return hir.HirWhile(test=cond, body=body)


def _convert_for(node: Node) -> list[hir.HirNode]:
    out: list[hir.HirNode] = []
    init_node = node.child_by_field_name("initializer")
    cond_node = node.child_by_field_name("condition")
    update_node = node.child_by_field_name("update")
    body_node = required_field(node, "body")
    if init_node is not None:
        out.extend(_convert_stmt(init_node))
    cond = (
        _convert_expr(cond_node)
        if cond_node is not None
        else hir.HirBoolLiteral(value=True)
    )
    inner = _convert_stmt(body_node)
    if update_node is not None:
        inner.append(_convert_expression_stmt(update_node))
    out.append(hir.HirWhile(test=cond, body=inner))
    return out


def _convert_variable_declaration(var_decl: Node) -> list[hir.HirNode]:
    """C# `variable_declaration` directly (no equals_value_clause wrapper) —
    the value sits as the trailing named child of `variable_declarator`."""
    type_node = required_field(var_decl, "type")
    annotation = _type_text(type_node)
    out: list[hir.HirNode] = []
    for c in named_children(var_decl):
        if c.type == "variable_declarator":
            name_node = required_field(c, "name")
            # Value is the last named child that isn't the name itself.
            init_node: Node | None = None
            for sub in named_children(c):
                if sub.id != name_node.id:
                    init_node = sub
            value = (
                _convert_expr(init_node)
                if init_node is not None
                else hir.HirIntLiteral(value=0)
            )
            out.append(
                hir.HirAssign(
                    target=text(name_node), value=value, annotation=annotation
                )
            )
    return out


def _convert_expression_stmt(node: Node) -> hir.HirNode:
    if node.type == "assignment_expression":
        return _convert_assignment(node)
    if node.type in ("prefix_unary_expression", "postfix_unary_expression"):
        return _convert_update(node)
    if node.type == "invocation_expression":
        return _convert_expr(node)
    raise UnsupportedConstruct(f"csharp expression statement {node.type}")


def _convert_assignment(node: Node) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    op = text(required_field(node, "operator"))
    if left.type != "identifier":
        raise UnsupportedConstruct(f"csharp assignment lhs {left.type}")
    aug = None if op == "=" else op[:-1]
    return hir.HirAssign(
        target=text(left), value=_convert_expr(right), annotation=None, augmented_op=aug
    )


def _convert_update(node: Node) -> hir.HirNode:
    operand = None
    op = None
    for c in node.children:
        if c.is_named:
            operand = c
        else:
            op = text(c)
    if operand is None or operand.type != "identifier" or op not in ("++", "--"):
        raise UnsupportedConstruct(f"csharp update {op!r}")
    sign = "+" if op == "++" else "-"
    return hir.HirAssign(
        target=text(operand),
        value=hir.HirIntLiteral(value=1),
        annotation=None,
        augmented_op=sign,
    )


# ---------- expressions ----------


def _convert_expr(node: Node) -> hir.HirNode:
    kind = node.type
    if kind == "integer_literal":
        return hir.HirIntLiteral(
            value=int(text(node).rstrip("uUlL").replace("_", ""), 0)
        )
    if kind == "real_literal":
        return hir.HirFloatLiteral(
            value=float(text(node).rstrip("fFdDmM").replace("_", ""))
        )
    if kind == "boolean_literal":
        return hir.HirBoolLiteral(value=text(node) == "true")
    if kind == "string_literal":
        raw = text(node)
        # tree-sitter-c-sharp wraps content with " — strip them.
        return hir.HirStringLiteral(value=raw[1:-1] if raw.startswith('"') else raw)
    if kind == "identifier":
        return hir.HirName(name=text(node))
    if kind == "parenthesized_expression":
        kids = named_children(node)
        if len(kids) == 1:
            return _convert_expr(kids[0])
    if kind == "binary_expression":
        return _convert_binary(node)
    if kind == "prefix_unary_expression":
        return _convert_unary(node)
    if kind == "postfix_unary_expression":
        raise UnsupportedConstruct("csharp postfix unary at expression position")
    if kind == "invocation_expression":
        return _convert_call(node)
    raise UnsupportedConstruct(f"csharp expr {kind}")


def _convert_binary(node: Node) -> hir.HirNode:
    left = _convert_expr(required_field(node, "left"))
    right = _convert_expr(required_field(node, "right"))
    op = text(required_field(node, "operator"))
    if op in COMPARE_OPS:
        return hir.HirCompare(op=op, left=left, right=right)
    if op in LOGICAL_OPS:
        return hir.HirBoolOp(op="and" if op == "&&" else "or", left=left, right=right)
    if op in ARITH_OPS:
        return hir.HirBinOp(op=op, left=left, right=right)
    raise UnsupportedConstruct(f"csharp binary op {op!r}")


def _convert_unary(node: Node) -> hir.HirNode:
    operand = None
    op = None
    for c in node.children:
        if c.is_named:
            operand = c
        else:
            op = text(c)
    if operand is None or op is None:
        raise UnsupportedConstruct("csharp unary missing parts")
    if op == "!":
        return hir.HirUnaryOp(op="not", operand=_convert_expr(operand))
    if op == "-":
        return hir.HirUnaryOp(op="-", operand=_convert_expr(operand))
    raise UnsupportedConstruct(f"csharp unary {op!r}")


def _convert_call(node: Node) -> hir.HirNode:
    func_node = required_field(node, "function")
    args_node = required_field(node, "arguments")
    name = text(func_node)
    args: list[hir.HirNode] = []
    for c in named_children(args_node):
        if c.type == "argument":
            kids = named_children(c)
            if kids:
                args.append(_convert_expr(kids[0]))
        else:
            args.append(_convert_expr(c))
    return hir.HirCall(func=name, args=args)


COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}
ARITH_OPS = {"+", "-", "*", "/", "%"}
LOGICAL_OPS = {"&&", "||"}


# ---------- types ----------


def _type_text(node: Node) -> str:
    if node.type == "predefined_type":
        return CS_TYPE_ALIASES.get(text(node), text(node))
    if node.type == "void_keyword":
        return "None"
    if node.type == "identifier":
        return CS_TYPE_ALIASES.get(text(node), text(node))
    # Fallback to text + alias lookup; this catches user-defined types we
    # don't yet model, surfacing a recognizable identifier.
    spelling = text(node)
    return CS_TYPE_ALIASES.get(spelling, spelling)

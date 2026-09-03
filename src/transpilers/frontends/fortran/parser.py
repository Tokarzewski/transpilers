"""Fortran source -> HIR via tree-sitter-fortran.

Initial subset: modern free-form Fortran 90+ — `function NAME(args) result(r)`
with `integer` / `real` / `logical` declarations for params and locals, plus
basic control flow (if/else, do-while, do-loop with bounds), assignments,
arithmetic / relational operators.

Three Fortran-specific shapes the parser normalizes:

  - Param types are declared separately from the signature. We scan
    `variable_declaration` statements in the function body to build a name→type
    table, then apply the types back to params.
  - There's no `return <expr>` statement — you assign to the result variable.
    A `HirReturn(HirName(result_name))` is synthesized at the end of body.
  - `do i = a, b` (inclusive both ends) lowers to a `for i in range(a, b+1)`-
    equivalent shape in the HIR. The +1 inclusive→exclusive adjustment lives
    in the frontend so MIR semantics stay clean.
"""

from __future__ import annotations

from tree_sitter import Language, Node
import tree_sitter_fortran

from transpilers.ir import hir
from transpilers.frontends._treesitter import (
    make_parser,
    named_children,
    required_field,
    text,
)


from transpilers.frontends.errors import UnsupportedConstruct


FORTRAN_TYPE_ALIASES: dict[str, str] = {
    "integer": "int",
    "real": "float",
    "double precision": "float",
    "logical": "bool",
    "character": "str",
}


_LANGUAGE = Language(tree_sitter_fortran.language())


def parse_fortran(source: str) -> hir.HirModule:
    parser = make_parser(_LANGUAGE)
    tree = parser.parse(bytes(source, "utf-8"))
    body: list[hir.HirNode] = []
    for c in named_children(tree.root_node):
        if c.type == "function":
            body.append(_convert_function(c))
            continue
        if c.type == "program":
            # Skip the program wrapper — its body uses Fortran I/O that
            # doesn't translate; surface only function defs nested inside.
            for inner in named_children(c):
                if inner.type == "function":
                    body.append(_convert_function(inner))
            continue
        if c.type in ("subroutine", "module"):
            raise UnsupportedConstruct(f"fortran top-level {c.type}")
        if c.type in ("comment",):
            continue
    return hir.HirModule(source_lang="fortran", body=body)


def _convert_function(node: Node) -> hir.HirFunction:
    header: Node | None = None
    result_name: str | None = None
    var_types: dict[str, str] = {}
    body_statements: list[Node] = []
    for c in named_children(node):
        if c.type == "function_statement":
            header = c
            # function_result is a child of function_statement here.
            for sub in named_children(c):
                if sub.type == "function_result":
                    inner = named_children(sub)
                    if inner:
                        result_name = text(inner[0])
        elif c.type == "variable_declaration":
            ty = _resolve_var_decl_type(c)
            for sub in named_children(c):
                if sub.type == "identifier":
                    var_types[text(sub)] = ty
        elif c.type == "end_function_statement":
            continue
        else:
            body_statements.append(c)

    if header is None:
        raise UnsupportedConstruct("fortran function missing function_statement")
    name = text(required_field(header, "name"))
    params_node = required_field(header, "parameters")
    param_names = [
        text(p) for p in named_children(params_node) if p.type == "identifier"
    ]

    # Result variable: either explicit via `result(r)` or implicit (same as fn).
    if result_name is None:
        result_name = name
    return_annotation = var_types.get(result_name)

    params = [hir.HirParam(name=p, annotation=var_types.get(p)) for p in param_names]

    # The Fortran result variable is implicitly declared by appearing in
    # `result(...)`. Other backends (Rust/C/Mojo) need an explicit
    # declaration before the variable can be assigned in a branch — otherwise
    # `if (cond) { r = a } else { r = b }` introduces `r` in branch scope
    # only.
    #
    # Optimization: if the first user statement is an unconditional `r = ...`,
    # let that assignment be the declaration — no synthesized init needed.
    # This drops the double-init noise Ghidra-output Fortran used to show.
    first_user_stmt = body_statements[0] if body_statements else None
    first_assigns_result = (
        first_user_stmt is not None
        and first_user_stmt.type == "assignment_statement"
        and first_user_stmt.child_by_field_name("left") is not None
        and first_user_stmt.child_by_field_name("left").text.decode("utf-8")
        == result_name
    )

    body: list[hir.HirNode]
    if first_assigns_result:
        body = []
        locals_seen = set(
            param_names
        )  # result_name picks up annotation on first user assign
    else:
        result_init = _default_value_for(return_annotation)
        body = [
            hir.HirAssign(
                target=result_name, value=result_init, annotation=return_annotation
            )
        ]
        locals_seen = set(param_names) | {result_name}

    # Stash return annotation in var_types so the first user assign carries it.
    if first_assigns_result and return_annotation is not None:
        var_types.setdefault(result_name, return_annotation)
    body.extend(
        _convert_block(body_statements, locals_seen=locals_seen, var_types=var_types)
    )
    # Implicit return of the result variable.
    body.append(hir.HirReturn(value=hir.HirName(name=result_name)))
    return hir.HirFunction(
        name=name, params=params, return_annotation=return_annotation, body=body
    )


def _default_value_for(annotation: str | None) -> hir.HirNode:
    if annotation in (None, "int"):
        return hir.HirIntLiteral(value=0)
    if annotation == "float":
        # We don't have a float literal HIR node yet; 0 as an int will likely
        # be widened by later passes when surrounding context is float, or
        # raise a clear error.
        return hir.HirIntLiteral(value=0)
    if annotation == "bool":
        return hir.HirBoolLiteral(value=False)
    if annotation == "str":
        return hir.HirStringLiteral(value="")
    return hir.HirIntLiteral(value=0)


def _resolve_var_decl_type(node: Node) -> str:
    type_node = required_field(node, "type")
    if type_node.type == "intrinsic_type":
        spelling = text(type_node).lower().strip()
        return FORTRAN_TYPE_ALIASES.get(spelling, spelling)
    raise UnsupportedConstruct(f"fortran type {type_node.type}")


def _convert_block(
    statements: list[Node], *, locals_seen: set[str], var_types: dict[str, str]
) -> list[hir.HirNode]:
    out: list[hir.HirNode] = []
    for stmt in statements:
        out.extend(_convert_stmt(stmt, locals_seen, var_types))
    return out


def _convert_stmt(
    node: Node, locals_seen: set[str], var_types: dict[str, str]
) -> list[hir.HirNode]:
    kind = node.type
    if kind == "assignment_statement":
        return [_convert_assignment(node, locals_seen, var_types)]
    if kind == "if_statement":
        return [_convert_if(node, locals_seen, var_types)]
    if kind == "do_loop":
        return [_convert_do_loop(node, locals_seen, var_types)]
    if kind == "comment":
        return []
    raise UnsupportedConstruct(f"fortran stmt {kind}")


def _convert_assignment(
    node: Node, locals_seen: set[str], var_types: dict[str, str]
) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    if left.type != "identifier":
        raise UnsupportedConstruct(f"fortran assignment lhs {left.type}")
    name = text(left)
    annotation = None
    if name not in locals_seen:
        annotation = var_types.get(name)
        locals_seen.add(name)
    return hir.HirAssign(target=name, value=_convert_expr(right), annotation=annotation)


def _convert_if(
    node: Node, locals_seen: set[str], var_types: dict[str, str]
) -> hir.HirNode:
    cond_node: Node | None = None
    body_stmts: list[Node] = []
    else_stmts: list[Node] = []
    for c in named_children(node):
        if c.type == "parenthesized_expression" and cond_node is None:
            cond_node = c
        elif c.type == "else_clause":
            else_stmts = [sub for sub in named_children(c) if sub.type != "comment"]
        elif c.type == "end_if_statement":
            continue
        elif cond_node is not None:
            body_stmts.append(c)
    if cond_node is None:
        raise UnsupportedConstruct("fortran if missing condition")
    cond = _convert_expr(cond_node)
    body = _convert_block(body_stmts, locals_seen=locals_seen, var_types=var_types)
    orelse = _convert_block(else_stmts, locals_seen=locals_seen, var_types=var_types)
    return hir.HirIf(test=cond, body=body, orelse=orelse)


def _convert_do_loop(
    node: Node, locals_seen: set[str], var_types: dict[str, str]
) -> hir.HirNode:
    """Two forms:
    do while (cond) ... end do   -> HirWhile
    do i = a, b [, step] ... end do  -> HirFor (range; inclusive endpoint
         adjusted to exclusive by +1)
    """
    header: Node | None = None
    body_stmts: list[Node] = []
    for c in named_children(node):
        if c.type == "do_statement":
            header = c
        elif c.type == "end_do_loop_statement":
            continue
        else:
            body_stmts.append(c)
    if header is None:
        raise UnsupportedConstruct("fortran do_loop missing header")
    inner = list(named_children(header))
    if not inner:
        raise UnsupportedConstruct("fortran do header empty")
    head = inner[0]
    if head.type == "while_statement":
        # `do while (cond) ... end do` — cond is wrapped in parens.
        cond_kids = named_children(head)
        if not cond_kids:
            raise UnsupportedConstruct("fortran do-while missing condition")
        cond = _convert_expr(cond_kids[0])
        body = _convert_block(body_stmts, locals_seen=locals_seen, var_types=var_types)
        return hir.HirWhile(test=cond, body=body)
    if head.type == "loop_control_expression":
        kids = named_children(head)
        if len(kids) < 3:
            raise UnsupportedConstruct("fortran do-iter missing parts")
        target_node = kids[0]
        start_node = kids[1]
        stop_node = kids[2]
        # Fortran's `a, b` is inclusive on both ends; our HirFor over range(...)
        # treats stop as exclusive, matching Python. Adjust by +1.
        stop_expr = hir.HirBinOp(
            op="+", left=_convert_expr(stop_node), right=hir.HirIntLiteral(value=1)
        )
        iter_expr = hir.HirCall(
            func="range", args=[_convert_expr(start_node), stop_expr]
        )
        locals_seen.add(text(target_node))
        body = _convert_block(body_stmts, locals_seen=locals_seen, var_types=var_types)
        return hir.HirFor(target=text(target_node), iter=iter_expr, body=body)
    raise UnsupportedConstruct(f"fortran do header {head.type}")


# ---------- expressions ----------


def _convert_expr(node: Node) -> hir.HirNode:
    kind = node.type
    if kind == "number_literal":
        raw = text(node).replace("_", "")
        # Fortran allows `1.5_dp` (kind suffix already stripped above) and
        # exponent forms `1.5e0`, `1d-3`.
        if "." in raw or "e" in raw.lower() or "d" in raw.lower():
            normalized = raw.lower().replace("d", "e")
            return hir.HirFloatLiteral(value=float(normalized))
        return hir.HirIntLiteral(value=int(raw, 0))
    if kind == "boolean_literal" or kind == "logical_literal":
        return hir.HirBoolLiteral(value=".true." in text(node).lower())
    if kind == "string_literal":
        raw = text(node)
        return hir.HirStringLiteral(value=raw[1:-1])
    if kind == "identifier":
        return hir.HirName(name=text(node))
    if kind == "parenthesized_expression":
        kids = named_children(node)
        if len(kids) == 1:
            return _convert_expr(kids[0])
    if kind in ("math_expression", "relational_expression"):
        return _convert_binary(node, kind == "relational_expression")
    if kind == "unary_expression":
        return _convert_unary(node)
    if kind == "logical_expression":
        return _convert_binary(node, False, logical=True)
    raise UnsupportedConstruct(f"fortran expr {kind}")


def _convert_binary(
    node: Node, is_relational: bool, *, logical: bool = False
) -> hir.HirNode:
    left = _convert_expr(required_field(node, "left"))
    right = _convert_expr(required_field(node, "right"))
    op_raw = text(required_field(node, "operator"))
    op = _FORTRAN_OPS.get(op_raw.lower(), op_raw)
    if op in COMPARE_OPS or is_relational:
        return hir.HirCompare(op=op, left=left, right=right)
    if op in ("and", "or") or logical:
        # Fortran logical ops are .and. / .or. — already normalized via the map.
        return hir.HirBoolOp(op=op, left=left, right=right)
    if op in ARITH_OPS:
        return hir.HirBinOp(op=op, left=left, right=right)
    raise UnsupportedConstruct(f"fortran binary {op!r}")


def _convert_unary(node: Node) -> hir.HirNode:
    kids = named_children(node)
    if len(kids) != 1:
        raise UnsupportedConstruct("fortran unary missing parts")
    op_node = None
    for c in node.children:
        if not c.is_named:
            op_node = c
            break
    op = text(op_node).lower() if op_node else None
    if op == "-":
        return hir.HirUnaryOp(op="-", operand=_convert_expr(kids[0]))
    if op == ".not.":
        return hir.HirUnaryOp(op="not", operand=_convert_expr(kids[0]))
    raise UnsupportedConstruct(f"fortran unary {op!r}")


_FORTRAN_OPS = {
    ".eq.": "==",
    ".ne.": "!=",
    ".lt.": "<",
    ".le.": "<=",
    ".gt.": ">",
    ".ge.": ">=",
    ".and.": "and",
    ".or.": "or",
    "==": "==",
    "/=": "!=",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "**": "**",
}


COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}
ARITH_OPS = {"+", "-", "*", "/", "%", "**"}

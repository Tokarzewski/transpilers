"""Go source -> HIR via tree-sitter-go.

Initial subset: top-level `func name(params) ResultType { ... }`, primitive
types (int / int64 / float64 / bool / string), return, if/else, for-with-
condition (Go's `for cond { }` is our `while`), C-style for, short var
declarations (`:=`), assignments, binary / comparison / logical ops, ints,
bools, strings, identifiers, function calls.
"""

from __future__ import annotations

from tree_sitter import Language, Node
import tree_sitter_go

from transpilers.ir import hir
from transpilers.frontends._markers import FlattenBlock
from transpilers.frontends._treesitter import (
    make_parser,
    named_children,
    required_field,
    text,
)


from transpilers.frontends.errors import UnsupportedConstruct


GO_TYPE_ALIASES: dict[str, str] = {
    "int": "int",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "uint": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "byte": "int",
    "rune": "int",
    "float32": "float",
    "float64": "float",
    "bool": "bool",
    "string": "str",
}


_LANGUAGE = Language(tree_sitter_go.language())


def parse_go(source: str) -> hir.HirModule:
    parser = make_parser(_LANGUAGE)
    tree = parser.parse(bytes(source, "utf-8"))
    body: list[hir.HirNode] = []
    for c in named_children(tree.root_node):
        if c.type == "function_declaration":
            body.append(_convert_function(c))
            continue
        if c.type in ("package_clause", "import_declaration", "comment"):
            continue
        if c.type in (
            "method_declaration",
            "type_declaration",
            "const_declaration",
            "var_declaration",
        ):
            raise UnsupportedConstruct(f"top-level {c.type}")
        raise UnsupportedConstruct(f"top-level {c.type}")
    return hir.HirModule(source_lang="go", body=body)


def _convert_function(node: Node) -> hir.HirFunction:
    name_node = required_field(node, "name")
    params_node = required_field(node, "parameters")
    body_node = required_field(node, "body")
    result_node = node.child_by_field_name("result")
    params = _convert_params(params_node)
    return hir.HirFunction(
        name=text(name_node),
        params=params,
        return_annotation=_type_text(result_node)
        if result_node is not None
        else "None",
        body=_convert_block(body_node),
    )


def _convert_params(node: Node) -> list[hir.HirParam]:
    out: list[hir.HirParam] = []
    for c in named_children(node):
        if c.type == "parameter_declaration":
            # Go allows `a, b int` — one type for several names.
            type_node = required_field(c, "type")
            annotation = _type_text(type_node)
            for sub in named_children(c):
                if sub.type == "identifier":
                    out.append(hir.HirParam(name=text(sub), annotation=annotation))
    return out


def _convert_block(node: Node) -> list[hir.HirNode]:
    if node.type not in ("block", "statement_list"):
        raise UnsupportedConstruct(f"go block {node.type}")
    out: list[hir.HirNode] = []
    for c in named_children(node):
        if c.type == "statement_list":
            # Go's grammar wraps statements in `statement_list` inside `block`.
            out.extend(_convert_block(c))
        else:
            out.extend(_convert_stmt(c))
    return out


def _convert_stmt(node: Node) -> list[hir.HirNode]:
    kind = node.type
    if kind == "return_statement":
        kids = named_children(node)
        if not kids:
            return [hir.HirReturn(value=None)]
        # Go return can be multi-value — only single is supported.
        if kids[0].type == "expression_list":
            inner = named_children(kids[0])
            if len(inner) != 1:
                raise UnsupportedConstruct("multi-value go return")
            return [hir.HirReturn(value=_convert_expr(inner[0]))]
        return [hir.HirReturn(value=_convert_expr(kids[0]))]
    if kind == "if_statement":
        return [_convert_if(node)]
    if kind == "for_statement":
        return _convert_for(node)
    if kind == "short_var_declaration":
        return _convert_short_var(node)
    if kind == "var_declaration":
        return _convert_var_decl(node)
    if kind == "assignment_statement":
        result = _convert_assignment(node)
        if isinstance(result, FlattenBlock):
            return result.stmts
        return [result]
    if kind == "inc_statement" or kind == "dec_statement":
        return [_convert_inc_dec(node)]
    if kind == "expression_statement":
        kids = named_children(node)
        if not kids:
            return []
        if kids[0].type == "call_expression":
            return [_convert_expr(kids[0])]
        raise UnsupportedConstruct(f"go expression stmt {kids[0].type}")
    if kind == "block":
        return _convert_block(node)
    if kind in ("comment",):
        return []
    raise UnsupportedConstruct(f"go stmt {kind}")


def _convert_if(node: Node) -> hir.HirNode:
    cond = _convert_expr(required_field(node, "condition"))
    body = _convert_block(required_field(node, "consequence"))
    orelse_node = node.child_by_field_name("alternative")
    orelse: list[hir.HirNode] = []
    if orelse_node is not None:
        if orelse_node.type == "block":
            orelse = _convert_block(orelse_node)
        elif orelse_node.type == "if_statement":
            orelse = [_convert_if(orelse_node)]
    return hir.HirIf(test=cond, body=body, orelse=orelse)


def _convert_for(node: Node) -> list[hir.HirNode]:
    """Go for forms:
    for { ... }                   -> infinite — refuse
    for cond { ... }              -> while(cond)
    for init; cond; update { ... }-> init; while(cond) { body; update }
    for x := range ... { ... }    -> refuse (ranges over collections)
    """
    body_node = required_field(node, "body")
    inner = _convert_block(body_node)
    # Look at the header: it may be a single 'condition' child or a 'for_clause'.
    header = None
    for c in named_children(node):
        if c.type in ("for_clause", "range_clause"):
            header = c
            break
        if c.type != "block" and c.type != "comment":
            # Some Go grammars place a bare condition expression here.
            header = c
            break
    if header is None:
        raise UnsupportedConstruct("`for {}` (infinite loop) not supported")
    if header.type == "range_clause":
        raise UnsupportedConstruct(
            "`for ... := range ...` not supported in initial subset"
        )
    if header.type == "for_clause":
        init_node = header.child_by_field_name("initializer")
        cond_node = header.child_by_field_name("condition")
        update_node = header.child_by_field_name("update")
        out: list[hir.HirNode] = []
        if init_node is not None:
            out.extend(_convert_stmt(init_node))
        cond = (
            _convert_expr(cond_node)
            if cond_node is not None
            else hir.HirBoolLiteral(value=True)
        )
        if update_node is not None:
            inner.extend(_convert_stmt(update_node))
        out.append(hir.HirWhile(test=cond, body=inner))
        return out
    # Bare condition: `for cond { ... }`.
    return [hir.HirWhile(test=_convert_expr(header), body=inner)]


def _convert_short_var(node: Node) -> list[hir.HirNode]:
    """`x := value` or `x, y := a, b` or `x, _ := fn(...)` (multi-value).

    Multi-value forms (common in Go: `val, err := someFunc()`) lower as the
    first concrete name bound to the first value; remaining slots and `_`
    blanks are dropped. Lossy but practical — full multi-return support
    needs tuple types in the IR."""
    left = required_field(node, "left")
    right = required_field(node, "right")
    left_names = [text(c) for c in named_children(left) if c.type == "identifier"]
    right_exprs = list(named_children(right))
    # Drop `_` blank-identifier slots from the LHS — they signal "ignore this
    # return value" rather than a real binding.
    real_names = [n for n in left_names if n != "_"]
    if not real_names or not right_exprs:
        # Pure side-effect call with all-blank LHS — skip the statement.
        return []
    return [
        hir.HirAssign(
            target=real_names[0], value=_convert_expr(right_exprs[0]), annotation=None
        )
    ]


def _convert_var_decl(node: Node) -> list[hir.HirNode]:
    """`var x int = 0` or `var x int` or `var (...)` group."""
    out: list[hir.HirNode] = []
    for spec in named_children(node):
        if spec.type == "var_spec":
            type_node = spec.child_by_field_name("type")
            annotation = _type_text(type_node) if type_node is not None else None
            value_node = spec.child_by_field_name("value")
            name_nodes = [c for c in named_children(spec) if c.type == "identifier"]
            if len(name_nodes) != 1:
                raise UnsupportedConstruct("multi-name go var spec")
            if value_node is not None:
                value_kids = named_children(value_node)
                value = (
                    _convert_expr(value_kids[0])
                    if value_kids
                    else hir.HirIntLiteral(value=0)
                )
            else:
                value = hir.HirIntLiteral(value=0)
            out.append(
                hir.HirAssign(
                    target=text(name_nodes[0]), value=value, annotation=annotation
                )
            )
    return out


def _convert_assignment(node: Node) -> hir.HirNode:
    left = required_field(node, "left")
    right = required_field(node, "right")
    left_kids = named_children(left)
    right_kids = named_children(right)
    real_lhs = [c for c in left_kids if not (c.type == "identifier" and text(c) == "_")]
    if not real_lhs or not right_kids:
        return hir.HirAssign(
            target="_", value=hir.HirIntLiteral(value=0), annotation=None
        )
    op = None
    for child in node.children:
        if not child.is_named:
            t = text(child)
            if t.endswith("=") or t == "=":
                op = t
                break
    if op is None:
        op = "="
    aug = None if op == "=" else op[:-1]
    # Multi-target tuple form: `a, b = b, a` (and the subscript variant
    # `xs[i], xs[j] = xs[j], xs[i]`) → desugar to a sequence of writes via
    # tmps, mirroring the Python frontend's _tuple_assign.
    if len(real_lhs) > 1 and aug is None and len(right_kids) == len(real_lhs):
        return _tuple_swap(real_lhs, right_kids)
    target = real_lhs[0]
    rhs = _convert_expr(right_kids[0])
    if target.type == "identifier":
        return hir.HirAssign(
            target=text(target), value=rhs, annotation=None, augmented_op=aug
        )
    if target.type == "index_expression":
        return hir.HirSubscriptAssign(
            obj=_convert_expr(required_field(target, "operand")),
            index=_convert_expr(required_field(target, "index")),
            value=rhs,
        )
    if target.type == "selector_expression":
        return hir.HirFieldAssign(
            obj=_convert_expr(required_field(target, "operand")),
            field=text(required_field(target, "field")),
            value=rhs,
        )
    raise UnsupportedConstruct(f"go assignment lhs {target.type}")


def _tuple_swap(targets: list[Node], rhs_nodes: list[Node]) -> hir.HirNode:
    """`a, b = b, a` (or subscript variant) → tmp staging then writes."""
    tmps = [f"__xpile_swap_{i}" for i in range(len(targets))]
    stmts: list[hir.HirNode] = []
    for tmp, rhs in zip(tmps, rhs_nodes):
        stmts.append(
            hir.HirAssign(target=tmp, value=_convert_expr(rhs), annotation=None)
        )
    for slot, tmp in zip(targets, tmps):
        rhs_expr = hir.HirName(name=tmp)
        if slot.type == "identifier":
            stmts.append(
                hir.HirAssign(target=text(slot), value=rhs_expr, annotation=None)
            )
        elif slot.type == "index_expression":
            stmts.append(
                hir.HirSubscriptAssign(
                    obj=_convert_expr(required_field(slot, "operand")),
                    index=_convert_expr(required_field(slot, "index")),
                    value=rhs_expr,
                )
            )
        elif slot.type == "selector_expression":
            stmts.append(
                hir.HirFieldAssign(
                    obj=_convert_expr(required_field(slot, "operand")),
                    field=text(required_field(slot, "field")),
                    value=rhs_expr,
                )
            )
        else:
            raise UnsupportedConstruct(f"go tuple-assign slot {slot.type}")
    return FlattenBlock(stmts=stmts)


def _convert_inc_dec(node: Node) -> hir.HirNode:
    """`x++` / `x--`."""
    kids = named_children(node)
    if not kids or kids[0].type != "identifier":
        raise UnsupportedConstruct("go inc/dec on non-identifier")
    sign = "+" if node.type == "inc_statement" else "-"
    return hir.HirAssign(
        target=text(kids[0]),
        value=hir.HirIntLiteral(value=1),
        annotation=None,
        augmented_op=sign,
    )


# ---------- expressions ----------


def _convert_expr(node: Node) -> hir.HirNode:
    kind = node.type
    if kind == "int_literal":
        raw = text(node).replace("_", "")
        return hir.HirIntLiteral(value=int(raw, 0))
    if kind == "float_literal":
        return hir.HirFloatLiteral(value=float(text(node).replace("_", "")))
    if kind == "interpreted_string_literal":
        raw = text(node)
        return hir.HirStringLiteral(value=raw[1:-1])
    if kind == "true":
        return hir.HirBoolLiteral(value=True)
    if kind == "false":
        return hir.HirBoolLiteral(value=False)
    if kind == "identifier":
        spelling = text(node)
        if spelling == "true":
            return hir.HirBoolLiteral(value=True)
        if spelling == "false":
            return hir.HirBoolLiteral(value=False)
        return hir.HirName(name=spelling)
    if kind == "parenthesized_expression":
        kids = named_children(node)
        if len(kids) == 1:
            return _convert_expr(kids[0])
    if kind == "selector_expression":
        # `obj.field` access at expression position.
        operand = required_field(node, "operand")
        field = required_field(node, "field")
        return hir.HirFieldAccess(value=_convert_expr(operand), field=text(field))
    if kind == "binary_expression":
        return _convert_binary(node)
    if kind == "unary_expression":
        return _convert_unary(node)
    if kind == "call_expression":
        return _convert_call(node)
    if kind == "index_expression":
        operand = required_field(node, "operand")
        index = required_field(node, "index")
        return hir.HirSubscript(
            value=_convert_expr(operand), index=_convert_expr(index)
        )
    if kind == "composite_literal":
        # `[]int{1, 2, 3}` / `T{...}` — only the slice/array literal form
        # maps cleanly to HirList. Structs would need user-type tracking.
        type_node = node.child_by_field_name("type")
        body_node = node.child_by_field_name("body")
        if type_node is not None and type_node.type in ("slice_type", "array_type"):
            elements = []
            if body_node is not None:
                for c in named_children(body_node):
                    if c.type == "literal_element":
                        inner = named_children(c)
                        if inner:
                            elements.append(_convert_expr(inner[0]))
                    elif c.type != "comment":
                        elements.append(_convert_expr(c))
            return hir.HirList(elements=elements)
    raise UnsupportedConstruct(f"go expr {kind}")


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
    raise UnsupportedConstruct(f"go binary {op!r}")


def _convert_unary(node: Node) -> hir.HirNode:
    operand = required_field(node, "operand")
    op = text(required_field(node, "operator"))
    if op == "!":
        return hir.HirUnaryOp(op="not", operand=_convert_expr(operand))
    if op == "-":
        return hir.HirUnaryOp(op="-", operand=_convert_expr(operand))
    raise UnsupportedConstruct(f"go unary {op!r}")


def _convert_call(node: Node) -> hir.HirNode:
    func_node = required_field(node, "function")
    args_node = required_field(node, "arguments")
    args = [_convert_expr(c) for c in named_children(args_node) if c.type != "comment"]
    if func_node.type == "identifier":
        name = text(func_node)
        # `int64(x)` / `float64(x)` — Go casts parse as calls; unwrap.
        if name in GO_TYPE_ALIASES and len(args) == 1:
            return args[0]
        return hir.HirCall(func=name, args=args)
    if func_node.type == "selector_expression":
        # `pkg.Func(args)` or `obj.Method(args)` — lower as a method call on
        # the LHS. The pkg-level qualifier collapses onto the method name in
        # downstream emission, which is approximate but lets the corpus
        # parse.
        operand = func_node.child_by_field_name("operand")
        field = func_node.child_by_field_name("field")
        receiver = (
            _convert_expr(operand) if operand is not None else hir.HirName(name="_")
        )
        method = text(field) if field is not None else "_"
        return hir.HirMethodCall(receiver=receiver, method=method, args=args)
    raise UnsupportedConstruct(f"go call target {func_node.type}")


COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}
ARITH_OPS = {"+", "-", "*", "/", "%"}
LOGICAL_OPS = {"&&", "||"}


# ---------- types ----------


def _type_text(node: Node) -> str:
    if node is None:
        return "None"
    if node.type in ("type_identifier", "identifier"):
        spelling = text(node)
        return GO_TYPE_ALIASES.get(spelling, spelling)
    if node.type == "qualified_type":
        # `os.File` etc. — out of scope.
        raise UnsupportedConstruct(f"go qualified type {text(node)}")
    if node.type == "slice_type":
        # `[]int` → `list[int]` (Python annotation form for downstream typing).
        elem_node = node.child_by_field_name("element")
        return f"list[{_type_text(elem_node)}]"
    if node.type == "array_type":
        elem_node = node.child_by_field_name("element")
        return f"list[{_type_text(elem_node)}]"
    if node.type == "pointer_type":
        # `*T` → `T` — drops nullability and aliasing (#7). The IR has no
        # OptionT / RefT yet; mirrors the C++ frontend's `&`/`*` drop.
        return _type_text(node.child_by_field_name("type"))
    raise UnsupportedConstruct(f"go type {node.type}")

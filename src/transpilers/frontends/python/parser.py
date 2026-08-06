"""Python source -> HIR via libcst.

Subset supported:
  - function defs with annotated params and return types
  - simple statements: return, expression
  - control flow: if/elif/else, while, for-in-range, for-each over a bare-name
    collection, and `for i, x in enumerate(<name>)` (both desugar to an
    indexed range loop in HIR->MIR)
  - assignment: plain, annotated, augmented
  - expressions: binary ops, comparisons, boolean ops, unary not/-
  - literals: int, bool, string
  - lists, subscript indexing, function calls

Anything else raises UnsupportedConstruct. That's the right failure mode —
later passes (algorithmic inference, LLM hooks) handle gaps; the parser must
not silently drop nodes it doesn't understand.
"""

from __future__ import annotations

import libcst as cst

from transpilers.ir import hir
from transpilers.frontends._markers import FlattenBlock, PassMarker


from transpilers.frontends.errors import UnsupportedConstruct


def parse_python(source: str) -> hir.HirModule:
    tree = cst.parse_module(source)
    body: list[hir.HirNode] = []
    for stmt in tree.body:
        # Skip top-level constructs that don't translate semantically: imports,
        # `if __name__ == "__main__"` guards, bare statement-position calls
        # (script entrypoints), and the shebang/coding comments libcst exposes
        # as EmptyLine. The pipeline transpiles function definitions; module-
        # level side effects are dropped here rather than partway through.
        if isinstance(stmt, cst.SimpleStatementLine):
            inner = stmt.body
            if all(isinstance(s, (cst.Import, cst.ImportFrom)) for s in inner):
                continue
            if all(isinstance(s, cst.Expr) for s in inner):
                # Bare function-call statements at module scope: drop.
                continue
        if isinstance(stmt, cst.If):
            # Skip the conventional `if __name__ == "__main__":` guard.
            if _is_main_guard(stmt):
                continue
        try:
            body.append(_convert(stmt))
        except UnsupportedConstruct:
            # Walk inside FunctionDef even if we can't parse module-level
            # constructs around it. For non-FunctionDef siblings we skip.
            if isinstance(stmt, cst.FunctionDef):
                raise
    return hir.HirModule(source_lang="python", body=body)


def _is_main_guard(node: cst.If) -> bool:
    """Detect `if __name__ == "__main__":` — a script-entry idiom we drop."""
    test = node.test
    if not isinstance(test, cst.Comparison) or len(test.comparisons) != 1:
        return False
    left = test.left
    target = test.comparisons[0].comparator
    if not isinstance(left, cst.Name) or left.value != "__name__":
        return False
    if not isinstance(target, cst.SimpleString):
        return False
    return target.value.strip("\"'") == "__main__"


def _convert(node: cst.CSTNode) -> hir.HirNode:
    if isinstance(node, cst.SimpleStatementLine):
        if len(node.body) != 1:
            raise UnsupportedConstruct("compound simple statements")
        return _convert(node.body[0])

    if isinstance(node, cst.FunctionDef):
        return _convert_function(node)

    if isinstance(node, cst.Return):
        return hir.HirReturn(value=_convert(node.value) if node.value else None)

    if isinstance(node, cst.If):
        return _convert_if(node)

    if isinstance(node, cst.While):
        return hir.HirWhile(test=_convert(node.test), body=_convert_block(node.body))

    if isinstance(node, cst.For):
        return _convert_for(node)

    if isinstance(node, cst.Assign):
        if len(node.targets) != 1:
            raise UnsupportedConstruct("multi-target assignment")
        target = node.targets[0].target
        if isinstance(target, cst.Name):
            return hir.HirAssign(
                target=target.value, value=_convert(node.value), annotation=None
            )
        if isinstance(target, cst.Subscript):
            # `xs[i] = v` — only plain single-index form (no slice).
            if len(target.slice) != 1 or not isinstance(
                target.slice[0].slice, cst.Index
            ):
                raise UnsupportedConstruct("slice / multi-index subscript-assignment")
            idx = target.slice[0].slice.value
            return hir.HirSubscriptAssign(
                obj=_convert(target.value),
                index=_convert(idx),
                value=_convert(node.value),
            )
        if isinstance(target, cst.Attribute):
            return hir.HirFieldAssign(
                obj=_convert(target.value),
                field=target.attr.value,
                value=_convert(node.value),
            )
        if isinstance(target, cst.Tuple):
            return _tuple_assign(target, node.value)
        raise UnsupportedConstruct(f"assignment target {type(target).__name__}")

    if isinstance(node, cst.AnnAssign):
        if not isinstance(node.target, cst.Name):
            raise UnsupportedConstruct(
                f"annotated-assignment target {type(node.target).__name__}"
            )
        if node.value is None:
            raise UnsupportedConstruct("declaration without initializer")
        return hir.HirAssign(
            target=node.target.value,
            value=_convert(node.value),
            annotation=_annotation_text(node.annotation.annotation),
        )

    if isinstance(node, cst.AugAssign):
        if not isinstance(node.target, cst.Name):
            raise UnsupportedConstruct(
                f"aug-assignment target {type(node.target).__name__}"
            )
        return hir.HirAssign(
            target=node.target.value,
            value=_convert(node.value),
            annotation=None,
            augmented_op=_op_symbol(node.operator),
        )

    if isinstance(node, cst.Pass):
        return PassMarker()

    if isinstance(node, cst.Break):
        return hir.HirBreak()

    if isinstance(node, cst.Continue):
        return hir.HirContinue()

    if isinstance(node, cst.Expr):
        return _convert(node.value)

    if isinstance(node, cst.BinaryOperation):
        return hir.HirBinOp(
            op=_op_symbol(node.operator),
            left=_convert(node.left),
            right=_convert(node.right),
        )

    if isinstance(node, cst.Comparison):
        if len(node.comparisons) != 1:
            raise UnsupportedConstruct("chained comparison (a < b < c)")
        target = node.comparisons[0]
        return hir.HirCompare(
            op=_cmp_symbol(target.operator),
            left=_convert(node.left),
            right=_convert(target.comparator),
        )

    if isinstance(node, cst.BooleanOperation):
        op = "and" if isinstance(node.operator, cst.And) else "or"
        return hir.HirBoolOp(
            op=op, left=_convert(node.left), right=_convert(node.right)
        )

    if isinstance(node, cst.IfExp):
        # `<body> if <test> else <orelse>` → the shared `__ternary__` builtin
        # (test, then, else), which every backend already lowers to its native
        # conditional expression.
        return hir.HirCall(
            func="__ternary__",
            args=[_convert(node.test), _convert(node.body), _convert(node.orelse)],
        )

    if isinstance(node, cst.UnaryOperation):
        if isinstance(node.operator, cst.Not):
            op = "not"
        elif isinstance(node.operator, cst.Minus):
            op = "-"
        else:
            raise UnsupportedConstruct(f"unary op {type(node.operator).__name__}")
        return hir.HirUnaryOp(op=op, operand=_convert(node.expression))

    if isinstance(node, cst.Name):
        if node.value == "True":
            return hir.HirBoolLiteral(value=True)
        if node.value == "False":
            return hir.HirBoolLiteral(value=False)
        if node.value == "None":
            return hir.HirNullLiteral()
        return hir.HirName(name=node.value)

    if isinstance(node, cst.Integer):
        return hir.HirIntLiteral(value=int(node.value))

    if isinstance(node, cst.Float):
        return hir.HirFloatLiteral(value=float(node.value))

    if isinstance(node, cst.SimpleString):
        return hir.HirStringLiteral(value=_unquote(node.value))

    if isinstance(node, cst.Call):
        if isinstance(node.func, cst.Name):
            return hir.HirCall(
                func=node.func.value, args=[_convert(a.value) for a in node.args]
            )
        if isinstance(node.func, cst.Attribute):
            # `obj.method(args)` — method call.
            receiver = _convert(node.func.value)
            return hir.HirMethodCall(
                receiver=receiver,
                method=node.func.attr.value,
                args=[_convert(a.value) for a in node.args],
            )
        raise UnsupportedConstruct(f"call target {type(node.func).__name__}")
    if isinstance(node, cst.Attribute):
        # `obj.field` access at expression position.
        return hir.HirFieldAccess(value=_convert(node.value), field=node.attr.value)

    if isinstance(node, cst.List):
        return hir.HirList(elements=[_convert(e.value) for e in node.elements])

    if isinstance(node, cst.Subscript):
        if len(node.slice) != 1 or not isinstance(node.slice[0].slice, cst.Index):
            raise UnsupportedConstruct("slice / multi-index subscript")
        idx = node.slice[0].slice.value
        return hir.HirSubscript(value=_convert(node.value), index=_convert(idx))

    if isinstance(node, cst.With):
        # `with cm() as name: body` lowers to `name = cm(); body`. Context-
        # manager cleanup is dropped, but the `as` binding flows through so
        # the body actually parses with the variable in scope.
        if isinstance(node.body, cst.IndentedBlock):
            prelude: list[hir.HirNode] = []
            for item in node.items:
                ctx_expr = _convert(item.item)
                if item.asname is not None:
                    target = item.asname.name
                    if isinstance(target, cst.Name):
                        prelude.append(
                            hir.HirAssign(
                                target=target.value, value=ctx_expr, annotation=None
                            )
                        )
                # No `as`: the context expression is evaluated for side
                # effect; we drop it.
            body_stmts = [_convert(s) for s in node.body.body]
            return FlattenBlock(stmts=prelude + body_stmts)
    raise UnsupportedConstruct(f"{type(node).__name__}")


# Block-flatten markers live in transpilers.frontends._markers — both
# `with` desugar, tuple-unpacking, and the `pass` filter use them.


def _tuple_assign(target: cst.Tuple, value: cst.BaseExpression) -> hir.HirNode:
    """`a, b = expr1, expr2` (or LHS subscripts) → temps then writes.

    Each LHS slot may be a Name (HirAssign), Subscript (HirSubscriptAssign),
    or Attribute (HirFieldAssign). The RHS must be a parallel tuple literal.
    """
    targets = [e.value for e in target.elements if isinstance(e, cst.Element)]
    if not isinstance(value, cst.Tuple):
        raise UnsupportedConstruct("tuple unpacking from non-tuple RHS")
    rhs_exprs = [e.value for e in value.elements if isinstance(e, cst.Element)]
    if len(rhs_exprs) != len(targets):
        raise UnsupportedConstruct("tuple unpacking arity mismatch")
    # Stash each RHS in a fresh tmp first, then assign tmps to targets —
    # this models Python's "evaluate all RHS, then bind" exactly.
    tmps = [f"__xpile_swap_{i}" for i in range(len(targets))]
    stmts: list[hir.HirNode] = []
    for tmp, rhs in zip(tmps, rhs_exprs):
        stmts.append(hir.HirAssign(target=tmp, value=_convert(rhs), annotation=None))
    for slot, tmp in zip(targets, tmps):
        rhs = hir.HirName(name=tmp)
        if isinstance(slot, cst.Name):
            stmts.append(hir.HirAssign(target=slot.value, value=rhs, annotation=None))
        elif isinstance(slot, cst.Subscript):
            if len(slot.slice) != 1 or not isinstance(slot.slice[0].slice, cst.Index):
                raise UnsupportedConstruct("slice subscript in tuple unpacking")
            stmts.append(
                hir.HirSubscriptAssign(
                    obj=_convert(slot.value),
                    index=_convert(slot.slice[0].slice.value),
                    value=rhs,
                )
            )
        elif isinstance(slot, cst.Attribute):
            stmts.append(
                hir.HirFieldAssign(
                    obj=_convert(slot.value),
                    field=slot.attr.value,
                    value=rhs,
                )
            )
        else:
            raise UnsupportedConstruct(f"tuple unpacking slot {type(slot).__name__}")
    return FlattenBlock(stmts=stmts)


def _convert_function(fn: cst.FunctionDef) -> hir.HirFunction:
    sig = fn.params
    # Variadic forms can't be modelled in the typed subset — refuse loudly
    # rather than silently drop them (the prior behaviour for keyword-only
    # params, which produced a no-arg signature with a body using them).
    if isinstance(sig.star_arg, cst.Param):
        raise UnsupportedConstruct("*args variadic parameter")
    if sig.star_kwarg is not None:
        raise UnsupportedConstruct("**kwargs variadic parameter")
    # Positional-only, normal, and keyword-only params all become ordinary
    # typed params. Python's keyword-only call convention (the `*,` marker) is
    # relaxed to positional-or-keyword; the computation is identical and every
    # target backend renders a flat typed parameter list.
    cst_params = [
        *getattr(sig, "posonly_params", ()),
        *sig.params,
        *getattr(sig, "kwonly_params", ()),
    ]
    params = [_convert_param(p) for p in cst_params]
    ret = _annotation_text(fn.returns.annotation) if fn.returns else None
    body = _convert_block(fn.body)
    return hir.HirFunction(
        name=fn.name.value, params=params, return_annotation=ret, body=body
    )


def _convert_param(p: cst.Param) -> hir.HirParam:
    ann = _annotation_text(p.annotation.annotation) if p.annotation else None
    return hir.HirParam(name=p.name.value, annotation=ann)


def _convert_block(body: cst.BaseSuite) -> list[hir.HirNode]:
    if not isinstance(body, cst.IndentedBlock):
        raise UnsupportedConstruct("non-indented block")
    out: list[hir.HirNode] = []
    for stmt in body.body:
        node = _convert(stmt)
        if isinstance(node, FlattenBlock):
            out.extend(node.stmts)
            continue
        # Drop bare string-literal expression statements (docstrings):
        # no runtime effect but leak into emit as raw quoted text on
        # targets without a comment-statement form.
        if isinstance(node, hir.HirStringLiteral):
            continue
        if isinstance(node, PassMarker):
            continue
        out.append(node)
    return out


def _convert_if(node: cst.If) -> hir.HirIf:
    test = _convert(node.test)
    body = _convert_block(node.body)
    orelse: list[hir.HirNode] = []
    if node.orelse is not None:
        if isinstance(node.orelse, cst.If):
            # elif → nested HirIf
            orelse = [_convert_if(node.orelse)]
        elif isinstance(node.orelse, cst.Else):
            orelse = _convert_block(node.orelse.body)
        else:
            raise UnsupportedConstruct(f"if orelse {type(node.orelse).__name__}")
    return hir.HirIf(test=test, body=body, orelse=orelse)


def _is_call_named(node: cst.BaseExpression, name: str) -> bool:
    return (
        isinstance(node, cst.Call)
        and isinstance(node.func, cst.Name)
        and node.func.value == name
    )


def _convert_for(node: cst.For) -> hir.HirNode:
    if node.orelse is not None:
        raise UnsupportedConstruct("for-else clause")
    target, it = node.target, node.iter

    # `for <name> in range(...)` — the original indexed-range form, untouched.
    if isinstance(target, cst.Name) and _is_call_named(it, "range"):
        return hir.HirFor(
            target=target.value, iter=_convert(it), body=_convert_block(node.body)
        )

    # `for <index>, <value> in enumerate(<name>)` — indexed iteration.
    if isinstance(target, cst.Tuple) and _is_call_named(it, "enumerate"):
        elts = [e.value for e in target.elements if isinstance(e, cst.Element)]
        if len(elts) != 2 or not all(isinstance(e, cst.Name) for e in elts):
            raise UnsupportedConstruct(
                "enumerate target must be two names `index, value`"
            )
        eargs = [a.value for a in it.args]
        if len(eargs) != 1:
            raise UnsupportedConstruct("enumerate(seq, start) is not supported yet")
        if not isinstance(eargs[0], cst.Name):
            raise UnsupportedConstruct(
                "enumerate(<expr>): only a bare-name iterable is supported"
            )
        return hir.HirForEach(
            value_name=elts[1].value,
            iterable=eargs[0].value,
            index_name=elts[0].value,
            body=_convert_block(node.body),
        )

    # `for <value> in <name>` — plain foreach over a bare-name collection.
    if isinstance(target, cst.Name) and isinstance(it, cst.Name):
        return hir.HirForEach(
            value_name=target.value,
            iterable=it.value,
            index_name=None,
            body=_convert_block(node.body),
        )

    if isinstance(target, cst.Tuple):
        raise UnsupportedConstruct("for tuple target requires enumerate(<name>)")
    raise UnsupportedConstruct(
        f"for-loop over {type(it).__name__}: only range(...), a bare name, "
        "or enumerate(<name>) are supported"
    )


def _annotation_text(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Subscript):
        base = _annotation_text(node.value)
        parts = []
        for s in node.slice:
            if not isinstance(s.slice, cst.Index):
                raise UnsupportedConstruct("non-index subscript in annotation")
            parts.append(_annotation_text(s.slice.value))
        return f"{base}[{', '.join(parts)}]"
    raise UnsupportedConstruct(f"annotation {type(node).__name__}")


def _op_symbol(op: cst.BaseBinaryOp | cst.BaseAugOp) -> str:
    table = {
        cst.Add: "+",
        cst.Subtract: "-",
        cst.Multiply: "*",
        cst.Divide: "/",
        cst.Modulo: "%",
        cst.FloorDivide: "//",
        cst.Power: "**",
        cst.BitAnd: "&",
        cst.BitOr: "|",
        cst.BitXor: "^",
        cst.LeftShift: "<<",
        cst.RightShift: ">>",
        cst.AddAssign: "+",
        cst.SubtractAssign: "-",
        cst.MultiplyAssign: "*",
        cst.DivideAssign: "/",
        cst.FloorDivideAssign: "//",
        cst.ModuloAssign: "%",
        cst.PowerAssign: "**",
        cst.BitAndAssign: "&",
        cst.BitOrAssign: "|",
        cst.BitXorAssign: "^",
        cst.LeftShiftAssign: "<<",
        cst.RightShiftAssign: ">>",
    }
    for kls, sym in table.items():
        if isinstance(op, kls):
            return sym
    raise UnsupportedConstruct(f"op {type(op).__name__}")


def _cmp_symbol(op: cst.BaseCompOp) -> str:
    table = {
        cst.Equal: "==",
        cst.NotEqual: "!=",
        cst.LessThan: "<",
        cst.LessThanEqual: "<=",
        cst.GreaterThan: ">",
        cst.GreaterThanEqual: ">=",
        cst.Is: "==",
        cst.IsNot: "!=",  # `x is None` → `x == None` at MIR
    }
    for kls, sym in table.items():
        if isinstance(op, kls):
            return sym
    raise UnsupportedConstruct(f"comparison {type(op).__name__}")


def _unquote(literal: str) -> str:
    # libcst's SimpleString.value includes the surrounding quotes; strip them.
    # Real escape handling lives in a future pass — current subset is exact-text.
    if literal.startswith(("'''", '"""')):
        return literal[3:-3]
    return literal[1:-1]

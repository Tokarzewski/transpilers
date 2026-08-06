"""Shared precedence-aware emission for arithmetic/comparison/logical
binops across every target backend.

Each backend gives this helper:
  - `emit_expr`: the target's expression-emit function (recursive)
  - `op_of(node)`: returns the operator string if `node` is one of the
    target's BinOp/Compare/BoolOp nodes, otherwise None

The helper returns the rendered child string, optionally wrapped in
parentheses to preserve the original grouping.

The single shared precedence table tries to express the C/Python/most-
language consensus. Targets with unusual precedence (e.g. Fortran
`.and.`/`.or.` vs C `&&`/`||`) still fit because they only differ in
the spelling, not the precedence order.
"""

from __future__ import annotations

from typing import Callable


_PREC = {
    # Logical
    "or": 1,
    "||": 1,
    ".or.": 1,
    "and": 2,
    "&&": 2,
    ".and.": 2,
    # Equality / comparison (treat as the same band)
    "==": 3,
    "!=": 3,
    "/=": 3,
    "<": 3,
    "<=": 3,
    ">": 3,
    ">=": 3,
    ".eq.": 3,
    ".ne.": 3,
    ".lt.": 3,
    ".le.": 3,
    ".gt.": 3,
    ".ge.": 3,
    # Bitwise
    "|": 4,
    "^": 5,
    "&": 6,
    "<<": 7,
    ">>": 7,
    # Arithmetic
    "+": 8,
    "-": 8,
    "*": 9,
    "/": 9,
    "%": 9,
    "//": 9,
    # Concat / power (no current use but reserved)
    "**": 10,
    # Pseudo-op marker for unary; binds tighter than any binop so we
    # always parenthesize a non-atomic operand.
    "__unary__": 11,
    # Rust `as` cast — emitted as binop with op="as"; binds loosely.
    "as": 0,
}


def paren_emit(
    child,
    parent_op: str,
    *,
    on_right: bool,
    emit_expr: Callable,
    op_of: Callable,
) -> str:
    """Render `child` for inclusion inside a binop with `parent_op`.
    Adds parentheses iff the child's precedence requires them to
    preserve the original grouping. All emitted ops are left-associative,
    so equal-precedence on the right needs parens; on the left does not.
    """
    child_op = op_of(child)
    if child_op is None:
        return emit_expr(child)
    cp = _PREC.get(child_op, 0)
    pp = _PREC.get(parent_op, 0)
    if cp > pp:
        return emit_expr(child)
    if cp == pp and not on_right:
        return emit_expr(child)
    return f"({emit_expr(child)})"

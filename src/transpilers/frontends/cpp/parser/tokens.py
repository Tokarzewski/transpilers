"""Token / source-location / operator helpers (leaf utility)."""
from __future__ import annotations

import clang.cindex as ci

from transpilers.frontends.errors import UnsupportedConstruct

CursorKind = ci.CursorKind

def _strip_unexposed(cursor: ci.Cursor) -> ci.Cursor:
    """Recurse past UNEXPOSED_EXPR wrappers libclang inserts for things
    like implicit conversions — they hide the real operator kind."""
    while cursor.kind == CursorKind.UNEXPOSED_EXPR:
        inner = list(cursor.get_children())
        if not inner:
            return cursor
        cursor = inner[0]
    return cursor

COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}

ARITH_OPS = {"+", "-", "*", "/", "%"}

LOGICAL_OPS = {"&&", "||"}

ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}

def _decl_name(cursor: ci.Cursor) -> str | None:
    """Find the identifier name behind a DeclRefExpr / nested unwrapping."""
    if cursor.kind == CursorKind.DECL_REF_EXPR:
        return cursor.spelling
    if cursor.kind in (CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR):
        kids = list(cursor.get_children())
        if len(kids) == 1:
            return _decl_name(kids[0])
    return None

def _is_this_deref(cursor: ci.Cursor) -> bool:
    """True for `*this` (however parenthesized/unexposed-wrapped) -- the
    common "replace my whole value" idiom for a mutate-via-copy-assign
    method (`void Multiply(...) { (*this) = Multiplied(...); }`, OCCT's
    `gp_Quaternion` and common elsewhere). Used as an assignment LHS to
    recognize a self-reassignment that `_decl_name` can't name (there's no
    identifier behind a dereference)."""
    while cursor.kind in (CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR):
        kids = list(cursor.get_children())
        if len(kids) != 1:
            return False
        cursor = kids[0]
    if cursor.kind != CursorKind.UNARY_OPERATOR:
        return False
    kids = list(cursor.get_children())
    return len(kids) == 1 and kids[0].kind == CursorKind.CXX_THIS_EXPR

def _tokens_for(cursor: ci.Cursor) -> list:
    """*cursor*'s tokens, falling back to a widened whole-line retokenize
    when the direct extent yields nothing.

    libclang's tokenizer can return an *empty* result for a sub-expression
    extent that touches a command-line (`-D`) macro's expansion point --
    seen with `M_PI`-style constants (this frontend defines several via
    `-D` for `<cmath>` portability) used inline in a binary/unary
    expression, e.g. `M_PI - Angle(theOther)`. The same source range
    re-tokenized from a wider, macro-boundary-clear start succeeds fine, so
    retry from the start of the cursor's first line to the end of its last
    line before giving up -- callers still filter by exact token location,
    so widening the search window can't return a wrong token, only find
    one that direct tokenization missed."""
    toks = list(cursor.get_tokens())
    if toks:
        return toks
    try:
        tu = cursor.translation_unit
        f = cursor.extent.start.file
        if f is None:
            return []
        start = ci.SourceLocation.from_position(tu, f, cursor.extent.start.line, 1)
        end = ci.SourceLocation.from_position(tu, f, cursor.extent.end.line + 1, 1)
        return list(tu.get_tokens(extent=ci.SourceRange.from_locations(start, end)))
    except Exception:
        return []

def _literal_token(cursor: ci.Cursor):
    """The single token spelling out a literal cursor (INTEGER_LITERAL,
    FLOATING_LITERAL, ...), or None if it truly has none.

    A literal cursor's *own* extent can fail to tokenize directly under the
    same macro-boundary quirk `_tokens_for` works around (a command-line
    `-D`-defined constant like `M_PI` used as a literal): its narrow extent
    starts exactly at the macro expansion point. `_tokens_for`'s widened
    retokenize recovers the real token list in that case, but a widened
    list may contain unrelated tokens before the literal -- so match by the
    literal's own start location instead of taking "the first token"."""
    start = cursor.extent.start
    for tok in _tokens_for(cursor):
        if (tok.location.line, tok.location.column) == (start.line, start.column):
            return tok
    return None

def _binop_token(cursor: ci.Cursor) -> str:
    """The operator token sits between the two child cursors. We slice
    tokens by their source position to find it.

    BINARY_OPERATOR has no direct `.operator` accessor in older libclang
    bindings; we use tokens for portability."""
    kids = list(cursor.get_children())
    if len(kids) != 2:
        raise UnsupportedConstruct(f"binary operator with {len(kids)} children")
    left_end = kids[0].extent.end
    right_start = kids[1].extent.start
    for tok in _tokens_for(cursor):
        loc = tok.location
        if _loc_ge(loc, left_end) and _loc_lt(loc, right_start):
            return tok.spelling
    raise UnsupportedConstruct("could not locate binary-operator token")

def _unary_token(cursor: ci.Cursor) -> str:
    """For a unary op, the operator is either before or after the single
    operand. We pick whichever non-operand token we find first within the
    cursor's extent."""
    kids = list(cursor.get_children())
    if len(kids) != 1:
        raise UnsupportedConstruct(f"unary operator with {len(kids)} children")
    operand = kids[0]
    o_start, o_end = operand.extent.start, operand.extent.end
    for tok in _tokens_for(cursor):
        loc = tok.location
        if not (_loc_ge(loc, o_start) and _loc_lt(loc, o_end)):
            return tok.spelling
    raise UnsupportedConstruct("could not locate unary-operator token")

def _loc_ge(a: ci.SourceLocation, b: ci.SourceLocation) -> bool:
    return (a.line, a.column) >= (b.line, b.column)

def _loc_lt(a: ci.SourceLocation, b: ci.SourceLocation) -> bool:
    return (a.line, a.column) < (b.line, b.column)


__all__ = ['_strip_unexposed', 'COMPARE_OPS', 'ARITH_OPS', 'LOGICAL_OPS', 'ASSIGN_OPS', '_decl_name', '_is_this_deref', '_binop_token', '_unary_token', '_loc_ge', '_loc_lt', '_tokens_for', '_literal_token']

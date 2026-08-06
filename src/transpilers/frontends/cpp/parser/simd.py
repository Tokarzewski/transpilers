"""Lift Intel SIMD intrinsics (`_mm256_*`, ...) to semantic HIR operations."""

from __future__ import annotations

from transpilers.ir import hir

_SIMD_BINOP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "and": "&",
    "or": "|",
    "xor": "^",
}


def _lift_simd_intrinsic(name: str, args: list[hir.HirNode]) -> hir.HirNode | None:
    """Recognize Intel SIMD intrinsics and lift them to semantic HIR
    operations. Returns None if the name doesn't match an intrinsic
    pattern; the caller falls back to a plain HirCall."""
    if not name.startswith(("_mm_", "_mm128_", "_mm256_", "_mm512_")):
        return None
    parts = [p for p in name.split("_") if p]
    if len(parts) < 2:
        return None
    op = parts[1]
    if op in _SIMD_BINOP and len(args) == 2:
        return hir.HirBinOp(op=_SIMD_BINOP[op], left=args[0], right=args[1])
    if op == "set" and args:
        # Intel's `_mm256_set_pd(d, c, b, a)` reverses lane order.
        return hir.HirCall(func="simd_pack", args=list(reversed(args)))
    if op == "set1" and len(args) == 1:
        return hir.HirCall(func="simd_splat", args=args)
    if op == "setzero" and not args:
        return hir.HirCall(func="simd_zero", args=[])
    if op in ("sqrt", "abs", "ceil", "floor") and len(args) == 1:
        return hir.HirCall(func=f"simd_{op}", args=args)
    return None

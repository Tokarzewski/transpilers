"""Tests for interprocedural purity inference (`infer_contracts.py`).

`_propagate_interprocedural`'s MirCall case reads `callee.contract.pure` to
decide whether a call site is pure -- but until this fix, no function's own
`.contract` was ever written by this pass, so it stayed at the MirNode
default (WILDCARD, pure=False) forever regardless of what the function's
body actually did. Every callee looked impure unconditionally, so the
fixed-point loop in `infer_contracts()` converged instantly without ever
computing real interprocedural purity. There was no test coverage for this
module at all before this fix.
"""

from __future__ import annotations

from transpilers.frontends.python import parse_python
from transpilers.passes import hir_to_mir, infer_contracts, infer_types


def _contracts(source: str):
    hir_mod = parse_python(source)
    mir_mod = hir_to_mir(hir_mod)
    infer_types(mir_mod)
    infer_contracts(mir_mod)
    return {fn.name: fn.contract.pure for fn in mir_mod.functions}


def test_function_calling_only_pure_arithmetic_is_pure():
    src = (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "def caller(x: int) -> int:\n"
        "    return add(x, 1)\n"
    )
    purity = _contracts(src)
    assert purity["add"] is True
    # A prior version left every function at pure=False forever, since a
    # callee's contract was never actually computed from its own body.
    assert purity["caller"] is True


def test_function_calling_unknown_external_is_impure():
    src = "def caller(x: int) -> int:\n    return unknown_external(x)\n"
    purity = _contracts(src)
    assert purity["caller"] is False


def test_transitive_purity_propagates_through_call_chain():
    """f -> g -> pure leaf. Purity must propagate two hops through the
    fixed-point loop, not just one -- this is exactly what MAX_ITER exists
    for, and exactly what silently never happened before this fix."""
    src = (
        "def leaf(a: int) -> int:\n"
        "    return a + 1\n"
        "def middle(a: int) -> int:\n"
        "    return leaf(a)\n"
        "def outer(a: int) -> int:\n"
        "    return middle(a)\n"
    )
    purity = _contracts(src)
    assert purity == {"leaf": True, "middle": True, "outer": True}


def test_method_call_makes_function_impure():
    src = "def mutate(xs: list) -> None:\n    xs.append(1)\n"
    purity = _contracts(src)
    assert purity["mutate"] is False

"""Never-refuse pipeline tests.

A C++ function that mixes supported code with one unsupported construct must
still transpile to every target, with the unsupported part replaced by a
`TODO[port]` hole (the Raw node), instead of the whole function refusing.
"""

from __future__ import annotations

import textwrap

import pytest

from transpilers.cli.main import (
    transpile,
    transpile_cpp_to_mojo,
    transpile_cpp_to_rust,
)
from transpilers.frontends.cpp.parser.core import parse_cpp
from transpilers.ir import hir


# A function body mixing supported statements with an unsupported try/catch.
_TRY_SRC = textwrap.dedent(
    """
    int compute(int n) {
        int total = 0;
        try {
            total = n * 2;
        } catch (...) {
            total = -1;
        }
        total = total + 1;
        return total;
    }
    """
).lstrip()


# A function whose return value uses an unsupported construct (a lambda).
_LAMBDA_SRC = textwrap.dedent(
    """
    int run() {
        int base = 10;
        auto f = [](int x) { return x + 1; };
        return base;
    }
    """
).lstrip()


def test_frontend_emits_hir_raw_not_refuse():
    """The unsupported statement becomes a HirRaw hole; the rest survives."""
    parsed = parse_cpp(_TRY_SRC)
    # Issue #50: parse_cpp now returns (HirModule, TypeGroundTruth).
    module = parsed[0] if isinstance(parsed, tuple) else parsed
    fn = next(n for n in module.body if isinstance(n, hir.HirFunction))
    # The body keeps the supported statements (decl, += , return) and gains
    # exactly one HirRaw hole for the try/catch.
    raws = [n for n in fn.body if isinstance(n, hir.HirRaw)]
    assert len(raws) == 1
    assert "try" in raws[0].snippet and "catch" in raws[0].snippet
    # Supported statements are still present (the function did not abort).
    assert any(isinstance(n, hir.HirReturn) for n in fn.body)
    assert any(isinstance(n, hir.HirAssign) for n in fn.body)


def test_try_catch_transpiles_to_mojo_with_hole():
    out = transpile_cpp_to_mojo(_TRY_SRC)
    # Function shape and supported lines survive.
    assert "def compute(n: Int) -> Int:" in out
    assert "var total: Int = 0" in out
    assert "return total" in out
    # The unsupported construct is a TODO[port] hole, not a refusal.
    assert "TODO[port]" in out
    assert "pass  # TODO[port]" in out


def test_try_catch_transpiles_to_rust_with_hole():
    out = transpile_cpp_to_rust(_TRY_SRC)
    assert "fn compute(n: i64) -> i64" in out
    assert "let mut total: i64 = 0;" in out
    assert "return total;" in out
    assert "TODO[port]" in out
    assert "unimplemented!()" in out


@pytest.mark.parametrize(
    "target", ["mojo", "rust", "c", "fortran", "python"]
)
def test_never_refuse_reaches_all_targets(target):
    """One unsupported construct never aborts transpilation to any target."""
    out = transpile(_TRY_SRC, source_lang="cpp", target=target)
    assert "TODO[port]" in out
    # The supported `return total` / `total` name still appears.
    assert "total" in out


def test_unsupported_construct_does_not_abort_function():
    """The lambda construct yields a hole but the function still transpiles."""
    out = transpile_cpp_to_rust(_LAMBDA_SRC)
    assert "fn run() -> i64" in out
    assert "TODO[port]" in out
    # The supported declaration of `base` survives.
    assert "base" in out

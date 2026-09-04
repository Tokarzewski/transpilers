"""C frontend tests. Validates that a second source language flows through the
shared MIR / inference / LIR pipeline without any pipeline-level changes —
the architectural payoff of the three-tier IR."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import transpile_c_to_rust
from transpilers.verify import rust_compiles


def _rust(src: str) -> str:
    return transpile_c_to_rust(textwrap.dedent(src).lstrip())


# ---------- shape ----------


def test_c_add_to_rust():
    out = _rust("int add(int a, int b) { return a + b; }")
    assert "fn add(a: i64, b: i64) -> i64" in out
    # C int is fixed-width -> wrapping_* for safety from arbitrary-precision
    assert "wrapping_add(b)" in out


def test_c_if_else_to_rust():
    out = _rust(
        """
        int max2(int a, int b) {
            if (a > b) {
                return a;
            } else {
                return b;
            }
        }
        """
    )
    assert "if a > b {" in out
    assert "} else {" in out


def test_c_while_to_rust():
    out = _rust(
        """
        int factorial(int n) {
            int result = 1;
            int i = 1;
            while (i <= n) {
                result = result * i;
                i = i + 1;
            }
            return result;
        }
        """
    )
    assert "let mut result: i64 = 1;" in out
    assert "while i <= n {" in out


def test_c_for_loop_desugars_to_while():
    """C-style `for (init; cond; step)` desugars to init + while at the
    frontend, so the rest of the pipeline never sees C-style for."""
    out = _rust(
        """
        int sum_to(int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total = total + i;
            }
            return total;
        }
        """
    )
    assert "let mut total: i64 = 0;" in out
    assert "let mut i: i64 = 0;" in out
    assert "while i < n {" in out
    assert "i += 1;" in out


def test_c_logical_ops():
    out = _rust(
        """
        int in_range(int x, int lo, int hi) {
            return x >= lo && x <= hi;
        }
        """
    )
    assert "x >= lo && x <= hi" in out


def test_c_long_collapses_to_int():
    """`long` aliases onto `int` via C_TYPE_ALIASES."""
    out = _rust("long add(long a, long b) { return a + b; }")
    assert "fn add(a: i64, b: i64) -> i64" in out


def test_c_void_return():
    out = _rust(
        """
        void do_nothing() {
            return;
        }
        """
    )
    assert "fn do_nothing()" in out
    assert "return;" in out


# ---------- compile checks ----------


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc not installed")
@pytest.mark.parametrize(
    "src",
    [
        "int add(int a, int b) { return a + b; }",
        """
        int max2(int a, int b) {
            if (a > b) return a;
            else return b;
        }
        """,
        """
        int factorial(int n) {
            int result = 1;
            int i = 1;
            while (i <= n) {
                result = result * i;
                i = i + 1;
            }
            return result;
        }
        """,
        """
        int sum_to(int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total = total + i;
            }
            return total;
        }
        """,
    ],
)
def test_c_to_rust_compiles(src: str):
    out = _rust(src)
    result = rust_compiles(out)
    assert result.ok, f"rustc rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- cross-frontend: inference works for C too ----------


def test_interprocedural_inference_on_c():
    """C usually has annotations everywhere, but the interprocedural pass
    still runs and shouldn't break anything."""
    out = _rust(
        """
        int square(int x) {
            return x * x;
        }

        int sum_of_squares(int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total = total + square(i);
            }
            return total;
        }
        """
    )
    assert "fn square(x: i64) -> i64" in out
    assert "wrapping_mul(x)" in out  # C int -> wrapping arithmetic
    assert "wrapping_add(square(i))" in out

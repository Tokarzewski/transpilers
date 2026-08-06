"""Python -> Zig pipeline tests. Mirrors test_python_to_rust at smaller scope:
proves the LIR-family pattern (shared MIR, target-specific LIR/emit) works
for a second target."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import transpile_python_to_zig
from transpilers.verify import zig_compiles


def _t(src: str, **kwargs) -> str:
    return transpile_python_to_zig(textwrap.dedent(src).lstrip(), **kwargs)


def _zig() -> bool:
    return shutil.which("zig") is not None


# ---------- emission shape ----------


def test_zig_add():
    out = _t("def add(a: int, b: int) -> int:\n    return a + b\n")
    assert "fn add(a: i64, b: i64) i64" in out
    assert "return a + b;" in out


def test_zig_if_else():
    out = _t(
        """
        def max2(a: int, b: int) -> int:
            if a > b:
                return a
            else:
                return b
        """
    )
    assert "if (a > b) {" in out
    assert "} else {" in out


def test_zig_var_vs_const_inference():
    """Reassigned target -> `var`; single-assignment would be `const`. The
    existing mutability inference is shared with Rust via the MIR."""
    out = _t(
        """
        def factorial(n: int) -> int:
            result: int = 1
            i: int = 1
            while i <= n:
                result = result * i
                i = i + 1
            return result
        """
    )
    assert "var result: i64 = 1;" in out
    assert "var i: i64 = 1;" in out
    assert "while (i <= n) {" in out


def test_zig_for_range_uses_native_syntax():
    out = _t(
        """
        def sum_range(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """
    )
    assert "for (0..n) |i| {" in out


def test_zig_list_type_and_indexing():
    out = _t(
        """
        def sum_list(xs: list[int]) -> int:
            total: int = 0
            for i in range(len(xs)):
                total = total + xs[i]
            return total
        """
    )
    assert "xs: []const i64" in out
    assert "xs[@intCast(i)]" in out
    assert "xs.len" in out


# ---------- end-to-end compile ----------


@pytest.mark.skipif(not _zig(), reason="zig not installed")
@pytest.mark.parametrize(
    "src",
    [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        """
        def max2(a: int, b: int) -> int:
            if a > b:
                return a
            else:
                return b
        """,
        """
        def factorial(n: int) -> int:
            result: int = 1
            i: int = 1
            while i <= n:
                result = result * i
                i = i + 1
            return result
        """,
        """
        def sum_range(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """,
        """
        def in_range(x: int, lo: int, hi: int) -> bool:
            return x >= lo and x <= hi
        """,
    ],
)
def test_zig_compiles(src: str):
    out = _t(src)
    result = zig_compiles(out)
    assert result.ok, f"zig rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- inference shared across targets ----------


@pytest.mark.skipif(not _zig(), reason="zig not installed")
def test_inference_works_with_zig_target():
    """Same MIR feeds both Rust and Zig — verify the inference still resolves
    unannotated Python when targeting Zig."""
    out = _t(
        """
        def add_one(x):
            return x + 1
        """
    )
    assert "fn add_one(x: i64) i64" in out
    result = zig_compiles(out)
    assert result.ok, result.stderr


@pytest.mark.skipif(not _zig(), reason="zig not installed")
def test_interprocedural_inference_works_with_zig():
    out = _t(
        """
        def square(x):
            return x * x

        def total(n):
            sum_: int = 0
            for i in range(n):
                sum_ = sum_ + square(i)
            return sum_
        """
    )
    assert "fn square(x: i64) i64" in out
    assert "fn total(n: i64) i64" in out
    result = zig_compiles(out)
    assert result.ok, result.stderr

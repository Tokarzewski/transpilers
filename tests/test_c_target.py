"""C as a target — Python -> C and C -> C round-trip.

This is the system's fourth target dimension. C -> C in particular validates
that the architecture is lossless enough to round-trip a real source through
the IR and recover compilable output."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import transpile_c_to_c, transpile_python_to_c
from transpilers.verify import c_compiles


def _py(src: str) -> str:
    return transpile_python_to_c(textwrap.dedent(src).lstrip())


def _c(src: str) -> str:
    return transpile_c_to_c(textwrap.dedent(src).lstrip())


def _cc_available() -> bool:
    return (
        shutil.which("cc") is not None
        or shutil.which("gcc") is not None
        or shutil.which("clang") is not None
    )


def test_python_to_c_preamble():
    out = _py("def add(a: int, b: int) -> int:\n    return a + b\n")
    assert "#include <stdint.h>" in out
    assert "#include <stdbool.h>" in out
    assert "int64_t add(int64_t a, int64_t b)" in out


def test_python_for_range_emits_native_c_for():
    out = _py(
        """
        def sum_to(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """
    )
    assert "for (int64_t i = 0; i < n; i++)" in out


def test_c_to_c_round_trip():
    """The output should be clean idiomatic C, not a verbatim copy: declarations
    use int64_t, the body is semantically equivalent."""
    out = _c(
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
    assert "int64_t factorial(int64_t n)" in out
    assert "int64_t result = 1;" in out
    assert "while (i <= n)" in out


def test_python_unannotated_to_c_works_via_inference():
    out = _py(
        """
        def add_one(x):
            return x + 1
        """
    )
    assert "int64_t add_one(int64_t x)" in out
    assert "return x + 1;" in out


def test_c_void_return_round_trips():
    out = _c(
        """
        void do_nothing() {
            return;
        }
        """
    )
    assert "void do_nothing(void)" in out
    assert "return;" in out


def test_c_target_refuses_string_concat():
    """C strings need an allocator for concat (snprintf/asprintf); raise
    rather than emit broken code."""
    with pytest.raises(NotImplementedError, match="string concatenation in C"):
        _py(
            """
            def greet(name: str) -> str:
                return "hello, " + name
            """
        )


def test_c_target_lists_emit_slice():
    """`list[T]` lowers to one of the fixed slice typedefs (`slice_i64_t`
    for `list[int]`); subscript reads route through `.data[i]` and `len(xs)`
    becomes `(int64_t)xs.len`."""
    out = _py(
        """
        def first(xs: list[int]) -> int:
            return xs[0]
        """
    )
    assert "slice_i64_t" in out
    assert ".data[0]" in out


# ---------- compile checks ----------


def test_python_foreach_typed_binding_on_zero_inference_target():
    # The desugared `for x in xs` binding must carry the iterable's element
    # type on C — a target with no type inference. A type-blind binding would
    # surface here (`double x = ...` is the proof it didn't).
    out = _py(
        """
        def total_scaled(xs: list[float]) -> float:
            total: float = 0.0
            for x in xs:
                total = total + x * 2.0
            return total
        """
    )
    assert "double x = xs.data[" in out


def test_python_enumerate_uses_index_on_c():
    out = _py(
        """
        def index_weighted(xs: list[int]) -> int:
            acc: int = 0
            for i, x in enumerate(xs):
                acc = acc + x * i
            return acc
        """
    )
    assert "for (int64_t i = 0; i < (int64_t)xs.len; i++)" in out
    assert "int64_t x = xs.data[i];" in out


@pytest.mark.skipif(not _cc_available(), reason="no C compiler on PATH")
@pytest.mark.parametrize(
    "py_src",
    [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        """
        def total_scaled(xs: list[float]) -> float:
            total: float = 0.0
            for x in xs:
                total = total + x * 2.0
            return total
        """,
        """
        def index_weighted(xs: list[int]) -> int:
            acc: int = 0
            for i, x in enumerate(xs):
                acc = acc + x * i
            return acc
        """,
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
        def sum_to(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """,
    ],
)
def test_python_to_c_compiles(py_src: str):
    out = _py(py_src)
    result = c_compiles(out)
    assert result.ok, f"cc rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _cc_available(), reason="no C compiler on PATH")
@pytest.mark.parametrize(
    "c_src",
    [
        "int add(int a, int b) { return a + b; }",
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
def test_c_to_c_compiles(c_src: str):
    out = _c(c_src)
    result = c_compiles(out)
    assert result.ok, f"cc rejected:\n{out}\n\nstderr:\n{result.stderr}"

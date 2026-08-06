"""Fortran / Go / VB / Assembly frontends + Go / Python targets.

Closes the original goal list (Fortran, VB) and adds modern targets
(Go, Python). Assembly is intentionally stubbed — the test verifies the
refusal includes the architectural explanation."""

from __future__ import annotations

import os
import shutil
import textwrap

import pytest

from transpilers.cli.main import transpile
from transpilers.verify import go_compiles, python_compiles, rust_compiles


def _has(name: str) -> bool:
    return shutil.which(name) is not None


def _t(source_lang: str, src: str, target: str = "rust") -> str:
    return transpile(
        textwrap.dedent(src).lstrip(), source_lang=source_lang, target=target
    )


# ---------- Fortran ----------


def test_fortran_add_to_rust():
    out = _t(
        "fortran",
        """
        function add(a, b) result(r)
            integer :: a, b, r
            r = a + b
        end function
        """,
    )
    assert "fn add(a: i64, b: i64) -> i64" in out
    assert "return r;" in out


def test_fortran_do_iter_loop():
    """`do i = 0, n - 1` — inclusive endpoint adjusted to exclusive."""
    out = _t(
        "fortran",
        """
        function sum_to(n) result(total)
            integer :: n, total, i
            total = 0
            do i = 0, n - 1
                total = total + i
            end do
        end function
        """,
    )
    # `n - 1 + 1` becomes wrapping arithmetic due to arbitrary-precision int contract
    assert "wrapping_sub(1)).wrapping_add(1)" in out


def test_fortran_result_var_pre_declared():
    """The Fortran result variable needs an outer declaration so branches can
    assign to it without scoping it locally."""
    out = _t(
        "fortran",
        """
        function max2(a, b) result(r)
            integer :: a, b, r
            if (a > b) then
                r = a
            else
                r = b
            end if
        end function
        """,
    )
    # The synthesized `let mut r: i64 = 0;` lands at function-scope.
    assert "let mut r: i64 = 0;" in out
    assert "r = a;" in out
    assert "r = b;" in out


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_fortran_factorial_compiles():
    src = """
        function factorial(n) result(r)
            integer :: n, r, i
            r = 1
            i = 1
            do while (i <= n)
                r = r * i
                i = i + 1
            end do
        end function
        """
    out = _t("fortran", src)
    result = rust_compiles(out)
    assert result.ok, result.stderr


# ---------- Go (source) ----------


def test_go_function_to_rust():
    out = _t(
        "go",
        """
        package main
        func add(a int64, b int64) int64 {
            return a + b
        }
        """,
    )
    assert "fn add(a: i64, b: i64) -> i64" in out


def test_go_short_var_declaration():
    """Go's C-style for desugars at the frontend to init + while — same pattern
    as C/C++/Java. The target's emitter can re-emerge a for-range if it
    chooses, but here we're checking the source-side conversion."""
    out = _t(
        "go",
        """
        package main
        func sumTo(n int64) int64 {
            total := int64(0)
            for i := int64(0); i < n; i++ {
                total = total + i
            }
            return total
        }
        """,
    )
    assert "let mut total: i64 = 0;" in out
    assert "while i < n {" in out
    assert "i += 1;" in out


def test_go_while_via_for_cond():
    """Go's `for cond { }` is our `while`."""
    out = _t(
        "go",
        """
        package main
        func factorial(n int64) int64 {
            var result int64 = 1
            var i int64 = 1
            for i <= n {
                result = result * i
                i = i + 1
            }
            return result
        }
        """,
    )
    assert "while i <= n {" in out


# ---------- Go (target) ----------


def test_python_to_go_emission():
    out = _t(
        "python",
        """
        def add(a: int, b: int) -> int:
            return a + b
        """,
        target="go",
    )
    assert "package main" in out
    assert "func add(a int64, b int64) int64 {" in out


def test_python_to_go_for_loop_uses_int64_cast():
    """Loop var needs `int64(0)` for Go's strict type checks against int64 bounds."""
    out = _t(
        "python",
        """
        def sum_to(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """,
        target="go",
    )
    assert "for i := int64(0); i < n; i++" in out


@pytest.mark.skipif(not _has("go"), reason="go not installed")
def test_python_to_go_compiles():
    out = _t(
        "python",
        """
        def factorial(n: int) -> int:
            result: int = 1
            i: int = 1
            while i <= n:
                result = result * i
                i = i + 1
            return result
        """,
        target="go",
    )
    result = go_compiles(out)
    assert result.ok, result.stderr


# ---------- Python (target) ----------


def test_c_to_python_round_trip_shape():
    out = _t(
        "c",
        """
        int add(int a, int b) {
            return a + b;
        }
        """,
        target="python",
    )
    assert "def add(a: int, b: int) -> int:" in out
    assert "return a + b" in out


def test_python_to_python_round_trip_compiles():
    out = _t(
        "python",
        """
        def add(a: int, b: int) -> int:
            return a + b
        """,
        target="python",
    )
    result = python_compiles(out)
    assert result.ok, result.stderr


def test_python_target_first_assignment_carries_annotation():
    """Annotation lands only on the first occurrence; later assignments are bare."""
    out = _t(
        "python",
        """
        def f(n: int) -> int:
            x: int = 0
            x = x + n
            return x
        """,
        target="python",
    )
    # First-occurrence annotation present, second not.
    lines = [line.strip() for line in out.splitlines()]
    assert "x: int = 0" in lines
    # Aug-assign emission: `x = x + n` collapses to `x += n` in idiomatic Python.
    assert "x += n" in lines


# ---------- VB ----------


def test_vb_function_to_rust():
    out = _t(
        "vb",
        """
        Function Add(a As Integer, b As Integer) As Integer
            Return a + b
        End Function
        """,
    )
    assert "fn Add(a: i64, b: i64) -> i64" in out


def test_vb_while_and_for():
    out = _t(
        "vb",
        """
        Function Factorial(n As Integer) As Integer
            Dim result As Integer = 1
            Dim i As Integer = 1
            While i <= n
                result = result * i
                i = i + 1
            End While
            Return result
        End Function
        """,
    )
    assert "while i <= n {" in out


def test_vb_for_to_inclusive_endpoint():
    """`For i = 0 To n - 1` is inclusive both ends — converted to range(..., n)."""
    out = _t(
        "vb",
        """
        Function SumTo(n As Integer) As Integer
            Dim total As Integer = 0
            Dim i As Integer
            For i = 0 To n - 1
                total = total + i
            Next
            Return total
        End Function
        """,
    )
    # `n - 1 + 1` becomes wrapping arithmetic
    assert "wrapping_sub(1)).wrapping_add(1)" in out


# ---------- Assembly via Ghidra ----------


def test_asm_requires_binary_path():
    """The asm frontend expects a path to a binary, not raw assembly text —
    the actual decompilation runs through Ghidra. Verify the path check
    surfaces a clear error rather than silently doing the wrong thing."""
    from transpilers.frontends.asm.parser import UnsupportedConstruct

    with pytest.raises(UnsupportedConstruct, match="path to a binary"):
        _t("asm", "/this/path/does/not/exist")


def test_asm_pipeline_end_to_end():
    """If Ghidra is installed, decompile a tiny ELF and verify the pipeline
    produces parseable HIR. Skipped without Ghidra so CI doesn't depend on it."""
    from pathlib import Path

    if not Path("/opt/ghidra/support/analyzeHeadless").exists():
        pytest.skip("Ghidra not installed")
    if not shutil.which("cc"):
        pytest.skip("no C compiler")

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        c_src = Path(td) / "tiny.c"
        binary = Path(td) / "tiny"
        c_src.write_text(
            "int add(int a, int b) { return a + b; }\nint main() { return add(2, 3); }\n"
        )
        subprocess.run(
            ["cc", "-O0", "-fno-pic", "-no-pie", str(c_src), "-o", str(binary)],
            check=True,
        )

        # PyGhidra has a high cold-start cost; skip by default so `just check`
        # stays fast, but only when the caller hasn't explicitly asked for
        # it. A prior version of this skip fired unconditionally regardless
        # of Ghidra actually being installed (checked above), so the
        # Assembly frontend got zero coverage even in environments where
        # Ghidra is present and this would run quickly. Opt in with:
        #     RUN_SLOW_GHIDRA_TESTS=1 pytest tests/test_extra_languages.py -k asm_pipeline
        # or run the equivalent CLI command directly:
        #     transpile <binary> --source asm --target rust --verify
        if not os.environ.get("RUN_SLOW_GHIDRA_TESTS"):
            pytest.skip(
                "PyGhidra cold-start too slow for default test run; "
                "set RUN_SLOW_GHIDRA_TESTS=1 to run it (Ghidra is installed here)"
            )

        out = _t("asm", str(binary), target="rust")
        assert out.strip()
        assert "add" in out

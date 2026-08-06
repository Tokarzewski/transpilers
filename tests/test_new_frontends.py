"""Java, C#, TypeScript, and JavaScript frontends — all tree-sitter based.

These four frontends triple the source-language coverage and demonstrate that
the HIR/MIR/LIR pipeline accepts any reasonable source grammar without
pipeline-level changes. JS is the inference-stress-test (no annotations);
the others arrive with native types we can map directly."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import (
    transpile_csharp,
    transpile_java,
    transpile_javascript,
    transpile_typescript,
)
from transpilers.verify import mojo_compiles, rust_compiles


def _t(fn, src: str, target: str = "rust") -> str:
    return fn(textwrap.dedent(src).lstrip(), target=target)


def _has(name: str) -> bool:
    return shutil.which(name) is not None


# ---------- Java ----------


def test_java_add_to_rust():
    out = _t(
        transpile_java,
        """
        class M {
            static int add(int a, int b) {
                return a + b;
            }
        }
        """,
    )
    assert "fn add(a: i64, b: i64) -> i64" in out
    # Java int is fixed-width -> wrapping_* for safety
    assert "wrapping_add(b)" in out


def test_java_for_loop_desugars():
    out = _t(
        transpile_java,
        """
        class M {
            static int sumTo(int n) {
                int total = 0;
                for (int i = 0; i < n; i++) {
                    total = total + i;
                }
                return total;
            }
        }
        """,
    )
    assert "while i < n {" in out


def test_java_boolean_type():
    out = _t(
        transpile_java,
        """
        class M {
            static boolean positive(int x) {
                return x > 0;
            }
        }
        """,
    )
    assert "fn positive(x: i64) -> bool" in out


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_java_to_rust_compiles():
    src = """
        class M {
            static int factorial(int n) {
                int result = 1;
                int i = 1;
                while (i <= n) {
                    result = result * i;
                    i = i + 1;
                }
                return result;
            }
        }
        """
    out = _t(transpile_java, src)
    result = rust_compiles(out)
    assert result.ok, result.stderr


# ---------- C# ----------


def test_csharp_add_to_rust():
    out = _t(
        transpile_csharp,
        """
        class M {
            static int Add(int a, int b) {
                return a + b;
            }
        }
        """,
    )
    assert "fn Add(a: i64, b: i64) -> i64" in out


def test_csharp_for_loop_desugars():
    out = _t(
        transpile_csharp,
        """
        class M {
            static int F(int n) {
                int total = 0;
                for (int i = 0; i < n; i++) {
                    total = total + i;
                }
                return total;
            }
        }
        """,
    )
    assert "while i < n {" in out


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_csharp_to_rust_compiles():
    out = _t(
        transpile_csharp,
        """
        class M {
            static int Add(int a, int b) {
                return a + b;
            }
        }
        """,
    )
    result = rust_compiles(out)
    assert result.ok, result.stderr


# ---------- TypeScript ----------


def test_typescript_basic():
    out = _t(
        transpile_typescript,
        """
        function add(a: number, b: number): number {
            return a + b;
        }
        """,
    )
    assert "fn add(a: i64, b: i64) -> i64" in out


def test_typescript_strict_equality_collapses():
    """`===` / `!==` collapse onto `==` / `!=` in the IR."""
    out = _t(
        transpile_typescript,
        """
        function eq(a: number, b: number): boolean {
            return a === b;
        }
        """,
    )
    assert "a == b" in out


def test_typescript_for_loop():
    out = _t(
        transpile_typescript,
        """
        function sumTo(n: number): number {
            let total: number = 0;
            for (let i: number = 0; i < n; i++) {
                total = total + i;
            }
            return total;
        }
        """,
    )
    assert "while i < n {" in out


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_typescript_to_rust_compiles():
    out = _t(
        transpile_typescript,
        """
        function factorial(n: number): number {
            let result: number = 1;
            let i: number = 1;
            while (i <= n) {
                result = result * i;
                i = i + 1;
            }
            return result;
        }
        """,
    )
    result = rust_compiles(out)
    assert result.ok, result.stderr


# ---------- JavaScript ----------


def test_javascript_inference_drives_everything():
    """JS has no annotations — `addOne(x)` becomes `x: i64 -> i64` purely via
    the type-inference pass."""
    out = _t(
        transpile_javascript,
        """
        function addOne(x) {
            return x + 1;
        }
        """,
    )
    assert "fn addOne(x: i64) -> i64" in out


def test_javascript_recursion_via_interprocedural():
    out = _t(
        transpile_javascript,
        """
        function fact(n) {
            if (n <= 1) {
                return 1;
            }
            return n * fact(n - 1);
        }
        """,
    )
    assert "fn fact(n: i64) -> i64" in out


def test_javascript_for_loop_with_let():
    out = _t(
        transpile_javascript,
        """
        function sumTo(n) {
            let total = 0;
            for (let i = 0; i < n; i++) {
                total = total + i;
            }
            return total;
        }
        """,
    )
    assert "while i < n {" in out
    assert "fn sumTo(n: i64) -> i64" in out


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_javascript_to_rust_compiles():
    out = _t(
        transpile_javascript,
        """
        function fact(n) {
            if (n <= 1) {
                return 1;
            }
            return n * fact(n - 1);
        }
        """,
    )
    result = rust_compiles(out)
    assert result.ok, result.stderr


# ---------- cross-target: each new frontend reaches every backend ----------


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
@pytest.mark.parametrize(
    "fn,src",
    [
        (
            transpile_java,
            """
            class M {
                static int add(int a, int b) {
                    return a + b;
                }
            }
            """,
        ),
        (
            transpile_csharp,
            """
            class M {
                static int Add(int a, int b) {
                    return a + b;
                }
            }
            """,
        ),
        (
            transpile_typescript,
            """
            function add(a: number, b: number): number {
                return a + b;
            }
            """,
        ),
        (
            transpile_javascript,
            """
            function addOne(x) {
                return x + 1;
            }
            """,
        ),
    ],
)
def test_new_frontend_to_mojo_compiles(fn, src):
    out = _t(fn, src, target="mojo")
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"

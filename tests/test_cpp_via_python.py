"""Python-as-pivot tests: C++ → Python → <target>.

Routing through Python as a shared IR (the "Python-as-pivot" path validated by
the CodePivot paper, arXiv:2604.18027) replaces N×M direct language pairs with
N + M stages. These tests prove a simple function survives both stages for
multiple targets, and that the CLI ``--path python_pivot`` flag drives the same
path for any target."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import (
    main,
    transpile_cpp_to_python_to_mojo,
    transpile_cpp_via_python,
)
from transpilers.verify import mojo_compiles, rust_compiles


_ADD = "int add(int a, int b) { return a + b; }"


def _pivot(src: str, target: str) -> tuple[str, str]:
    return transpile_cpp_via_python(textwrap.dedent(src).lstrip(), target)


def _has(name: str) -> bool:
    return shutil.which(name) is not None


# ---------- shape: the Python pivot is a readable, explicit intermediate ----------


def test_pivot_produces_python_intermediate():
    python_ir, _ = _pivot(_ADD, "rust")
    assert "def add(a: int, b: int) -> int:" in python_ir
    assert "return a + b" in python_ir


# ---------- shape: pivot reaches multiple targets ----------


def test_cpp_via_python_to_rust():
    python_ir, out = _pivot(_ADD, "rust")
    assert "def add(a: int, b: int) -> int:" in python_ir
    assert "fn add(a: i64, b: i64) -> i64" in out
    # Arbitrary-precision int -> wrapping_* for safety
    assert "wrapping_add(b)" in out


def test_cpp_via_python_to_mojo():
    python_ir, out = _pivot(_ADD, "mojo")
    assert "def add(a: int, b: int) -> int:" in python_ir
    assert "def add(a: Int, b: Int) -> Int:" in out
    assert "return a + b" in out


@pytest.mark.parametrize(
    "target,needle",
    [
        ("c", "int64_t add(int64_t a, int64_t b)"),
        ("zig", "fn add(a: i64, b: i64) i64"),
        ("go", "func add(a int64, b int64) int64"),
        ("fortran", "function add(a, b) result(result_)"),
    ],
)
def test_cpp_via_python_to_other_targets(target: str, needle: str):
    """The pivot is target-agnostic: any backend Python can lower to works."""
    _, out = _pivot(_ADD, target)
    assert needle in out


# ---------- backward-compat wrapper ----------


def test_legacy_wrapper_matches_general_pivot():
    py_a, mojo_a = transpile_cpp_to_python_to_mojo(_ADD)
    py_b, mojo_b = _pivot(_ADD, "mojo")
    assert py_a == py_b
    assert mojo_a == mojo_b


# ---------- CLI flag drives the pivot for any target ----------


def test_cli_python_pivot_rust(tmp_path, capsys):
    src = tmp_path / "add.cpp"
    src.write_text(_ADD)
    rc = main([str(src), "--path", "python_pivot", "--target", "rust"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fn add(a: i64, b: i64) -> i64" in out


def test_cli_python_pivot_mojo(tmp_path, capsys):
    src = tmp_path / "add.cpp"
    src.write_text(_ADD)
    rc = main([str(src), "--path", "python_pivot", "--target", "mojo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "def add(a: Int, b: Int) -> Int:" in out


def test_cli_python_pivot_requires_cpp(tmp_path, capsys):
    src = tmp_path / "add.py"
    src.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    rc = main([str(src), "--path", "python_pivot", "--target", "rust"])
    assert rc == 2
    assert "requires a C++ source" in capsys.readouterr().err


# ---------- compile checks (gated on toolchain availability) ----------


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_cpp_via_python_to_rust_compiles():
    _, out = _pivot(_ADD, "rust")
    result = rust_compiles(out)
    assert result.ok, result.stderr


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_via_python_to_mojo_compiles():
    _, out = _pivot(_ADD, "mojo")
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"

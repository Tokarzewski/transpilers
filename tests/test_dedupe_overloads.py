"""Regression tests for `dedupe_overloads` (issue #80).

C++ allows overloading by signedness alone (`int` vs `unsigned int`), but
every backend here maps both to the same target scalar type, so two such
overloads previously emitted two methods with an *identical* signature in
the same struct -- a guaranteed duplicate-definition compile error in
Mojo/Rust/Zig (confirmed against the real Mojo compiler). Found stress-
testing against github.com/wassimj/Topologic's `Bitwise::NOT(int)` /
`Bitwise::NOT(unsigned int)`.
"""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import transpile_cpp_to_mojo, transpile_cpp_to_rust
from transpilers.verify import mojo_compiles, rust_compiles


def _mojo(src: str) -> str:
    return transpile_cpp_to_mojo(textwrap.dedent(src).lstrip())


def _rust(src: str) -> str:
    return transpile_cpp_to_rust(textwrap.dedent(src).lstrip())


_SIGNEDNESS_OVERLOAD_SRC = """
    class Bitwise {
    public:
        static int NOT(const int kArgument1);
        static unsigned int NOT(const unsigned int kArgument1);
    };
    int Bitwise::NOT(const int kArgument1) { return -kArgument1; }
    unsigned int Bitwise::NOT(const unsigned int kArgument1) { return -kArgument1; }
"""


def test_signedness_overload_renamed_not_duplicated():
    out = _mojo(_SIGNEDNESS_OVERLOAD_SRC)
    # NOT is declared `static` -- no `self`, called as Bitwise.NOT(x).
    assert "def NOT(kArgument1: Int) -> Int:" in out
    assert "def NOT_overload2(kArgument1: Int) -> Int:" in out
    assert out.count("def NOT(") == 1


@pytest.mark.skipif(shutil.which("mojo") is None, reason="mojo not installed")
def test_signedness_overload_mojo_actually_compiles():
    out = _mojo(_SIGNEDNESS_OVERLOAD_SRC)
    result = mojo_compiles(out)
    assert result.ok, result.stderr


def test_signedness_overload_renamed_in_rust_too():
    # Same fix, applied generically at the MIR level -- not Mojo-specific.
    out = _rust(_SIGNEDNESS_OVERLOAD_SRC)
    assert out.count("fn NOT(") == 1
    assert "fn NOT_overload2(" in out
    result = rust_compiles(out)
    assert result.ok, result.stderr


def test_distinct_methods_with_different_names_are_untouched():
    out = _mojo("""
        class Calc {
        public:
            int add(int a, int b) { return a + b; }
            int sub(int a, int b) { return a - b; }
        };
    """)
    assert "overload" not in out
    assert "def add(self, a: Int, b: Int) -> Int:" in out
    assert "def sub(self, a: Int, b: Int) -> Int:" in out

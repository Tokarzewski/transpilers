"""Regression tests for newly unblocked frontend constructs (issue #9).

One assertion per construct per frontend. Keeps the matrix metric
honest as parser coverage continues to grow.
"""

from __future__ import annotations

import textwrap


from transpilers.cli.main import transpile_python_to_rust


def _py_to_rust(src: str) -> str:
    return transpile_python_to_rust(textwrap.dedent(src).lstrip())


# ---------- Python frontend ----------


def test_py_pass_statement():
    out = _py_to_rust(
        """
        def noop():
            pass
        """
    )
    assert "fn noop()" in out
    assert "pass" not in out


def test_py_tuple_swap_names():
    out = _py_to_rust(
        """
        def swap(a: int, b: int) -> int:
            a, b = b, a
            return a
        """
    )
    assert "__xpile_swap_0" in out
    assert "__xpile_swap_1" in out


def test_py_tuple_swap_subscript_lhs():
    out = _py_to_rust(
        """
        def swap_at(xs: list[int], i: int, j: int) -> int:
            xs[i], xs[j] = xs[j], xs[i]
            return xs[0]
        """
    )
    assert "xs[(i) as usize]" in out
    assert "xs[(j) as usize]" in out


def test_py_power_operator():
    out = _py_to_rust(
        """
        def cube(x: int) -> int:
            return x ** 3
        """
    )
    assert "**" in out or "pow(" in out


def test_py_is_none_uses_null_literal_not_zero():
    """Regression: `x is None` used to lower to `x == 0`. After the
    HirNullLiteral fix it lowers to a comparison against a null
    sentinel (`None` in Rust), not the integer 0."""
    out = _py_to_rust(
        """
        def is_missing(x: int) -> bool:
            return x is None
        """
    )
    assert "None" in out
    # The comparison must not collapse to `== 0`.
    assert "== 0" not in out


# ---------- C frontend ----------


def test_c_block_comment_stripped():
    from transpilers.cli.main import transpile_c_to_rust

    src = textwrap.dedent(
        """
        /* preamble block comment
           with multiple lines */
        int add(int a, int b) {
            return a + b;
        }
        """
    ).lstrip()
    out = transpile_c_to_rust(src)
    assert "fn add" in out
    assert "preamble" not in out


def test_c_ternary_lowers_to_if_expr():
    from transpilers.cli.main import transpile_c_to_rust

    src = textwrap.dedent(
        """
        int max2(int a, int b) {
            return a > b ? a : b;
        }
        """
    ).lstrip()
    out = transpile_c_to_rust(src)
    # Should compile-style ternary, NOT a literal __ternary__ call.
    assert "__ternary__" not in out
    assert "if " in out


def test_c_array_ref():
    from transpilers.cli.main import transpile_c_to_rust

    src = textwrap.dedent(
        """
        int first(int *xs) {
            return xs[0];
        }
        """
    ).lstrip()
    out = transpile_c_to_rust(src)
    assert "[" in out


# ---------- Java frontend ----------


def test_java_ternary():
    from transpilers.cli.main import transpile_java as _java

    def transpile_java_to_rust(src):
        return _java(src, target="rust")

    src = textwrap.dedent(
        """
        class C {
            int max2(int a, int b) {
                return a > b ? a : b;
            }
        }
        """
    ).lstrip()
    out = transpile_java_to_rust(src)
    assert "__ternary__" not in out
    assert "if " in out


def test_java_null_literal_is_not_zero():
    """Java `null` used to lower to integer 0; now lowers via
    HirNullLiteral so reference comparisons stay semantically distinct."""
    from transpilers.cli.main import transpile_java as _java

    def transpile_java_to_rust(src):
        return _java(src, target="rust")

    src = textwrap.dedent(
        """
        class C {
            boolean isMissing(Integer x) {
                return x == null;
            }
        }
        """
    ).lstrip()
    out = transpile_java_to_rust(src)
    # We emit `None` as the null sentinel for Rust — not `0`.
    assert "None" in out


def test_java_array_initializer():
    from transpilers.cli.main import transpile_java as _java

    def transpile_java_to_rust(src):
        return _java(src, target="rust")

    src = textwrap.dedent(
        """
        class C {
            int first() {
                int[] xs = {1, 2, 3};
                return xs[0];
            }
        }
        """
    ).lstrip()
    out = transpile_java_to_rust(src)
    assert "vec![" in out or "[1," in out


def test_java_subscript_lhs():
    from transpilers.cli.main import transpile_java as _java

    def transpile_java_to_rust(src):
        return _java(src, target="rust")

    src = textwrap.dedent(
        """
        class C {
            void zeroFirst(int[] xs) {
                xs[0] = 0;
            }
        }
        """
    ).lstrip()
    out = transpile_java_to_rust(src)
    assert "xs[" in out


# ---------- C++ frontend ----------


def test_cpp_postfix_in_expr_defers_effect():
    """Issue #4: postfix ++/-- in expression context must NOT silently drop
    the side effect.  The old behaviour raised; the new behaviour defers the
    increment to a statement after the expression so the pre-increment value
    is used at the use site (correct post-increment semantics)."""
    from transpilers.cli.main import transpile

    src = textwrap.dedent(
        """
        int postfix_add(int i) {
            int x = i++;
            return x;
        }
        """
    ).lstrip()
    # Must not raise and must produce valid Python that returns the pre-increment value.
    out = transpile(src, source_lang="cpp", target="python")
    ns: dict = {}
    exec(compile(out, "<test>", "exec"), ns)  # noqa: S102
    fn = ns.get("postfix_add")
    assert fn is not None, f"function not found in:\n{out}"
    assert fn(3) == 3, f"expected pre-increment value 3, got {fn(3)}\n{out}"

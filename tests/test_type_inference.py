"""Type-inference pass tests.

Two layers under test:
  - algorithmic dataflow (no LLM): runs by default, resolves cases where an
    integer literal or known-typed expression anchors a binop or compare
  - LLM fallback (injected fake): exercised without API credentials so CI is
    deterministic and free
"""

from __future__ import annotations

import textwrap

import pytest

from transpilers.cli.main import transpile_python_to_rust
from transpilers.ir.types import BoolT, FloatT, IntT, ListT, NoneT, StrT, Type


def _t(src: str, **kwargs) -> str:
    return transpile_python_to_rust(textwrap.dedent(src).lstrip(), **kwargs)


# ---------- algorithmic only ----------


def test_param_inferred_from_int_literal_binop():
    """`x + 1` anchors `x: int`; return type follows."""
    out = _t(
        """
        def add_one(x):
            return x + 1
        """
    )
    assert "fn add_one(x: i64) -> i64" in out
    # Arbitrary-precision int -> wrapping arithmetic
    assert "wrapping_add(1)" in out


def test_local_var_inferred_from_literal_assignment():
    out = _t(
        """
        def thing():
            x = 0
            return x + x
        """
    )
    assert "fn thing() -> i64" in out
    assert "let mut x" in out or "let x" in out


def test_param_inferred_via_comparison():
    """Comparisons constrain both sides to the same type."""
    out = _t(
        """
        def positive(x):
            return x > 0
        """
    )
    assert "fn positive(x: i64) -> bool" in out
    assert "x > 0" in out


def test_for_range_target_typed_automatically():
    out = _t(
        """
        def f(n: int):
            total = 0
            for i in range(n):
                total = total + i
            return total
        """
    )
    assert "fn f(n: i64) -> i64" in out
    assert "for i in 0..n" in out


def test_range_arg_anchors_unannotated_param():
    """`for i in range(n)` constrains `n: int` even when `n` has no annotation."""
    out = _t(
        """
        def sum_to(n):
            total = 0
            for i in range(n):
                total = total + i
            return total
        """
    )
    assert "fn sum_to(n: i64) -> i64" in out


def test_ambiguous_list_return_types_stay_unresolved():
    """Two branches returning `list[int]` vs `list[str]` must NOT collapse
    to one type just because both are `ListT` at the top level. A prior
    version of `_resolve_return` compared `type(t) is type(first)`
    (same dataclass) instead of value equality, so `ListT(elem=IntT())`
    and `ListT(elem=StrT())` looked "the same" and it silently picked
    whichever branch was seen first -- a miscompilation (the function gets
    typed `-> list[i64]` even though the else-branch returns strings), not
    just an unfilled hole."""
    with pytest.raises(ValueError, match="unresolved type hole"):
        _t(
            """
            def pick(flag: bool):
                if flag:
                    return [1, 2]
                else:
                    return ["a", "b"]
            """
        )


def test_unresolvable_still_raises_without_llm():
    """`a + b` with no anchor can't be inferred — algorithmic path must surface
    the hole rather than guess. This is the contract."""
    with pytest.raises(ValueError, match="unresolved type hole"):
        _t(
            """
            def f(a, b):
                return a + b
            """
        )


# ---------- LLM fallback (injected fake) ----------


def test_llm_fallback_resolves_ambiguous_params():
    """Fake inferencer returns IntT for every hole. The pipeline should ride
    through to emission."""

    calls: list[tuple[str, dict]] = []

    def fake(name: str, ctx: dict) -> Type:
        calls.append((name, ctx))
        return IntT()

    out = _t(
        """
        def add(a, b):
            return a + b
        """,
        llm_fill=fake,
    )
    assert "fn add(a: i64, b: i64) -> i64" in out
    # Both params + return queried as separate holes.
    assert {c[0] for c in calls} >= {"a", "b", "__return__"}


def test_llm_fallback_can_choose_floats():
    """The fake returns FloatT, demonstrating the lattice is honored end-to-end."""
    out = _t(
        """
        def scale(x, factor):
            return x * factor
        """,
        llm_fill=lambda name, ctx: FloatT(),
    )
    assert "fn scale(x: f64, factor: f64) -> f64" in out


def test_llm_only_asked_for_residual_holes():
    """If algorithmic inference resolves everything, the LLM must not be called.
    The fake raises if invoked, asserting that property."""

    def must_not_call(name, ctx):
        raise AssertionError(f"LLM unexpectedly invoked for {name}")

    out = _t(
        """
        def add_one(x):
            return x + 1
        """,
        llm_fill=must_not_call,
    )
    assert "fn add_one(x: i64) -> i64" in out


def test_llm_response_validator_rejects_bad_type():
    """The production wiring's validator should fail closed on bad LLM output;
    test it directly via the parse function."""
    from transpilers.llm.inference import _parse_type

    with pytest.raises(ValueError, match="unknown type"):
        _parse_type('{"type": "complex"}')


def test_llm_response_validator_accepts_lattice():
    from transpilers.llm.inference import _parse_type

    cases = {
        '{"type": "int"}': IntT,
        '{"type": "float"}': FloatT,
        '{"type": "bool"}': BoolT,
        '{"type": "str"}': StrT,
        '{"type": "none"}': NoneT,
        '{"type": "list[int]"}': ListT,
    }
    for raw, expected_kind in cases.items():
        result = _parse_type(raw)
        assert isinstance(result, expected_kind), raw


# ---------- interprocedural ----------


def test_callee_signature_shapes_caller_return():
    """Forward propagation: caller's `return add_one(5)` picks up that
    add_one's inferred return is int, so caller's return is int."""
    out = _t(
        """
        def add_one(x):
            return x + 1

        def call_it():
            return add_one(5)
        """
    )
    assert "fn add_one(x: i64) -> i64" in out
    assert "fn call_it() -> i64" in out


def test_caller_arg_anchors_callee_param():
    """Backward propagation: `f` has nothing local to anchor its param, but a
    caller passes 5 — that pins `f`'s param to int."""
    out = _t(
        """
        def f(a):
            return a

        def g():
            return f(5)
        """
    )
    assert "fn f(a: i64) -> i64" in out
    assert "fn g() -> i64" in out


def test_recursion_converges():
    """Self-call resolves via fixed point: the `n - 1` arith anchors `n`, the
    `return 1` anchors return type, the recursive call then propagates."""
    out = _t(
        """
        def fact(n):
            if n <= 1:
                return 1
            return n * fact(n - 1)
        """
    )
    assert "fn fact(n: i64) -> i64" in out


def test_chain_of_calls_propagates_types():
    """A → B → C with no internal anchors anywhere — the only concrete type is
    the literal at the bottom of the chain; backward propagation walks it up."""
    out = _t(
        """
        def c(x):
            return x

        def b(x):
            return c(x)

        def a():
            return b(7)
        """
    )
    assert "fn c(x: i64) -> i64" in out
    assert "fn b(x: i64) -> i64" in out
    assert "fn a() -> i64" in out


def test_caller_callee_pair_compiles():
    """End-to-end: emitted Rust must actually compile."""
    import shutil

    if shutil.which("rustc") is None:
        pytest.skip("rustc not installed")
    from transpilers.verify import rust_compiles

    out = _t(
        """
        def add_one(x):
            return x + 1

        def call_it():
            return add_one(5)
        """
    )
    result = rust_compiles(out)
    assert result.ok, result.stderr

"""Tests for behavioral-equivalence verification (issue #48).

Compile-rate is not correctness; these tests assert the harness actually runs
the source oracle and the transpiled target and compares input→output behavior.

Python→Python checks are toolchain-free and always run. Python→Rust checks
compile a generated harness with ``rustc`` and are skipped when it is absent.
"""

from __future__ import annotations

import shutil

import pytest

from transpilers.verify.behavioral import (
    DIV_FLOORED_MOD,
    DIV_TARGET_ERROR,
    DIV_VALUE,
    BehavioralReport,
    Divergence,
    canonical_token,
    check_behavioral_equivalence,
    classify_divergence,
    generate_inputs,
    infer_param_tags,
    make_behavioral_verifier,
)

HAVE_RUST = shutil.which("rustc") is not None
rust_only = pytest.mark.skipif(not HAVE_RUST, reason="rustc not available")


# ---------------------------------------------------------------------------
# Input generation + signature inference
# ---------------------------------------------------------------------------


def test_generate_inputs_is_deterministic_and_staggered():
    a = generate_inputs(["int", "int"], n=6, seed=7)
    b = generate_inputs(["int", "int"], n=6, seed=7)
    assert a == b  # seeded → reproducible
    assert len(a) == 6
    # Staggered: at least one tuple has differing arguments (so arg-order
    # bugs are reachable).
    assert any(t[0] != t[1] for t in a)


def test_generate_inputs_includes_edge_cases():
    vals = {t[0] for t in generate_inputs(["int"], n=7, seed=1)}
    assert 0 in vals and 1 in vals and -1 in vals


def test_nullary_function_gets_one_call():
    assert generate_inputs([], n=5) == [()]


def test_infer_param_tags_from_annotations():
    src = "def f(a: int, b: float, c: bool, d: str, e: list[int]):\n    return a\n"
    assert infer_param_tags(src, "f") == ["int", "float", "bool", "str", "list[int]"]


def test_infer_param_tags_unannotated_defaults_int():
    assert infer_param_tags("def g(x, y):\n    return x\n", "g") == ["int", "int"]


def test_infer_param_tags_missing_function():
    assert infer_param_tags("def g():\n    return 1\n", "nope") is None


def test_canonical_token_normalizes_across_spellings():
    assert canonical_token(True, "bool") == "true"
    assert canonical_token(6, "int") == "6"
    assert canonical_token(1 / 3, "float") == "0.333333"
    assert canonical_token([1, 2], "list[int]") == "[1,2]"


# ---------------------------------------------------------------------------
# Python → Python (toolchain-free)
# ---------------------------------------------------------------------------

ADD = "def add(a, b):\n    return a + b\n"


def test_identical_code_is_equivalent():
    r = check_behavioral_equivalence(
        ADD, source_lang="python", target="python", target_code=ADD, func_name="add"
    )
    assert r.ok
    assert r.matched == r.total > 0
    assert r.pass_rate == 1.0


def test_divergent_target_is_caught():
    wrong = "def add(a, b):\n    return a - b\n"
    r = check_behavioral_equivalence(
        ADD, source_lang="python", target="python", target_code=wrong, func_name="add"
    )
    assert not r.ok
    assert r.matched < r.total
    assert r.divergences
    # The divergence carries the concrete input + expected/actual.
    d = r.divergences[0]
    assert len(d.args) == 2
    assert d.expected != d.actual


def test_float_tolerance_allows_tiny_differences():
    src = "def half(x):\n    return x / 2.0\n"
    # A target that adds a sub-tolerance epsilon still passes.
    tgt = "def half(x):\n    return x / 2.0 + 1e-9\n"
    r = check_behavioral_equivalence(
        src,
        source_lang="python",
        target="python",
        target_code=tgt,
        func_name="half",
        param_tags=["float"],
    )
    assert r.ok


def test_oracle_that_always_raises_yields_no_samples():
    src = "def boom(a, b):\n    raise ValueError('always')\n"
    r = check_behavioral_equivalence(
        src, source_lang="python", target="python", target_code=src, func_name="boom"
    )
    assert r.total == 0
    assert not r.ok
    assert "raised" in r.reason


def test_inputs_where_source_raises_are_dropped():
    # Source raises on b == 0; target matches everywhere it is defined.
    src = "def d(a, b):\n    return a // b\n"
    r = check_behavioral_equivalence(
        src, source_lang="python", target="python", target_code=src, func_name="d"
    )
    # Some inputs have b == 0 and are dropped, but the survivors all match.
    assert r.ok
    assert 0 < r.total < 12


def test_python_runner_times_out_on_infinite_loop(monkeypatch):
    """`PythonRunner.run()` execs untrusted-shaped source in-process with no
    subprocess boundary to kill. Without a wall-clock guard, a pathological
    input (here, an infinite loop) hangs the caller forever. The timeout is
    monkeypatched down so this test itself runs fast rather than actually
    waiting out the real 5s default."""
    import transpilers.verify.behavioral as behavioral

    monkeypatch.setattr(behavioral, "_PY_EXEC_TIMEOUT_S", 0.2)
    src = "def spin(a, b):\n    while True:\n        pass\n"
    samples = behavioral.PythonRunner().run(
        src, "spin", [(1, 2)], param_tags=["int", "int"], ret_tag="int"
    )
    assert len(samples) == 1
    assert samples[0].ok is False
    assert "timeout" in samples[0].error


def test_list_return_value_comparison():
    src = "def sq(xs):\n    return [x * x for x in xs]\n"
    bad = "def sq(xs):\n    return [x + x for x in xs]\n"
    ok = check_behavioral_equivalence(
        src,
        source_lang="python",
        target="python",
        target_code=src,
        func_name="sq",
        param_tags=["list[int]"],
    )
    assert ok.ok
    diff = check_behavioral_equivalence(
        src,
        source_lang="python",
        target="python",
        target_code=bad,
        func_name="sq",
        param_tags=["list[int]"],
    )
    assert not diff.ok


# ---------------------------------------------------------------------------
# Unsupported / missing target reporting (never a false divergence)
# ---------------------------------------------------------------------------


def test_non_python_source_is_unsupported_not_failed():
    r = check_behavioral_equivalence(
        "int add(int a){return a;}",
        source_lang="c",
        target="python",
        target_code="x",
        func_name="add",
    )
    assert r.supported is False
    assert r.total == 0


def test_unknown_target_is_unsupported():
    r = check_behavioral_equivalence(
        ADD, source_lang="python", target="haskell", target_code="x", func_name="add"
    )
    assert r.supported is False


# ---------------------------------------------------------------------------
# Python → Rust (requires rustc)
# ---------------------------------------------------------------------------


@rust_only
def test_python_to_rust_correct():
    rust = "fn add(a: i64, b: i64) -> i64 { a + b }"
    r = check_behavioral_equivalence(
        ADD, source_lang="python", target="rust", target_code=rust, func_name="add"
    )
    assert r.supported
    assert r.ok, r.summary()


@rust_only
def test_python_to_rust_divergence_caught():
    rust = "fn add(a: i64, b: i64) -> i64 { a * b }"
    r = check_behavioral_equivalence(
        ADD, source_lang="python", target="rust", target_code=rust, func_name="add"
    )
    assert not r.ok
    assert r.divergences


@rust_only
def test_python_to_rust_compile_failure_is_divergence():
    rust = "fn add(a: i64, b: i64) -> i64 { this is not rust }"
    r = check_behavioral_equivalence(
        ADD, source_lang="python", target="rust", target_code=rust, func_name="add"
    )
    assert not r.ok
    # A target that does not even build is a behavioral failure, not a crash.
    assert r.divergences


@rust_only
def test_python_to_rust_list_param_by_reference():
    # The transpiler emits collection params by reference (`& Vec<i64>`); the
    # harness must match that, not report a false divergence.
    src = (
        "def total(xs):\n    s = 0\n    for x in xs:\n        s = s + x\n    return s\n"
    )
    rust = "fn total(xs: & Vec<i64>) -> i64 { let mut s = 0i64; for x in xs { s = s + x; } s }"
    r = check_behavioral_equivalence(
        src,
        source_lang="python",
        target="rust",
        target_code=rust,
        func_name="total",
        param_tags=["list[int]"],
    )
    assert r.supported
    assert r.ok, r.summary()


@rust_only
def test_python_to_rust_bool_return():
    src = "def is_even(n):\n    return n % 2 == 0\n"
    rust = "fn is_even(n: i64) -> bool { n % 2 == 0 }"
    r = check_behavioral_equivalence(
        src,
        source_lang="python",
        target="rust",
        target_code=rust,
        func_name="is_even",
        param_tags=["int"],
    )
    assert r.ok, r.summary()


# ---------------------------------------------------------------------------
# Repair-loop adapter
# ---------------------------------------------------------------------------


def test_verifier_passes_on_match():
    verify = make_behavioral_verifier(
        ADD, source_lang="python", target="python", func_name="add"
    )
    outcome = verify(ADD)
    assert outcome.ok


def test_verifier_emits_run_mismatch_signal_on_divergence():
    verify = make_behavioral_verifier(
        ADD, source_lang="python", target="python", func_name="add"
    )
    outcome = verify("def add(a, b):\n    return a - b\n")
    assert not outcome.ok
    assert outcome.signal is not None
    assert outcome.signal.kind == "run_mismatch"
    assert outcome.expected and outcome.actual
    assert outcome.signal.input_diff  # carries the diverging input


def test_verifier_unsupported_signature_does_not_fail_by_default():
    # No rust toolchain assumption: use an unknown target → unsupported.
    verify = make_behavioral_verifier(
        ADD, source_lang="python", target="haskell", func_name="add"
    )
    assert verify(ADD).ok  # gate does not punish what it cannot drive
    strict = make_behavioral_verifier(
        ADD,
        source_lang="python",
        target="haskell",
        func_name="add",
        require_supported=True,
    )
    assert not strict(ADD).ok


def test_report_summary_is_human_readable():
    r = BehavioralReport(ok=False, total=10, matched=7, divergences=[])
    assert "7/10" in r.summary()


# ---------------------------------------------------------------------------
# Divergence classification (issue #48 slice: name the root cause)
# ---------------------------------------------------------------------------


def test_classify_floored_mod_divergence():
    # gcd(-1, 2): Python floored % gives 1; Rust/C truncated gives -1.
    # The token delta (1 - (-1) = 2) is a multiple of the divisor 2, and an
    # operand is negative -> the floored/truncated fingerprint.
    div = Divergence(args=(-1, 2), expected="1", actual="-1")
    assert classify_divergence(div, "int") == DIV_FLOORED_MOD


def test_classify_target_error_is_not_floored_mod():
    div = Divergence(args=(-1, 2), expected="1", actual="compile: error[E0428]")
    assert classify_divergence(div, "int") == DIV_TARGET_ERROR


def test_classify_no_result_is_target_error():
    div = Divergence(args=(1,), expected="2", actual="<no result>")
    assert classify_divergence(div, "int") == DIV_TARGET_ERROR


def test_classify_positive_operands_is_plain_value_mismatch():
    # No negative operand -> floored and truncated agree -> not the mod class.
    div = Divergence(args=(5, 2), expected="2", actual="3")
    assert classify_divergence(div, "int") == DIV_VALUE


def test_classify_non_int_return_is_value_mismatch():
    div = Divergence(args=(-1, 2), expected="true", actual="false")
    assert classify_divergence(div, "bool") == DIV_VALUE


def test_report_carries_divergence_class_on_int_mismatch():
    # Source uses Python floored //; a "truncated" target diverges on negatives.
    src = "def fdiv(a: int, b: int) -> int:\n    return a // b\n"
    wrong = (
        "def fdiv(a, b):\n"
        "    q = abs(a) // abs(b)\n"
        "    return q if (a < 0) == (b < 0) else -q\n"  # C-style truncation
    )
    rep = check_behavioral_equivalence(
        src,
        source_lang="python",
        target="python",
        target_code=wrong,
        func_name="fdiv",
    )
    assert not rep.ok
    assert rep.divergence_class == DIV_FLOORED_MOD
    assert DIV_FLOORED_MOD in rep.summary()

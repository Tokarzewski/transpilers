"""Failure-taxonomy tests (issue #42): exception, compiler-stderr and
run-result classification, plus the staged classify_unit driver."""

from __future__ import annotations

import subprocess
import textwrap

from transpilers.frontends.errors import UnsupportedConstruct
from transpilers.verify.taxonomy import (
    BUCKETS,
    classify_compile_stderr,
    classify_exception,
    classify_run,
    classify_unit,
)


def _t(src: str) -> str:
    return textwrap.dedent(src).lstrip()


# ---------- classify_exception ----------


def test_unsupported_construct_is_parse_bucket():
    bucket, construct = classify_exception(
        "parse", UnsupportedConstruct("chained comparison (a < b < c)")
    )
    assert bucket == "parse"
    assert "chained comparison" in construct


def test_syntax_error_is_parse_bucket():
    bucket, _ = classify_exception("parse", SyntaxError("invalid syntax"))
    assert bucket == "parse"


def test_frontend_parser_error_types_are_parse_bucket():
    # libcst / pycparser raise their own exception types — match by name.
    class ParserSyntaxError(Exception):
        pass

    class ParseError(Exception):
        pass

    assert (
        classify_exception("parse", ParserSyntaxError("Syntax Error @ 1:1."))[0]
        == "parse"
    )
    assert (
        classify_exception("parse", ParseError(":23:1: Invalid function definition"))[0]
        == "parse"
    )


def test_lowering_refusal_is_parse_bucket():
    bucket, construct = classify_exception(
        "hir-to-mir", NotImplementedError("HIR expr HirLambda")
    )
    assert bucket == "parse"
    assert construct == "HIR expr HirLambda"


def test_unresolved_type_hole():
    bucket, construct = classify_exception(
        "lower", ValueError("unresolved type hole: param a of f")
    )
    assert bucket == "unfilled-UnknownT-hole"
    assert construct == "param a of f"


def test_timeout_bucket():
    exc = subprocess.TimeoutExpired(cmd="rustc", timeout=60)
    assert classify_exception("compile", exc)[0] == "timeout"
    assert classify_exception("run", TimeoutError())[0] == "timeout"


def test_symbol_message_is_unresolved_symbol():
    bucket, _ = classify_exception(
        "lower", RuntimeError("unknown function `frobnicate`")
    )
    assert bucket == "unresolved-symbol"


def test_unexpected_exception_is_internal_error():
    bucket, construct = classify_exception("emit", KeyError("whoops"))
    assert bucket == "internal-error"
    assert construct.startswith("KeyError")


# ---------- classify_compile_stderr ----------


def test_rustc_type_error_is_type_inference_miss():
    stderr = "error[E0308]: mismatched types\n --> lib.rs:3:5\n"
    bucket, detail = classify_compile_stderr(stderr)
    assert bucket == "type-inference-miss"
    assert "E0308" in detail


def test_rustc_missing_symbol_is_unresolved_symbol():
    stderr = "error[E0425]: cannot find value `foo` in this scope\n"
    assert classify_compile_stderr(stderr)[0] == "unresolved-symbol"


def test_go_undefined_is_unresolved_symbol():
    assert (
        classify_compile_stderr("./main.go:5:2: undefined: fmt\n")[0]
        == "unresolved-symbol"
    )


def test_symbol_beats_type_when_both_present():
    stderr = (
        "error[E0425]: cannot find value `n` in this scope\n"
        "error[E0308]: mismatched types\n"
    )
    assert classify_compile_stderr(stderr)[0] == "unresolved-symbol"


def test_other_compiler_error_is_target_compile_error():
    bucket, detail = classify_compile_stderr("error: expected `;`, found `}`\n")
    assert bucket == "target-compile-error"
    assert "expected" in detail


# ---------- classify_run ----------


def test_matching_output_is_ok():
    assert classify_run("42\n", "42\n") == ("ok", "")


def test_output_mismatch_reports_first_diverging_line():
    bucket, detail = classify_run("1\n2\n3\n", "1\n9\n3\n")
    assert bucket == "output-mismatch"
    assert "line 2" in detail


def test_runtime_crash_is_output_mismatch():
    bucket, detail = classify_run(
        "42\n", "panicked at 'index out of bounds'", exit_ok=False
    )
    assert bucket == "output-mismatch"
    assert "runtime failure" in detail


# ---------- classify_unit (staged driver) ----------


def test_unit_ok():
    rec = classify_unit(
        _t("def add(a: int, b: int) -> int:\n    return a + b\n"),
        source_lang="python",
        target="rust",
        compile=False,
    )
    assert rec.bucket == "ok"
    assert "fn add" in rec.output


def test_unit_parse_failure():
    rec = classify_unit(
        _t("def f(a: int) -> bool:\n    return 0 < a < 10\n"),
        source_lang="python",
        target="rust",
        compile=False,
    )
    assert rec.bucket == "parse"
    assert rec.stage == "parse"
    assert "chained comparison" in rec.construct


def test_unit_syntax_error():
    rec = classify_unit("def f(:\n", source_lang="python", target="rust", compile=False)
    assert rec.bucket == "parse"
    assert rec.stage == "parse"


def test_unit_unfilled_hole():
    rec = classify_unit(
        _t("def f(a, b):\n    return a + b\n"),
        source_lang="python",
        target="rust",
        compile=False,
    )
    assert rec.bucket == "unfilled-UnknownT-hole"
    assert rec.stage == "lower"


def test_unit_structural_gate_passes_on_clean_unit():
    rec = classify_unit(
        _t(
            """
            def f(n: int) -> int:
                total = 0
                for i in range(n):
                    if i > 2:
                        total += i
                return total
            """
        ),
        source_lang="python",
        target="rust",
        compile=False,
        structural=True,
    )
    assert rec.bucket == "ok"


def test_all_buckets_are_known():
    # The bucket strings are reporting keys — lock them down.
    assert BUCKETS == (
        "ok",
        "parse",
        "unresolved-symbol",
        "unfilled-UnknownT-hole",
        "type-inference-miss",
        "target-compile-error",
        "output-mismatch",
        "structural-divergence",
        "timeout",
        "internal-error",
    )

"""Tests for the toolchain-free dataset generator (issue #57).

The generator runs the transpiler's algorithmic path over a Python corpus and
emits source->{mojo,rust,c} SFT pairs, behaviorally self-verifying what is
runnable in pure Python. These tests assert the pure-Python logic — record
schema, the entry-point stripper, the BaseException-based timeout guard, and
honest verified-flagging — without requiring any external compiler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "build_algorithm_pairs", REPO / "scripts/sft/build_algorithm_pairs.py"
)
gen = importlib.util.module_from_spec(_spec)
sys.modules["build_algorithm_pairs"] = gen
_spec.loader.exec_module(gen)


# --- entry-point stripping ------------------------------------------------


def test_strip_main_rust_removes_fn_main():
    code = "fn f(x: i64) -> i64 { x + 1 }\nfn main() {\n    f(1);\n}\n"
    out = gen._strip_main(code, "rust")
    assert "fn main" not in out
    assert "fn f" in out


def test_strip_main_rust_no_main_is_unchanged():
    code = "fn f(x: i64) -> i64 { x + 1 }\n"
    assert "fn f" in gen._strip_main(code, "rust")


def test_strip_main_python_removes_def_main():
    code = "def f(x):\n    return x\n\ndef main():\n    print(f(1))\n"
    out = gen._strip_main(code, "python")
    assert "def main" not in out
    assert "def f" in out


def test_strip_main_handles_nested_braces():
    code = "fn g() -> i64 { if true { 1 } else { 2 } }\nfn main() { g(); }\n"
    out = gen._strip_main(code, "rust")
    assert "fn main" not in out
    assert "fn g" in out


# --- public-function discovery -------------------------------------------


def test_public_funcs_skips_main():
    src = "def a():\n    return 1\ndef main():\n    pass\n"
    assert gen._public_funcs(src) == ["a"]


def test_public_funcs_handles_syntax_error():
    assert gen._public_funcs("def (") == []


# --- timeout guard --------------------------------------------------------


def test_verify_timeout_is_base_exception():
    # Must NOT be a subclass of Exception, or the harness's broad
    # ``except Exception`` would swallow the alarm.
    assert issubclass(gen._VerifyTimeout, BaseException)
    assert not issubclass(gen._VerifyTimeout, Exception)


def test_time_limit_interrupts_infinite_loop():
    with pytest.raises(gen._VerifyTimeout):
        with gen._time_limit(1):
            x = 0
            while True:  # never terminates until the alarm fires
                x += 1


# --- verification honesty -------------------------------------------------


def test_verify_marks_non_python_source_no_runner():
    status, note = gen._verify("int f() { return 1; }", "cpp", "mojo", "...")
    assert status == gen.NO_RUNNER
    assert "no in-env runner" in note


def test_verify_marks_undriveable_target_no_runner():
    src = "def f(x: int) -> int:\n    return x\n"
    status, _ = gen._verify(src, "python", "mojo", "def f(x): return x")
    assert status == gen.NO_RUNNER  # no in-env mojo runner


def test_verify_passes_on_behaviorally_equivalent_python():
    src = "def inc(x: int) -> int:\n    return x + 1\n"
    out = gen.transpile(src, source_lang="python", target="python").strip()
    status, note = gen._verify(src, "python", "python", out)
    assert status == gen.VERIFIED
    assert "behavioral match" in note


def test_verify_fails_on_divergent_target():
    src = "def inc(x: int) -> int:\n    return x + 1\n"
    wrong = "def inc(x):\n    return x - 1\n"  # off by design
    status, _ = gen._verify(src, "python", "python", wrong)
    assert status == gen.FAILED  # ran and diverged -> quarantined, not training


def test_three_statuses_are_distinct():
    assert len({gen.VERIFIED, gen.NO_RUNNER, gen.FAILED}) == 3


# --- end-to-end build over a tiny corpus ----------------------------------


def test_build_emits_pairs_with_expected_schema(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "inc.py").write_text(
        "def inc(x: int) -> int:\n    return x + 1\n\ndef main():\n    print(inc(1))\n"
    )
    mojo_pairs = gen.build(corpus, "python", "mojo")
    py_pairs = gen.build(corpus, "python", "python")
    rec = mojo_pairs[0]
    assert rec.source_lang == "python"
    assert rec.target == "mojo"
    assert rec.func == "inc"
    assert rec.output.startswith("```mojo")
    assert "inc" in rec.input
    # python target is in-env verifiable; mojo is not.
    assert py_pairs[0].verified is True
    assert mojo_pairs[0].verified is False
    assert mojo_pairs[0].status == gen.NO_RUNNER


def test_build_cpp_to_mojo_direction(tmp_path):
    corpus = tmp_path / "cpp"
    corpus.mkdir()
    (corpus / "gcd.cpp").write_text(
        "int gcd(int a, int b) {\n    while (b != 0) {\n        int t = b;\n"
        "        b = a % b;\n        a = t;\n    }\n    return a;\n}\n"
    )
    pairs = gen.build(corpus, "cpp", "mojo")
    assert len(pairs) == 1
    assert pairs[0].source_lang == "cpp"
    assert pairs[0].target == "mojo"
    assert pairs[0].output.startswith("```mojo")
    assert "def gcd" in pairs[0].output
    # cpp source has no in-env behavioral oracle -> kept but flagged.
    assert pairs[0].status == gen.NO_RUNNER


def test_record_includes_verified_bool(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "inc.py").write_text("def inc(x: int) -> int:\n    return x + 1\n")
    rec = gen.build(corpus, "python", "python")[0]
    import json

    d = json.loads(gen._record(rec))
    assert d["status"] == gen.VERIFIED
    assert d["verified"] is True
    assert d["target"] == "python"

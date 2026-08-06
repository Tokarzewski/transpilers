"""Tests for the record/replay verifier's pure pieces (issue #65).

Codegen + fixture parsing are tested without g++/Mojo. The live record→replay
round-trip is exercised by scripts/sft/record_replay.py on real items.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sft"))

import record_replay as rr  # noqa: E402


def test_fmt_forces_float_literals():
    assert rr._fmt(2) == "2.0"  # integer-looking -> float
    assert rr._fmt(0.5) == "0.5"
    assert "e" in rr._fmt(5.6697e-8)  # exponent preserved, already float-ish


def test_parse_fixtures_groups_by_dep_and_splits_args_ret():
    text = "G\t1\t2\t3\nG\t4\t5\t9\nH\t2\t4\n"
    fx = rr.parse_fixtures(text)
    assert fx["G"] == [((1.0, 2.0), 3.0), ((4.0, 5.0), 9.0)]
    assert fx["H"] == [((2.0,), 4.0)]  # 1-arg dep


def test_gen_replay_mojo_one_param_matches_recorded_return():
    src = rr.gen_replay_mojo("H", 1, [((2.0,), 4.0), ((3.0,), 9.0)])
    assert "def H(a0: Float64) raises -> Float64:" in src
    assert "var k0 = [2.0, 3.0]" in src
    assert "var rv = [4.0, 9.0]" in src
    assert "abs(a0 - k0[i])" in src
    assert "return rv[i]" in src


def test_gen_replay_mojo_two_params_ands_conditions():
    src = rr.gen_replay_mojo("G", 2, [((1.0, 2.0), 3.0)])
    assert "def G(a0: Float64, a1: Float64) raises" in src
    assert "var k0 = [1.0]" in src and "var k1 = [2.0]" in src
    assert "abs(a0 - k0[i])" in src and "and" in src and "abs(a1 - k1[i])" in src


def test_gen_replay_mojo_no_calls_raises():
    src = rr.gen_replay_mojo("G", 1, [])
    assert "no recorded calls" in src


def test_gen_recorder_cpp_wraps_dep_and_logs():
    item = {
        "name": "t",
        "inputs": [[1.0]],
        "f": {
            "name": "F",
            "params": 1,
            "cpp": "double F(double x){ return G(x); }",
            "mojo": "",
        },
        "deps": [
            {
                "name": "G",
                "params": 1,
                "cpp_impl": "double G_impl(double a){ return a*2.0; }",
            }
        ],
    }
    cpp = rr.gen_recorder_cpp(item)
    assert "double G_impl(double a){ return a*2.0; }" in cpp
    assert "double G(double a0){ double _r = G_impl(a0);" in cpp  # wrapper calls impl
    assert "fprintf(_fx," in cpp  # logs the call
    assert 'printf("%.17g\\n", F(1.0));' in cpp  # drives F on input


def test_typed_lit_bool_and_int_literals():
    assert rr._typed_lit(True) == "True"  # bool before int (bool subclass of int)
    assert rr._typed_lit(False) == "False"
    assert rr._typed_lit(3) == "3"  # int -> bare Int literal, no ".0"
    assert rr._typed_lit(-7) == "-7"
    assert rr._typed_lit(2.0) == "2.0"  # float still forced to float literal
    assert rr._typed_lit(2) == "2"  # int stays int (unlike _fmt's float coercion)


def test_parse_fixtures_parses_bool_cells():
    text = "P\t1\t2\tTrue\nP\t3\t4\tFalse\n"
    fx = rr.parse_fixtures(text)
    # args stay float, return is a real Python bool (exact-match key)
    assert fx["P"] == [((1.0, 2.0), True), ((3.0, 4.0), False)]
    assert fx["P"][0][1] is True and fx["P"][1][1] is False


def test_gen_replay_mojo_bool_return_exact_match():
    src = rr.gen_replay_mojo("P", 1, [((2.0,), True), ((3.0,), False)])
    assert "def P(a0: Float64) raises -> Bool:" in src  # bool return type
    assert "var rv = [True, False]" in src  # bool literals
    assert "abs(a0 - k0[i])" in src  # float arg keeps tolerance


def test_gen_replay_mojo_int_args_and_return_exact_match():
    src = rr.gen_replay_mojo("Q", 2, [((1, 2.0), 5), ((3, 4.0), 7)])
    assert "def Q(a0: Int, a1: Float64) raises -> Int:" in src
    assert "var k0 = [1, 3]" in src  # int column, no ".0"
    assert "var rv = [5, 7]" in src  # int return literals
    assert "a0 == k0[i]" in src  # int arg uses exact match
    assert "abs(a1 - k1[i])" in src  # float arg keeps tolerance


def test_gen_replay_program_includes_prelude_shims_and_driver():
    item = {
        "name": "t",
        "inputs": [[2.0]],
        "f": {
            "name": "F",
            "params": 1,
            "cpp": "",
            "mojo": "def F(x: Float64) raises -> Float64:\n    return G(x)",
        },
        "deps": [{"name": "G", "params": 1, "cpp_impl": ""}],
    }
    prog = rr.gen_replay_program(item, {"G": [((2.0,), 4.0)]})
    assert "def G(a0: Float64) raises" in prog  # replay shim present
    assert "def F(x: Float64)" in prog  # F present
    assert "print(F(2.0))" in prog  # driver present

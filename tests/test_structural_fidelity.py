"""Structural-fidelity verifier tests (issue #45): skeleton extraction,
isomorphism checking, allowed idioms, and the CLI verify-gate wiring."""

from __future__ import annotations

import textwrap

import pytest

from transpilers.ir import hir, lir
from transpilers.pipeline.stages import TARGETS, run_stages
from transpilers.verify.structural import (
    check_structural_fidelity,
    hir_skeleton,
    lir_skeleton,
)


def _trace(src: str, target: str = "rust"):
    return run_stages(
        textwrap.dedent(src).lstrip(), source_lang="python", target=target
    )


_CONTROL_FLOW_SRC = """
def classify(n: int) -> int:
    total = 0
    for i in range(0, n, 1):
        if i > 2:
            total += i
        else:
            total -= 1
    while total > 100:
        total -= 10
    return total

def helper(x: int) -> int:
    return x * 2
"""

_FOREACH_SRC = """
def total(xs: list[int]) -> int:
    s = 0
    for v in xs:
        s += v
    return s
"""


# ---------- skeleton extraction ----------


def test_hir_skeleton_captures_functions_and_shape():
    trace = _trace(_CONTROL_FLOW_SRC)
    sk = hir_skeleton(trace.hir)
    assert set(sk.functions) == {"classify", "helper"}
    assert sk.functions["helper"] == ()
    assert sk.functions["classify"] == (
        ("loop", (("if", (), ()),)),
        ("loop", ()),
    )


def test_lir_skeleton_matches_hir_skeleton():
    trace = _trace(_CONTROL_FLOW_SRC)
    assert lir_skeleton(trace.lir).functions == hir_skeleton(trace.hir).functions


# ---------- isomorphism across every target ----------


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_control_flow_skeleton_preserved(target):
    trace = _trace(_CONTROL_FLOW_SRC, target)
    report = check_structural_fidelity(trace.hir, trace.lir)
    assert report.ok, report.summary()


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_foreach_desugar_is_allowed_idiom(target):
    trace = _trace(_FOREACH_SRC, target)
    report = check_structural_fidelity(trace.hir, trace.lir)
    assert report.ok, report.summary()


# ---------- divergence detection (hand-built skeletons) ----------


def _hir_two_fns() -> hir.HirModule:
    return hir.HirModule(
        source_lang="python",
        body=[
            hir.HirFunction(
                "f",
                params=[],
                return_annotation=None,
                body=[
                    hir.HirWhile(test=hir.HirBoolLiteral(True), body=[]),
                ],
            ),
            hir.HirFunction("g", params=[], return_annotation=None, body=[]),
        ],
    )


def _rust_fn(name: str, body=()) -> lir.RustFn:
    return lir.RustFn(name=name, params=[], return_type="()", body=list(body))


def test_dropped_function_diverges():
    report = check_structural_fidelity(
        _hir_two_fns(),
        lir.RustModule(
            items=[
                _rust_fn("f", [lir.RustWhile(test=lir.RustBoolLiteral(True), body=[])])
            ]
        ),
    )
    assert not report.ok
    assert any(
        d.kind == "dropped-function" and d.where == "g" for d in report.divergences
    )


def test_added_function_diverges():
    report = check_structural_fidelity(
        _hir_two_fns(),
        lir.RustModule(
            items=[
                _rust_fn("f", [lir.RustWhile(test=lir.RustBoolLiteral(True), body=[])]),
                _rust_fn("g"),
                _rust_fn("sneaky_extra"),
            ]
        ),
    )
    assert not report.ok
    assert any(
        d.kind == "added-function" and d.where == "sneaky_extra"
        for d in report.divergences
    )


def test_flattened_control_flow_diverges():
    report = check_structural_fidelity(
        _hir_two_fns(),
        lir.RustModule(items=[_rust_fn("f"), _rust_fn("g")]),  # f's while got flattened
    )
    assert not report.ok
    assert any(
        d.kind == "control-flow-shape" and d.where == "f" for d in report.divergences
    )


def test_renamed_function_reports_drop_and_add():
    report = check_structural_fidelity(
        _hir_two_fns(),
        lir.RustModule(
            items=[
                _rust_fn("f", [lir.RustWhile(test=lir.RustBoolLiteral(True), body=[])]),
                _rust_fn("g_renamed"),
            ]
        ),
    )
    kinds = {d.kind for d in report.divergences}
    assert kinds == {"dropped-function", "added-function"}


# ---------- struct / method idioms ----------


def _hir_struct_module() -> hir.HirModule:
    return hir.HirModule(
        source_lang="cpp",
        body=[
            hir.HirStruct(
                name="Point",
                fields=[hir.HirParam("x", "int"), hir.HirParam("y", "int")],
                methods=[
                    hir.HirFunction(
                        "norm2",
                        params=[hir.HirParam("self", "Point")],
                        return_annotation="int",
                        body=[],
                    )
                ],
            )
        ],
    )


def test_rust_struct_impl_split_is_allowed():
    lir_mod = lir.RustModule(
        items=[
            lir.RustStruct(name="Point", fields=[("x", "i64"), ("y", "i64")]),
            lir.RustImpl(struct_name="Point", methods=[_rust_fn("norm2")]),
        ]
    )
    report = check_structural_fidelity(_hir_struct_module(), lir_mod)
    assert report.ok, report.summary()


def test_fortran_method_as_free_function_is_allowed():
    lir_mod = lir.FortranModule(
        items=[
            lir.FortranType(name="Point", fields=[("x", "integer")], methods=[]),
            lir.FortranFn(
                name="norm2",
                params=[("self", "type(Point)")],
                return_type="integer",
                result_name="result_",
                locals=[],
                body=[],
            ),
        ]
    )
    report = check_structural_fidelity(_hir_struct_module(), lir_mod)
    assert report.ok, report.summary()


def test_dropped_struct_diverges():
    report = check_structural_fidelity(_hir_struct_module(), lir.RustModule(items=[]))
    assert not report.ok
    assert any(
        d.kind == "dropped-struct" and d.where == "Point" for d in report.divergences
    )
    assert any(
        d.kind == "dropped-function" and d.where == "Point.norm2"
        for d in report.divergences
    )


# ---------- CLI verify-gate wiring (--fidelity) ----------


def test_cli_verify_runs_structural_gate(tmp_path, capsys):
    from transpilers.cli.main import main

    f = tmp_path / "prog.py"
    f.write_text(textwrap.dedent(_CONTROL_FLOW_SRC).lstrip())
    # python target: verification needs no external toolchain.
    assert main([str(f), "--target", "python", "--verify"]) == 0
    assert "[verify] ok" in capsys.readouterr().err


def test_cli_fidelity_idiomatic_accepted(tmp_path, capsys):
    from transpilers.cli.main import main

    f = tmp_path / "prog.py"
    f.write_text(textwrap.dedent(_CONTROL_FLOW_SRC).lstrip())
    assert (
        main([str(f), "--target", "python", "--verify", "--fidelity", "idiomatic"]) == 0
    )

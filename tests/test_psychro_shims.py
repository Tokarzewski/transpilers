"""Tests for the psychrometric dependency-boundary shim (issue #66).

Verifies the Python reference implementation of ``PsyPsatFnTemp`` against
well-known water saturation-pressure values, the C++-oracle branch boundaries,
and that the emitted Mojo/C++ shim text carries the same constants/branches.
All pure-Python — the Mojo↔C++ compile-match gate is deferred to a toolchain box.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "psychro_shims", REPO / "scripts/sft/psychro_shims.py"
)
ps = importlib.util.module_from_spec(_spec)
sys.modules["psychro_shims"] = ps
_spec.loader.exec_module(ps)


# --- numeric correctness vs well-known saturation pressures ---------------


@pytest.mark.parametrize(
    "temp_c, expected_pa, tol",
    [
        (0.0, 611.2, 1.0),  # triple-point-ish, textbook ~611 Pa
        (25.0, 3169.0, 5.0),  # ASHRAE ~3.17 kPa
        (100.0, 101325.0, 200.0),  # boiling at 1 atm
        (-10.0, 259.9, 1.0),  # over ice
    ],
)
def test_psat_matches_reference_values(temp_c, expected_pa, tol):
    assert ps.psy_psat_fn_temp(temp_c) == pytest.approx(expected_pa, abs=tol)


def test_psat_is_monotonic_increasing_in_range():
    vals = [ps.psy_psat_fn_temp(t) for t in range(-50, 200, 10)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_psat_low_branch_is_clamped_constant():
    # Tkel < 173.15  ->  T < -100 C  -> fixed C++ constant.
    assert ps.psy_psat_fn_temp(-150.0) == 0.001405102123874164


def test_psat_high_branch_is_clamped_constant():
    # Tkel > 473.15  ->  T > 200 C  -> fixed C++ constant.
    assert ps.psy_psat_fn_temp(250.0) == 1555073.745636215


def test_branch_boundary_continuity_around_triple_point():
    # The two exp branches meet near the triple point; values should be close.
    just_below = ps.psy_psat_fn_temp(0.005)  # Tkel just under 273.16
    just_above = ps.psy_psat_fn_temp(0.02)  # Tkel just over 273.16
    assert just_above > just_below
    assert abs(just_above - just_below) < 5.0  # ~Pa, no discontinuity jump


# --- shim codegen ---------------------------------------------------------


def test_mojo_shim_has_signature_and_constants():
    src = ps.mojo_shim()
    assert "fn PsyPsatFnTemp(T: Float64) -> Float64:" in src
    assert "exp(" in src and "log(" in src
    assert "0.001405102123874164" in src  # low clamp
    assert "1555073.745636215" in src  # high clamp
    assert str(ps.KELVIN) in src


def test_cpp_shim_has_signature_and_constants():
    src = ps.cpp_shim()
    assert "inline double PsyPsatFnTemp(double T)" in src
    assert "std::exp(" in src and "std::log(" in src
    assert "0.001405102123874164" in src
    assert "1555073.745636215" in src


def test_shims_share_the_same_coefficients():
    # Every saturation coefficient must appear verbatim in BOTH emitted shims so
    # they compile to the same numerics as the Python reference.
    mojo, cpp = ps.mojo_shim(), ps.cpp_shim()
    for c in (ps._C1, ps._C7, ps._C8, ps._C13):
        assert str(c) in mojo
        assert str(c) in cpp


def test_shim_is_real_not_the_pvstar_stub():
    # The old stub returned a fixed 611.65 for any T. The real shim must vary.
    assert ps.psy_psat_fn_temp(0.0) != ps.psy_psat_fn_temp(50.0)
    assert "611.65" not in ps.mojo_shim()

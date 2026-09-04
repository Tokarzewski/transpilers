"""Shared toolchain-path resolution (_tool.resolve_tool) — the Windows shim fix.

On Windows, ``shutil.which()`` finds ``.BAT``/``.CMD`` shims (e.g. pixi's
``mojo.bat`` launcher) that a bare-name ``subprocess`` call cannot launch:
``CreateProcess`` auto-appends only ``.exe``, so ``["mojo", ...]`` fails even
though ``which("mojo")`` succeeded. Resolving to the absolute which() path
makes the shim launchable (cmd.exe handles .BAT/.CMD) and is a no-op when
the tool is absent. Every native-compiler verify gate must therefore launch
its tool through ``resolve_tool`` — these tests pin that behavior.
"""

from __future__ import annotations

import os
import sys
import unittest.mock as mock

import pytest

from transpilers.verify import c as c_gate
from transpilers.verify import fortran as fortran_gate
from transpilers.verify import mojo as mojo_gate
from transpilers.verify import rust as rust_gate
from transpilers.verify import taxonomy
from transpilers.verify._tool import resolve_tool

MOCK = mock.MagicMock(returncode=0, stderr="", stdout="")


# --------------------------------------------------------------------------- #
# resolve_tool itself
# --------------------------------------------------------------------------- #


def test_resolve_tool_returns_which_path_when_found():
    # e.g. a pixi shim: which() sees it, a bare-name launch would not.
    with mock.patch(
        "transpilers.verify._tool.shutil.which",
        return_value=r"C:\pixi\bin\mojo.bat",
    ):
        assert resolve_tool("mojo") == r"C:\pixi\bin\mojo.bat"


def test_resolve_tool_falls_back_to_bare_name_when_absent():
    with mock.patch("transpilers.verify._tool.shutil.which", return_value=None):
        assert resolve_tool("rustc") == "rustc"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PATH lookup only")
def test_resolve_tool_finds_bat_shim_on_real_path(tmp_path):
    # End-to-end against a real shutil.which on the real PATH: a directory
    # containing only mojo.bat must be picked up and returned in full.
    shim = tmp_path / "mojo.bat"
    shim.write_text("@echo off\r\n", encoding="utf-8")
    with mock.patch.dict("os.environ", {"PATH": str(tmp_path)}):
        # which() may return the PATHEXT casing (.BAT), not the on-disk
        # casing — compare case-insensitively like the filesystem does.
        assert os.path.normcase(resolve_tool("mojo")) == os.path.normcase(str(shim))


# --------------------------------------------------------------------------- #
# Every gate launches through resolve_tool (argv[0] is the which() path)
# --------------------------------------------------------------------------- #


def _which_is(path):
    return mock.patch("transpilers.verify._tool.shutil.which", return_value=path)


def test_mojo_gate_launches_resolved_shim_path():
    with _which_is(r"C:\pixi\bin\mojo.bat"), mock.patch(
        "transpilers.verify.mojo.subprocess.run", return_value=MOCK
    ) as run, mock.patch(
        "transpilers.verify.mojo.mojo_available", return_value=True
    ):
        mojo_gate.mojo_compiles("def main():\n    pass\n")
    assert run.call_args.args[0][0] == r"C:\pixi\bin\mojo.bat"


def test_rust_gate_launches_resolved_path():
    with _which_is(r"C:\rustup\bin\rustc.exe"), mock.patch(
        "transpilers.verify.rust.subprocess.run", return_value=MOCK
    ) as run:
        rust_gate.rust_compiles("fn f() {}")
    assert run.call_args.args[0][0] == r"C:\rustup\bin\rustc.exe"


def test_c_gate_launches_resolved_path():
    with _which_is("/usr/bin/cc"), mock.patch(
        "transpilers.verify.c.subprocess.run", return_value=MOCK
    ) as run:
        c_gate.c_compiles("int f(void) { return 0; }")
    assert run.call_args.args[0][0] == "/usr/bin/cc"


def test_fortran_gate_launches_resolved_path():
    with _which_is("/usr/bin/gfortran"), mock.patch(
        "transpilers.verify.fortran.subprocess.run", return_value=MOCK
    ) as run:
        fortran_gate.fortran_compiles("subroutine f()\nend subroutine f\n")
    assert run.call_args.args[0][0] == "/usr/bin/gfortran"


def test_c_gate_reports_missing_compiler_not_bare_name():
    # With nothing on PATH, the C gate must report "no compiler found" —
    # resolve_tool's bare-name fallback must not leak into the launch args.
    with _which_is(None), mock.patch(
        "transpilers.verify.c.subprocess.run", return_value=MOCK
    ) as run:
        result = c_gate.c_compiles("int f(void) { return 0; }")
    assert not result.ok
    assert "no C compiler" in result.stderr
    assert not run.called


# --------------------------------------------------------------------------- #
# Availability checks stay honest about shims
# --------------------------------------------------------------------------- #


def test_taxonomy_mojo_branch_defers_to_working_toolchain_probe():
    # compiler_available("mojo") must use the --version probe, not a bare
    # which() that a name-collision shim would pass.
    with mock.patch(
        "transpilers.verify.mojo.mojo_available", return_value=True
    ) as probe:
        assert taxonomy.compiler_available("mojo") is True
    probe.assert_called_once()

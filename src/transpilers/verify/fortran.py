"""Verify emitted Fortran source compiles with gfortran."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from transpilers.verify._tool import resolve_tool


@dataclass
class CompileResult:
    ok: bool
    stderr: str


def _fortran_compiler() -> str | None:
    # resolve_tool returns the which() absolute path, or the bare name
    # unchanged when nothing matched — the comparison detects "not found".
    for name in ("gfortran", "flang"):
        if (path := resolve_tool(name)) != name:
            return path
    return None


def fortran_available() -> bool:
    return _fortran_compiler() is not None


def fortran_compiles(source: str) -> CompileResult:
    compiler = _fortran_compiler()
    if compiler is None:
        return CompileResult(ok=False, stderr="no Fortran compiler on PATH")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "lib.f90"
        src.write_text(source)
        out = subprocess.run(
            [compiler, "-c", "-ffree-form", str(src), "-o", str(Path(td) / "out.o")],
            capture_output=True,
            text=True,
            cwd=td,
            timeout=30,
        )
        return CompileResult(ok=out.returncode == 0, stderr=out.stderr)

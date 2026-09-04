"""Verify emitted Mojo source compiles."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from transpilers.verify._tool import resolve_tool


@dataclass
class CompileResult:
    ok: bool
    stderr: str


@lru_cache(maxsize=1)
def mojo_available() -> bool:
    """True only if a *working* mojo toolchain is on PATH.

        ``shutil.which`` alone is not enough: an unrelated program that happens
    to be named ``mojo`` (a name-collision shim) passes the PATH check but
    fails at launch or answers ``--version`` with something else. Probe once
    and cache — a real toolchain prints a version banner and exits 0.
    """
    exe = shutil.which("mojo")
    if exe is None:
        return False
    try:
        probe = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=15
        )
    except OSError, subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0 and "mojo" in (probe.stdout or "").lower()


def mojo_compiles(source: str) -> CompileResult:
    if not mojo_available():
        return CompileResult(ok=False, stderr="mojo not found on PATH")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "lib.mojo"
        src.write_text(source)
        # `mojo build` requires a main(); for library-style code we use
        # `mojo build -o` writing to /dev/null is awkward, so we wrap the
        # file's defs by adding a trivial main if absent. The shorter and
        # more robust path is `mojo run` against a wrapper that imports
        # — but for emit-level testing we just type-check by building.
        # Trick: append a tiny `def main(): pass` so the file is buildable.
        if "def main" not in source:
            (src).write_text(source + "\ndef main():\n    pass\n")
        out = subprocess.run(
            [resolve_tool("mojo"), "build", str(src), "-o", str(Path(td) / "out")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return CompileResult(ok=out.returncode == 0, stderr=out.stderr)

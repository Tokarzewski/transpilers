"""Verify emitted C source compiles cleanly with the system C compiler."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileResult:
    ok: bool
    stderr: str


def _cc() -> str | None:
    for candidate in ("cc", "gcc", "clang"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def c_compiles(source: str) -> CompileResult:
    cc = _cc()
    if cc is None:
        return CompileResult(ok=False, stderr="no C compiler found on PATH")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "lib.c"
        src.write_text(source)
        out = subprocess.run(
            [
                cc,
                "-c",
                "-std=c11",
                "-Wall",
                "-Werror=implicit-function-declaration",
                str(src),
                "-o",
                str(Path(td) / "out.o"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return CompileResult(ok=out.returncode == 0, stderr=out.stderr)

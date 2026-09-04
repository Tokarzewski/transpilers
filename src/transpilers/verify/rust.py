"""Verify emitted Rust source actually compiles. Non-negotiable for any LLM-touched output."""

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


def rust_compiles(source: str) -> CompileResult:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "lib.rs"
        src.write_text(source)
        out = subprocess.run(
            [
                resolve_tool("rustc"),
                "--crate-type",
                "lib",
                "--edition",
                "2021",
                str(src),
                "-o",
                str(Path(td) / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return CompileResult(ok=out.returncode == 0, stderr=out.stderr)

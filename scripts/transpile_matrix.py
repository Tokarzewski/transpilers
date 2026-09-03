"""Run every supported-extension file under a directory tree through
the transpiler and (optionally) the target's compiler, then print a
PASS/FAIL matrix.

Status values
-------------

- ``XPILE-FAIL``: transpile errored (parser/MIR/LIR refused the input).
- ``BUILD-FAIL``: transpile produced output, but the target compiler
  rejected it. Surfaces silent emit bugs (e.g., dead-call ternary
  literals) that the old ``PASS`` metric hid.
- ``BUILD-OK``:   transpile + compile both succeeded.

Use ``--no-compile`` to skip the compile step and fall back to the
previous transpile-only behavior. Targets we don't know how to compile
(every target except ``rust`` and ``c`` today) silently use the
transpile-only path.

Usage:
    uv run python scripts/transpile_matrix.py <root> [<target>] [--no-compile]
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


EXT_MAP = {
    "c": "c", "cpp": "cpp", "cs": "csharp", "f90": "fortran",
    "java": "java", "js": "javascript", "py": "python",
    "ts": "typescript", "vb": "vb",
}

_CPP_EXTS = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"}
_PY_EXTS = {".py"}


def _topo_ordered(root: pathlib.Path, files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return files in dependency order (callees before callers) via the call graph."""
    try:
        _src = pathlib.Path(__file__).resolve().parent.parent / "src"
        import sys
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from transpilers.graph.code_graph import file_topological_order
        cpp_count = sum(1 for f in files if f.suffix in _CPP_EXTS)
        py_count = sum(1 for f in files if f.suffix in _PY_EXTS)
        lang = "cpp" if cpp_count >= py_count else "python"
        ordered = file_topological_order(root, lang=lang)
        ordered_strs = {str(f) for f in ordered}
        tail = [f for f in files if str(f) not in ordered_strs]
        return ordered + tail
    except Exception:
        return files


def _compile_rust(src: str, td: pathlib.Path) -> tuple[bool, str]:
    p = td / "lib.rs"
    p.write_text(src)
    proc = subprocess.run(
        ["rustc", "--edition", "2021", "--crate-type", "lib",
         str(p), "-o", str(td / "lib")],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode == 0, proc.stderr


def _compile_c(src: str, td: pathlib.Path) -> tuple[bool, str]:
    p = td / "lib.c"
    p.write_text(src)
    proc = subprocess.run(
        ["gcc", "-std=c11", "-c", "-Wno-implicit-function-declaration",
         str(p), "-o", str(td / "lib.o")],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode == 0, proc.stderr


COMPILERS = {
    "rust": _compile_rust,
    "c": _compile_c,
}


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--no-compile"]
    compile_outputs = "--no-compile" not in argv[1:]
    root = pathlib.Path(args[0] if args else "examples/speed-comparison")
    target = args[1] if len(args) > 1 else "rust"
    results: list[tuple[str, str, str]] = []
    compiler = COMPILERS.get(target) if compile_outputs else None

    all_files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and EXT_MAP.get(f.suffix.lstrip("."))
    ]
    ordered_files = _topo_ordered(root, all_files)

    for f in ordered_files:
        src = EXT_MAP.get(f.suffix.lstrip("."))
        if not src:
            continue
        proc = subprocess.run(
            ["uv", "run", "transpile", str(f), "--source", src, "--target", target],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout).strip().splitlines()
            first_err = next((l for l in stderr[::-1] if l.strip()), "")[:80]
            status = f"XPILE-FAIL  {first_err}"
        elif compiler is None:
            status = "PASS"
        else:
            with tempfile.TemporaryDirectory() as td:
                ok, err = compiler(proc.stdout, pathlib.Path(td))
            if ok:
                status = "BUILD-OK"
            else:
                first_err = next((l for l in err.strip().splitlines()[::-1] if l.strip()), "")[:80]
                status = f"BUILD-FAIL  {first_err}"
        results.append((str(f.relative_to(root)), src, status))

    print(f"{'file':<32} {'source':<11} status")
    print("-" * 110)
    for name, src, status in results:
        print(f"{name:<32} {src:<11} {status}")

    if compiler is None:
        passes = sum(1 for _, _, s in results if s == "PASS")
        print(f"\n{passes} / {len(results)} files transpile to {target}")
    else:
        builds = sum(1 for _, _, s in results if s == "BUILD-OK")
        xpiles = sum(1 for _, _, s in results if not s.startswith("XPILE-FAIL"))
        print(
            f"\n{xpiles} / {len(results)} transpile to {target} ; "
            f"{builds} / {len(results)} also compile"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

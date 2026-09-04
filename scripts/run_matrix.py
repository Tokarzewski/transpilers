"""End-to-end correctness check: transpile a Python program to every
target, compile and run each, then verify all targets produce the same
stdout the reference Python program produced.

Usage:
    uv run python scripts/run_matrix.py <root>

The script walks the tree under <root>, takes every `*.py` file that
defines a `main()` callable from the standard `def main():` form,
transpiles it through each supported target, compiles+runs it, and
prints a matrix of `PASS / FAIL(<reason>) / SKIP` per (file, target).

Only failures with a non-matching stdout are real bugs; a compile
failure on a target with a known feature gap is reported as such and
isn't a regression unless the target previously passed.
"""

from __future__ import annotations

import pathlib
import shlex
import subprocess
import sys
import tempfile


WRAP_MAIN = {
    "rust": "\nfn main_entry() {{ {body} }}\n",
}


def run_python(path: pathlib.Path) -> tuple[bool, str]:
    # The corpus files define `main()` but don't call it at module scope.
    # Append a top-level call so the reference run produces output to
    # compare against. The transpilers do this implicitly: Rust/C
    # treat `main` as the entry point, Mojo/Python need the explicit call
    # — they pick it up from the Builder's wrapper too.
    src = path.read_text() + "\n\nmain()\n"
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True, timeout=30,
    )
    return proc.returncode == 0, (proc.stdout if proc.returncode == 0 else proc.stderr)


def transpile(path: pathlib.Path, target: str) -> tuple[bool, str]:
    proc = subprocess.run(
        ["uv", "run", "transpile", str(path), "--source", "python", "--target", target],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode == 0, (proc.stdout if proc.returncode == 0 else proc.stderr)


def _build_and_run_rust(src: str, td: pathlib.Path) -> tuple[bool, str]:
    main_path = td / "main.rs"
    # If `main` already exists in the transpiled source, use it directly.
    # Otherwise we'd need a wrapper that calls the entry point — for the
    # algorithm corpus, every example defines main() so we just emit as-is.
    main_path.write_text(src)
    exe = td / "prog"
    build = subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(main_path), "-o", str(exe)],
        capture_output=True, text=True, timeout=60,
    )
    if build.returncode != 0:
        return False, f"compile: {build.stderr.strip().splitlines()[-1] if build.stderr else 'unknown'}"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
    return run.returncode == 0, (run.stdout if run.returncode == 0 else run.stderr)


def _build_and_run_c(src: str, td: pathlib.Path) -> tuple[bool, str]:
    p = td / "main.c"
    p.write_text(src)
    exe = td / "prog"
    build = subprocess.run(
        ["gcc", "-std=c11", "-O2", str(p), "-o", str(exe)],
        capture_output=True, text=True, timeout=60,
    )
    if build.returncode != 0:
        return False, f"compile: {build.stderr.strip().splitlines()[-1] if build.stderr else 'unknown'}"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
    return run.returncode == 0, (run.stdout if run.returncode == 0 else run.stderr)


def _build_and_run_python(src: str, td: pathlib.Path) -> tuple[bool, str]:
    p = td / "main.py"
    p.write_text(src + "\n\nif __name__ == '__main__':\n    main()\n")
    run = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=15)
    return run.returncode == 0, (run.stdout if run.returncode == 0 else run.stderr)


def _build_and_run_fortran(src: str, td: pathlib.Path) -> tuple[bool, str]:
    p = td / "main.f90"
    p.write_text(src)
    exe = td / "main"
    if "program prog" in src:
        # Full program with entry point — compile and run.
        build = subprocess.run(
            ["gfortran", "-ffree-line-length-none", "-frealloc-lhs", str(p), "-o", str(exe)],
            capture_output=True, text=True, timeout=60,
        )
        if build.returncode != 0:
            return False, f"compile: {build.stderr.strip().splitlines()[-1] if build.stderr else 'unknown'}"
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
        return run.returncode == 0, (run.stdout if run.returncode == 0 else run.stderr)
    else:
        # Library-only (no main) — compile as object file to check correctness.
        build = subprocess.run(
            ["gfortran", "-ffree-line-length-none", "-frealloc-lhs", "-c", str(p), "-o", str(td / "lib.o")],
            capture_output=True, text=True, timeout=60,
        )
        if build.returncode != 0:
            return False, f"compile: {build.stderr.strip().splitlines()[-1] if build.stderr else 'unknown'}"
        return True, "<compile-only>"


def _build_and_run_mojo(src: str, td: pathlib.Path) -> tuple[bool, str]:
    p = td / "main.mojo"
    p.write_text(src)
    run = subprocess.run(
        ["mojo", "run", str(p)],
        capture_output=True, text=True, timeout=120,
    )
    return run.returncode == 0, (run.stdout if run.returncode == 0 else run.stderr)


BUILDERS = {
    "rust": _build_and_run_rust,
    "c": _build_and_run_c,
    "python": _build_and_run_python,
    "fortran": _build_and_run_fortran,
    "mojo": _build_and_run_mojo,
}


def evaluate(py_path: pathlib.Path, targets: list[str]) -> dict[str, str]:
    py_ok, py_out = run_python(py_path)
    if not py_ok:
        return {tgt: "SKIP-py-fail" for tgt in targets}
    results: dict[str, str] = {}
    for tgt in targets:
        ok, transpiled = transpile(py_path, tgt)
        if not ok:
            tail = transpiled.strip().splitlines()
            results[tgt] = f"XPILE: {tail[-1][:60] if tail else '<no output>'}"
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                td_p = pathlib.Path(td)
                ok, stdout = BUILDERS[tgt](transpiled, td_p)
        except subprocess.TimeoutExpired:
            results[tgt] = "HANG"
            continue
        if not ok:
            results[tgt] = f"BUILD: {stdout[:60]}"
            continue
        if stdout == "<compile-only>":
            results[tgt] = "COMPILE-OK"
        elif stdout.strip() == py_out.strip():
            results[tgt] = "PASS"
        else:
            results[tgt] = f"DIFF: got {stdout.strip()[:30]!r} want {py_out.strip()[:30]!r}"
    return results


def _topo_ordered_py(root: pathlib.Path, files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return Python files in dependency order via the call graph."""
    try:
        import sys
        _src = pathlib.Path(__file__).resolve().parent.parent / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from transpilers.graph.code_graph import file_topological_order
        ordered = file_topological_order(root, lang="python")
        ordered_strs = {str(f) for f in ordered}
        tail = [f for f in files if str(f) not in ordered_strs]
        return ordered + tail
    except Exception:
        return files


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else "examples/algorithms")
    targets = ["rust", "c", "python", "fortran", "mojo"]
    raw_py_files = sorted(p for p in root.rglob("*.py") if "main()" in p.read_text())
    py_files = _topo_ordered_py(root, raw_py_files)

    width_file = max(len(p.name) for p in py_files) + 2
    col_w = 14
    header = "file".ljust(width_file) + " ".join(t[:col_w - 1].ljust(col_w) for t in targets)
    print(header)
    print("-" * len(header))

    totals = {t: 0 for t in targets}
    runnable = {t: 0 for t in targets}
    for f in py_files:
        results = evaluate(f, targets)
        row = f.name.ljust(width_file)
        for tgt in targets:
            r = results[tgt]
            short = r if len(r) <= col_w - 1 else r[:col_w - 1]
            row += short.ljust(col_w)
            if r == "PASS" or r == "COMPILE-OK":
                totals[tgt] += 1
            if r == "PASS":
                runnable[tgt] += 1
        print(row)

    print()
    print("totals (build OK or runs+match): " + ", ".join(f"{t}={totals[t]}/{len(py_files)}" for t in targets))
    print("runnable (output matches python): " + ", ".join(f"{t}={runnable[t]}/{len(py_files)}" for t in targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

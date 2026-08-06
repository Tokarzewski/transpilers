"""Assembly frontend — staged transpilation via Ghidra (PyGhidra).

The pipeline:

    binary → Ghidra (PyGhidra-driven decompile) → C-like text → cleanup → parse_c → HIR

Ghidra recovers function structure and types from the binary; we re-use the
existing C frontend for everything downstream. Decompiled output is rarely
perfectly clean C — Ghidra uses non-standard types (`undefined`, `ulong`),
emits `WARNING:` comments, and sometimes inserts `goto` or inline asm
blocks. A small cleanup pass collapses the easy cases before handing off to
pycparser; anything left unparseable raises with the Ghidra output preserved
so the user can hand-edit.

This handles the **decompilation** style of asm transpilation (compiler-
generated / stripped binaries). For hand-written SIMD kernels with visible
structure, a direct asm-to-IR frontend would be a better fit — see the
discussion in the project README.

The Ghidra JVM is started lazily on first call. Subsequent calls reuse the
same JVM but open separate transient projects per binary.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from transpilers.ir import hir
from transpilers.frontends.c import parse_c


from transpilers.frontends.errors import UnsupportedConstruct


GHIDRA_HOME = Path(os.environ.get("GHIDRA_INSTALL_DIR", "/opt/ghidra"))


def parse_asm(source: str) -> hir.HirModule:
    """`source` is interpreted as a path to a binary (ELF / PE / Mach-O)
    rather than raw text. The CLI special-cases asm so the file is never
    read as text and won't trip on non-UTF-8 bytes."""
    binary_path = Path(source.strip())
    if not binary_path.exists():
        raise UnsupportedConstruct(
            f"asm frontend expects a path to a binary; {source!r} does not exist"
        )
    if not GHIDRA_HOME.exists():
        raise UnsupportedConstruct(
            f"Ghidra not found at {GHIDRA_HOME}. Install via `pacman -S ghidra` "
            f"(Arch) or set GHIDRA_INSTALL_DIR."
        )

    c_text = _decompile_via_pyghidra(binary_path)
    cleaned = _clean_ghidra_c(c_text)
    if not cleaned.strip():
        raise UnsupportedConstruct(
            "Ghidra produced no function output; binary may be stripped or "
            "Ghidra's auto-analysis didn't discover any functions."
        )
    try:
        module = parse_c(cleaned)
    except Exception as e:
        raise UnsupportedConstruct(
            "Ghidra output couldn't be parsed as C. This is the common failure "
            "mode of staged asm transpilation — decompiled code uses non-"
            "standard types and idioms. First parse error:\n"
            f"  {e}\n\n"
            "Cleaned Ghidra output (first 60 lines):\n"
            + "\n".join("  " + line for line in cleaned.splitlines()[:60])
        ) from e
    module.source_lang = "asm-via-ghidra"
    return module


# ---------- PyGhidra invocation ----------

_PYGHIDRA_STARTED = False


def _ensure_pyghidra_started() -> None:
    """Lazy JVM startup. PyGhidra accepts GHIDRA_INSTALL_DIR via env var; we
    set it before importing in case the user hasn't."""
    global _PYGHIDRA_STARTED
    if _PYGHIDRA_STARTED:
        return
    os.environ.setdefault("GHIDRA_INSTALL_DIR", str(GHIDRA_HOME))
    import pyghidra  # noqa: F401 — side-effect: JVM setup

    pyghidra.start(verbose=False)
    _PYGHIDRA_STARTED = True


# CRT / linker-injected stubs that Ghidra always recovers but the user never
# wants. Filter at the source so they don't pollute the parse.
CRT_FUNCTION_NAMES = {
    "_init",
    "_fini",
    "_start",
    "_dl_relocate_static_pie",
    "deregister_tm_clones",
    "register_tm_clones",
    "__do_global_dtors_aux",
    "frame_dummy",
    "__libc_csu_init",
    "__libc_csu_fini",
    "_DT_INIT",
    "call_gmon_start",
    "__do_global_ctors_aux",
}


def _decompile_via_pyghidra(binary_path: Path) -> str:
    _ensure_pyghidra_started()
    import pyghidra
    from ghidra.app.decompiler import DecompInterface
    from ghidra.util.task import ConsoleTaskMonitor

    fragments: list[str] = []
    with pyghidra.open_program(str(binary_path), analyze=True) as flat_api:
        program = flat_api.getCurrentProgram()
        decomp = DecompInterface()
        decomp.openProgram(program)
        monitor = ConsoleTaskMonitor()
        for fn in program.getFunctionManager().getFunctions(True):
            if fn.isExternal() or fn.isThunk():
                continue
            name = str(fn.getName())
            if name in CRT_FUNCTION_NAMES:
                continue
            if name.startswith("__") or name.startswith("_GLOBAL_"):
                continue
            result = decomp.decompileFunction(fn, 60, monitor)
            if not result.decompileCompleted():
                continue
            fragments.append(str(result.getDecompiledFunction().getC()))
    return "\n\n".join(fragments)


# ---------- cleanup ----------

# Ghidra's pseudo-C uses these non-standard width types; collapse onto
# stdint-compatible names that pycparser accepts.
GHIDRA_TYPE_REWRITES = {
    "undefined": "int",
    "undefined1": "char",
    "undefined2": "short",
    "undefined4": "int",
    "undefined8": "long",
    "byte": "unsigned char",
    "sbyte": "signed char",
    "uchar": "unsigned char",
    "ushort": "unsigned short",
    "uint": "unsigned int",
    "ulong": "unsigned long",
    "ulonglong": "unsigned long long",
    "longlong": "long long",
    "bool": "_Bool",
    "code": "void *",
    "wchar_t": "int",
}


# Standard-C int sizes that we know are valid identifiers in a clean type
# context. Everything else that *looks* like a type but isn't in this set
# becomes `void *` so pycparser can swallow it (the user can refine later).
KNOWN_C_TYPES = {
    "void",
    "char",
    "short",
    "int",
    "long",
    "float",
    "double",
    "_Bool",
    "signed",
    "unsigned",
    "const",
    "volatile",
    "struct",
    "union",
    "enum",
    "static",
    "extern",
    "register",
    "inline",
    # stdint flavors after our rewrites:
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "FILE",
}


def _clean_ghidra_c(text: str) -> str:
    # Strip `/* WARNING ... */` blocks — they sometimes contain characters
    # that confuse pycparser even though plain block comments are fine.
    text = re.sub(r"/\*\s*WARNING.*?\*/\s*", "", text, flags=re.DOTALL)
    # Replace Ghidra-specific type spellings with C-portable equivalents.
    for src, dst in GHIDRA_TYPE_REWRITES.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    # Strip calling-convention modifiers pycparser doesn't recognize.
    for cc in (
        "__fastcall ",
        "__cdecl ",
        "__stdcall ",
        "__thiscall ",
        "__attribute__((__noreturn__)) ",
    ):
        text = text.replace(cc, "")

    # Replace unknown library-recovered type names (`EVP_PKEY_CTX`, etc.) with
    # `void *`. Heuristic: identifiers that appear in a pointer-decl position
    # `<TYPE> *<ident>` where <TYPE> is uppercase / underscored and not a
    # known C type. This handles the common case without needing a full
    # typedef table.
    def _opaque_type(match: re.Match) -> str:
        ident = match.group(1)
        if ident in KNOWN_C_TYPES:
            return match.group(0)
        return f"void *{match.group(2)}"

    text = re.sub(r"\b([A-Z][A-Za-z0-9_]*)\s*\*\s*(\w+)", _opaque_type, text)
    return text


# Optional helper for raw asm source: assemble then decompile. Exposed for
# future use; the SIMD-direct frontend will likely supersede this.
def _assemble_and_decompile(asm_source: str) -> hir.HirModule:
    if not shutil.which("as"):
        raise UnsupportedConstruct("`as` (GNU assembler) not found")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "input.s"
        obj = Path(td) / "input.o"
        src.write_text(asm_source)
        result = subprocess.run(
            ["as", str(src), "-o", str(obj)], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise UnsupportedConstruct(f"assembly failed:\n{result.stderr}")
        return parse_asm(str(obj))

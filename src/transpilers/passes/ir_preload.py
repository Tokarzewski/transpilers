"""Extract function type signatures from LLVM IR to pre-populate MIR inference.

Compiles a C/C++ source file to LLVM IR (clang -O0 -emit-llvm -S) and parses
function definitions to map parameter and return types onto transpiler types.
These hints feed into infer_types() before the algorithmic and LLM passes,
eliminating most UnknownT holes for well-typed C/C++ code without extra LLM calls.

Usage::

    from transpilers.passes.ir_preload import extract_ir_types
    hints = extract_ir_types(Path("foo.cpp"))
    # {"_Z3addii": ([IntT(), IntT()], IntT()), ...}
    infer_types(mir_mod, ir_hints=hints)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from transpilers.ir.types import BoolT, FloatT, IntT, NoneT, Type, UnknownT

# ---------------------------------------------------------------------------
# LLVM primitive type → transpiler type
# ---------------------------------------------------------------------------

_LLVM_PRIMITIVES: dict[str, Type] = {
    "void": NoneT(),
    "i1": BoolT(),
    "i8": IntT(),
    "i16": IntT(),
    "i32": IntT(),
    "i64": IntT(),
    "i128": IntT(),
    "float": FloatT(),
    "double": FloatT(),
    "half": FloatT(),
    "fp128": FloatT(),
    "x86_fp80": FloatT(),
    "bfloat": FloatT(),
}

# LLVM IR function definition line (handles common attribute keywords)
_DEFINE_RE = re.compile(
    r"^define\b[^@]*@([^\s(]+)\(([^)]*)\)",
    re.MULTILINE,
)

# First type token before % in a parameter description, e.g. "i32 noundef %x"
_PARAM_TYPE_RE = re.compile(r"^\s*(\S+)(?:\s+\S+)*\s+%")

_ATTR_WORDS = frozenset(
    "dso_local hidden protected internal weak linkonce_odr available_externally "
    "noundef nonnull noalias readonly writeonly speculatable willreturn "
    "mustprogress nofree nosync nounwind "
    "zeroext signext inreg byval sret".split()
)


def _parse_type(tok: str) -> Type:
    t = tok.strip().rstrip("*").strip()
    if any(c in tok for c in ("*", "%", "{", "[", "<")):
        return UnknownT()
    return _LLVM_PRIMITIVES.get(t, UnknownT())


def _parse_ret_type(define_line: str) -> Type:
    """Extract return type from the token sequence between 'define' and '@'."""
    m = re.match(r"define\b(.+?)@", define_line)
    if not m:
        return UnknownT()
    attrs = m.group(1)
    tokens = [t for t in attrs.split() if t not in _ATTR_WORDS]
    tok = tokens[-1] if tokens else ""
    return _parse_type(tok)


def _parse_ir(ir_text: str) -> dict[str, tuple[list[Type], Type]]:
    result: dict[str, tuple[list[Type], Type]] = {}
    for m in _DEFINE_RE.finditer(ir_text):
        name = m.group(1)
        params_str = m.group(2).strip()

        line_start = ir_text.rfind("\n", 0, m.start()) + 1
        define_line = ir_text[line_start : m.start() + len("define") + 200]
        ret_type = _parse_ret_type(define_line)

        param_types: list[Type] = []
        if params_str and params_str != "...":
            for part in params_str.split(","):
                part = part.strip()
                if not part or part == "...":
                    continue
                pm = _PARAM_TYPE_RE.match(part)
                if pm:
                    param_types.append(_parse_type(pm.group(1)))
                else:
                    toks = [t for t in part.split() if t not in _ATTR_WORDS]
                    param_types.append(_parse_type(toks[0]) if toks else UnknownT())

        result[name] = (param_types, ret_type)
    return result


def extract_ir_types(
    source_path: Path | str,
    *,
    clang: str = "clang",
    extra_args: list[str] | None = None,
) -> dict[str, tuple[list[Type], Type]]:
    """Compile *source_path* to LLVM IR and return function type signatures.

    Parameters
    ----------
    source_path:
        A C or C++ source file.
    clang:
        clang binary name or path (default: ``"clang"``).
    extra_args:
        Additional arguments forwarded to clang, e.g. ``["-I/usr/include"]``.

    Returns
    -------
    dict mapping mangled function name → ``([param_types], return_type)``.
    Returns ``{}`` on any failure (clang not found, compile error).
    """
    source_path = Path(source_path)
    cmd = [clang, "-O0", "-emit-llvm", "-S", "-o", "-", str(source_path)]
    if extra_args:
        cmd.extend(extra_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError, subprocess.TimeoutExpired:
        return {}
    if proc.returncode != 0:
        return {}
    return _parse_ir(proc.stdout)


def demangle_name(mangled: str) -> str | None:
    """Return the demangled base function name via c++filt, or None on failure."""
    if not mangled.startswith("_Z"):
        return None
    try:
        proc = subprocess.run(
            ["c++filt", mangled], capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0 and proc.stdout.strip():
            demangled = proc.stdout.strip()
            return demangled.split("(")[0].split("::")[-1]
    except FileNotFoundError, subprocess.TimeoutExpired:
        pass
    return None

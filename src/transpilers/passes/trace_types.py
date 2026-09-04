"""Python trace-driven type inference.

Records runtime types by executing Python under ``sys.settrace`` instrumentation,
maps observed Python types to the transpiler's MIR type lattice (IntT, FloatT,
BoolT, StrT, ListT, NoneT, ...), and returns an ``IrHints`` dict suitable for
feeding into ``infer_types(..., ir_hints=...)``.

This provides ground-truth typing for Python code that lacks annotations,
dramatically improving transpilation to statically-typed targets (Rust, C,
Mojo, Fortran).
"""

from __future__ import annotations

import sys
import types
from typing import Any

from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    StrT,
    Type,
    UnknownT,
)

IrHints = dict[str, tuple[list[Type], Type]]


def _python_type_to_mir_type(value: Any) -> Type:
    """Map a Python runtime value to the closest MIR type lattice member."""
    if value is None:
        return NoneT()
    if isinstance(value, bool):
        return BoolT()
    if isinstance(value, int):
        return IntT()
    if isinstance(value, float):
        return FloatT()
    if isinstance(value, str):
        return StrT()
    if isinstance(value, (list, tuple)):
        elem_tys = [_python_type_to_mir_type(v) for v in value]
        concrete = [t for t in elem_tys if not isinstance(t, UnknownT)]
        elem: Type = concrete[0] if concrete else UnknownT(hint="empty sequence")
        return ListT(elem=elem)
    return UnknownT(hint=f"python type {type(value).__name__}")


def _merge_types(observed: list[Type]) -> Type:
    """Merge multiple observed types into the most specific common type.

    Rules:
    - All same kind -> return the first (preserving element types).
    - IntT + FloatT -> FloatT() (numeric promotion).
    - ListT with different element types -> recursively merge elements.
    - Any other mix -> UnknownT(hint=...).
    """
    if not observed:
        return UnknownT(hint="no observations")
    first = observed[0]
    if all(type(t) is type(first) for t in observed):
        if isinstance(first, ListT):
            elem_tys = [t.elem for t in observed]
            return ListT(elem=_merge_types(elem_tys))
        return first
    kinds = {type(t) for t in observed}
    if kinds <= {IntT, FloatT}:
        return FloatT()
    return UnknownT(hint=f"mixed: {sorted(k.__name__ for k in kinds)}")


class _PythonTypeTracer:
    """sys.settrace-based tracer that records param/return types.

    Only records events for frames whose co_filename matches the source
    file being traced -- internal Python frames are ignored.
    """

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        # Per-function: list of observed arg-type lists (one per call).
        self._param_types: dict[str, list[list[Type]]] = {}
        # Per-function: list of observed return types (one per return).
        self._return_types: dict[str, list[Type]] = {}

    def _trace(self, frame: types.FrameType, event: str, arg: Any) -> Any:
        """The settrace-compatible trace function."""
        code = frame.f_code
        if not code.co_filename.endswith(self.source_path):
            return None  # Don't descend into other modules.

        fn_name = code.co_name

        if event == "call":
            if fn_name.startswith("__") or fn_name == "<module>":
                return None
            arg_types: list[Type] = []
            n_args = code.co_argcount
            for i in range(n_args):
                name = code.co_varnames[i]
                val = frame.f_locals.get(name)
                arg_types.append(_python_type_to_mir_type(val))
            if fn_name not in self._param_types:
                self._param_types[fn_name] = []
            self._param_types[fn_name].append(arg_types)
            return self._trace

        if event == "return":
            if fn_name.startswith("__"):
                return None
            ret_type = _python_type_to_mir_type(arg)
            if fn_name not in self._return_types:
                self._return_types[fn_name] = []
            self._return_types[fn_name].append(ret_type)

        return self._trace

    def get_hints(self) -> IrHints:
        """Aggregate all observed types into an IrHints dict."""
        hints: IrHints = {}
        all_names: set[str] = set(self._param_types.keys()) | set(
            self._return_types.keys()
        )
        for name in sorted(all_names):
            ptypes_list = self._param_types.get(name, [])
            rtypes_list = self._return_types.get(name, [])

            merged_params: list[Type] = []
            if ptypes_list:
                max_params = max(len(p) for p in ptypes_list)
                for i in range(max_params):
                    collected = [p[i] for p in ptypes_list if i < len(p)]
                    merged_params.append(_merge_types(collected))

            merged_return = (
                _merge_types(rtypes_list)
                if rtypes_list
                else UnknownT(hint="no return observed")
            )
            hints[name] = (merged_params, merged_return)
        return hints


def trace_types(
    source_code: str,
    source_path: str = "<source.py>",
    call_main: bool = True,
) -> IrHints:
    """Execute Python source_code under type tracing and return type hints.

    Compiles and execs the source to register function definitions.
    If the module exposes a main() callable, it is invoked (when
    call_main is True) to exercise code paths.

    Returns {function_name: ([param_types], return_type)} for each
    function called at least once during tracing. Functions never
    called do not appear -- their types remain UnknownT holes.
    """
    tracer = _PythonTypeTracer(source_path)

    globals_dict: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__file__": source_path,
    }

    old_trace = sys.gettrace()
    sys.settrace(tracer._trace)
    try:
        code_obj = compile(source_code, source_path, "exec")
        exec(code_obj, globals_dict)

        if call_main and "main" in globals_dict and callable(globals_dict["main"]):
            try:
                globals_dict["main"]()
            except SystemExit, Exception:
                pass
    except SyntaxError, Exception:
        pass
    finally:
        sys.settrace(old_trace)

    return tracer.get_hints()


def trace_types_from_file(
    filepath: str,
    call_main: bool = True,
) -> IrHints:
    """Read Python source from filepath and run type tracing.

    Convenience wrapper around trace_types that reads the file content
    and uses its path as the source path for trace filtering.
    """
    from pathlib import Path

    source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    return trace_types(source, source_path=filepath, call_main=call_main)

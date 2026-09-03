"""Pipeline passes between IR tiers. Each pass declares its LLM hooks explicitly."""

from .dedupe_overloads import dedupe_overloads
from .hir_to_mir import hir_to_mir
from .infer_contracts import infer_contracts
from .infer_types import infer_types
from .ir_preload import extract_ir_types
from .llm_rename import llm_rename
from .mir_to_c_lir import mir_to_c_lir
from .mir_to_fortran_lir import mir_to_fortran_lir
from .mir_to_mojo_lir import mir_to_mojo_lir
from .mir_to_python_lir import mir_to_python_lir
from .mir_to_rust_lir import mir_to_rust_lir
from .trace_types import trace_types_from_file as trace_types_from_file

__all__ = [
    "dedupe_overloads",
    "extract_ir_types",
    "hir_to_mir",
    "infer_contracts",
    "infer_types",
    "llm_rename",
    "mir_to_c_lir",
    "mir_to_fortran_lir",
    "mir_to_mojo_lir",
    "mir_to_python_lir",
    "mir_to_rust_lir",
    "trace_types_from_file",
]

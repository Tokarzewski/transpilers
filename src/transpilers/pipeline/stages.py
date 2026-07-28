"""Stage-decomposed transpilation pipeline.

The canonical frontend / target registries live here, together with
``run_stages`` — the single-pass pipeline that keeps every intermediate
artifact (HIR, MIR, LIR, emitted text) instead of collapsing straight to a
string. Two consumers need the intermediates:

* the structural-fidelity verifier (``transpilers.verify.structural``)
  compares the source HIR skeleton against the target LIR skeleton;
* the failure taxonomy (``transpilers.verify.taxonomy``) attributes a
  failure to the exact stage that raised.

``transpilers.cli.main`` re-exports ``FRONTENDS`` / ``EXT_TO_SOURCE`` /
``TARGETS`` for backward compatibility and implements ``transpile`` as
``run_stages(...).output``.
"""

from __future__ import annotations

from dataclasses import dataclass

from transpilers.backends.c import emit_c
from transpilers.backends.fortran import emit_fortran
from transpilers.backends.go import emit_go
from transpilers.backends.mojo import emit_mojo
from transpilers.backends.python import emit_python
from transpilers.backends.rust import emit_rust
from transpilers.backends.zig import emit_zig
from transpilers.frontends.asm import parse_asm
from transpilers.frontends.c import parse_c
from transpilers.frontends.cpp import parse_cpp
from transpilers.frontends.csharp import parse_csharp
from transpilers.frontends.fortran import parse_fortran
from transpilers.frontends.go import parse_go
from transpilers.frontends.java import parse_java
from transpilers.frontends.javascript import parse_javascript
from transpilers.frontends.python import parse_python
from transpilers.frontends.typescript import parse_typescript
from transpilers.frontends.vb import parse_vb
from transpilers.ir.hir import HirModule
from transpilers.ir.lir.base import LirNode
from transpilers.ir.mir import MirModule
from transpilers.ir.provenance import ProvenanceMap
from transpilers.passes import (
    dedupe_overloads,
    hir_to_mir,
    infer_contracts,
    infer_types,
    llm_rename,
    mir_to_c_lir,
    mir_to_fortran_lir,
    mir_to_go_lir,
    mir_to_mojo_lir,
    mir_to_python_lir,
    mir_to_rust_lir,
    mir_to_zig_lir,
)
from transpilers.verify import (
    c_compiles,
    fortran_compiles,
    go_compiles,
    mojo_compiles,
    python_compiles,
    rust_compiles,
    zig_compiles,
)

__all__ = [
    "EXT_TO_SOURCE",
    "FRONTENDS",
    "STAGES",
    "TARGETS",
    "StageTrace",
    "run_stages",
]


FRONTENDS = {
    "python": parse_python,
    "c": parse_c,
    "cpp": parse_cpp,
    "java": parse_java,
    "csharp": parse_csharp,
    "typescript": parse_typescript,
    "javascript": parse_javascript,
    "fortran": parse_fortran,
    "go": parse_go,
    "vb": parse_vb,
    "asm": parse_asm,
}

EXT_TO_SOURCE = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f": "fortran",
    ".go": "go",
    ".vb": "vb",
    ".vbs": "vb",
    ".asm": "asm",
    ".s": "asm",
    ".S": "asm",
}

TARGETS = {
    "rust": (mir_to_rust_lir, emit_rust, rust_compiles),
    "zig": (mir_to_zig_lir, emit_zig, zig_compiles),
    "c": (mir_to_c_lir, emit_c, c_compiles),
    "mojo": (mir_to_mojo_lir, emit_mojo, mojo_compiles),
    "go": (mir_to_go_lir, emit_go, go_compiles),
    "python": (mir_to_python_lir, emit_python, python_compiles),
    "fortran": (mir_to_fortran_lir, emit_fortran, fortran_compiles),
}

# Pipeline stage names, in execution order. The taxonomy records which stage
# a failure came from; keep these strings stable — they are reporting keys.
STAGES = ("parse", "hir-to-mir", "infer-types", "lower", "emit", "compile", "run")


def _walk_provenance_nodes(node: object, out: list[object]) -> None:
    """Collect all IR nodes that carry ``_hir_provenance_id`` or
    ``_hir_node_id`` by DFS walk."""
    if hasattr(node, "_hir_provenance_id") or hasattr(node, "_hir_node_id"):
        out.append(node)
    # Recurse into child collections or attributes.
    if isinstance(node, (list, tuple)):
        for child in node:
            _walk_provenance_nodes(child, out)
    elif hasattr(node, "__dataclass_fields__"):
        for field_name in node.__dataclass_fields__:
            val = getattr(node, field_name, None)
            if isinstance(val, (list, tuple)):
                _walk_provenance_nodes(val, out)
            elif hasattr(val, "__dataclass_fields__"):
                _walk_provenance_nodes(val, out)


def _build_provenance_map(
    hir_mod: HirModule, mir_mod: MirModule, lir_mod: LirNode
) -> ProvenanceMap:
    """Walk all three IR tiers and build a ``ProvenanceMap``.

    Records every HIR node by its ``_hir_node_id``, then links MIR and LIR
    nodes to their originating HIR nodes via ``_hir_provenance_id``.
    """
    pm = ProvenanceMap()

    # Record all HIR nodes.
    hir_nodes: list[object] = []
    _walk_provenance_nodes(hir_mod, hir_nodes)
    by_hir_id: dict[int, object] = {}
    for hir_node in hir_nodes:
        hid = getattr(hir_node, "_hir_node_id", 0)
        if hid > 0:
            prov = pm.record_node(
                hir_node,
                hir_id=hid,
                hir_type=type(hir_node).__name__,
                source_span=None,
                hir_repr=repr(hir_node)[:120],
            )
            by_hir_id[hid] = prov

    # Record MIR nodes pointing back to their HIR provenance. Looked up via
    # by_hir_id (built once above) rather than a linear re-scan of every
    # already-recorded provenance per node -- the latter is O(mir_nodes *
    # hir_nodes), which is fine for a hand-written algorithm slice but
    # becomes minutes-long on a real-world amalgamated translation unit
    # (thousands of nodes on each side, e.g. transpiling a whole C++
    # package with --include-impls) for no benefit over a dict lookup.
    mir_nodes: list[object] = []
    _walk_provenance_nodes(mir_mod, mir_nodes)
    for mir_node in mir_nodes:
        hid = getattr(mir_node, "_hir_provenance_id", 0)
        prov = by_hir_id.get(hid) if hid > 0 else None
        if prov is not None:
            pm.record(mir_node, prov)

    # Record LIR nodes pointing back to their HIR provenance (via MIR).
    lir_nodes: list[object] = []
    _walk_provenance_nodes(lir_mod, lir_nodes)
    for lir_node in lir_nodes:
        hid = getattr(lir_node, "_hir_provenance_id", 0)
        prov = by_hir_id.get(hid) if hid > 0 else None
        if prov is not None:
            pm.record(lir_node, prov)

    return pm
@dataclass
class StageTrace:
    """Every intermediate artifact of one source→target transpilation."""

    source_lang: str
    target: str
    hir: HirModule
    mir: MirModule
    lir: LirNode
    output: str
    provenance_map: ProvenanceMap | None = None


def run_stages(
    source: str,
    *,
    source_lang: str = "python",
    target: str = "rust",
    llm_fill=None,
    llm_rename_fill=None,
    ir_hints=None,
    trace_types_hints=None,
) -> StageTrace:
    """Run the full pipeline, returning all intermediate artifacts.

    Parameters
    ----------
    source:
        Source code text.
    source_lang:
        Source language identifier (``"python"``, ``"c"``, …).
    target:
        Target language identifier (``"rust"``, ``"mojo"``, …).
    llm_fill:
        Optional LLM callback for filling ``UnknownT`` type holes.
    llm_rename_fill:
        Optional LLM callback for renaming opaque variables.
    ir_hints:
        Pre-populated type hints from LLVM IR (for C/C++ frontends).
    trace_types_hints:
        Pre-populated type hints from Python trace-driven execution.
        Merged with *ir_hints* before inference — trace-driven hints take
        precedence since they are ground-truth observations.

    Raises whatever the failing stage raises — callers that need failure
    attribution should drive the stages via ``transpilers.verify.taxonomy``.
    """
    from transpilers.passes.cpp_ground_truth import apply_ground_truth
    parse = FRONTENDS[source_lang]
    lower, emit, _ = TARGETS[target]
    # The C++ frontend returns (HirModule, TypeGroundTruth). The
    # other frontends still return a bare HirModule; normalise to a
    # (HirModule, truth-or-None) pair here so the rest of the
    # pipeline doesn't need to know about the C++ quirk.
    parsed = parse(source)
    if isinstance(parsed, tuple) and len(parsed) == 2:
        hir_mod, cpp_truth = parsed
    else:
        hir_mod, cpp_truth = parsed, None
    mir_mod = hir_to_mir(hir_mod)
    # Merge ir_hints (LLVM IR derived) with trace_types_hints (runtime derived).
    merged_hints = dict(ir_hints or {})
    if trace_types_hints:
        merged_hints.update(trace_types_hints)
    # Apply C++ ground truth *before* the inference pass: the
    # inference pass benefits from the resolved types and uses them
    # to anchor its own propagation, so filling holes first means
    # fewer UnknownT holes reach the algorithmic dataflow. For other
    # source languages this is a no-op (cpp_truth is None).
    if cpp_truth is not None:
        apply_ground_truth(mir_mod, cpp_truth, hir_mod)
    infer_types(mir_mod, llm_fill=llm_fill, ir_hints=merged_hints if merged_hints else None)
    # Semantic-contract inference: runs after type resolution so it can
    # map resolved types to contracts (arbitrary-precision int → overflow
    # guard, Python ref → borrow annotation, etc.).
    infer_contracts(mir_mod)
    if llm_rename_fill is not None:
        llm_rename(mir_mod, llm_fill=llm_rename_fill)
    # Must run after type inference (needs resolved param/return types to
    # detect a collision) and before lowering (every backend assumes unique
    # names within a struct/module).
    dedupe_overloads(mir_mod)
    lir_mod = lower(mir_mod)
    provenance = _build_provenance_map(hir_mod, mir_mod, lir_mod)
    return StageTrace(
        source_lang=source_lang,
        target=target,
        hir=hir_mod,
        mir=mir_mod,
        lir=lir_mod,
        output=emit(lir_mod),
        provenance_map=provenance,
    )

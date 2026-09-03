"""Structural-fidelity verifier — skeleton isomorphism between HIR and LIR.

Checks that the transpiled output preserves the *coarse structure* of the
source: the set of functions and structs, which struct owns which method, and
the control-flow shape (the nesting tree of if / while / for) inside every
function body. Statement- and expression-level idiom mapping is explicitly
allowed; this verifier only flags **parent-level** rewrites — added, dropped,
or merged functions, and flattened or invented control flow.

Matching is *name-keyed*: a function corresponds to the target function of
the same name (the engine preserves names today). When the node-level
HIR→LIR provenance map (issue #43) lands, matching can switch to provenance
ids and additionally catch renames-with-identical-shape.

Allowed idiom set (divergences that do NOT fail the check):

* All loop kinds count as one ``loop`` shape — the MIR pass desugars
  foreach into an indexed range loop, and stepped ranges lower to a
  transparent ``init + while`` block; both preserve the nesting tree, which is what we
  assert. (Flattening a loop into straight-line code is still caught.)
* A struct method may surface as a *free function* of the same name —
  Fortran has no type-bound procedures in the supported slice, so methods
  emit as free functions taking ``type(Name)`` first.
* Rust splitting a struct into ``struct`` + ``impl`` blocks.
* Statements that are not control flow may be dropped or rewritten
  (e.g. Mojo drops bare ``assert`` calls).

Works on every LIR dialect by exploiting the uniform ``<Prefix><Kind>``
class-naming convention (``RustIf``, ``MojoWhile``, ``FortranForRange``...).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transpilers.ir import hir
from transpilers.ir.lir.base import LirNode

__all__ = [
    "Divergence",
    "StructuralReport",
    "check_structural_fidelity",
    "hir_skeleton",
    "lir_skeleton",
]


# Longest-first so "Fortran" wins over "For..." typos and "C" comes last.
_DIALECT_PREFIXES = ("Fortran", "Mojo", "Rust", "Py", "C")

# LIR node kinds that open a struct-like item. "Class" is Python's, "Type"
# is Fortran's (which carries no methods — see allowed idiom set above).
_STRUCT_KINDS = {"Struct", "Class", "Type"}


def _lir_kind(node: object) -> str:
    """``RustForRange`` → ``ForRange``; unknown/private classes map to ``""``."""
    name = type(node).__name__
    for prefix in _DIALECT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return ""


# --------------------------------------------------------------------------- #
# Skeletons
# --------------------------------------------------------------------------- #

# A control-flow shape is a tuple of entries; each entry is
#   ("if", then_shape, else_shape) | ("loop", body_shape)
# Loop kinds (for / foreach / while) are unified — see allowed idiom set.
Shape = tuple


@dataclass
class Skeleton:
    """Coarse structure of a module: functions, structs, per-function CFG."""

    functions: dict[str, Shape] = field(default_factory=dict)
    structs: set[str] = field(default_factory=set)
    # (struct_name, method_name) -> shape
    methods: dict[tuple[str, str], Shape] = field(default_factory=dict)


def _hir_shape(body: list[hir.HirNode]) -> Shape:
    out: list[tuple] = []
    for node in body:
        if isinstance(node, hir.HirIf):
            out.append(("if", _hir_shape(node.body), _hir_shape(node.orelse)))
        elif isinstance(node, (hir.HirWhile, hir.HirFor, hir.HirForEach)):
            out.append(("loop", _hir_shape(node.body)))
    return tuple(out)


def hir_skeleton(module: hir.HirModule) -> Skeleton:
    sk = Skeleton()
    for item in module.body:
        if isinstance(item, hir.HirFunction):
            sk.functions[item.name] = _hir_shape(item.body)
        elif isinstance(item, hir.HirStruct):
            sk.structs.add(item.name)
            for m in item.methods:
                sk.methods[(item.name, m.name)] = _hir_shape(m.body)
    return sk


def _lir_shape(body: list[LirNode]) -> Shape:
    out: list[tuple] = []
    for node in body:
        kind = _lir_kind(node)
        if kind == "If":
            out.append(
                ("if", _lir_shape(node.body), _lir_shape(getattr(node, "orelse", [])))
            )
        elif kind in ("While", "ForRange"):
            out.append(("loop", _lir_shape(node.body)))
        else:
            # Transparent statement blocks (the init+while desugar of a
            # stepped range lives in a private block node): splice their items.
            items = getattr(node, "items", None)
            if isinstance(items, list):
                out.extend(_lir_shape(items))
    return tuple(out)


def lir_skeleton(module: LirNode) -> Skeleton:
    sk = Skeleton()
    for item in getattr(module, "items", []):
        kind = _lir_kind(item)
        if kind == "Fn":
            sk.functions[item.name] = _lir_shape(item.body)
        elif kind in _STRUCT_KINDS:
            sk.structs.add(item.name)
            for m in getattr(item, "methods", []):
                sk.methods[(item.name, m.name)] = _lir_shape(m.body)
        elif kind == "Impl":  # Rust: methods live in a separate impl block
            for m in item.methods:
                sk.methods[(item.struct_name, m.name)] = _lir_shape(m.body)
    return sk


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class Divergence:
    kind: str  # dropped-function | added-function | dropped-struct | added-struct | control-flow-shape
    where: str  # function / struct name ("Struct.method" for methods)
    detail: str = ""

    def __str__(self) -> str:
        msg = f"{self.kind}: {self.where}"
        return f"{msg} ({self.detail})" if self.detail else msg


@dataclass
class StructuralReport:
    ok: bool
    divergences: list[Divergence]

    def summary(self) -> str:
        if self.ok:
            return "structural fidelity: ok"
        lines = [f"structural fidelity: {len(self.divergences)} divergence(s)"]
        lines += [f"  - {d}" for d in self.divergences]
        return "\n".join(lines)


def _shape_str(shape: Shape) -> str:
    """Compact one-line rendering: ``if(while();for())``."""
    parts = []
    for entry in shape:
        if entry[0] == "if":
            parts.append(f"if({_shape_str(entry[1])}|{_shape_str(entry[2])})")
        else:
            parts.append(f"{entry[0]}({_shape_str(entry[1])})")
    return ";".join(parts)


def check_structural_fidelity(
    hir_module: hir.HirModule, lir_module: LirNode
) -> StructuralReport:
    """Compare source HIR skeleton against target LIR skeleton."""
    src, dst = hir_skeleton(hir_module), lir_skeleton(lir_module)
    divergences: list[Divergence] = []
    consumed_free_fns: set[str] = set()

    # Structs ------------------------------------------------------------- #
    for name in sorted(src.structs - dst.structs):
        divergences.append(Divergence("dropped-struct", name))
    for name in sorted(dst.structs - src.structs):
        divergences.append(Divergence("added-struct", name))

    # Methods (may legally surface as free functions — Fortran) ------------ #
    for (owner, mname), shape in src.methods.items():
        where = f"{owner}.{mname}"
        if (owner, mname) in dst.methods:
            got = dst.methods[(owner, mname)]
        elif mname in dst.functions:
            got = dst.functions[mname]
            consumed_free_fns.add(mname)
        else:
            divergences.append(Divergence("dropped-function", where))
            continue
        if got != shape:
            divergences.append(
                Divergence(
                    "control-flow-shape",
                    where,
                    f"source {_shape_str(shape) or 'flat'} != target {_shape_str(got) or 'flat'}",
                )
            )
    for owner, mname in sorted(set(dst.methods) - set(src.methods)):
        divergences.append(Divergence("added-function", f"{owner}.{mname}"))

    # Free functions -------------------------------------------------------- #
    for name, shape in src.functions.items():
        if name not in dst.functions:
            divergences.append(Divergence("dropped-function", name))
            continue
        got = dst.functions[name]
        if got != shape:
            divergences.append(
                Divergence(
                    "control-flow-shape",
                    name,
                    f"source {_shape_str(shape) or 'flat'} != target {_shape_str(got) or 'flat'}",
                )
            )
    for name in sorted(set(dst.functions) - set(src.functions) - consumed_free_fns):
        divergences.append(Divergence("added-function", name))

    return StructuralReport(ok=not divergences, divergences=divergences)

"""Failure taxonomy — bucket every verify-gate failure (issue #42).

Real-world pass rates were a single number; this module turns every failure
into a (bucket, stage, construct) triple so the corpus sweeps can say *which*
failure class dominates each (source-lang, target) pair.

Buckets (fixed — these strings are reporting keys, do not rename):

* ``parse``                  — the algorithmic pipeline refused the construct.
  This covers the frontend proper (``UnsupportedConstruct`` / ``SyntaxError``)
  *and* refusals later in HIR→MIR / lowering / emit (``NotImplementedError``
  on a construct or type the target can't map). The ``stage`` field records
  where the refusal happened.
* ``unresolved-symbol``      — a name/function couldn't be resolved, either
  in-pipeline or by the target compiler (rustc E0425, gcc ``undefined reference``...).
* ``unfilled-UnknownT-hole`` — an ``UnknownT`` survived inference and reached
  a target type map ("unresolved type hole").
* ``type-inference-miss``    — inference *filled* the types but the target
  compiler rejected with a type error (rustc E0308, gcc incompatible-type...).
* ``target-compile-error``   — any other target-compiler rejection.
* ``output-mismatch``        — compiled, ran, produced different output than
  the source-language reference run (or crashed at runtime).
* ``structural-divergence``  — the structural-fidelity verifier (issue #45)
  found the output skeleton non-isomorphic to the source.
* ``timeout``                — any stage exceeded its time budget.
* ``internal-error``         — unexpected engine exception; always a bug.
* ``ok``                     — no failure.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from transpilers.frontends.errors import UnsupportedConstruct

__all__ = [
    "BUCKETS",
    "TaxonomyRecord",
    "classify_exception",
    "classify_compile_stderr",
    "classify_run",
    "classify_unit",
    "compiler_available",
]


BUCKETS = (
    "ok",
    "parse",
    "unresolved-symbol",
    "unfilled-UnknownT-hole",
    "type-inference-miss",
    "target-compile-error",
    "output-mismatch",
    "structural-divergence",
    "timeout",
    "internal-error",
)


@dataclass
class TaxonomyRecord:
    """One classified (file, source-lang, target) verification outcome."""

    source_lang: str
    target: str
    bucket: str
    stage: str  # parse | hir-to-mir | infer-types | lower | emit | compile | structural | run
    construct: str = ""  # offending construct / symbol / type hole, when known
    detail: str = ""  # first error line, free-form
    output: str | None = None  # emitted target source, when emission succeeded


# --------------------------------------------------------------------------- #
# Exception classification (in-pipeline failures)
# --------------------------------------------------------------------------- #

_HOLE_RE = re.compile(r"unresolved type hole(?::\s*(?P<hint>.*))?", re.DOTALL)
_SYMBOL_MSG_RE = re.compile(
    r"unknown function|undefined symbol|unresolved symbol|unknown call"
)


def _first_line(text: str) -> str:
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def classify_exception(stage: str, exc: BaseException) -> tuple[str, str]:
    """Map an in-pipeline exception to ``(bucket, construct)``."""
    if isinstance(exc, (subprocess.TimeoutExpired, TimeoutError)):
        return "timeout", ""
    msg = str(exc)
    if isinstance(exc, ValueError):
        m = _HOLE_RE.search(msg)
        if m:
            return "unfilled-UnknownT-hole", (m.group("hint") or "").strip()
    if isinstance(exc, NameError) or _SYMBOL_MSG_RE.search(msg):
        return "unresolved-symbol", _first_line(msg)
    # libcst / pycparser / tree-sitter raise their own syntax-error types
    # (ParserSyntaxError, ParseError, ...); match by class name so every
    # frontend's parse failure lands in `parse`.
    if "SyntaxError" in type(exc).__name__ or "ParseError" in type(exc).__name__:
        return "parse", _first_line(msg)
    if isinstance(
        exc, (UnsupportedConstruct, SyntaxError, NotImplementedError, ValueError)
    ):
        # Deliberate refusals: the pipeline won't model this construct. The
        # `stage` field distinguishes frontend parse from lowering/emit gaps.
        return "parse", _first_line(msg)
    return "internal-error", f"{type(exc).__name__}: {_first_line(msg)}"


# --------------------------------------------------------------------------- #
# Target-compiler stderr classification
# --------------------------------------------------------------------------- #

_SYMBOL_PATTERNS = [
    r"\bE0425\b",
    r"\bE0433\b",
    r"\bE0412\b",  # rustc: name/path/type not found
    r"cannot find (?:function|value|type|macro|struct)",  # rustc prose
    r"undeclared identifier",
    r"undeclared \(first use",  # gcc
    r"implicit declaration of function",  # gcc
    r"undefined reference",  # ld
    r"use of unknown declaration",  # mojo
    r"has no IMPLICIT type",  # gfortran
]
_TYPE_PATTERNS = [
    r"\bE0308\b",
    r"mismatched types",  # rustc
    r"incompatible type",
    r"but argument is of type",  # gcc
    r"cannot be converted",
    r"invalid conversion",  # mojo
    r"Type mismatch",  # gfortran
]
_SYMBOL_RE = re.compile("|".join(_SYMBOL_PATTERNS))
_TYPE_RE = re.compile("|".join(_TYPE_PATTERNS))


def classify_compile_stderr(stderr: str) -> tuple[str, str]:
    """Map a target-compiler rejection to ``(bucket, detail)``.

    Symbol resolution wins over type errors: a missing symbol typically
    cascades into bogus type complaints, not vice versa.
    """
    detail = next(
        (line.strip() for line in stderr.splitlines() if "error" in line.lower()),
        _first_line(stderr),
    )
    if _SYMBOL_RE.search(stderr):
        return "unresolved-symbol", detail
    if _TYPE_RE.search(stderr):
        return "type-inference-miss", detail
    return "target-compile-error", detail


# --------------------------------------------------------------------------- #
# Run-result classification
# --------------------------------------------------------------------------- #


def classify_run(
    expected: str, actual: str, *, exit_ok: bool = True
) -> tuple[str, str]:
    """Compare a target run against the source-language reference run."""
    if not exit_ok:
        return "output-mismatch", f"runtime failure: {_first_line(actual)}"
    if expected.strip() == actual.strip():
        return "ok", ""
    exp, act = expected.strip().splitlines(), actual.strip().splitlines()
    for i, (e, a) in enumerate(zip(exp, act)):
        if e != a:
            return "output-mismatch", f"line {i + 1}: expected {e!r}, got {a!r}"
    return "output-mismatch", f"line count: expected {len(exp)}, got {len(act)}"


# --------------------------------------------------------------------------- #
# Staged driver
# --------------------------------------------------------------------------- #

_COMPILER_BINS = {
    "rust": ("rustc",),
    "c": ("gcc", "clang", "cc"),
    "mojo": ("mojo",),
    "fortran": ("gfortran", "flang"),
    "python": (),  # host interpreter — always available
}


def compiler_available(target: str) -> bool:
    if target == "mojo":
        from transpilers.verify.mojo import mojo_available

        return mojo_available()
    bins = _COMPILER_BINS.get(target, ())
    return not bins or any(shutil.which(b) for b in bins)


def classify_unit(
    source: str,
    *,
    source_lang: str = "python",
    target: str = "rust",
    compile: bool = True,
    structural: bool = False,
    llm_fill=None,
    ir_hints=None,
) -> TaxonomyRecord:
    """Run the staged pipeline on one unit and classify the outcome.

    Stages: parse → hir-to-mir → infer-types → lower → emit, then optionally
    the target-compiler gate and the structural-fidelity gate. The first
    failing stage determines the record; ``bucket == "ok"`` means everything
    requested passed.
    """
    from transpilers.passes import hir_to_mir, infer_types
    from transpilers.pipeline.stages import FRONTENDS, TARGETS

    def fail(
        stage: str, exc: BaseException, output: str | None = None
    ) -> TaxonomyRecord:
        bucket, construct = classify_exception(stage, exc)
        return TaxonomyRecord(
            source_lang,
            target,
            bucket,
            stage,
            construct=construct,
            detail=_first_line(str(exc)),
            output=output,
        )

    lower, emit, verify_fn = TARGETS[target]
    # The C++ frontend returns (HirModule, TypeGroundTruth). Other
    # frontends still return a bare HirModule. Normalise so the rest
    # of the staged driver can treat the parse step the same way.
    try:
        parsed = FRONTENDS[source_lang](source)
    except Exception as exc:
        return fail("parse", exc)
    if isinstance(parsed, tuple) and len(parsed) == 2:
        hir_mod, cpp_truth = parsed
    else:
        hir_mod, cpp_truth = parsed, None
    try:
        mir_mod = hir_to_mir(hir_mod)
    except Exception as exc:
        return fail("hir-to-mir", exc)
    # Apply the C++ ground truth (issue #50). No-op for non-C++
    # frontends.
    if cpp_truth is not None:
        from transpilers.passes.cpp_ground_truth import apply_ground_truth

        try:
            apply_ground_truth(mir_mod, cpp_truth, hir_mod)
        except Exception as exc:
            return fail("ground-truth", exc)
    try:
        infer_types(mir_mod, llm_fill=llm_fill, ir_hints=ir_hints)
    except Exception as exc:
        return fail("infer-types", exc)
    try:
        lir_mod = lower(mir_mod)
    except Exception as exc:
        return fail("lower", exc)
    try:
        output = emit(lir_mod)
    except Exception as exc:
        return fail("emit", exc)

    if compile and compiler_available(target):
        try:
            result = verify_fn(output)
        except subprocess.TimeoutExpired, TimeoutError:
            return TaxonomyRecord(
                source_lang, target, "timeout", "compile", output=output
            )
        if not result.ok:
            bucket, detail = classify_compile_stderr(result.stderr)
            return TaxonomyRecord(
                source_lang, target, bucket, "compile", detail=detail, output=output
            )

    if structural:
        from transpilers.verify.structural import check_structural_fidelity

        report = check_structural_fidelity(hir_mod, lir_mod)
        if not report.ok:
            first = report.divergences[0]
            return TaxonomyRecord(
                source_lang,
                target,
                "structural-divergence",
                "structural",
                construct=first.where,
                detail="; ".join(str(d) for d in report.divergences[:5]),
                output=output,
            )

    return TaxonomyRecord(source_lang, target, "ok", "", output=output)

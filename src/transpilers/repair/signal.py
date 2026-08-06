"""Repair signal — the structured failure context fed back to the LLM (issue #47).

The original ``repair.repair`` prompt dumped raw stderr into a single string
and asked the LLM to fix the broken code. The verification-driven loop
(issue #47) is more disciplined: it captures *what kind* of failure happened
(compiler error vs run divergence vs unfilled type hole vs structural
divergence) and feeds a structured :class:`RepairSignal` back to the LLM so
each tier can produce a targeted fix.

This module is the single source of truth for that signal shape and for the
helpers that build one from a verifier's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from transpilers.verify.taxonomy import classify_compile_stderr, classify_run

# The signal kinds the loop recognises. Strings are reporting keys — do not
# rename without updating flywheel consumers (#51).
SignalKind = Literal[
    "compile_error",
    "run_mismatch",
    "structural_divergence",
    "unfilled_hole",
    "internal",
]


@dataclass
class RepairSignal:
    """Structured failure context re-fed to the LLM on each retry.

    Fields
    ------
    kind:
        Which gate failed. The prompt template picks a different framing per
        kind (compiler diagnostic vs expected/actual diff vs hole context).
    diagnostic:
        The raw failure text — compiler stderr, the first diverging line, or
        the structural report summary. Always non-empty for a real signal;
        the only empty-diagnostic case is a clean pass (no signal needed).
    hole_context:
        Typed-hole context, when *kind* is ``unfilled_hole``. Captures the
        name, the surrounding HIR node, and the inference confidence so the
        LLM can fill the missing piece without hallucinating type semantics.
    input_diff:
        For ``run_mismatch``: the input that drove the divergence (or
        ``"<no test input>"`` when the verifier is compile-only).
    expected:
        Expected stdout, when known.
    actual:
        Actual stdout, when known.
    bucket:
        The taxonomy bucket the failure landed in (e.g. ``type-inference-miss``,
        ``unresolved-symbol``, ``output-mismatch``). Promoted to the prompt
        header so the LLM can reason about which class of fix to apply.
    stage:
        Pipeline stage the failure happened at (``compile``, ``run``,
        ``structural``, ``lower``…). The first failing stage wins; the loop
        does not chain signals across stages.
    attempt:
        1-based attempt index (1 == first try, 2 == first retry, …). The
        prompt template uses it to remind the model of the budget.
    extra:
        Free-form per-kind payload. Reserved for forward-compat.
    """

    kind: SignalKind
    diagnostic: str
    hole_context: dict[str, Any] = field(default_factory=dict)
    input_diff: str = ""
    expected: str = ""
    actual: str = ""
    bucket: str = ""
    stage: str = ""
    attempt: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def is_clean(self) -> bool:
        """A "signal" representing a verified pass — empty diagnostic."""
        return self.kind == "" and not self.diagnostic

    def to_prompt_dict(self) -> dict[str, Any]:
        """Serialise for the prompt template (drops empty fields for brevity)."""
        out: dict[str, Any] = {"kind": self.kind, "attempt": self.attempt}
        if self.diagnostic:
            out["diagnostic"] = self.diagnostic
        if self.bucket:
            out["bucket"] = self.bucket
        if self.stage:
            out["stage"] = self.stage
        if self.input_diff:
            out["input_diff"] = self.input_diff
        if self.expected:
            out["expected"] = self.expected
        if self.actual:
            out["actual"] = self.actual
        if self.hole_context:
            out["hole_context"] = self.hole_context
        if self.extra:
            out["extra"] = self.extra
        return out


# ---------------------------------------------------------------------------
# Builders from verifier output
# ---------------------------------------------------------------------------


def signal_from_compile(
    stderr: str,
    *,
    attempt: int = 1,
    stage: str = "compile",
) -> RepairSignal:
    """Build a :class:`RepairSignal` from a target-compiler rejection."""
    bucket, detail = classify_compile_stderr(stderr or "")
    return RepairSignal(
        kind="compile_error",
        diagnostic=detail or (stderr or "").strip(),
        bucket=bucket,
        stage=stage,
        attempt=attempt,
    )


def signal_from_run(
    expected: str,
    actual: str,
    *,
    attempt: int = 1,
    input_text: str = "",
    exit_ok: bool = True,
    stage: str = "run",
) -> RepairSignal:
    """Build a :class:`RepairSignal` from a target-runtime mismatch.

    The caller is expected to have already established that *expected* and
    *actual* diverge (or the run failed); this helper is for the divergence
    branch.
    """
    bucket, detail = classify_run(expected, actual, exit_ok=exit_ok)
    return RepairSignal(
        kind="run_mismatch",
        diagnostic=detail,
        bucket=bucket,
        stage=stage,
        attempt=attempt,
        input_diff=input_text or "<no test input>",
        expected=expected,
        actual=actual,
    )


def signal_from_structural(
    summary: str,
    *,
    attempt: int = 1,
    stage: str = "structural",
    divergences: list[str] | None = None,
) -> RepairSignal:
    """Build a :class:`RepairSignal` from a structural-fidelity report."""
    return RepairSignal(
        kind="structural_divergence",
        diagnostic=summary,
        bucket="structural-divergence",
        stage=stage,
        attempt=attempt,
        extra={"divergences": list(divergences or [])},
    )


def signal_from_hole(
    hole_name: str,
    hole_hint: str,
    *,
    attempt: int = 1,
    stage: str = "lower",
    extra: dict[str, Any] | None = None,
) -> RepairSignal:
    """Build a :class:`RepairSignal` for an unfilled ``UnknownT`` hole.

    The hole context is what the prompt template uses to teach the LLM
    *which* type to infer (name + surrounding HIR snippet) so it can fill
    the hole with a real lattice type, not a guess.
    """
    return RepairSignal(
        kind="unfilled_hole",
        diagnostic=f"unresolved type hole: {hole_name}: {hole_hint}".rstrip(": "),
        bucket="unfilled-UnknownT-hole",
        stage=stage,
        attempt=attempt,
        hole_context={"name": hole_name, "hint": hole_hint, **(extra or {})},
    )


def signal_from_internal(
    exc: BaseException,
    *,
    attempt: int = 1,
    stage: str = "internal",
) -> RepairSignal:
    """Build a :class:`RepairSignal` from an in-pipeline exception that was
    already bucketed by ``classify_exception``. The exception's
    ``__class__.__name__`` is the construct hint."""
    from transpilers.verify.taxonomy import _first_line, classify_exception

    bucket, construct = classify_exception(stage, exc)
    return RepairSignal(
        kind="internal",
        diagnostic=construct or _first_line(str(exc)),
        bucket=bucket,
        stage=stage,
        attempt=attempt,
        extra={"exception": type(exc).__name__, "message": str(exc)},
    )

"""Verification-driven repair loop with escalating tiers (issue #47).

This is the public surface for the new repair behaviour described in the
issue. It re-prompts on verification failure with a *structured* signal
(compiler diagnostic vs diverging input+expected/actual vs unfilled
``UnknownT`` hole context), escalates the model tier per retry
(CACHED -> LOCAL_FINETUNED -> FRONTIER), stops on the first verified
pass, records every verified repair to the flywheel JSONL log (issue #51),
and refuses on exhaustion (no silent broken output).

The loop is the *one-shot* baseline's replacement: it keeps the same
public surface (compile-then-fix) but adds the tier escalation, the
structured signal, the bounded budget, and the flywheel. The legacy
:mod:`transpilers.repair.repair` module is preserved as the API
backward-compat shim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable

from transpilers.llm.client import ModelTier, TieredLlmClient
from transpilers.repair.flywheel import Flywheel
from transpilers.repair.outcomes import RepairOutcome, RepairTracker
from transpilers.repair.signal import (
    RepairSignal,
    signal_from_compile,
)

# Prompt template: extends the legacy repair.md with a per-signal framing
# block. Rendered with ``string.Template`` after a tiny pre-processor
# strips the irrelevant ``{% if %}`` blocks.
_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "llm" / "prompts" / "repair_with_signal.md"
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EscalatingRepairAttempt:
    """Record of one iteration of the escalating loop."""

    attempt: int
    tier: ModelTier
    code: str
    verified: bool
    signal: RepairSignal | None = None
    error: str = ""
    fix_applied: str = ""
    cache_hit: bool = False
    elapsed_ms: int = 0


@dataclass
class EscalatingRepairResult:
    """Final outcome of :func:`escalating_repair`."""

    code: str
    passed: bool
    attempts: int
    fixing_tier: ModelTier | None
    history: list[EscalatingRepairAttempt] = field(default_factory=list)
    refused: bool = False
    flywheel_recorded: bool = False

    def summary(self) -> str:
        """One-line summary for the CLI's stderr."""
        last = self.history[-1] if self.history else None
        tier = self.fixing_tier.value if self.fixing_tier else "none"
        if self.passed:
            return (
                f"[escalating-repair] passed on attempt {self.attempts} at tier {tier}"
            )
        if self.refused:
            return (
                f"[escalating-repair] refused after {self.attempts} attempts "
                f"(last tier {tier}, last error: "
                f"{(last.error[:80] if last else '?')})"
            )
        return f"[escalating-repair] failed after {self.attempts} attempts"


# ---------------------------------------------------------------------------
# Verifier protocol
# ---------------------------------------------------------------------------


@dataclass
class VerificationOutcome:
    """The loop's view of a verify run on a piece of translated code."""

    ok: bool
    signal: RepairSignal | None = None
    # For a compile-only verifier the expected/actual fields are blank.
    expected: str = ""
    actual: str = ""


Verifier = Callable[[str], VerificationOutcome]


def _wrap_legacy_verifier(target: str) -> Verifier:
    """Build a :class:`Verifier` for *target* from the legacy compile gate."""
    from transpilers.verify import (
        c_compiles,
        fortran_compiles,
        go_compiles,
        mojo_compiles,
        python_compiles,
        rust_compiles,
        zig_compiles,
    )

    legacy = {
        "rust": rust_compiles,
        "zig": zig_compiles,
        "c": c_compiles,
        "mojo": mojo_compiles,
        "go": go_compiles,
        "python": python_compiles,
        "fortran": fortran_compiles,
    }[target]

    def verify(code: str) -> VerificationOutcome:
        result = legacy(code)
        if result.ok:
            return VerificationOutcome(ok=True)
        return VerificationOutcome(
            ok=False,
            signal=signal_from_compile(result.stderr, attempt=1),
        )

    return verify


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


# Trim the leading/trailing markdown fences the LLM sometimes wraps the
# answer in.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)


def _strip_fences(code: str) -> str:
    return _FENCE_RE.sub("", code.strip()).strip()


def _render_prompt(
    *,
    source_lang: str,
    target: str,
    source_code: str,
    broken_code: str,
    signal: RepairSignal,
    attempt: int,
    max_attempts: int,
    tier: ModelTier,
) -> str:
    """Build the signal-aware repair prompt.

    Falls back to the legacy ``repair.md`` template if the signal-aware
    template is missing (e.g. a partial checkout).
    """
    if not _PROMPT_PATH.exists():
        return _legacy_prompt(
            source_lang=source_lang,
            target=target,
            source_code=source_code,
            broken_code=broken_code,
            error_message=signal.diagnostic,
            attempt=attempt,
            max_attempts=max_attempts,
        )
    raw_template = _PROMPT_PATH.read_text()
    template = _strip_irrelevant_blocks(raw_template, signal.kind)
    ctx = {
        "source_lang": source_lang,
        "target_lang": target,
        "source_code": source_code,
        "broken_code": broken_code,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "tier": tier.value,
        "signal_kind": signal.kind,
        "signal_bucket": signal.bucket,
        "signal_stage": signal.stage,
        "signal_diagnostic": signal.diagnostic,
        "signal_input": signal.input_diff,
        "signal_expected": signal.expected,
        "signal_actual": signal.actual,
        "signal_hole_context": _format_hole_context(signal.hole_context),
        "signal_exception": signal.extra.get("exception", "?"),
    }
    t_template = _to_dollar_placeholders(template)
    try:
        return Template(t_template).safe_substitute(ctx)
    except KeyError, ValueError:
        return _legacy_prompt(
            source_lang=source_lang,
            target=target,
            source_code=source_code,
            broken_code=broken_code,
            error_message=signal.diagnostic,
            attempt=attempt,
            max_attempts=max_attempts,
        )


def _format_hole_context(hole: dict[str, Any]) -> str:
    if not hole:
        return "{}"
    import json

    return json.dumps(hole, indent=2)


def _strip_irrelevant_blocks(template: str, signal_kind: str) -> str:
    """Strip ``{% if signal_kind == "..." %} ... {% endif %}`` blocks whose
    condition does not match *signal_kind*."""
    lines = template.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{% if "):
            # ``{% if signal_kind == "..." %}`` → condition body is between
            # ``if`` and ``%``. ``stripped[3:]`` skips ``{% ``, then we look
            # for the closing ``%``.
            body = stripped[3:]
            end = body.find("%")
            cond = body[2:end].strip()  # skip "if"
            keep = _eval_cond(cond, signal_kind)
            skip = not keep
            continue
        if stripped == "{% endif %}":
            skip = False
            continue
        if not skip:
            out.append(line)
    return "\n".join(out) + "\n"


def _eval_cond(cond: str, signal_kind: str) -> bool:
    """Evaluate ``signal_kind == "compile_error"`` and similar one-liners."""
    if "==" not in cond:
        return False
    left, right = (s.strip() for s in cond.split("==", 1))
    rhs = right.strip().strip('"').strip("'")
    return left == "signal_kind" and rhs == signal_kind


def _to_dollar_placeholders(template: str) -> str:
    """Translate ``{name}`` and ``{{name}}`` placeholders to ``$name`` for
    :class:`string.Template`.

    Handles:
    * ``{{name}}`` → ``$name`` (the template convention used in this file)
    * ``{name}``  → ``$name`` (passthrough)
    * ``{{``      → ``{`` (literal opening brace)
    * ``}}``      → ``}`` (literal closing brace)
    * ``{}``      → ``{}`` (empty brace pair, not a placeholder)
    """
    out: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        c = template[i]
        # ``{{name}}`` — double-brace placeholder
        if (
            c == "{"
            and i + 1 < n
            and template[i + 1] == "{"
            and i + 2 < n
            and (template[i + 2].isalpha() or template[i + 2] == "_")
        ):
            j = template.find("}}", i + 2)
            if j != -1:
                name = template[i + 2 : j]
                if all(ch.isalnum() or ch == "_" for ch in name):
                    out.append("$" + name)
                    i = j + 2
                    continue
        # ``{{`` — literal opening brace
        if c == "{" and i + 1 < n and template[i + 1] == "{":
            out.append("{")
            i += 2
            continue
        # ``}}`` — literal closing brace
        if c == "}" and i + 1 < n and template[i + 1] == "}":
            out.append("}")
            i += 2
            continue
        # ``{}`` — empty brace pair
        if c == "{" and i + 1 < n and template[i + 1] == "}":
            out.append("{}")
            i += 2
            continue
        # ``{name}`` — single-brace placeholder
        if (
            c == "{"
            and i + 1 < n
            and (template[i + 1].isalpha() or template[i + 1] == "_")
        ):
            j = template.find("}", i + 1)
            if j == -1:
                out.append(c)
                i += 1
                continue
            name = template[i + 1 : j]
            if all(ch.isalnum() or ch == "_" for ch in name):
                out.append("$" + name)
                i = j + 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _legacy_prompt(
    *,
    source_lang: str,
    target: str,
    source_code: str,
    broken_code: str,
    error_message: str,
    attempt: int,
    max_attempts: int,
) -> str:
    """Build the legacy repair prompt as a fallback."""
    legacy_path = (
        Path(__file__).resolve().parent.parent / "llm" / "prompts" / "repair.md"
    )
    if legacy_path.exists():
        template = legacy_path.read_text()
    else:
        template = (
            "Original ({{source_lang}}):\n```\n{{source_code}}\n```\n\n"
            "Broken translation ({{target_lang}}, attempt {{attempt}}/{{max_passes}}):\n"
            "```\n{{broken_code}}\n```\n\n"
            "Error:\n```\n{{error_message}}\n```\n\n"
            "Output only the corrected {{target_lang}} code."
        )
    return (
        template.replace("{{source_lang}}", source_lang)
        .replace("{{target_lang}}", target)
        .replace("{{source_code}}", source_code)
        .replace("{{broken_code}}", broken_code)
        .replace("{{error_message}}", error_message)
        .replace("{{attempt}}", str(attempt))
        .replace("{{max_passes}}", str(max_attempts))
    )


# ---------------------------------------------------------------------------
# The escalating loop
# ---------------------------------------------------------------------------


def _call_llm(
    client: TieredLlmClient,
    prompt: str,
    *,
    tier: ModelTier,
) -> str:
    """Call *tier* with *prompt* (a pre-rendered string), return raw text.

    Caching is handled inside :meth:`TieredLlmClient.call`. The tier is
    expected to be non-CACHED — the CACHED path is taken via
    :meth:`TieredLlmClient.cache_lookup` before the call.
    """
    return client.call(prompt, tier=tier)


def escalating_repair(
    source: str,
    *,
    source_lang: str,
    target: str,
    tiered_client: TieredLlmClient,
    verifier: Verifier | None = None,
    initial_translate: Callable[[], str] | None = None,
    max_attempts: int = 5,
    flywheel: Flywheel | None = None,
    tracker: RepairTracker | None = None,
    enable_cache: bool = True,
) -> EscalatingRepairResult:
    """Run the verification-driven repair loop with escalating tiers.

    See the module docstring for the loop's invariants. Parameters:

    source, source_lang, target
        The source program and language pair.
    tiered_client
        A :class:`TieredLlmClient` with at least one non-CACHED tier.
    verifier
        Optional :class:`Verifier`. Defaults to the legacy compile gate
        for *target* (compile-only).
    initial_translate
        Callable that returns the first-pass translation. Defaults to
        :func:`transpilers.cli.main.transpile` (no LLM on the first
        attempt; the LLM is only consulted on retries).
    max_attempts
        Bounded retry budget. The loop stops at this count even if more
        tiers are available.
    flywheel
        Optional :class:`Flywheel` to record every verified repair.
    tracker
        Optional :class:`RepairTracker` (issue #51). Records one
        :class:`RepairOutcome` per unit — verdict ``algorithmic`` /
        ``llm`` / ``unrepaired`` — so the algorithmic-vs-LLM ratio is
        populated by the live loop, not just the batch corpus scripts.
        If ``None`` and ``$TRANSPILER_REPAIR_OUTCOMES_PATH`` is set, a
        tracker is created against that log.
    enable_cache
        When ``False``, the CACHED tier is bypassed (force a fresh LLM
        call). Useful for measuring the loop's incremental lift.
    """
    if verifier is None:
        verifier = _wrap_legacy_verifier(target)
    if initial_translate is None:
        from transpilers.cli.main import transpile as _default_translate

        def initial_translate() -> str:
            return _default_translate(source, source_lang=source_lang, target=target)

    if flywheel is None:
        import os

        path = os.environ.get("TRANSPILER_FLYWHEEL_PATH")
        if path:
            flywheel = Flywheel(path=path)

    if tracker is None:
        import os

        outcomes_path = os.environ.get("TRANSPILER_REPAIR_OUTCOMES_PATH")
        if outcomes_path:
            tracker = RepairTracker(log_path=outcomes_path)

    available = tiered_client.available_tiers()
    if not available:
        return _finalize(
            _trivial_loop(
                source=source,
                source_lang=source_lang,
                target=target,
                initial_translate=initial_translate,
                verifier=verifier,
                max_attempts=max_attempts,
                flywheel=flywheel,
            ),
            tracker=tracker,
            source=source,
            source_lang=source_lang,
            target=target,
        )

    non_cached = [t for t in available if t != ModelTier.CACHED]
    initial_tier = non_cached[0] if non_cached else available[0]
    retry_tiers = non_cached if non_cached else available

    history: list[EscalatingRepairAttempt] = []
    current_code = ""
    last_signal: RepairSignal | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            tier = initial_tier
        else:
            tier = retry_tiers[min(attempt - 2, len(retry_tiers) - 1)]

        if attempt == 1:
            current_code = initial_translate()
            cache_hit = False
            fix_applied = ""
        else:
            assert last_signal is not None
            prompt = _render_prompt(
                source_lang=source_lang,
                target=target,
                source_code=source,
                broken_code=current_code,
                signal=last_signal,
                attempt=attempt,
                max_attempts=max_attempts,
                tier=tier,
            )
            cache_hit = False
            fix_applied = ""
            if enable_cache:
                cached = tiered_client.cache_lookup(prompt, tier=ModelTier.CACHED)
                if cached is not None:
                    fix_applied = _strip_fences(cached)
                    cache_hit = True
                else:
                    for cached_tier in available:
                        if cached_tier == ModelTier.CACHED:
                            continue
                        cached = tiered_client.cache_lookup(prompt, tier=cached_tier)
                        if cached is not None:
                            fix_applied = _strip_fences(cached)
                            cache_hit = True
                            break
            if not cache_hit:
                try:
                    raw = _call_llm(tiered_client, prompt, tier=tier)
                except Exception as exc:
                    history.append(
                        EscalatingRepairAttempt(
                            attempt=attempt,
                            tier=tier,
                            code=current_code,
                            verified=False,
                            signal=last_signal,
                            error=f"llm error: {exc}",
                            fix_applied="",
                            cache_hit=False,
                        )
                    )
                    continue
                fix_applied = _strip_fences(raw)
            current_code = fix_applied or current_code

        import time as _time

        t0 = _time.monotonic()
        outcome = verifier(current_code)
        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        verified = outcome.ok
        error_msg = (
            outcome.signal.diagnostic if (outcome.signal and not verified) else ""
        )

        history.append(
            EscalatingRepairAttempt(
                attempt=attempt,
                tier=tier,
                code=current_code,
                verified=verified,
                signal=outcome.signal,
                error=error_msg,
                fix_applied=fix_applied,
                cache_hit=cache_hit,
                elapsed_ms=elapsed_ms,
            )
        )

        if verified:
            return _finalize(
                _on_verified_pass(
                    attempt=attempt,
                    tier=tier,
                    current_code=current_code,
                    history=history,
                    source=source,
                    source_lang=source_lang,
                    target=target,
                    last_signal=last_signal,
                    tiered_client=tiered_client,
                    flywheel=flywheel,
                    enable_cache=enable_cache,
                    max_attempts=max_attempts,
                ),
                tracker=tracker,
                source=source,
                source_lang=source_lang,
                target=target,
            )

        last_signal = outcome.signal or RepairSignal(
            kind="compile_error", diagnostic=error_msg, attempt=attempt
        )
        last_signal.attempt = attempt

    return _finalize(
        EscalatingRepairResult(
            code=current_code,
            passed=False,
            attempts=max_attempts,
            fixing_tier=available[min(len(available) - 1, max_attempts - 1)],
            history=history,
            refused=True,
        ),
        tracker=tracker,
        source=source,
        source_lang=source_lang,
        target=target,
    )


def _finalize(
    result: EscalatingRepairResult,
    *,
    tracker: RepairTracker | None,
    source: str,
    source_lang: str,
    target: str,
) -> EscalatingRepairResult:
    """Emit one :class:`RepairOutcome` per unit, then return *result* unchanged.

    The verdict is derived purely from *result*, so every terminal path
    (trivial loop, verified pass, refusal) records the same way:

    * not passed                 -> ``unrepaired``
    * passed on attempt 1        -> ``algorithmic`` (attempt 1 is always the
                                    no-LLM initial translation)
    * passed on a later attempt  -> ``llm`` (an LLM-derived fix verified)

    The loop never applies deterministic rule patches itself (those live in
    the batch corpus scripts), so the ``rule`` verdict is not produced here.
    """
    if tracker is None:
        return result

    import hashlib

    history = result.history
    # Count actual LLM round-trips: retries (attempt > 1) that were not served
    # from cache. Cache hits replay a prior LLM answer without a fresh call.
    n_llm_calls = sum(1 for h in history if h.attempt > 1 and not h.cache_hit)
    # Bucket / construct come from the last failing signal we saw.
    bucket = ""
    for h in reversed(history):
        if h.signal is not None and not h.verified:
            bucket = h.signal.bucket
            break

    if not result.passed:
        verdict = "unrepaired"
    elif result.attempts <= 1:
        verdict = "algorithmic"
    else:
        verdict = "llm"

    fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    tracker.record(
        RepairOutcome(
            source_lang=source_lang,
            target=target,
            fingerprint=fingerprint,
            bucket=bucket,
            verdict=verdict,
            n_llm_calls=n_llm_calls,
            n_repair_passes=max(0, result.attempts - 1),
            wallclock_ms=sum(h.elapsed_ms for h in history),
            notes=(
                f"fixing_tier={result.fixing_tier.value}"
                if result.fixing_tier
                else "fixing_tier=none"
            ),
        )
    )
    return result


def _on_verified_pass(
    *,
    attempt: int,
    tier: ModelTier,
    current_code: str,
    history: list[EscalatingRepairAttempt],
    source: str,
    source_lang: str,
    target: str,
    last_signal: RepairSignal | None,
    tiered_client: TieredLlmClient,
    flywheel: Flywheel | None,
    enable_cache: bool,
    max_attempts: int,
) -> EscalatingRepairResult:
    """Handle the verified-pass branch: pin to CACHED, record flywheel."""
    if enable_cache and last_signal is not None and attempt > 1:
        broken_code = history[-2].code if len(history) >= 2 else current_code
        prompt = _render_prompt(
            source_lang=source_lang,
            target=target,
            source_code=source,
            broken_code=broken_code,
            signal=last_signal,
            attempt=attempt,
            max_attempts=max_attempts,
            tier=tier,
        )
        tiered_client.cache_store(prompt, current_code, tier=ModelTier.CACHED)

    flywheel_recorded = False
    if flywheel is not None and attempt > 1 and last_signal is not None:
        broken_code = history[-2].code if len(history) >= 2 else current_code
        broken_attempts = [
            {
                "attempt": h.attempt,
                "tier": h.tier.value,
                "code": h.code,
                "error": h.error,
                "cache_hit": h.cache_hit,
            }
            for h in history[:-1]
        ]
        prompt = _render_prompt(
            source_lang=source_lang,
            target=target,
            source_code=source,
            broken_code=broken_code,
            signal=last_signal,
            attempt=attempt,
            max_attempts=max_attempts,
            tier=tier,
        )
        flywheel.record(
            source_lang=source_lang,
            target=target,
            source=source,
            broken_attempts=broken_attempts,
            fixed_code=current_code,
            fixing_tier=tier,
            fixing_signal=last_signal,
            fix_prompt=prompt,
        )
        flywheel_recorded = True

    return EscalatingRepairResult(
        code=current_code,
        passed=True,
        attempts=attempt,
        fixing_tier=tier,
        history=history,
        refused=False,
        flywheel_recorded=flywheel_recorded,
    )


def _trivial_loop(
    *,
    source: str,
    source_lang: str,
    target: str,
    initial_translate: Callable[[], str],
    verifier: Verifier,
    max_attempts: int,
    flywheel: Flywheel | None,
) -> EscalatingRepairResult:
    """Fallback path when no LLM is configured: transpile once, verify,
    refuse on failure."""
    import time as _time

    current_code = initial_translate()
    t0 = _time.monotonic()
    outcome = verifier(current_code)
    elapsed_ms = int((_time.monotonic() - t0) * 1000)
    attempt = EscalatingRepairAttempt(
        attempt=1,
        tier=ModelTier.CACHED,
        code=current_code,
        verified=outcome.ok,
        signal=outcome.signal,
        error=(
            outcome.signal.diagnostic if (outcome.signal and not outcome.ok) else ""
        ),
        cache_hit=False,
        elapsed_ms=elapsed_ms,
    )
    if outcome.ok:
        return EscalatingRepairResult(
            code=current_code,
            passed=True,
            attempts=1,
            fixing_tier=None,
            history=[attempt],
            refused=False,
        )
    return EscalatingRepairResult(
        code=current_code,
        passed=False,
        attempts=1,
        fixing_tier=None,
        history=[attempt],
        refused=True,
    )

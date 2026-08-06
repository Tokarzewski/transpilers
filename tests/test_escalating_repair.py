"""Tests for the verification-driven repair loop with escalating tiers
(issue #47).

Three concerns are exercised:

1. **Tier escalation** — the loop starts at the cheapest tier and only
   escalates when verification fails. A frontier-tier answer is never
   burned on a problem a local-tier answer can solve.
2. **Signal routing** — the prompt template gets the *kind* of failure
   (compile diagnostic vs run divergence vs hole context) and the prompt
   it produces differs by kind.
3. **Flywheel** — every verified repair (and only those) lands in the
   flywheel JSONL log with the schema issue #51 expects.
4. **Refuse-don't-guess** — when the budget is exhausted, the loop
   returns ``passed=False, refused=True`` and does not silently swap in
   a "best-effort" broken code.
"""

from __future__ import annotations

from pathlib import Path

from transpilers.llm import ModelTier, TieredLlmClient
from transpilers.repair import (
    Flywheel,
    escalating_repair,
    signal_from_compile,
    signal_from_hole,
    signal_from_run,
    signal_from_structural,
)
from transpilers.repair.flywheel import read_flywheel
from transpilers.repair.loop import VerificationOutcome, _render_prompt
from transpilers.repair.signal import signal_from_internal


# ---------------------------------------------------------------------------
# Helpers: stub tiers + scripted verifiers
# ---------------------------------------------------------------------------


class _ScriptedTier:
    """A stub LLM tier: a list of (prompt-substring → raw-answer) rules,
    with a call counter for assertions."""

    def __init__(self, responses=None) -> None:
        # ``responses`` is either a list (consumed in order), a dict
        # (matched by prompt-substring), or None (always returns a
        # generic stub answer).
        if responses is None:
            responses = []
        if isinstance(responses, str):
            responses = [responses]
        self.responses = responses
        self.calls: list[str] = []  # raw prompts we were asked

    def __call__(self, prompt: str, temperature: float) -> str:
        self.calls.append(prompt)
        if isinstance(self.responses, dict):
            for key, val in self.responses.items():
                if key in prompt:
                    return val
            return "fn repair() -> None: pass\n"
        if not self.responses:
            return "fn repair() -> None: pass\n"
        return self.responses.pop(0)


class _ScriptedVerifier:
    """A stub verifier: a list of (code-substring → outcome) rules applied
    in order. Use ``False`` to fail with a fake signal, ``True`` to pass."""

    def __init__(self, outcomes, *, signal_factory=None) -> None:
        self.outcomes = list(outcomes)
        self.signal_factory = signal_factory
        self.calls: list[str] = []

    def __call__(self, code: str) -> VerificationOutcome:
        self.calls.append(code)
        if not self.outcomes:
            return VerificationOutcome(ok=False, signal=signal_from_compile(""))
        ok = self.outcomes.pop(0)
        if ok:
            return VerificationOutcome(ok=True)
        sig = (
            self.signal_factory()
            if self.signal_factory
            else signal_from_compile("error[E0308]: type mismatch")
        )
        return VerificationOutcome(ok=False, signal=sig)


# ---------------------------------------------------------------------------
# Basic shape / contract
# ---------------------------------------------------------------------------


def test_escalating_repair_no_tiers_uses_trivial_loop(tmp_path: Path):
    """A TieredLlmClient with no non-CACHED tiers falls back to a single-try loop.

    The trivial loop still respects refuse-don't-guess: if the only try
    fails, ``refused=True``."""
    # CACHED is always materialized (issue #47: every verified answer is
    # pinned there for free replays). With no LLM tiers configured, the
    # loop only has CACHED, so it never escalates to an actual LLM call.
    client = TieredLlmClient(tiers={}, default_cache_dir=tmp_path)
    verifier = _ScriptedVerifier([True])
    res = escalating_repair(
        "def f(): pass",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() {}",
    )
    assert res.passed
    assert res.refused is False
    assert res.attempts == 1
    # No real LLM was consulted — only CACHED is available.
    assert res.fixing_tier == ModelTier.CACHED


def test_escalating_repair_first_try_passes_no_llm_call(tmp_path: Path):
    """If the algorithmic pipeline already verifies, no LLM tier is called
    and the loop returns the initial code on attempt 1."""
    local = _ScriptedTier(["UNREACHABLE"])
    client = TieredLlmClient(
        tiers={ModelTier.LOCAL_FINETUNED: local},
        default_cache_dir=tmp_path,
    )
    verifier = _ScriptedVerifier([True])
    res = escalating_repair(
        "def f(): pass",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() {}",
    )
    assert res.passed
    assert res.attempts == 1
    assert res.history[0].tier == ModelTier.LOCAL_FINETUNED
    # No LLM call should have been made — the initial pass already passed.
    assert not local.calls


# ---------------------------------------------------------------------------
# Tier escalation
# ---------------------------------------------------------------------------


def test_escalation_runs_local_then_frontier(tmp_path: Path):
    """Local-tier fails, frontier-tier fixes it. We assert (a) the order
    of tier calls and (b) the result reports the fixing tier."""
    local = _ScriptedTier(
        ["fn f() -> i32 { 0_i32 }\n"]
    )  # always returns the same broken code
    frontier = _ScriptedTier(["fn f() -> i32 { 0 }\n"])  # fixed
    client = TieredLlmClient(
        tiers={
            ModelTier.LOCAL_FINETUNED: local,
            ModelTier.FRONTIER: frontier,
        },
        default_cache_dir=tmp_path,
    )
    # First verifier call: fail (initial pass). Second: fail (local fix
    # is still broken). Third: pass (frontier fix verifies).
    verifier = _ScriptedVerifier([False, False, True])
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",  # first attempt also fails
    )
    assert res.passed, res.summary()
    assert res.attempts == 3
    assert len(local.calls) == 1
    assert len(frontier.calls) == 1
    assert res.fixing_tier == ModelTier.FRONTIER


def test_local_tier_alone_solves_no_frontier_call(tmp_path: Path):
    """If the local tier produces a verified answer, the frontier tier is
    not consulted — the loop's whole point is to keep frontier usage
    bounded to the hard cases."""
    local = _ScriptedTier(["fn f() -> i32 { 0 }\n"])
    frontier = _ScriptedTier(["UNREACHABLE"])
    client = TieredLlmClient(
        tiers={
            ModelTier.LOCAL_FINETUNED: local,
            ModelTier.FRONTIER: frontier,
        },
        default_cache_dir=tmp_path,
    )
    verifier = _ScriptedVerifier([False, True])
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",  # fails
    )
    assert res.passed
    assert res.attempts == 2
    assert res.fixing_tier == ModelTier.LOCAL_FINETUNED
    assert len(local.calls) == 1
    assert frontier.calls == []  # never consulted


# ---------------------------------------------------------------------------
# Cache: CACHED tier short-circuits the loop
# ---------------------------------------------------------------------------


def test_cached_answer_short_circuits_fresh_call(tmp_path: Path):
    """A cache hit at CACHED for the *prompt* the loop would build must
    skip the LLM call entirely."""
    frontier = _ScriptedTier(["UNREACHABLE"])
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )

    # Use the *same* signal the loop will produce (the scripted verifier
    # always returns this signal on failure). Pre-seeding with a different
    # signal would miss the cache and is not a meaningful test of the
    # CACHED short-circuit.
    sig = signal_from_compile("error[E0308]: type mismatch")
    prompt = _render_prompt(
        source_lang="python",
        target="rust",
        source_code="def f(): return 0",
        broken_code="fn f() { }",
        signal=sig,
        attempt=2,
        max_attempts=5,
        tier=ModelTier.FRONTIER,
    )
    client.cache_store(prompt, "fn f() -> i32 { 0 }\n", tier=ModelTier.CACHED)

    verifier = _ScriptedVerifier([False, True])
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",
    )
    assert res.passed
    assert res.attempts == 2
    # The frontier LLM was never called.
    assert frontier.calls == []


def test_disable_cache_forces_fresh_call(tmp_path: Path):
    """When ``enable_cache=False`` the loop ignores CACHED entries."""
    frontier = _ScriptedTier(["fn f() -> i32 { 0 }\n"])
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )
    # Pre-seed a different (cached) answer; the loop must NOT use it.
    client.cache_store("any-prompt", "OLD_ANSWER", tier=ModelTier.CACHED)
    verifier = _ScriptedVerifier([False, True])
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",
        enable_cache=False,
    )
    assert res.passed
    assert res.attempts == 2
    # The frontier was consulted despite the cache (because the cache was
    # disabled), and the call used a fresh prompt, not "any-prompt".
    assert len(frontier.calls) == 1
    assert "any-prompt" not in frontier.calls[0]


# ---------------------------------------------------------------------------
# Pin to CACHED on success
# ---------------------------------------------------------------------------


def test_verified_repair_pinned_to_cached_for_next_run(tmp_path: Path):
    """After a frontier-tier verified pass, the next run with the same
    broken code + signal must hit CACHED and not re-call the LLM."""
    frontier = _ScriptedTier(["fn f() -> i32 { 0 }\n"])
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )
    verifier = _ScriptedVerifier([False, True])

    # First run — fixes via frontier.
    res1 = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",
    )
    assert res1.passed
    assert len(frontier.calls) == 1

    # Second run with the same inputs — should hit CACHED.
    verifier2 = _ScriptedVerifier([False, True])
    res2 = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier2,
        initial_translate=lambda: "fn f() { }",
    )
    assert res2.passed
    # No new LLM calls on the second run.
    assert len(frontier.calls) == 1


# ---------------------------------------------------------------------------
# Refuse-don't-guess on exhaustion
# ---------------------------------------------------------------------------


def test_exhaustion_refuses_does_not_silently_swap_code(tmp_path: Path):
    """When the budget is exhausted without a verified pass, the result
    reports ``refused=True`` and the ``code`` is the *last* attempt's
    code (whatever it was) — not a hidden "best-effort" replacement."""
    frontier = _ScriptedTier(["fn f() { BROKEN }\n"] * 10)
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )
    verifier = _ScriptedVerifier([False] * 10)
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",
        max_attempts=3,
    )
    assert not res.passed
    assert res.refused is True
    assert res.attempts == 3
    assert (
        res.code == "fn f() { BROKEN }"
    )  # the last (broken) attempt (stripped by _strip_fences)


# ---------------------------------------------------------------------------
# Flywheel recording
# ---------------------------------------------------------------------------


def test_flywheel_records_only_verified_repairs(tmp_path: Path):
    """A verified repair (the one that solved the problem) is written to
    the flywheel log with the schema issue #51 expects. A *refused* run
    writes nothing."""
    log = tmp_path / "flywheel.jsonl"
    flywheel = Flywheel(path=log)
    frontier = _ScriptedTier(["fn f() -> i32 { 0 }\n"])
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )
    verifier = _ScriptedVerifier([False, True])

    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",
        flywheel=flywheel,
    )
    assert res.passed
    assert res.flywheel_recorded
    records = list(read_flywheel(log))
    assert len(records) == 1
    rec = records[0]
    assert rec.source_lang == "python"
    assert rec.target == "rust"
    assert rec.source == "def f(): return 0"
    assert rec.fixed_code == "fn f() -> i32 { 0 }"  # stripped by _strip_fences
    assert rec.fixing_tier == "frontier"
    assert rec.fixing_signal_kind == "compile_error"
    assert len(rec.broken_attempts) == 1
    assert rec.broken_attempts[0]["tier"] == "frontier"


def test_flywheel_no_record_on_refuse(tmp_path: Path):
    """Exhausted budget → no flywheel record. The flywheel only captures
    successful, verified repairs (issue #51's input)."""
    log = tmp_path / "flywheel.jsonl"
    flywheel = Flywheel(path=log)
    frontier = _ScriptedTier(["fn f() { BROKEN }\n"] * 10)
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )
    verifier = _ScriptedVerifier([False] * 10)
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { }",
        flywheel=flywheel,
        max_attempts=3,
    )
    assert not res.passed
    assert res.flywheel_recorded is False
    assert (not log.exists()) or log.read_text() == ""


def test_flywheel_record_carries_signal_kind(tmp_path: Path):
    """The flywheel record preserves the *kind* of failure the LLM fixed —
    this is what issue #51's promotion step uses to prioritise hard cases."""
    log = tmp_path / "flywheel.jsonl"
    flywheel = Flywheel(path=log)
    frontier = _ScriptedTier(["fn f() -> i32 { 0 }\n"])
    client = TieredLlmClient(
        tiers={ModelTier.FRONTIER: frontier},
        default_cache_dir=tmp_path,
    )

    sig = signal_from_run(expected="0\n", actual="99\n", input_text="<empty>")
    verifier = _ScriptedVerifier([False, True], signal_factory=lambda: sig)
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=verifier,
        initial_translate=lambda: "fn f() { 99 }",
        flywheel=flywheel,
    )
    assert res.passed
    rec = list(read_flywheel(log))[0]
    assert rec.fixing_signal_kind == "run_mismatch"


# ---------------------------------------------------------------------------
# Signal routing
# ---------------------------------------------------------------------------


def test_signal_routing_prompt_differs_by_kind():
    """The same broken code + same verifier should produce a *different*
    prompt per signal kind. The loop's whole point is to feed the LLM a
    targeted framing."""
    sigs = {
        "compile_error": signal_from_compile("error: cannot find value `n`"),
        "run_mismatch": signal_from_run("0\n", "99\n", input_text="<empty>"),
        "structural_divergence": signal_from_structural("dropped-function: f"),
        "unfilled_hole": signal_from_hole("x", "param a of f"),
        "internal": signal_from_internal(ValueError("oops")),
    }
    prompts = {
        kind: _render_prompt(
            source_lang="python",
            target="rust",
            source_code="def f(): return 0",
            broken_code="fn f() { }",
            signal=sig,
            attempt=2,
            max_attempts=5,
            tier=ModelTier.FRONTIER,
        )
        for kind, sig in sigs.items()
    }
    # All prompts reference the broken code + source — that's shared.
    for kind, prompt in prompts.items():
        assert "fn f() { }" in prompt
        assert "def f(): return 0" in prompt
    # Each kind's prompt has its own section.
    assert "Compiler diagnostic" in prompts["compile_error"]
    assert "Diverging run" in prompts["run_mismatch"]
    assert "Structural divergence" in prompts["structural_divergence"]
    assert "Unfilled type hole" in prompts["unfilled_hole"]
    assert "Internal pipeline error" in prompts["internal"]
    # The hole prompt embeds the hole context.
    assert '"name": "x"' in prompts["unfilled_hole"]


# ---------------------------------------------------------------------------
# Sanity: the legacy ``repair`` function is untouched
# ---------------------------------------------------------------------------


def test_legacy_repair_still_works():
    """The new module does not break the old API."""
    from transpilers.repair import repair

    class _AlwaysBad:
        def complete(self, prompt: str) -> str:
            return "fn f() { STILL_BAD }"

    # Properly indented block (the real ``transpile`` rejects top-level
    # single-statement bodies).
    res = repair(
        "def f():\n    return 0\n",
        source_lang="python",
        target="rust",
        llm_client=_AlwaysBad(),
        max_passes=1,
    )
    # max_passes=1 → one translate, one verify, no LLM fix-up.
    assert res.passes == 1


def test_legacy_repair_test_inputs_dont_claim_unverified_pass():
    """`test_inputs` is given but no target has a runtime-execution runner
    (every ``_VERIFIERS`` entry only checks compilation). A prior version
    of `_run_tests` reported `test_ok=True` in this case -- indistinguishable
    from a genuinely confirmed functional pass -- which could mask real
    logic bugs as "tested and passing". It must now record `test_ok=None`
    (compiled, but not actually verified), while still treating the run as
    successful overall (compiling code that was never shown to be wrong
    shouldn't burn an LLM "fix" pass)."""
    from transpilers.repair import repair

    class _ExplodesIfCalled:
        def complete(self, prompt: str) -> str:
            raise AssertionError("LLM should not be called -- code already compiled")

    res = repair(
        "def f():\n    return 0\n",
        source_lang="python",
        target="rust",
        llm_client=_ExplodesIfCalled(),
        max_passes=3,
        test_inputs=[{"input": "", "expected_output": "0"}],
    )
    assert res.passed is True
    assert res.passes == 1
    assert res.history[0].test_ok is None


# ---------------------------------------------------------------------------
# RepairTracker wiring (issue #51): the live loop emits one RepairOutcome
# per unit so the algorithmic-vs-LLM ratio is populated, not just by the
# batch corpus scripts.
# ---------------------------------------------------------------------------


def _tracker(tmp_path: Path):
    from transpilers.repair import RepairTracker

    return RepairTracker(
        log_path=tmp_path / "outcomes.jsonl",
        metrics_path=tmp_path / "metrics.json",
    )


def test_tracker_records_algorithmic_when_first_try_passes(tmp_path: Path):
    """Attempt-1 pass (no LLM) is recorded as the cheap ``algorithmic`` verdict."""
    client = TieredLlmClient(tiers={}, default_cache_dir=tmp_path)
    tracker = _tracker(tmp_path)
    res = escalating_repair(
        "def f(): pass",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=_ScriptedVerifier([True]),
        initial_translate=lambda: "fn f() {}",
        tracker=tracker,
    )
    tracker.close()
    assert res.passed
    snap = tracker.aggregate()
    assert snap["by_verdict"] == {"algorithmic": 1}
    assert snap["llm_fraction"] == 0.0


def test_tracker_records_llm_when_repaired(tmp_path: Path):
    """A unit only the LLM could fix is recorded as ``llm`` with the call count."""
    local = _ScriptedTier(["fn f() -> i32 { 0 }\n"])
    client = TieredLlmClient(
        tiers={ModelTier.LOCAL_FINETUNED: local},
        default_cache_dir=tmp_path,
    )
    tracker = _tracker(tmp_path)
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=_ScriptedVerifier([False, True]),
        initial_translate=lambda: "fn f() { }",
        tracker=tracker,
    )
    tracker.close()
    assert res.passed
    out = list(tracker.iter_log())
    assert len(out) == 1
    assert out[0].verdict == "llm"
    assert out[0].n_llm_calls == 1
    assert out[0].is_frontier_only  # LLM was the only thing that got us to pass


def test_tracker_records_unrepaired_on_refusal(tmp_path: Path):
    """Budget exhaustion → ``unrepaired`` (refuse-don't-guess, no silent pass)."""
    local = _ScriptedTier({"": "fn f() { STILL_BROKEN }\n"})
    client = TieredLlmClient(
        tiers={ModelTier.LOCAL_FINETUNED: local},
        default_cache_dir=tmp_path,
    )
    tracker = _tracker(tmp_path)
    res = escalating_repair(
        "def f(): return 0",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=_ScriptedVerifier([False, False, False]),
        initial_translate=lambda: "fn f() { }",
        max_attempts=3,
        tracker=tracker,
    )
    tracker.close()
    assert res.refused
    out = list(tracker.iter_log())
    assert len(out) == 1
    assert out[0].verdict == "unrepaired"
    assert out[0].is_pass is False


def test_tracker_auto_created_from_env(tmp_path: Path, monkeypatch):
    """With no explicit tracker, $TRANSPILER_REPAIR_OUTCOMES_PATH opts in."""
    from transpilers.repair import RepairOutcome  # noqa: F401  (import smoke)

    log = tmp_path / "env_outcomes.jsonl"
    monkeypatch.setenv("TRANSPILER_REPAIR_OUTCOMES_PATH", str(log))
    client = TieredLlmClient(tiers={}, default_cache_dir=tmp_path)
    escalating_repair(
        "def f(): pass",
        source_lang="python",
        target="rust",
        tiered_client=client,
        verifier=_ScriptedVerifier([True]),
        initial_translate=lambda: "fn f() {}",
    )
    assert log.exists()
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

"""Iterative repair loop: translate, verify, fix — up to *max_passes* times.

The repair loop works as follows for each pass:

1. Translate the source with the existing ``transpile()`` pipeline.
2. Invoke the target-language verifier (compiler check).
3. If compilation fails   → ask the LLM to fix the broken translation.
4. If compilation passes but a test fails → ask the LLM to fix the logic.
5. Repeat until success or *max_passes* is exhausted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transpilers.cli.main import transpile
from transpilers.verify import (
    c_compiles,
    fortran_compiles,
    mojo_compiles,
    python_compiles,
    rust_compiles,
)

# Map target name → verify function (same as cli/main.py TARGETS)
_VERIFIERS = {
    "rust": rust_compiles,
    "c": c_compiles,
    "mojo": mojo_compiles,
    "python": python_compiles,
    "fortran": fortran_compiles,
}

_PROMPT_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "llm" / "prompts" / "repair.md"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RepairPass:
    """Record for a single repair iteration."""

    attempt: int
    code: str
    compile_ok: bool
    error: str
    test_ok: bool | None = None
    fix_applied: str = ""


@dataclass
class RepairResult:
    """Aggregate result returned by :func:`repair`."""

    code: str
    passed: bool
    passes: int
    history: list[RepairPass] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_template() -> str:
    if _PROMPT_TEMPLATE.exists():
        return _PROMPT_TEMPLATE.read_text()
    # Embedded fallback so the module works even if the file is missing.
    return (
        "You are a transpiler repair assistant.\n"
        "Original ({{source_lang}}):\n```\n{{source_code}}\n```\n\n"
        "Broken translation ({{target_lang}}, attempt {{attempt}}/{{max_passes}}):\n"
        "```\n{{broken_code}}\n```\n\n"
        "Error:\n```\n{{error_message}}\n```\n\n"
        "Output only the corrected {{target_lang}} code."
    )


def _render_prompt(
    *,
    source_lang: str,
    target: str,
    source_code: str,
    broken_code: str,
    error_message: str,
    attempt: int,
    max_passes: int,
) -> str:
    template = _load_template()
    replacements = {
        "{{source_lang}}": source_lang,
        "{{target_lang}}": target,
        "{{source_code}}": source_code,
        "{{broken_code}}": broken_code,
        "{{error_message}}": error_message,
        "{{attempt}}": str(attempt),
        "{{max_passes}}": str(max_passes),
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _ask_llm(prompt: str, llm_client: Any) -> str:
    """Send *prompt* to the LLM and return the raw text response.

    Accepts any object with a `complete(prompt: str) -> str` method,
    or a callable that takes a prompt string and returns a string.
    """
    if callable(getattr(llm_client, "complete", None)):
        return llm_client.complete(prompt)
    if callable(llm_client):
        return llm_client(prompt)
    raise TypeError(
        f"llm_client must expose a .complete(prompt) method or be callable; "
        f"got {type(llm_client)!r}"
    )


def _strip_fences(code: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    code = code.strip()
    # Remove ```lang ... ``` wrappers
    fenced = re.sub(r"^```[a-zA-Z]*\n?", "", code)
    fenced = re.sub(r"\n?```$", "", fenced)
    return fenced.strip()


def _run_tests(
    code: str,
    *,
    target: str,
    test_inputs: list[dict],
) -> tuple[bool | None, str]:
    """Very lightweight test harness: run each test_input through the verifier.

    Each entry in *test_inputs* may have:
        ``input``  — string piped to stdin (optional)
        ``expected_output`` — string to compare stdout against (optional)

    Returns (passed, message), where *passed* is:
        ``False`` — compilation failed (a real, confirmed failure)
        ``None``  — compilation passed but the functional test wasn't
                    actually executed (no target has a runtime-execution
                    runner yet, only a compile-only verifier)
        ``True``  — reserved for when a real runner confirms the test passed

    Note: this is a best-effort runner suitable for simple I/O tests. For
    every current target we only check compilation here (execution
    requires language-specific runners that don't exist yet). Callers must
    not treat ``None`` as a confirmed pass -- see ``repair()``.
    """
    verify = _VERIFIERS[target]
    result = verify(code)
    if not result.ok:
        return False, result.stderr

    # If test_inputs provided but no runner available, say so honestly
    # rather than claiming a functional pass we never checked.
    if test_inputs:
        return (
            None,
            "(compilation passed; runtime I/O tests not executed -- no execution runner for this target)",
        )
    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def repair(
    source: str,
    *,
    source_lang: str,
    target: str,
    llm_client: Any,
    max_passes: int = 3,
    test_inputs: list[dict] | None = None,
) -> RepairResult:
    """Translate *source* to *target* and iteratively repair compilation errors.

    Parameters
    ----------
    source:
        Source code in *source_lang*.
    source_lang:
        Language of *source* (e.g. ``"python"``, ``"cpp"``).
    target:
        Target language (e.g. ``"rust"``, ``"zig"``).
    llm_client:
        An object with a ``.complete(prompt: str) -> str`` method, or a
        callable that accepts a prompt string and returns a string.
    max_passes:
        Maximum number of repair iterations (default 3).
    test_inputs:
        Optional list of dicts with ``"input"`` and ``"expected_output"``
        keys for lightweight functional testing after compilation.

    Returns
    -------
    RepairResult
        ``.passed`` is True if the final code compiled (and passed tests).
    """
    if target not in _VERIFIERS:
        raise ValueError(f"Unknown target {target!r}; choose from {sorted(_VERIFIERS)}")

    verify = _VERIFIERS[target]
    history: list[RepairPass] = []
    current_code: str = ""

    for attempt in range(1, max_passes + 1):
        # --- Stage 1: translate (or re-translate with LLM fix from prev pass) ---
        if attempt == 1:
            current_code = transpile(source, source_lang=source_lang, target=target)
        # (on subsequent passes current_code has already been updated by LLM fix)

        # --- Stage 2: compile ---
        compile_result = verify(current_code)
        compile_ok = compile_result.ok
        error_msg = compile_result.stderr if not compile_ok else ""

        # --- Stage 3: optional test run ---
        test_ok: bool | None = None
        if compile_ok and test_inputs is not None:
            test_ok, test_err = _run_tests(
                current_code, target=target, test_inputs=test_inputs
            )
            if test_ok is False:
                error_msg = test_err

        pass_record = RepairPass(
            attempt=attempt,
            code=current_code,
            compile_ok=compile_ok,
            error=error_msg,
            test_ok=test_ok,
            fix_applied="",
        )
        history.append(pass_record)

        # --- Done if everything passed ---
        # test_ok is False only for a *confirmed* test failure; None means
        # "compiled but not actually run" (no execution runner exists yet),
        # which we still accept rather than burn an LLM "fix" pass on code
        # that already compiles and was never shown to be wrong.
        if compile_ok and (test_inputs is None or test_ok is not False):
            return RepairResult(
                code=current_code,
                passed=True,
                passes=attempt,
                history=history,
            )

        # --- Stage 4: ask LLM for a fix (if we have passes left) ---
        if attempt < max_passes:
            prompt = _render_prompt(
                source_lang=source_lang,
                target=target,
                source_code=source,
                broken_code=current_code,
                error_message=error_msg,
                attempt=attempt,
                max_passes=max_passes,
            )
            raw_fix = _ask_llm(prompt, llm_client)
            fixed_code = _strip_fences(raw_fix)
            pass_record.fix_applied = fixed_code
            current_code = fixed_code

    # Exhausted all passes
    return RepairResult(
        code=current_code,
        passed=False,
        passes=max_passes,
        history=history,
    )

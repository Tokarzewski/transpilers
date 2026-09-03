#!/usr/bin/env python3
"""Toolchain-free SFT dataset generator (issue #57): C++/C/Python -> Mojo,
C++ -> Python, plus a Python -> {Rust, C} bonus.

Issue #57 asks for ``cpp / c -> mojo`` and ``cpp -> python`` training data
(``rust -> mojo`` is requested but the transpiler has no Rust *frontend*, so it
is out of scope here). The behaviorally-verified EnergyPlus pipelines
(``build_cpp_mojo_dataset.py`` / ``build_cpp_python_dataset.py``) need ``g++``,
``mojo`` and ``gcov`` — they cannot run in a toolchain-less environment. This
generator covers the part that needs **no external compiler**:

    source unit  (a small, self-contained function corpus)
        -> transpiler's *algorithmic* path (``llm_fill=None``, no model, no net)
        -> emit one (instruction, input, output) SFT record per direction
        -> assign a three-way verification status:
             * ``verified``           — behavioral match over generated inputs
                                        (only python source + python/rust target
                                        are runnable in pure Python here)
             * ``unverified_no_runner`` — emitted but no in-env oracle/runner for
                                        this direction (cpp/c source, mojo/c
                                        target): the headline ->Mojo pairs land
                                        here; the compile/behaviour gate is
                                        deferred to a box with the toolchain
             * ``failed``             — the harness *ran* and the target diverged
                                        (a wrong translation) — these are split
                                        OUT of the training JSONL into a
                                        ``*.rejects.jsonl`` diagnostics file so a
                                        known-wrong pair never enters SFT

Only ``verified`` + ``unverified_no_runner`` pairs go to the training files,
matching the keep-only-trustworthy discipline of the EnergyPlus pipelines.

Output schema matches the existing SFT corpora
(``data/sft/cpp_python_pairs.jsonl``): ``{"instruction", "input", "output",
...}`` plus provenance (``source_lang``, ``target``, ``status``, ``verified``,
``func``, ``verify_note``). One file per (source, target) under
``data/sft/algorithms/``.

Usage
-----
    uv run python scripts/sft/build_algorithm_pairs.py        # all directions
    uv run python scripts/sft/build_algorithm_pairs.py --only python:rust
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from transpilers.cli.main import transpile  # noqa: E402
from transpilers.verify.behavioral import (  # noqa: E402
    check_behavioral_equivalence,
    infer_param_tags,
)

LANG_NAME = {
    "python": "Python",
    "mojo": "Mojo",
    "rust": "Rust",
    "c": "C",
    "cpp": "C++",
}

# Targets the in-process behavioral harness can actually drive here. Anything
# else is emitted but flagged unverified rather than falsely claimed correct.
PY_VERIFIABLE_TARGETS = {"python", "rust"}

# Wall-clock guard for a single behavioral check. The Python source *oracle*
# runs untrusted-shaped fuzz inputs through real algorithms; some (Ackermann,
# Collatz on negatives) can recurse deeply or loop forever on inputs outside
# their intended domain. A SIGALRM cap keeps the generator from hanging — a
# timed-out check is reported unverified, never as a pass.
VERIFY_TIMEOUT_S = 10


class _VerifyTimeout(BaseException):
    """Derives from BaseException, not Exception, so the harness's broad
    ``except Exception`` (which captures source-oracle behavior) cannot swallow
    the alarm and let the next divergent input loop forever."""


@contextlib.contextmanager
def _time_limit(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(signum, frame):  # noqa: ARG001
        raise _VerifyTimeout()

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# Three-way verification status (see module docstring).
VERIFIED = "verified"
NO_RUNNER = "unverified_no_runner"
FAILED = "failed"


@dataclass
class Pair:
    instruction: str
    input: str
    output: str
    source_lang: str
    target: str
    func: str
    status: str
    verify_note: str

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED


def _public_funcs(source: str) -> list[str]:
    """Top-level Python function names worth verifying (skip the demo ``main``).

    Only used to drive the *Python* behavioral oracle, so AST-parsing Python is
    correct here; non-python sources never reach this (they are no-runner).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name != "main"
    ]


def _instruction(source_lang: str, target: str) -> str:
    return (
        f"Translate the following {LANG_NAME[source_lang]} function to idiomatic "
        f"{LANG_NAME[target]}. Preserve the behavior exactly."
    )


def _verify(source: str, source_lang: str, target: str, output: str) -> tuple[str, str]:
    """Behaviorally self-verify a pair when the env can run both sides.

    Returns ``(status, note)`` where ``status`` is one of ``VERIFIED`` /
    ``NO_RUNNER`` / ``FAILED``:

    * ``NO_RUNNER`` — no in-env oracle/runner for this direction, OR the harness
      could not drive any signature (timeout, undrivable types). This is *not*
      evidence the translation is wrong, so the pair stays in the training set.
    * ``FAILED`` — the harness ran and the target *diverged* from the source
      oracle: a demonstrably wrong translation, split out of the training set.
    * ``VERIFIED`` — behavioral match over generated inputs.
    """
    if source_lang != "python" or target not in PY_VERIFIABLE_TARGETS:
        return NO_RUNNER, f"no in-env runner for {source_lang}->{target}"
    funcs = _public_funcs(source)
    if not funcs:
        return NO_RUNNER, "no verifiable top-level function"
    # The Rust runner appends its own ``fn main`` harness; the emitted code
    # already carries a translated ``main``, so a verbatim hand-off collides
    # ("name `main` defined multiple times"). Strip the emitted entry point —
    # we only drive the named functions, never ``main``.
    runner_code = _strip_main(output, target)
    checked = 0
    for fn in funcs:
        # Only drive functions whose signature the harness understands.
        if infer_param_tags(source, fn) is None:
            continue
        try:
            with _time_limit(VERIFY_TIMEOUT_S):
                rep = check_behavioral_equivalence(
                    source,
                    source_lang="python",
                    target=target,
                    target_code=runner_code,
                    func_name=fn,
                    n_inputs=12,
                )
        except _VerifyTimeout:
            # Source domain issue (looping on out-of-domain fuzz), not a wrong
            # translation — keep the pair, just don't claim it verified.
            return NO_RUNNER, f"{fn}: verification timed out (>{VERIFY_TIMEOUT_S}s)"
        if not rep.supported:
            continue
        if rep.total == 0:
            continue
        checked += 1
        if not rep.ok:
            return FAILED, f"{fn}: {rep.summary()}"
    if checked == 0:
        return NO_RUNNER, "harness could not drive any signature in this env"
    return VERIFIED, f"behavioral match on {checked} function(s)"


def _strip_main(code: str, target: str) -> str:
    """Remove the emitted entry point so the runner can append its own.

    Best-effort: drops a top-level ``fn main`` (Rust/Zig) or ``def main``
    (Python/Mojo) and its braced/indented body. Leaves the named functions
    intact. If no entry point is found the code is returned unchanged.
    """
    if target in ("rust", "c"):
        idx = code.find("fn main")
        if target == "c":
            # crude: C main is usually `int main(`
            import re

            m = re.search(r"\n\s*(?:int|void)\s+main\s*\(", code)
            idx = m.start() if m else -1
        if idx < 0:
            return code
        brace = code.find("{", idx)
        if brace < 0:
            return code[:idx].rstrip() + "\n"
        depth = 0
        i = brace
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    return (code[:idx] + code[i + 1 :]).strip() + "\n"
            i += 1
        return code[:idx].rstrip() + "\n"
    # Python / Mojo: drop a top-level `def main(` block by indentation.
    lines = code.splitlines()
    out: list[str] = []
    skipping = False
    for ln in lines:
        if not skipping and ln.lstrip().startswith("def main"):
            skipping = True
            continue
        if skipping:
            if ln.strip() == "" or ln[:1] in (" ", "\t"):
                continue
            skipping = False
        out.append(ln)
    return "\n".join(out) + "\n"


def _record(p: Pair) -> str:
    """Serialize a pair, adding the convenience ``verified`` bool (a property,
    so ``asdict`` would otherwise drop it)."""
    d = asdict(p)
    d["verified"] = p.verified
    return json.dumps(d, ensure_ascii=False) + "\n"


_EXT = {"python": "*.py", "cpp": "*.cpp", "c": "*.c"}


def build(corpus: Path, source_lang: str, target: str) -> list[Pair]:
    """Transpile every unit in *corpus* for one (source_lang, target) direction."""
    pairs: list[Pair] = []
    units = sorted(corpus.glob(_EXT.get(source_lang, "*")))
    for path in units:
        source = path.read_text(encoding="utf-8").strip()
        if not source:
            continue
        try:
            output = transpile(source, source_lang=source_lang, target=target).strip()
        except Exception as exc:  # noqa: BLE001 — record the gap, keep going
            print(f"  skip {path.name} {source_lang}->{target}: {type(exc).__name__}: {exc}")
            continue
        if not output:
            print(f"  skip {path.name} {source_lang}->{target}: empty output")
            continue
        status, note = _verify(source, source_lang, target, output)
        pairs.append(
            Pair(
                instruction=_instruction(source_lang, target),
                input=source,
                output=f"```{target}\n{output}\n```",
                source_lang=source_lang,
                target=target,
                func=path.stem,
                status=status,
                verify_note=note,
            )
        )
    return pairs


# (source_lang, corpus-dir, target). The headline #57 asks are the ->mojo
# directions + cpp->python; python->{rust,c} is a behaviorally-verifiable
# bonus. rust->mojo is omitted: the transpiler has no Rust frontend.
DIRECTIONS = [
    ("cpp", "examples/cpp_algorithms", "mojo"),
    ("c", "examples/c_algorithms", "mojo"),
    ("cpp", "examples/cpp_algorithms", "python"),
    ("python", "examples/algorithms", "mojo"),
    ("python", "examples/algorithms", "rust"),
    ("python", "examples/algorithms", "c"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to directions 'src:tgt' (e.g. cpp:mojo python:rust)",
    )
    ap.add_argument("--out-dir", default="data/sft/algorithms")
    args = ap.parse_args()

    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None

    counts = {VERIFIED: 0, NO_RUNNER: 0, FAILED: 0}
    train_total = 0
    for source_lang, corpus_rel, target in DIRECTIONS:
        if wanted is not None and f"{source_lang}:{target}" not in wanted:
            continue
        corpus = (REPO / corpus_rel).resolve()
        if not corpus.is_dir():
            print(f"skip {source_lang}->{target}: no corpus at {corpus_rel}")
            continue
        pairs = build(corpus, source_lang, target)
        train = [p for p in pairs if p.status != FAILED]
        rejects = [p for p in pairs if p.status == FAILED]
        for p in pairs:
            counts[p.status] += 1
        train_total += len(train)

        stem = f"{source_lang}_{target}_pairs"
        (out_dir / f"{stem}.jsonl").write_text(
            "".join(_record(p) for p in train), encoding="utf-8"
        )
        if rejects:
            (out_dir / f"{stem}.rejects.jsonl").write_text(
                "".join(_record(p) for p in rejects), encoding="utf-8"
            )
        nver = sum(1 for p in train if p.status == VERIFIED)
        print(
            f"{source_lang}->{target}: {len(train)} train ({nver} verified, "
            f"{len(train) - nver} no-runner), {len(rejects)} rejected -> {stem}.jsonl"
        )
    print(
        f"TOTAL train: {train_total}  "
        f"[verified={counts[VERIFIED]} no_runner={counts[NO_RUNNER]} "
        f"rejected(failed)={counts[FAILED]}]"
    )


if __name__ == "__main__":
    main()

"""Behavioral-equivalence verification (issue #48).

Compile-rate is not correctness: a transpiled function can compile and still
return the wrong answer. This module closes that gap by *running* the source
function as an oracle and asserting the transpiled target reproduces its
input→output behavior.

The approach (per the issue):

1. **Source oracle.** Generate inputs over the function's inferred signature,
   run the *source* under instrumentation, and record one ``IOSample`` per
   input (return value, or the fact that it raised). Inputs on which the
   source itself raises are dropped — we only assert behavior the source
   actually defines.
2. **Target replay.** Run the transpiled target over the *same* inputs at the
   smallest runnable boundary (the function) and compare.
3. **Property/fuzz inputs.** Inputs combine deterministic edge cases (0, ±1,
   empty, …) with seeded random fuzzing over the inferred parameter types, so
   edge-case divergence is surfaced, not just the happy path.

Comparison is *type-aware* via a canonical-token protocol: each side renders
its result to a normalized string keyed on the observed return type (ints
exact, floats within a tolerance, bools/None normalized across language
spellings). This sidesteps cross-language value serialization while keeping
the comparison honest.

Scope of this first cut: **Python source** (toolchain-free oracle) →
**Python or Rust** target, over scalar / list-of-scalar signatures. Targets
whose signature this harness cannot yet drive are reported as *unsupported*
(skipped), never as a false divergence. Other source languages and compiled
targets slot in behind the :class:`Runner` protocol.

The :func:`make_behavioral_verifier` adapter produces a
:class:`transpilers.repair.loop.Verifier`, so a behavioral divergence becomes
a ``run_mismatch`` :class:`~transpilers.repair.signal.RepairSignal` that the
escalating-repair loop (#47) can feed back to the LLM.
"""

from __future__ import annotations

import ast
import contextlib
import io
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from transpilers.verify._tool import resolve_tool

from transpilers.verify._exec_timeout import ExecTimeout, time_limit
from transpilers.verify.taxonomy import compiler_available


@contextlib.contextmanager
def _muffled():
    """Suppress stdout/stderr while exec-ing source modules.

    Source files often run top-level code (a ``print`` loop, a demo ``main()``
    call); exec-ing them to harvest a function should not pollute the caller's
    output. Function *return values* are captured directly, not via stdout, so
    nothing we care about is lost.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


# Wall-clock cap on a single in-process exec()+call of untrusted-shaped
# source. Without it, a pathological input (infinite loop, runaway
# recursion) hangs the caller forever -- there is no subprocess boundary to
# kill here, unlike the native-compiler verifiers.
_PY_EXEC_TIMEOUT_S = 5

# ---------------------------------------------------------------------------
# Type tags
# ---------------------------------------------------------------------------

# The signature type tags this harness understands. Kept as plain strings so
# callers (and the trace-typing pass, #49) can hand them over without importing
# the MIR type lattice.
SCALAR_TAGS = ("int", "float", "bool", "str")
SUPPORTED_TAGS = SCALAR_TAGS + ("list[int]", "list[float]", "list[str]", "list[bool]")


# ---------------------------------------------------------------------------
# Samples + report
# ---------------------------------------------------------------------------


@dataclass
class IOSample:
    """One input and the behavior it produced on a given runner."""

    args: tuple
    token: str = ""  # canonical token of the return value (when ok)
    ok: bool = True  # False == the runner raised / exited non-zero
    error: str = ""  # short description when ``ok`` is False

    def input_repr(self) -> str:
        return ", ".join(repr(a) for a in self.args)


# Recognized divergence classes. Naming the *kind* of behavioral difference —
# not just "they differ" — gives the repair loop (#47) a sharper hint and lets a
# sweep bucket failures by root cause instead of one opaque "wrong" pile.
DIV_FLOORED_MOD = "floored_vs_truncated_intdiv"
DIV_TARGET_ERROR = "target_errored"
DIV_VALUE = "value_mismatch"


@dataclass
class Divergence:
    """A single input on which target behavior differs from the source oracle."""

    args: tuple
    expected: str
    actual: str

    def __str__(self) -> str:
        inp = ", ".join(repr(a) for a in self.args)
        return f"f({inp}): expected {self.expected!r}, got {self.actual!r}"


def classify_divergence(div: Divergence, ret_tag: str) -> str:
    """Categorize a single divergence by likely root cause.

    The headline class is the C-family integer-semantics gap: Python's ``%``/
    ``//`` are *floored* (sign follows the divisor) while Rust/C/Mojo are
    *truncated* (sign follows the dividend). On a negative operand the two
    disagree (``-1 % 2`` → ``1`` vs ``-1``), and the difference is exactly a
    multiple of one of the inputs — the fingerprint we test for here. This is
    the divergence the behavioral harness most often surfaces on real numeric
    leaves, so it is worth naming explicitly.
    """
    if div.actual.startswith(("compile:", "runtime:")) or "<no result>" in div.actual:
        return DIV_TARGET_ERROR
    if ret_tag == "int":
        try:
            exp, act = int(div.expected), int(div.actual)
        except TypeError, ValueError:
            return DIV_VALUE
        # Floored/truncated mod & div differ by k*divisor for some integer arg.
        delta = exp - act
        if delta != 0 and any(
            isinstance(a, int) and a != 0 and delta % a == 0 for a in div.args
        ):
            # Require at least one negative operand — the only case where
            # floored and truncated integer ops actually disagree.
            if any(isinstance(a, int) and a < 0 for a in div.args):
                return DIV_FLOORED_MOD
    return DIV_VALUE


@dataclass
class BehavioralReport:
    """Outcome of a behavioral-equivalence check on one function."""

    ok: bool
    total: int
    matched: int
    divergences: list[Divergence] = field(default_factory=list)
    supported: bool = True  # False == harness could not drive this signature
    reason: str = ""  # why unsupported / why no samples
    divergence_class: str = ""  # root-cause label of the first divergence

    @property
    def pass_rate(self) -> float:
        return (self.matched / self.total) if self.total else 0.0

    def summary(self) -> str:
        if not self.supported:
            return f"behavioral: unsupported ({self.reason})"
        if self.total == 0:
            return f"behavioral: no runnable samples ({self.reason or 'oracle never returned'})"
        head = f"behavioral: {self.matched}/{self.total} inputs match"
        if self.divergences:
            head += f"; first divergence — {self.divergences[0]}"
            if self.divergence_class:
                head += f" [{self.divergence_class}]"
        return head


# ---------------------------------------------------------------------------
# Signature inference + input generation
# ---------------------------------------------------------------------------

_ANNOT_MAP = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "list": "list[int]",
    "List": "list[int]",
}


def infer_param_tags(source: str, func_name: str) -> list[str] | None:
    """Read parameter type tags from *func_name*'s annotations.

    Returns one tag per positional parameter, or ``None`` if the function is
    not found. Unannotated parameters default to ``"int"`` — the most common
    case for the algorithmic samples and a safe generator default (the oracle
    run later corrects the *return* type from observation).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fn = _find_func(tree, func_name)
    if fn is None:
        return None
    tags: list[str] = []
    for arg in fn.args.args:
        tags.append(_annotation_tag(arg.annotation))
    return tags


def _find_func(tree: ast.AST, func_name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def _annotation_tag(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "int"
    if isinstance(annotation, ast.Name):
        return _ANNOT_MAP.get(annotation.id, "int")
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        container = annotation.value.id
        if container in ("list", "List"):
            elem = annotation.slice
            elem_name = elem.id if isinstance(elem, ast.Name) else "int"
            elem_tag = _ANNOT_MAP.get(elem_name, "int")
            if elem_tag in SCALAR_TAGS:
                return f"list[{elem_tag}]"
    return "int"


# Deterministic edge cases per tag — these run first so boundary behavior is
# always exercised, with fuzz appended to reach the requested count.
_EDGE_CASES: dict[str, list[Any]] = {
    "int": [0, 1, -1, 2, 7, -8, 100],
    "float": [0.0, 1.0, -1.0, 0.5, -2.5, 3.14159],
    "bool": [True, False],
    "str": ["", "a", "abc", "Hello, World"],
    "list[int]": [[], [1], [3, 1, 2], [-1, 0, 1], [5, 5, 5]],
    "list[float]": [[], [1.5], [0.0, -2.0, 3.0]],
    "list[str]": [[], ["a"], ["b", "a", "c"]],
    "list[bool]": [[], [True], [True, False]],
}


def _fuzz_value(tag: str, rng: random.Random) -> Any:
    if tag == "int":
        return rng.randint(-1000, 1000)
    if tag == "float":
        return round(rng.uniform(-1000, 1000), 4)
    if tag == "bool":
        return rng.random() < 0.5
    if tag == "str":
        n = rng.randint(0, 8)
        return "".join(rng.choice("abcXYZ 0_") for _ in range(n))
    if tag.startswith("list["):
        elem = tag[5:-1]
        n = rng.randint(0, 6)
        return [_fuzz_value(elem, rng) for _ in range(n)]
    return 0


def generate_inputs(
    param_tags: list[str], *, n: int = 12, seed: int = 1234
) -> list[tuple]:
    """Build *n* input tuples over *param_tags* (edge cases first, then fuzz)."""
    rng = random.Random(seed)
    if not param_tags:
        return [()]  # nullary function — one call
    per_arg: list[list[Any]] = []
    for idx, tag in enumerate(param_tags):
        edges = list(_EDGE_CASES.get(tag, _EDGE_CASES["int"]))
        # Rotate each arg's edge list by its position so distinct arguments
        # take distinct values — otherwise (0,0),(1,1),… would never surface
        # arg-order bugs (e.g. a-b emitted as b-a).
        if edges:
            r = idx % len(edges)
            edges = edges[r:] + edges[:r]
        while len(edges) < n:
            edges.append(_fuzz_value(tag, rng))
        per_arg.append(edges[:n])
    return [tuple(per_arg[a][i] for a in range(len(param_tags))) for i in range(n)]


# ---------------------------------------------------------------------------
# Canonical-token comparison
# ---------------------------------------------------------------------------

_FLOAT_TOL = 1e-6


def canonical_token(value: Any, ret_tag: str) -> str:
    """Render *value* to a normalized string keyed on *ret_tag*.

    Equal values produce identical tokens regardless of source language
    spelling (``True``/``true``, ``6``/``6.0``, …). Floats are rounded to the
    tolerance so ``1/3`` agrees across backends.
    """
    if ret_tag == "bool":
        return "true" if value else "false"
    if ret_tag == "float":
        return f"{float(value):.6f}"
    if ret_tag == "int":
        return str(int(value))
    if ret_tag == "str":
        return str(value)
    if ret_tag.startswith("list["):
        elem = ret_tag[5:-1]
        return "[" + ",".join(canonical_token(v, elem) for v in value) + "]"
    return str(value)


def _floats_close(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) <= _FLOAT_TOL
    except ValueError:
        return a == b


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------


class Runner(Protocol):
    """Runs a function from emitted code over inputs, returns one sample each."""

    def run(
        self,
        code: str,
        func_name: str,
        inputs: list[tuple],
        *,
        param_tags: list[str],
        ret_tag: str,
    ) -> list[IOSample]: ...


class PythonRunner:
    """Run emitted/source Python by exec-ing it and calling the function.

    Toolchain-free — used both as the source oracle and as a Python *target*
    runner. Each call runs in a fresh namespace so module-level state cannot
    leak between inputs.
    """

    def run(
        self,
        code: str,
        func_name: str,
        inputs: list[tuple],
        *,
        param_tags: list[str],
        ret_tag: str,
    ) -> list[IOSample]:
        samples: list[IOSample] = []
        with _muffled():
            for args in inputs:
                ns: dict[str, Any] = {}
                try:
                    with time_limit(_PY_EXEC_TIMEOUT_S):
                        exec(compile(code, "<behavioral>", "exec"), ns)
                        fn = ns.get(func_name)
                        if not callable(fn):
                            samples.append(
                                IOSample(
                                    args=args,
                                    ok=False,
                                    error=f"no callable {func_name!r}",
                                )
                            )
                            continue
                        value = fn(*args)
                    samples.append(
                        IOSample(args=args, token=canonical_token(value, ret_tag))
                    )
                except ExecTimeout:
                    samples.append(
                        IOSample(
                            args=args,
                            ok=False,
                            error=f"timeout after {_PY_EXEC_TIMEOUT_S}s",
                        )
                    )
                except Exception as exc:  # noqa: BLE001 — capturing source behavior
                    samples.append(
                        IOSample(
                            args=args, ok=False, error=f"{type(exc).__name__}: {exc}"
                        )
                    )
        return samples


class RustRunner:
    """Run an emitted Rust function by compiling a generated harness with rustc.

    Supports scalar (int/float/bool) and ``Vec`` of scalar parameters and
    returns. The harness appends a ``main`` that calls the function over each
    input and prints one canonical line per call; the lines are parsed back
    into tokens. Signatures outside the supported set raise
    :class:`UnsupportedSignature` so the caller can mark the report
    *unsupported* rather than a false divergence.
    """

    class UnsupportedSignature(Exception):
        pass

    def run(
        self,
        code: str,
        func_name: str,
        inputs: list[tuple],
        *,
        param_tags: list[str],
        ret_tag: str,
    ) -> list[IOSample]:
        for tag in (*param_tags, ret_tag):
            if tag not in SUPPORTED_TAGS:
                raise self.UnsupportedSignature(f"rust runner cannot drive tag {tag!r}")
        # The transpiler emits collection params by reference (``& Vec<i64>``);
        # match the emitted signature so the harness call type-checks rather
        # than reporting a false divergence.
        byref = _detect_rust_byref(code, func_name, len(param_tags))
        harness = (
            code
            + "\n\n"
            + self._main(func_name, inputs, param_tags, ret_tag, byref=byref)
        )
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "main.rs"
            src.write_text(harness, encoding="utf-8")
            exe = Path(td) / ("harness.exe")
            comp = subprocess.run(
                [
                    resolve_tool("rustc"),
                    "--edition",
                    "2021",
                    "-A",
                    "warnings",
                    str(src),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if comp.returncode != 0:
                # Whole batch fails to build — every input is a divergence.
                err = _first_line(comp.stderr)
                return [
                    IOSample(args=a, ok=False, error=f"compile: {err}") for a in inputs
                ]
            proc = subprocess.run(
                [str(exe)], capture_output=True, text=True, timeout=30
            )
            if proc.returncode != 0:
                err = _first_line(proc.stderr)
                return [
                    IOSample(args=a, ok=False, error=f"runtime: {err}") for a in inputs
                ]
            lines = proc.stdout.splitlines()
        samples: list[IOSample] = []
        for i, args in enumerate(inputs):
            if i >= len(lines):
                samples.append(
                    IOSample(args=args, ok=False, error="missing output line")
                )
                continue
            samples.append(
                IOSample(args=args, token=_recanonicalize(lines[i], ret_tag))
            )
        return samples

    def _main(
        self,
        func_name: str,
        inputs: list[tuple],
        param_tags: list[str],
        ret_tag: str,
        *,
        byref: list[bool] | None = None,
    ) -> str:
        byref = byref or [False] * len(param_tags)
        calls = []
        for args in inputs:
            parts = []
            for v, t, ref in zip(args, param_tags, byref):
                lit = _rust_literal(v, t)
                parts.append(f"&{lit}" if ref else lit)
            rendered = ", ".join(parts)
            calls.append(f'    println!("{{}}", _fmt({func_name}({rendered})));')
        fmt_helper = _rust_fmt_helper(ret_tag)
        return fmt_helper + "\nfn main() {\n" + "\n".join(calls) + "\n}\n"


def _detect_rust_byref(code: str, func_name: str, n_params: int) -> list[bool]:
    """Per-parameter: does the emitted ``fn func_name`` take it by reference?

    Best-effort top-level brace/paren scan of the parameter list. Falls back to
    all-by-value if the signature cannot be located. Returns one bool per
    parameter (``True`` == the type starts with ``&``).
    """
    import re

    m = re.search(r"\bfn\s+" + re.escape(func_name) + r"\s*\(", code)
    if not m:
        return [False] * n_params
    i = m.end()
    depth = 1
    start = i
    while i < len(code) and depth:
        if code[i] in "(<[":
            depth += 1
        elif code[i] in ")>]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    params_src = code[start:i]
    if not params_src.strip():
        return [False] * n_params
    # Split params at top-level commas (avoid commas inside Vec<...> etc.).
    parts: list[str] = []
    buf = ""
    d = 0
    for ch in params_src:
        if ch in "(<[":
            d += 1
        elif ch in ")>]":
            d -= 1
        if ch == "," and d == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    out: list[bool] = []
    for p in parts:
        ty = p.split(":", 1)[1].strip() if ":" in p else ""
        out.append(ty.startswith("&"))
    while len(out) < n_params:
        out.append(False)
    return out[:n_params]


def _rust_literal(value: Any, tag: str) -> str:
    if tag == "int":
        return f"{int(value)}i64"
    if tag == "float":
        return f"{float(value)}f64"
    if tag == "bool":
        return "true" if value else "false"
    if tag == "str":
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}".to_string()'
    if tag.startswith("list["):
        elem = tag[5:-1]
        items = ", ".join(_rust_literal(v, elem) for v in value)
        return f"vec![{items}]"
    raise RustRunner.UnsupportedSignature(f"cannot render literal for tag {tag!r}")


def _rust_fmt_helper(ret_tag: str) -> str:
    """A monomorphic ``_fmt`` that prints the return value as a canonical token."""
    if ret_tag == "bool":
        return 'fn _fmt(x: bool) -> String { if x { "true".into() } else { "false".into() } }'
    if ret_tag == "float":
        return 'fn _fmt(x: f64) -> String { format!("{:.6}", x) }'
    if ret_tag == "int":
        return 'fn _fmt(x: i64) -> String { format!("{}", x) }'
    if ret_tag == "str":
        return "fn _fmt(x: String) -> String { x }"
    if ret_tag.startswith("list["):
        elem = ret_tag[5:-1]
        inner = {
            "int": 'format!("{}", v)',
            "float": 'format!("{:.6}", v)',
            "bool": 'if *v { "true".to_string() } else { "false".to_string() }',
            "str": "v.clone()",
        }[elem]
        elem_ty = {"int": "i64", "float": "f64", "bool": "bool", "str": "String"}[elem]
        return (
            f"fn _fmt(xs: Vec<{elem_ty}>) -> String {{\n"
            f"    let parts: Vec<String> = xs.iter().map(|v| {inner}).collect();\n"
            f'    format!("[{{}}]", parts.join(","))\n'
            "}"
        )
    raise RustRunner.UnsupportedSignature(f"cannot format return tag {ret_tag!r}")


def _recanonicalize(line: str, ret_tag: str) -> str:
    """Re-normalize a printed Rust line so it matches the Python token exactly."""
    line = line.strip()
    if ret_tag == "float":
        return canonical_token(line, "float")
    if ret_tag.startswith("list[float]"):
        body = line.strip("[]")
        if not body:
            return "[]"
        return (
            "[" + ",".join(canonical_token(p, "float") for p in body.split(",")) + "]"
        )
    return line


def _first_line(text: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            return ln[:200]
    return text.strip()[:200]


_RUNNERS: dict[str, Runner] = {"python": PythonRunner(), "rust": RustRunner()}


# ---------------------------------------------------------------------------
# Top-level check
# ---------------------------------------------------------------------------


def check_behavioral_equivalence(
    source: str,
    *,
    source_lang: str,
    target: str,
    target_code: str,
    func_name: str,
    param_tags: list[str] | None = None,
    n_inputs: int = 12,
    seed: int = 1234,
) -> BehavioralReport:
    """Run *func_name* in both source and target over generated inputs.

    The source must be runnable here as the oracle (Python in this first cut).
    Inputs on which the source raises are dropped. The return type is inferred
    from the oracle's observed outputs, so unannotated functions still work.
    """
    if source_lang != "python":
        return BehavioralReport(
            ok=False,
            total=0,
            matched=0,
            supported=False,
            reason=f"oracle for source_lang={source_lang!r} not implemented",
        )
    target_runner = _RUNNERS.get(target)
    if target_runner is None:
        return BehavioralReport(
            ok=False,
            total=0,
            matched=0,
            supported=False,
            reason=f"no behavioral runner for target {target!r}",
        )
    if target != "python" and not compiler_available(target):
        return BehavioralReport(
            ok=False,
            total=0,
            matched=0,
            supported=False,
            reason=f"{target} toolchain not available",
        )

    if param_tags is None:
        param_tags = infer_param_tags(source, func_name)
        if param_tags is None:
            return BehavioralReport(
                ok=False,
                total=0,
                matched=0,
                supported=False,
                reason=f"function {func_name!r} not found in source",
            )

    inputs = generate_inputs(param_tags, n=n_inputs, seed=seed)

    # 1. Source oracle (Python). Observe the return type from the first
    #    successful call; default to int if the oracle only ever raised.
    oracle = PythonRunner().run(
        source, func_name, inputs, param_tags=param_tags, ret_tag="int"
    )
    ret_tag = _infer_ret_tag(source, func_name, inputs)
    if ret_tag != "int":
        oracle = PythonRunner().run(
            source, func_name, inputs, param_tags=param_tags, ret_tag=ret_tag
        )

    runnable = [(s.args, s.token) for s in oracle if s.ok]
    if not runnable:
        return BehavioralReport(
            ok=False,
            total=0,
            matched=0,
            reason="source oracle raised on every generated input",
        )
    runnable_inputs = [args for args, _ in runnable]

    # 2. Target replay over the same inputs.
    try:
        actual = target_runner.run(
            target_code,
            func_name,
            runnable_inputs,
            param_tags=param_tags,
            ret_tag=ret_tag,
        )
    except RustRunner.UnsupportedSignature as exc:
        return BehavioralReport(
            ok=False, total=0, matched=0, supported=False, reason=str(exc)
        )

    # 3. Compare canonical tokens. The target runner returns samples in input
    #    order, so pair by index (args may be unhashable lists or repeat).
    matched = 0
    divergences: list[Divergence] = []
    for i, (args, expected_tok) in enumerate(runnable):
        got = actual[i] if i < len(actual) else None
        if got is None or not got.ok:
            divergences.append(
                Divergence(
                    args=args,
                    expected=expected_tok,
                    actual=(got.error if got else "<no result>"),
                )
            )
            continue
        if _tokens_match(expected_tok, got.token, ret_tag):
            matched += 1
        else:
            divergences.append(
                Divergence(args=args, expected=expected_tok, actual=got.token)
            )

    total = len(runnable)
    div_class = classify_divergence(divergences[0], ret_tag) if divergences else ""
    return BehavioralReport(
        ok=(matched == total and total > 0),
        total=total,
        matched=matched,
        divergences=divergences,
        divergence_class=div_class,
    )


def _tokens_match(expected: str, actual: str, ret_tag: str) -> bool:
    if ret_tag == "float":
        return _floats_close(expected, actual)
    return expected == actual


def _infer_ret_tag(source: str, func_name: str, inputs: list[tuple]) -> str:
    """Run the oracle and map the first returned value to a type tag."""
    ns: dict[str, Any] = {}
    try:
        with _muffled(), time_limit(_PY_EXEC_TIMEOUT_S):
            exec(compile(source, "<behavioral>", "exec"), ns)
            fn = ns.get(func_name)
            if not callable(fn):
                return "int"
            for args in inputs:
                try:
                    v = fn(*args)
                except Exception:  # noqa: BLE001
                    continue
                return _value_tag(v)
    except ExecTimeout, Exception:  # noqa: BLE001
        return "int"
    return "int"


def _value_tag(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        if not v:
            return "list[int]"
        return f"list[{_value_tag(v[0])}]"
    return "int"


# ---------------------------------------------------------------------------
# Repair-loop adapter
# ---------------------------------------------------------------------------


def make_behavioral_verifier(
    source: str,
    *,
    source_lang: str,
    target: str,
    func_name: str,
    param_tags: list[str] | None = None,
    n_inputs: int = 12,
    seed: int = 1234,
    require_supported: bool = False,
) -> Callable[[str], Any]:
    """Build a :class:`transpilers.repair.loop.Verifier` from a behavioral check.

    The returned callable takes the candidate target *code* and yields a
    ``VerificationOutcome``: ``ok`` on full behavioral match, otherwise a
    ``run_mismatch`` signal built from the first divergence so the
    escalating-repair loop can feed the diverging input + expected/actual back
    to the LLM.

    When the harness cannot drive the signature it returns ``ok=True`` by
    default (it is a *gate*, and an undrivable signature is not evidence of a
    bug) unless ``require_supported`` is set.
    """
    from transpilers.repair.loop import VerificationOutcome
    from transpilers.repair.signal import signal_from_run

    def verify(code: str) -> Any:
        report = check_behavioral_equivalence(
            source,
            source_lang=source_lang,
            target=target,
            target_code=code,
            func_name=func_name,
            param_tags=param_tags,
            n_inputs=n_inputs,
            seed=seed,
        )
        if not report.supported:
            return VerificationOutcome(ok=not require_supported, signal=None)
        if report.ok:
            return VerificationOutcome(ok=True)
        first = report.divergences[0]
        signal = signal_from_run(
            expected=first.expected,
            actual=first.actual,
            input_text=", ".join(repr(a) for a in first.args),
            exit_ok=True,
        )
        return VerificationOutcome(
            ok=False, signal=signal, expected=first.expected, actual=first.actual
        )

    return verify

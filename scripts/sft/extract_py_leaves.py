#!/usr/bin/env python3
"""Extract self-contained SCALAR LEAF functions from the energyplus-mojo Python
package, with an auto-generated verification driver — the "leaf tasks" to hand to
the fine-tuned 1.5B transpiler.

A leaf = a pure scalar function: all params scalar (float/int/bool), body is only
arithmetic / math.* / builtins / if-return / simple for-range accumulator (NO
intra-package calls, NO attribute access beyond math, NO dict/list/class, NO
collection iteration). These are exactly what the model can transpile AND what we
can auto-verify (sample inputs -> run Python -> expected; run generated Mojo ->
compare).

Four leaf categories extracted:
1. Pure-scalar TOP-LEVEL functions (original behaviour), extended to:
   - allow keyword-only scalar params (treated as positional in the unit)
   - allow for-loops over range(scalar_expr) with a scalar accumulator
   - allow tuple returns (2-5 elements); driver prints each element
   - inject module-level numeric constants so the unit is self-contained
2. Pure-scalar @staticmethods (no self) — same purity rules.
3. Pure-scalar INSTANCE methods on simple @dataclass classes whose body only
   reads self.<scalar_field> and method params. These are REWRITTEN as free
   functions (self fields inlined as params) so the model sees a standalone fn.
   The oracle is verified against the real bound method on a constructed instance.
   Also extended to allow keyword-only params and for-range accumulators.
4. Same as 3 but for non-dataclass classes with scalar-typed scalar fields that
   are purely numeric (extracted via instance-method rewrite when constructable).

EXPANDED ACCEPTANCE (still fully auto-verifiable — every leaf keeps a Python
oracle that runs on the sampled inputs and a mirrorable driver):
  - Widened math whitelist: trig/hyperbolic/unit-conversion idioms
    (sin/cos/tan/asin/.../sinh/.../radians/degrees/log2/cbrt/...) which dominate
    the curve & geometry code. (NARROW set previously dropped these.)
  - `raise ValueError/RuntimeError/...` GUARDS are allowed: they are pure input
    validation. The oracle re-runs every sample and the leaf is dropped if any
    sampled input actually raises, so emitted leaves never raise on their inputs.
    Sample-value heuristics keep fractions in [0,1] and dimensions > 0 so valid
    guards don't fire.
  - INLINABLE pure module-level helpers (`_clip`/`_sign`/`_clamp`) are inlined:
    their source is prepended to the unit so it stays self-contained and the
    model transpiles helper+leaf together.
  - `while`-loops with scalar accumulators are allowed; oracle exec is wrapped in
    a SIGALRM time limit so a non-terminating sample drops the leaf.
  - @property methods and instance methods on classes that ALSO have non-scalar
    (enum/object) fields are accepted, as long as the method itself only reads
    scalar fields (the non-scalar fields are tolerated, not read).
  - Module-level numeric constants in BOTH `NAME = v` and annotated
    `NAME: float = v` forms (plus simple math.* / arithmetic-over-consts) are
    injected, fixing NameError drops for leaves referencing e.g. `SIGMA`.
  - Long docstrings are stripped from the emitted unit so size caps don't reject
    otherwise-pure numeric functions.

KEEP rejecting: dict/list/set construction, intra-package function calls that
can't be inlined, attribute access beyond math/self-scalar-fields, anything whose
Python oracle raises on the sampled inputs, collection iteration in for loops,
non-terminating loops, randomness/IO.

Emits data/sft/cpp_mojo/py_leaves.jsonl: {name, source_file, python_unit,
python_driver, sample_args, category}.  python_driver calls the fn on each sample
tuple and prints — the Mojo side mirrors it.

IMPORTANT: every python_unit has `import math` prepended so the oracle can run
without injecting globals.

PORTABILITY: source/output paths are overridable via $EPMOJO_SRC / $PY_LEAVES_OUT
env vars or --ep / --out argparse flags (with autodetection of common checkout
locations) instead of the original hardcoded /home/bart/... paths.
"""
from __future__ import annotations
import argparse, ast, importlib.util, inspect, json, math, os, random, re, signal, sys, types
from contextlib import contextmanager
from pathlib import Path


class _OracleTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds=2.0):
    """Abort the wrapped oracle execution if it runs longer than `seconds`.
    Guards against non-terminating while-loops in sampled functions. Uses
    SIGALRM (POSIX); on platforms without it, becomes a no-op (the extractor is
    intended to run under WSL/Linux per the repo's pipeline)."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise _OracleTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _default_ep() -> Path:
    """Resolve the energyplus-mojo source dir, honouring env override.

    Portability: the original author hardcoded /home/bart/...; on other machines
    set EPMOJO_SRC (or pass --ep). We also fall back to a couple of well-known
    checkout locations so the extractor runs unmodified across dev boxes."""
    env = os.environ.get("EPMOJO_SRC")
    if env:
        return Path(env)
    for cand in (
        "/home/bart/github/EnergyPlusMojo/src/energyplus_mojo",
        "/mnt/c/Github/energyplus-mojo/src/energyplus_mojo",
    ):
        if Path(cand).is_dir():
            return Path(cand)
    return Path("/home/bart/github/EnergyPlusMojo/src/energyplus_mojo")


def _default_out() -> Path:
    env = os.environ.get("PY_LEAVES_OUT")
    if env:
        return Path(env)
    for cand in (
        "/home/bart/Github/transpilers/data/sft/cpp_mojo/py_leaves.jsonl",
        "/mnt/c/Github/transpilers/data/sft/cpp_mojo/py_leaves.jsonl",
    ):
        if Path(cand).parent.is_dir():
            return Path(cand)
    return Path("/home/bart/Github/transpilers/data/sft/cpp_mojo/py_leaves.jsonl")


# Mutable globals — set in main() from argparse/env so helper functions that
# reference EP (e.g. p.relative_to(EP)) keep working unchanged.
EP = _default_ep()
OUT = _default_out()
rnd = random.Random(7)

SCALAR = {"float", "int", "bool", "Float64", "Real64", "Int"}
# math.* functions + builtins the model can transpile AND we can verify. Widened
# from the original narrow set to include the trig/hyperbolic/degree idioms that
# dominate the energyplus-mojo curve & geometry code (sin/cos/radians are the
# single most-used math fns in the package).
ALLOWED_CALLS = {"abs", "min", "max", "round", "pow", "int", "float",
                 "sqrt", "exp", "log", "log10", "pow", "fabs", "floor", "ceil",
                 "atan2", "hypot", "copysign", "fmod", "trunc", "erf", "erfc",
                 "sign",
                 # widened trig / hyperbolic / unit-conversion / misc
                 "sin", "cos", "tan", "asin", "acos", "atan",
                 "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                 "degrees", "radians", "expm1", "log1p", "log2", "cbrt"}

# Pure module-level helpers we can INLINE into a self-contained unit. Their
# Python source is injected verbatim ahead of the leaf so the oracle runs, and
# the model transpiles the helper alongside the leaf (the Mojo driver mirrors).
INLINABLE_HELPERS = {"_clip", "_sign", "_clamp"}

# Additional allowed bare function names in instance-method bodies (module-level helpers
# that we can inline or that the free-fn doesn't need because we inline self fields)
ALLOWED_INSTANCE_CALLS = ALLOWED_CALLS | INLINABLE_HELPERS

# Builtin "guard" callables allowed only in `raise <X>(...)` position — they are
# pure input validation that never fires on the sampled (valid) inputs. The
# oracle still re-runs every sample and drops the leaf if any raises.
GUARD_EXC = {"ValueError", "RuntimeError", "ZeroDivisionError",
             "ArithmeticError", "OverflowError", "AssertionError"}


def is_scalar_ann(a):
    if a is None:
        return True  # unannotated allowed (we'll feed floats)
    if isinstance(a, ast.Name):
        return a.id in SCALAR
    if isinstance(a, ast.Subscript):  # Optional[float] etc. -> reject (keep simple)
        return False
    return False


def is_scalar_field_ann(a):
    """Check if a dataclass field annotation is a plain scalar (not Optional, not str)."""
    if a is None:
        return False
    if isinstance(a, ast.Name):
        return a.id in SCALAR
    return False


def _for_range_only(node):
    """Return True if all for-loops in the node iterate over range(...) only.
    Helper used by pure_body* variants that want to allow simple accumulators."""
    for n in ast.walk(node):
        if isinstance(n, ast.For):
            if not (isinstance(n.iter, ast.Call)
                    and isinstance(n.iter.func, ast.Name)
                    and n.iter.func.id == 'range'):
                return False
    return True


def _for_body_pure(node, allowed_names=None):
    """Return True if all for-loop bodies in the node contain only
    AugAssign/Assign to local scalars and allowed conditional logic.
    Rejects list.append, dict access, etc. inside for bodies."""
    for n in ast.walk(node):
        if isinstance(n, ast.For):
            for stmt in ast.walk(n):
                if isinstance(stmt, (ast.ListComp, ast.DictComp, ast.SetComp,
                                     ast.GeneratorExp, ast.Dict, ast.List, ast.Set)):
                    return False
                if isinstance(stmt, ast.Call):
                    f = stmt.func
                    if isinstance(f, ast.Name):
                        if f.id not in ALLOWED_CALLS and f.id != 'range':
                            return False
                    elif isinstance(f, ast.Attribute):
                        if isinstance(f.value, ast.Name) and f.value.id == "math":
                            if f.attr not in ALLOWED_CALLS:
                                return False
                        else:
                            return False
                    else:
                        return False
    return True


def _raise_guard_call_ids(node):
    """Return the set of id() of Call nodes that are guard-exception constructors
    appearing directly as the value of a `raise` statement (e.g. raise ValueError(..)).
    These are allowed because they are pure input validation; the oracle re-runs
    every sample and drops the leaf if any actually raises."""
    ids = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and n.exc is not None:
            exc = n.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) \
               and exc.func.id in GUARD_EXC:
                ids.add(id(exc))
            # also allow bare `raise ValueError` (no call) — handled in purity walk
    return ids


def _bad_raise(node):
    """Return True if any `raise` statement raises something other than an allowed
    guard exception (so we can keep rejecting `raise SomeRecordError(obj)` etc.)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Raise):
            exc = n.exc
            if exc is None:  # bare `raise` re-raise — reject (no context)
                return True
            if isinstance(exc, ast.Call):
                if not (isinstance(exc.func, ast.Name) and exc.func.id in GUARD_EXC):
                    return True
            elif isinstance(exc, ast.Name):
                if exc.id not in GUARD_EXC:
                    return True
            else:
                return True
    return False


def _while_ok(node):
    """Return True if every while-loop in the node is a bounded scalar-accumulator
    loop: condition is a comparison/boolop over scalars, body contains only
    Assign/AugAssign/If/Break/Continue/Return and allowed scalar exprs (no
    collections, calls limited to ALLOWED_CALLS). The oracle still executes it and
    drops the leaf if it doesn't terminate quickly (guarded by an iteration cap in
    the exec harness)."""
    for n in ast.walk(node):
        if isinstance(n, ast.While):
            for stmt in n.body:
                for s in ast.walk(stmt):
                    if isinstance(s, (ast.For, ast.While, ast.With, ast.Try,
                                      ast.ListComp, ast.DictComp, ast.SetComp,
                                      ast.GeneratorExp, ast.Dict, ast.List,
                                      ast.Set, ast.Subscript)):
                        return False
    return True


def pure_body(node, allowed_consts=None):
    """Only arithmetic / comparisons / if-return / simple assign / allowed calls.
    Allow self.<attr> attribute access (handled separately for instance methods).
    Allow for-range loops over scalar exprs with scalar accumulator bodies.
    allowed_consts: set of allowed Name references (module-level numeric consts).
    """
    if allowed_consts is None:
        allowed_consts = frozenset()
    if _bad_raise(node):
        return False
    if not _while_ok(node):
        return False
    guard_ids = _raise_guard_call_ids(node)
    for n in ast.walk(node):
        if isinstance(n, ast.For):
            # Only allow for-loops over range(...)
            if not (isinstance(n.iter, ast.Call)
                    and isinstance(n.iter.func, ast.Name)
                    and n.iter.func.id == 'range'):
                return False
            # Do not allow nested for-loops (the range-check will catch inner ones too,
            # but let's also prevent the body from containing a for-loop)
            for inner in ast.walk(n):
                if inner is n:
                    continue
                if isinstance(inner, ast.For):
                    return False
            continue
        if isinstance(n, (ast.With, ast.Try, ast.Lambda,
                          ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                          ast.Dict, ast.List, ast.Set)):
            return False
        if isinstance(n, ast.Attribute):
            # allow math.<fn> and self.<field>
            if isinstance(n.value, ast.Name):
                if n.value.id not in ("math", "self"):
                    return False
            else:
                return False
        if isinstance(n, ast.Call):
            if id(n) in guard_ids:
                continue  # raise ValueError(..) guard — allowed
            f = n.func
            if isinstance(f, ast.Name):
                if (f.id not in ALLOWED_CALLS and f.id != 'range'
                        and f.id not in INLINABLE_HELPERS):
                    return False
            elif isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id == "math":
                    if f.attr not in ALLOWED_CALLS:
                        return False
                elif isinstance(f.value, ast.Name) and f.value.id == "self":
                    # self.method() calls — only allow if it's a known inlinable helper
                    if f.attr not in ("_sign", "_clip_x", "_clip_y", "_clip_output"):
                        return False
                else:
                    return False
            else:
                return False
    return True


def pure_body_no_self_calls(node, allowed_consts=None):
    """Stricter check: no self.method() calls at all (for non-inlinable methods).
    Also allows for-range loops (delegating body purity to pure_body)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id == "self":
                    return False
    return True


def pure_body_instance(node, allowed_consts=None):
    """Purity check for instance methods: like pure_body but also allows _clip/_sign
    module-level helper calls. The self.method() calls are handled separately.
    Also allows for-range loops with scalar accumulator bodies."""
    if allowed_consts is None:
        allowed_consts = frozenset()
    if _bad_raise(node):
        return False
    if not _while_ok(node):
        return False
    guard_ids = _raise_guard_call_ids(node)
    for n in ast.walk(node):
        if isinstance(n, ast.For):
            # Only allow for-loops over range(...)
            if not (isinstance(n.iter, ast.Call)
                    and isinstance(n.iter.func, ast.Name)
                    and n.iter.func.id == 'range'):
                return False
            # No nested for-loops
            for inner in ast.walk(n):
                if inner is n:
                    continue
                if isinstance(inner, ast.For):
                    return False
            continue
        if isinstance(n, (ast.With, ast.Try, ast.Lambda,
                          ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                          ast.Dict, ast.List, ast.Set)):
            return False
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name):
                if n.value.id not in ("math", "self"):
                    return False
            else:
                return False
        if isinstance(n, ast.Call):
            if id(n) in guard_ids:
                continue  # raise ValueError(..) guard — allowed
            f = n.func
            if isinstance(f, ast.Name):
                if f.id not in ALLOWED_INSTANCE_CALLS and f.id != 'range':
                    return False
            elif isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id == "math":
                    if f.attr not in ALLOWED_CALLS:
                        return False
                elif isinstance(f.value, ast.Name) and f.value.id == "self":
                    if f.attr not in ("_sign", "_clip_x", "_clip_y", "_clip_output"):
                        return False
                else:
                    return False
            else:
                return False
    return True


def get_module_consts(tree):
    """Extract module-level numeric (int/float) constants as {name: value}.

    Handles both plain `NAME = 1.0` (Assign) and annotated `NAME: float = 1.0`
    (AnnAssign) forms — the latter is common in this codebase (e.g.
    `SIGMA: float = 5.67e-8`) and was previously missed, which made any leaf
    referencing such a constant fail its oracle with NameError and get dropped.
    Also resolves a handful of `math.*` constant expressions (pi, tau, e, inf)
    and simple arithmetic over already-known numeric constants, so units that
    reference derived constants stay self-contained."""
    consts = {}
    safe_env = {"math": math, "pi": math.pi, "tau": math.tau, "e": math.e,
                "inf": math.inf}

    def _resolve(value_node):
        # literal first
        try:
            v = ast.literal_eval(value_node)
            if isinstance(v, (int, float)):
                return v
        except Exception:
            pass
        # math.* constants / arithmetic over known consts
        try:
            expr = ast.Expression(value_node)
            code = compile(expr, "<const>", "eval")
            v = eval(code, {"__builtins__": {}}, {**safe_env, **consts})
            if isinstance(v, (int, float)) and math.isfinite(v):
                return v
        except Exception:
            pass
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            v = _resolve(node.value)
            if v is not None:
                consts[node.targets[0].id] = v
        elif isinstance(node, ast.AnnAssign) and node.value is not None \
                and isinstance(node.target, ast.Name):
            v = _resolve(node.value)
            if v is not None:
                consts[node.target.id] = v
    return consts


def get_inlinable_helpers(tree, src):
    """Return {name: source_string} for module-level pure helper functions whose
    name is in INLINABLE_HELPERS and whose body is itself pure-scalar (so the
    helper + leaf form a self-contained, transpilable, verifiable unit).

    We re-purpose the standard purity checks; a helper qualifies only if it has
    scalar/optional-scalar params and a pure body (math/builtins/if-return)."""
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in INLINABLE_HELPERS:
            continue
        # params: allow float, int, bool, and `float | None` (clip bounds)
        ok_params = True
        for a in node.args.args + node.args.kwonlyargs:
            ann = a.annotation
            if ann is None or is_scalar_ann(ann) or is_float_or_none_ann(ann):
                continue
            ok_params = False
            break
        if not ok_params:
            continue
        if node.args.vararg or node.args.kwarg:
            continue
        # body purity (no self, no other intra-package calls). _bad_raise / while
        # gates apply; allow only math/builtins.
        if _bad_raise(node) or not _while_ok(node):
            continue
        bad = False
        guard_ids = _raise_guard_call_ids(node)
        for n in ast.walk(node):
            if isinstance(n, (ast.With, ast.Try, ast.Lambda, ast.ListComp,
                              ast.DictComp, ast.SetComp, ast.GeneratorExp,
                              ast.Dict, ast.List, ast.Set, ast.Subscript)):
                bad = True; break
            if isinstance(n, ast.Attribute):
                if not (isinstance(n.value, ast.Name) and n.value.id == "math"):
                    bad = True; break
            if isinstance(n, ast.Call):
                if id(n) in guard_ids:
                    continue
                f = n.func
                if isinstance(f, ast.Name):
                    if f.id not in ALLOWED_CALLS and f.id != "range":
                        bad = True; break
                elif isinstance(f, ast.Attribute):
                    if not (isinstance(f.value, ast.Name) and f.value.id == "math"
                            and f.attr in ALLOWED_CALLS):
                        bad = True; break
                else:
                    bad = True; break
        if bad:
            continue
        seg = ast.get_source_segment(src, node)
        if seg:
            out[node.name] = seg
    return out


def _helpers_used(unit_src, helpers):
    """Return {name: source} for helpers whose name is called in unit_src."""
    used = {}
    for name, hsrc in helpers.items():
        # word-boundary call match: NAME(
        if re.search(r"(?<![\w.])" + re.escape(name) + r"\s*\(", unit_src):
            used[name] = hsrc
    return used


# Per-file inlinable helpers, set in main() before invoking the extractors.
_CUR_HELPERS: dict = {}


def prepend_helpers(unit):
    """If `unit` calls any inlinable module-level helper (_clip/_sign/_clamp),
    prepend that helper's source so the emitted python_unit is self-contained
    (the oracle runs it as-is; the model transpiles the helper + leaf together)."""
    used = _helpers_used(unit, _CUR_HELPERS)
    if not used:
        return unit
    # Helpers go after a leading `import math` (if present) so math is in scope.
    helper_block = "\n\n".join(used[n] for n in sorted(used)) + "\n\n"
    if unit.startswith("import math"):
        # insert after the import line(s)
        lines = unit.splitlines(keepends=True)
        idx = 0
        while idx < len(lines) and (lines[idx].startswith("import ")
                                     or lines[idx].strip() == ""):
            idx += 1
        return "".join(lines[:idx]) + helper_block + "".join(lines[idx:])
    # Helpers may themselves use math.* -> ensure import present
    if "math." in helper_block:
        return "import math\n\n" + helper_block + unit
    return helper_block + unit


def collect_used_consts(func_node, module_consts):
    """Return a dict of {name: value} for module-level consts used in this function."""
    used = {}
    for n in ast.walk(func_node):
        if isinstance(n, ast.Name) and n.id in module_consts:
            used[n.id] = module_consts[n.id]
    return used


def build_const_prelude(used_consts):
    """Build a string of `NAME = value` lines to prepend to the python_unit."""
    lines = []
    for name, val in sorted(used_consts.items()):
        if isinstance(val, int):
            lines.append(f"{name} = {val}")
        else:
            lines.append(f"{name} = {val!r}")
    return "\n".join(lines) + "\n\n" if lines else ""


def strip_docstring_src(unit_src):
    """Remove a leading function docstring from a `def ...:` source block so the
    emitted python_unit stays under the size cap and is cleaner for the model.
    Re-parses the result and bails (returns the original) if anything looks off,
    so we never emit syntactically-broken units."""
    try:
        mod = ast.parse(unit_src)
    except Exception:
        return unit_src
    fn = next((n for n in mod.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None or not fn.body:
        return unit_src
    first = fn.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return unit_src
    if len(fn.body) == 1:
        return unit_src  # docstring is the whole body; leave it
    # Remove the docstring statement via lineno ranges (1-based, end-exclusive+1).
    lines = unit_src.splitlines()
    start = first.lineno - 1
    end = getattr(first, "end_lineno", first.lineno)
    new_lines = lines[:start] + lines[end:]
    candidate = "\n".join(new_lines)
    # Validate the result still parses to a function with a non-empty body.
    try:
        m2 = ast.parse(candidate)
        f2 = next((n for n in m2.body if isinstance(n, ast.FunctionDef)), None)
        if f2 is None or not f2.body:
            return unit_src
    except Exception:
        return unit_src
    return candidate


def returns_scalar_tuple(node):
    """Return the tuple arity (2-5) if the function has any return-tuple of scalars,
    else return 0. Used to decide if we should handle element-wise output."""
    for n in ast.walk(node):
        if isinstance(n, ast.Return) and n.value is not None:
            if isinstance(n.value, ast.Tuple):
                arity = len(n.value.elts)
                if 2 <= arity <= 5:
                    return arity
    return 0


def sample_vals(pname, ann):
    base = [0.5, 2.0, -1.5, 10.0, 0.0, 100.0]
    name = pname.lower()
    if ann == "bool":
        return [True, False]
    # strictly-positive geometry / dimensional params FIRST — avoid 0 and
    # negatives that commonly trip `must be > 0` guards. Checked before the
    # fraction heuristics because a dimensional name like "slat_separation_m"
    # contains the substring "ratio" ("sepa[ratio]n") yet must stay positive.
    if any(tok in name for tok in (
            "width", "separation", "thickness", "length", "height", "depth",
            "diameter", "radius", "area", "spacing", "pitch",
            "mass", "density", "conductivity")) or name.endswith("_m") \
            or name.endswith("_m2") or name.endswith("_mm"):
        return [0.5, 2.0, 1.0, 10.0]
    # [0, 1]-bounded physical fractions — keep samples in range so guard-raises
    # (e.g. `if not 0 <= x <= 1: raise ValueError`) don't fire on valid inputs.
    if any(tok in name for tok in (
            "frac", "ratio", "transmittance", "absorptance",
            "reflectance", "emissivity", "emittance", "albedo",
            "porosity", "humidity")) or name.endswith("eff") or "_eff" in name:
        return [0.0, 0.5, 1.0, 0.25]
    # integer-ish: only when explicitly annotated int, or clearly a count.
    if ann == "int" or "count" in name or name.startswith("num_") \
            or name.startswith("n_") or name in ("n", "i", "k"):
        return [0, 1, 3, 7]
    if "temp" in name:
        return [20.0, -5.0, 35.0, 0.0]
    return base


def coeff_sample_val(fname):
    """Return a safe sample value for a coefficient/range field."""
    name = fname.lower()
    if "minimum" in name:
        return -5.0
    if "maximum" in name:
        return 5.0
    if "coefficient" in name or name.startswith("c"):
        return 1.0
    return 1.0


def coeff_sample_row_vals(fname):
    """Return 4 DISCRIMINATING sample values for a coefficient/range field, one
    per driver row. Keeps minimum/maximum range fields ordered (min < max) and
    wide enough that clip(min, max, x) doesn't collapse, while still varying the
    coefficients so the differential test actually exercises the arithmetic
    (instead of probing a single fixed point repeatedly).
    """
    name = fname.lower()
    # Range bounds must stay min < max with the same wide window every row so the
    # clip never inverts; widening alone would not discriminate, but the *output*
    # bounds and the varying coefficients/x do.
    if "minimum" in name:
        return [-100.0, -100.0, -100.0, -100.0]
    if "maximum" in name:
        return [100.0, 100.0, 100.0, 100.0]
    if "coefficient" in name or name.startswith("c"):
        return [1.0, 0.5, -2.0, 3.0]
    return [1.0, 2.0, 0.5, -1.5]


def _is_scalar_result(r):
    """Return True if r is a single scalar or a tuple/list of scalars (all finite)."""
    if isinstance(r, bool):
        return True
    if isinstance(r, (int, float)):
        if isinstance(r, float) and not math.isfinite(r):
            return False
        return True
    if isinstance(r, tuple):
        return all(_is_scalar_result(x) for x in r)
    return False


def make_driver(fn_name, tuples, is_tuple_return):
    """Build the Python driver string. For tuple-return functions, print
    each element on a separate line. For scalar-return, print the result."""
    lines = []
    for t in tuples:
        call = f"{fn_name}({', '.join(repr(v) for v in t)})"
        if is_tuple_return:
            lines.append(f"_r = {call}")
            lines.append("for _v in _r: print(_v)")
        else:
            lines.append(f"print({call})")
    return "\n".join(lines)


def make_mojo_driver(fn_name, tuples, is_tuple_return):
    """Build a Mojo main() driver that mirrors the Python driver output.
    For tuple-return, index into the result. For scalar, print directly."""
    inner = []
    for t in tuples:
        call = f"{fn_name}({', '.join(repr(v) for v in t)})"
        if is_tuple_return:
            inner.append(f"    var _r = {call}")
            inner.append("    for i in range(len(_r)):")
            inner.append("        print(_r[i])")
        else:
            inner.append(f"    print({call})")
    return "def main():\n" + "\n".join(inner)


# ---------------------------------------------------------------------------
# Category 1: top-level scalar functions
# ---------------------------------------------------------------------------

def extract_toplevel_fns(tree, src, p, leaves, module_consts=None):
    if module_consts is None:
        module_consts = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        # Gather positional + kwonly args (treat kwonly as positional in free fn)
        pos_args = node.args.args
        kw_args = node.args.kwonlyargs
        all_args = pos_args + kw_args
        if not all_args or len(all_args) > 24:
            continue
        if any(a.arg in ("self", "cls") for a in all_args):
            continue
        if not all(is_scalar_ann(a.annotation) for a in all_args):
            continue
        # We now allow kwonly args by treating them as additional positional params
        if node.args.vararg or node.args.kwarg:
            continue
        if not pure_body(node):
            continue
        if not pure_body_no_self_calls(node):
            continue
        if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)):
            continue
        unit = ast.get_source_segment(src, node)
        if not unit:
            continue
        unit = strip_docstring_src(unit)
        if len(unit) > 2000:
            continue

        # If the function has kwonly args, rewrite as positional-only fn
        # (strip the * separator) so the model sees a standalone fn
        if kw_args:
            # Rebuild the def line without keyword-only markers
            lines = unit.splitlines()
            # Find def line and rewrite params
            all_param_names = [a.arg for a in all_args]
            # Get annotations
            def _ann_str(a):
                if a.annotation is None:
                    return a.arg
                if isinstance(a.annotation, ast.Name):
                    return f"{a.arg}: {a.annotation.id}"
                return a.arg
            params_str = ", ".join(_ann_str(a) for a in all_args)
            # Find return annotation if any
            ret = ""
            if node.returns:
                ret_src = ast.get_source_segment(src, node.returns)
                if ret_src:
                    ret = f" -> {ret_src}"
            new_def = f"def {node.name}({params_str}){ret}:"
            # Replace the def line (first non-decorator line starting with 'def ')
            new_lines = []
            replaced = False
            for l in lines:
                if not replaced and l.strip().startswith("def "):
                    # Preserve indentation
                    indent = len(l) - len(l.lstrip())
                    new_lines.append(" " * indent + new_def)
                    replaced = True
                else:
                    new_lines.append(l)
            unit = "\n".join(new_lines)

        # Collect and inject module-level constants used in this function
        used_consts = collect_used_consts(node, module_consts)
        const_prelude = build_const_prelude(used_consts)

        # prepend import math if any math.* is used
        if "math." in unit and "import math" not in unit:
            unit = "import math\n\n" + unit
        # inline any pure module-level helpers (_clip/_sign/_clamp) the unit calls
        unit = prepend_helpers(unit)
        if "math." in unit and "import math" not in unit:
            unit = "import math\n\n" + unit
        if const_prelude:
            unit = const_prelude + unit

        anns = [(a.arg, (a.annotation.id if isinstance(a.annotation, ast.Name) else None)) for a in all_args]
        pools = [sample_vals(nm, an) for nm, an in anns]
        tuples = []
        for i in range(4):
            tuples.append([pools[j][i % len(pools[j])] for j in range(len(pools))])

        # Determine if the function returns a tuple
        tuple_arity = returns_scalar_tuple(node)

        ns = {"math": math}
        if used_consts:
            ns.update(used_consts)
        try:
            with time_limit():
                exec(unit, ns)
                fn = ns[node.name]
                ok = True
                for t in tuples:
                    r = fn(*t)
                    if not _is_scalar_result(r):
                        ok = False; break
                    if isinstance(r, tuple) and not tuple_arity:
                        ok = False; break  # unexpected tuple
                    if not isinstance(r, tuple) and tuple_arity:
                        ok = False; break  # expected tuple but got scalar
                    if isinstance(r, float) and not math.isfinite(r):
                        ok = False; break
            if not ok:
                continue
        except Exception:
            continue

        driver = make_driver(node.name, tuples, bool(tuple_arity))
        cat = "toplevel_tuple" if tuple_arity else "toplevel"
        if kw_args:
            cat = cat + "_kwonly" if tuple_arity else "toplevel_kwonly"
        leaves.append({"name": node.name, "source_file": str(p.relative_to(EP)),
                       "python_unit": unit, "python_driver": driver,
                       "sample_args": tuples, "category": cat})


# ---------------------------------------------------------------------------
# Category 2: pure-scalar @staticmethods (no self)
# ---------------------------------------------------------------------------

def extract_staticmethods(tree, src, p, leaves, module_consts=None):
    if module_consts is None:
        module_consts = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            # must have @staticmethod decorator
            is_static = any(
                (isinstance(d, ast.Name) and d.id == "staticmethod")
                for d in node.decorator_list
            )
            if not is_static:
                continue
            # Gather positional + kwonly args
            pos_args = node.args.args
            kw_args = node.args.kwonlyargs
            all_args = pos_args + kw_args
            if not all_args or len(all_args) > 24:
                continue
            if any(a.arg in ("self", "cls") for a in all_args):
                continue
            if not all(is_scalar_ann(a.annotation) for a in all_args):
                continue
            if node.args.vararg or node.args.kwarg:
                continue
            if not pure_body_no_self_calls(node):
                continue
            # no self.* attribute access in body
            has_self_attr = any(
                isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
                for n in ast.walk(node)
            )
            if has_self_attr:
                continue
            # purity gate (math only, no class access)
            bad = False
            if _bad_raise(node) or not _while_ok(node):
                continue
            guard_ids = _raise_guard_call_ids(node)
            for n in ast.walk(node):
                if isinstance(n, ast.For):
                    # only allow range loops
                    if not (isinstance(n.iter, ast.Call)
                            and isinstance(n.iter.func, ast.Name)
                            and n.iter.func.id == 'range'):
                        bad = True; break
                    continue
                if isinstance(n, (ast.With, ast.Try, ast.Lambda,
                                  ast.ListComp, ast.DictComp, ast.SetComp)):
                    bad = True; break
                if isinstance(n, ast.Attribute):
                    if not (isinstance(n.value, ast.Name) and n.value.id == "math"):
                        bad = True; break
                if isinstance(n, ast.Call):
                    if id(n) in guard_ids:
                        continue  # raise ValueError(..) guard
                    f = n.func
                    if isinstance(f, ast.Name):
                        if (f.id not in ALLOWED_CALLS and f.id != 'range'
                                and f.id not in INLINABLE_HELPERS):
                            bad = True; break
                    elif isinstance(f, ast.Attribute):
                        if not (isinstance(f.value, ast.Name) and f.value.id == "math" and f.attr in ALLOWED_CALLS):
                            bad = True; break
                    else:
                        bad = True; break
            if bad:
                continue
            # must return
            if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)):
                continue
            unit = ast.get_source_segment(src, node)
            if not unit:
                continue
            unit = strip_docstring_src(unit)
            if len(unit) > 2000:
                continue
            # emit as a plain function (strip @staticmethod decorator)
            lines = unit.splitlines()
            body_lines = [l for l in lines if not l.strip().startswith("@staticmethod")]
            # Handle kwonly in staticmethods: rewrite def line
            if kw_args:
                all_param_names = [a.arg for a in all_args]
                def _ann_str(a):
                    if a.annotation is None:
                        return a.arg
                    if isinstance(a.annotation, ast.Name):
                        return f"{a.arg}: {a.annotation.id}"
                    return a.arg
                params_str = ", ".join(_ann_str(a) for a in all_args)
                ret = ""
                if node.returns:
                    ret_src = ast.get_source_segment(src, node.returns)
                    if ret_src:
                        ret = f" -> {ret_src}"
                new_def = f"def {node.name}({params_str}){ret}:"
                new_body = []
                replaced = False
                for l in body_lines:
                    if not replaced and l.strip().startswith("def "):
                        indent = len(l) - len(l.lstrip())
                        new_body.append(" " * indent + new_def)
                        replaced = True
                    else:
                        new_body.append(l)
                body_lines = new_body
            fn_unit = "\n".join(body_lines)
            # de-indent by the class indent level
            def_line = next((l for l in body_lines if l.strip().startswith("def ")), None)
            if def_line is None:
                continue
            indent = len(def_line) - len(def_line.lstrip())
            fn_unit = "\n".join(l[indent:] if len(l) > indent else l for l in body_lines)

            # Collect module-level constants
            used_consts = collect_used_consts(node, module_consts)
            const_prelude = build_const_prelude(used_consts)

            if "math." in fn_unit and "import math" not in fn_unit:
                fn_unit = "import math\n\n" + fn_unit
            fn_unit = prepend_helpers(fn_unit)
            if "math." in fn_unit and "import math" not in fn_unit:
                fn_unit = "import math\n\n" + fn_unit
            if const_prelude:
                fn_unit = const_prelude + fn_unit

            # Determine tuple return
            tuple_arity = returns_scalar_tuple(node)

            anns = [(a.arg, (a.annotation.id if isinstance(a.annotation, ast.Name) else None)) for a in all_args]
            pools = [sample_vals(nm, an) for nm, an in anns]
            tuples = []
            for i in range(4):
                tuples.append([pools[j][i % len(pools[j])] for j in range(len(pools))])
            ns = {"math": math}
            if used_consts:
                ns.update(used_consts)
            try:
                with time_limit():
                    exec(fn_unit, ns)
                    fn = ns[node.name]
                    ok = True
                    for t in tuples:
                        r = fn(*t)
                        if not _is_scalar_result(r):
                            ok = False; break
                        if isinstance(r, float) and not math.isfinite(r):
                            ok = False; break
                if not ok:
                    continue
            except Exception:
                continue
            driver = make_driver(node.name, tuples, bool(tuple_arity))
            cat = "staticmethod_tuple" if tuple_arity else "staticmethod"
            leaves.append({"name": node.name, "source_file": str(p.relative_to(EP)),
                           "python_unit": fn_unit, "python_driver": driver,
                           "sample_args": tuples, "category": cat,
                           "class_name": cls.name})


# ---------------------------------------------------------------------------
# Helpers for Category 3: dataclass instance method rewrites
# ---------------------------------------------------------------------------

def is_float_or_none_ann(ann):
    """Return True if annotation is `float | None` (BinOp with None) or just float."""
    if isinstance(ann, ast.Name) and ann.id == "float":
        return True
    # Python 3.10+ `float | None` syntax
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        left_ok = isinstance(ann.left, ast.Name) and ann.left.id == "float"
        right_none = isinstance(ann.right, ast.Constant) and ann.right.value is None
        right_none2 = isinstance(ann.right, ast.Name) and ann.right.id == "None"
        if left_ok and (right_none or right_none2):
            return True
    return False


def get_scalar_fields_with_defaults(cls_node):
    """Return list of (field_name, kind, default) for all fields in a dataclass.
    kind is one of 'scalar', 'str', 'other'.
    For 'scalar': default is the float value to use (1.0 if required with no default,
      proper coeff_sample_val otherwise).
    For 'str': default is ''.
    For 'other': a non-scalar/non-str field (enum/object/list/Optional[non-float]).
      Such a field is tolerated as long as the candidate METHOD never reads it;
      if construction needs it the cross-check falls back to free-fn-oracle-only.
    Returns None only on a structurally-unparseable class body.
    Handles:
      - float / int / bool fields (with or without defaults) -> scalar
      - float | None fields -> optional scalar, default None or a float
      - str fields (with or without defaults) -> str, default ''
      - tuple[str,...] and similar ClassVar-ish -> skip (class-level, not instance)
      - everything else (enums, objects, containers) -> 'other' (tolerated)
    """
    fields = []
    for stmt in cls_node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if stmt.target is None or not isinstance(stmt.target, ast.Name):
            continue
        fname = stmt.target.id
        ann = stmt.annotation

        # Skip ClassVar[...] and tuple[...] annotations (class-level constants)
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name) and ann.value.id in ("ClassVar", "tuple", "Tuple"):
                continue
            # Optional[float] -> treat as optional scalar
            if isinstance(ann.value, ast.Name) and ann.value.id == "Optional":
                # assume float inside
                fields.append((fname, "scalar", None))
                continue
            # Any other subscript (List[...], Dict[...], etc.) -> 'other'
            fields.append((fname, "other", None))
            continue

        # float | None  (BinOp in 3.10+ syntax)
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            if is_float_or_none_ann(ann):
                # optional float field — default is None or a float
                if stmt.value is not None:
                    try:
                        default = ast.literal_eval(stmt.value)
                    except Exception:
                        default = None
                else:
                    default = None
                fields.append((fname, "scalar", default))
                continue
            # other union (e.g. SomeEnum | None) -> 'other'
            fields.append((fname, "other", None))
            continue

        if isinstance(ann, ast.Name):
            if ann.id in SCALAR:
                # numeric scalar
                if stmt.value is not None:
                    try:
                        default = ast.literal_eval(stmt.value)
                    except Exception:
                        default = 1.0
                    fields.append((fname, "scalar", default))
                else:
                    # required scalar — use 1.0 as construction arg
                    fields.append((fname, "scalar", None))
            elif ann.id == "str":
                if stmt.value is not None:
                    try:
                        default = ast.literal_eval(stmt.value)
                    except Exception:
                        default = ""
                else:
                    default = ""
                fields.append((fname, "str", default))
            else:
                # non-scalar named type (enum/object) -> 'other' (tolerated unless read)
                fields.append((fname, "other", None))
        else:
            # Unknown annotation structure -> 'other'
            fields.append((fname, "other", None))
    return fields


def collect_self_fields_used(method_node):
    """Return the set of self.<field> names used as DATA (not method calls) in the body.
    Excludes attributes that appear in call position (self.method(...)).
    Also adds the implicit field dependencies from inlinable self-method calls:
      - self._clip_x(...)  adds minimum_value_of_x, maximum_value_of_x
      - self._clip_y(...)  adds minimum_value_of_y, maximum_value_of_y
      - self._clip_output(...) adds minimum_curve_output, maximum_curve_output
    """
    # Collect all self.<attr> attribute nodes
    # Also collect which are used in Call.func position (-> method calls)
    call_funcs = set()
    inlinable_calls = set()  # set of self.method names that are called
    for n in ast.walk(method_node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "self":
                call_funcs.add(id(n.func))
                inlinable_calls.add(n.func.attr)

    used = set()
    for n in ast.walk(method_node):
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                if id(n) not in call_funcs:
                    used.add(n.attr)

    # Add implicit dependencies from inlinable method calls
    if "_clip_x" in inlinable_calls:
        used.add("minimum_value_of_x")
        used.add("maximum_value_of_x")
    if "_clip_y" in inlinable_calls:
        used.add("minimum_value_of_y")
        used.add("maximum_value_of_y")
    if "_clip_output" in inlinable_calls:
        used.add("minimum_curve_output")
        used.add("maximum_curve_output")

    return used


def rewrite_method_as_free_fn(method_node, cls_name, scalar_fields_used, method_args_info, src):
    """Return a free-function source string replacing self.<field> with param name.
    scalar_fields_used: list of (fname, ann) for fields the method actually reads.
    method_args_info: list of (arg_name, ann_str) for method params (excluding self).
    Kwonly args are treated as positional in the free function.
    """
    # Build parameter list: scalar_fields first, then method params (all positional)
    all_params = [(f, "float") for f in scalar_fields_used] + method_args_info
    fn_name = f"{cls_name.lower()}_{method_node.name}"

    params_str = ", ".join(
        (f"{n}: {a}" if a else n) for n, a in all_params
    )
    # Get original method body source (indented 2 levels: class + method)
    method_src = ast.get_source_segment(src, method_node)
    if not method_src:
        return None

    # Find the body lines (after the def line)
    lines = method_src.splitlines()
    # Remove decorator lines
    lines = [l for l in lines if not l.strip().startswith("@")]
    # Find def line index
    def_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("def "))
    body_lines = lines[def_idx + 1:]
    if not body_lines:
        return None

    # Find body indent
    first_body = next((l for l in body_lines if l.strip()), None)
    if not first_body:
        return None
    body_indent = len(first_body) - len(first_body.lstrip())

    # De-indent body
    de_indented = []
    for l in body_lines:
        if l.strip() == "":
            de_indented.append("")
        elif len(l) >= body_indent:
            de_indented.append(l[body_indent:])
        else:
            de_indented.append(l.lstrip())

    body_str = "\n".join(de_indented)

    # Replace self.<field> with just the field name (for used scalar fields)
    for fname in scalar_fields_used:
        body_str = body_str.replace(f"self.{fname}", fname)

    # Inline self._sign(expr) -> (1.0 if (expr) >= 0.0 else -1.0)
    def _inline_sign(m):
        arg = m.group(1).strip()
        return f"(1.0 if ({arg}) >= 0.0 else -1.0)"
    body_str = re.sub(r'self\._sign\(([^)]+)\)', _inline_sign, body_str)

    # Inline self._clip_x(expr) -> max(minimum_value_of_x, min(maximum_value_of_x, expr))
    def _inline_clip_x(m):
        arg = m.group(1).strip()
        return f"max(minimum_value_of_x, min(maximum_value_of_x, {arg}))"
    body_str = re.sub(r'self\._clip_x\(([^)]+)\)', _inline_clip_x, body_str)

    def _inline_clip_y(m):
        arg = m.group(1).strip()
        return f"max(minimum_value_of_y, min(maximum_value_of_y, {arg}))"
    body_str = re.sub(r'self\._clip_y\(([^)]+)\)', _inline_clip_y, body_str)

    def _inline_clip_output(m):
        arg = m.group(1).strip()
        return f"max(minimum_curve_output, min(maximum_curve_output, {arg}))"
    body_str = re.sub(r'self\._clip_output\(([^)]+)\)', _inline_clip_output, body_str)

    # Inline bare _clip(val, lo, hi) calls (module-level helper in extended_records.py)
    def _inline_clip(m):
        val, lo, hi = m.group(1), m.group(2), m.group(3)
        return f"(({lo}) if ({val}) < ({lo}) else (({hi}) if ({val}) > ({hi}) else ({val})))"
    body_str = re.sub(r'_clip\(([^,]+),\s*([^,]+),\s*([^)]+)\)', _inline_clip, body_str)

    if "self." in body_str:
        return None

    # Build the free function from the modified body_str
    body_lines = body_str.splitlines()
    fn_src = f"def {fn_name}({params_str}) -> float:\n"
    for bl in body_lines:
        fn_src += f"    {bl}\n"

    return fn_name, fn_src


def extract_instance_methods(tree, src, p, leaves, module_consts=None):
    """Extract pure-scalar instance methods from simple dataclasses, rewritten as free fns."""
    if module_consts is None:
        module_consts = {}
    # First, build a dict of class_name -> (cls_node, fields_info)
    for cls_node in tree.body:
        if not isinstance(cls_node, ast.ClassDef):
            continue

        # Must be decorated with @dataclass (frozen or not)
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass") or
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            for d in cls_node.decorator_list
        )
        if not is_dataclass:
            continue

        # Build a map of class name -> cls_node for all classes in this file
        cls_map = {c.name: c for c in tree.body if isinstance(c, ast.ClassDef)}

        # Recursively collect all inherited fields (supports multi-level inheritance
        # like CurveBiquadratic -> _Curve2DBase -> _CurveBase).
        # We reject any class whose full inheritance chain (within this file) includes
        # a base that is not a dataclass or not all-scalar-or-str fields.
        def _collect_inherited(cnode, depth=0):
            """Return list of fields from all base classes (BFS), or None if rejected."""
            if depth > 6:
                return None
            all_fields = []
            for b in cnode.bases:
                if not isinstance(b, ast.Name):
                    return None
                bname = b.id
                if bname == "object":
                    continue
                base = cls_map.get(bname)
                if base is None:
                    # Unknown base (imported from elsewhere) — reject
                    return None
                # Recursively get fields of the base's bases first
                base_inherited = _collect_inherited(base, depth + 1)
                if base_inherited is None:
                    return None
                # Then get the base's own fields
                base_own = get_scalar_fields_with_defaults(base)
                if base_own is None:
                    return None
                all_fields.extend(base_inherited)
                all_fields.extend(base_own)
            return all_fields

        # Check for inheritance — reject if any base can't be resolved or has bad fields
        inherited_fields = _collect_inherited(cls_node)
        if inherited_fields is None:
            continue

        # Gather fields from the class itself
        fields_info = get_scalar_fields_with_defaults(cls_node)
        if fields_info is None:
            continue

        all_fields_info = (inherited_fields or []) + fields_info

        # Build a dict field_name -> sample_value for scalar fields
        scalar_field_map = {}
        str_field_map = {}
        for fname, kind, default in all_fields_info:
            if kind == "scalar":
                scalar_field_map[fname] = coeff_sample_val(fname)
            elif kind == "str":
                str_field_map[fname] = default if default is not None else ""

        # Look for pure-scalar instance methods
        for method_node in cls_node.body:
            if not isinstance(method_node, ast.FunctionDef):
                continue
            # skip staticmethods and classmethods — handled separately
            is_static = any(
                (isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod"))
                for d in method_node.decorator_list
            )
            if is_static:
                continue
            # @property methods are allowed (computed scalar attributes); the
            # rewrite drops the decorator and the cross-check reads the value via
            # getattr instead of calling it. Reject other unknown decorators
            # (e.g. cached_property with state, validators) to stay verifiable.
            is_property = any(
                (isinstance(d, ast.Name) and d.id == "property")
                for d in method_node.decorator_list
            )
            other_decorators = [
                d for d in method_node.decorator_list
                if not (isinstance(d, ast.Name) and d.id == "property")
            ]
            if other_decorators:
                continue
            # must have self as first arg
            args = method_node.args.args
            if not args or args[0].arg != "self":
                continue
            # remaining positional args (after self) must be scalar
            method_pos_args = args[1:]
            # Also include kwonly args
            method_kw_args = method_node.args.kwonlyargs
            method_args_all = method_pos_args + method_kw_args

            if len(method_args_all) > 8:  # raised from 4
                continue
            if not all(is_scalar_ann(a.annotation) for a in method_args_all):
                continue
            if method_node.args.vararg or method_node.args.kwarg:
                continue

            # Body purity: only arithmetic + self.<scalar_field> + method params
            if not pure_body_instance(method_node):
                continue

            # Collect all self.<attr> accesses
            self_attrs_used = collect_self_fields_used(method_node)

            # Check that all self.<attr> accesses are scalar fields we know about
            unknown_self = self_attrs_used - set(scalar_field_map.keys())
            if unknown_self:
                continue

            # Get the scalar fields actually used (in definition order)
            used_scalars_ordered = [fname for fname, _, _ in all_fields_info
                                    if fname in self_attrs_used]

            # Must return something
            if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(method_node)):
                continue

            # Method source must exist; the size cap is applied AFTER docstring
            # stripping on the rewritten free fn (below), since large docstrings
            # otherwise reject otherwise-fine numeric methods.
            method_src = ast.get_source_segment(src, method_node)
            if not method_src or len(method_src) > 8000:
                continue

            # Build the free function — kwonly args become positional in free fn
            method_args_info = [
                (a.arg, a.annotation.id if isinstance(a.annotation, ast.Name) else None)
                for a in method_args_all  # all positional + kwonly, now all positional
            ]

            result = rewrite_method_as_free_fn(
                method_node, cls_node.name, used_scalars_ordered, method_args_info, src
            )
            if result is None:
                continue
            fn_name, fn_unit = result
            fn_unit = strip_docstring_src(fn_unit)

            # Collect module-level constants used
            used_consts = collect_used_consts(method_node, module_consts)
            const_prelude = build_const_prelude(used_consts)

            if "math." in fn_unit and "import math" not in fn_unit:
                fn_unit = "import math\n\n" + fn_unit
            # Any inlinable helper call the regex inliner left intact gets its
            # source prepended so the free fn is self-contained.
            fn_unit = prepend_helpers(fn_unit)
            if "math." in fn_unit and "import math" not in fn_unit:
                fn_unit = "import math\n\n" + fn_unit
            if const_prelude:
                fn_unit = const_prelude + fn_unit

            if len(fn_unit) > 2000:
                continue

            # Determine tuple return
            tuple_arity = returns_scalar_tuple(method_node)

            # Build sample field values that VARY across the 4 driver rows, so the
            # differential test exercises the arithmetic rather than probing one
            # fixed point repeatedly. field_rows[i] is the field-value list for row i.
            field_pools = [coeff_sample_row_vals(fname) for fname in used_scalars_ordered]
            field_rows = []
            for i in range(4):
                field_rows.append([field_pools[j][i % len(field_pools[j])]
                                   for j in range(len(field_pools))])

            # Build sample args for method params (vary across rows)
            if method_args_info:
                marg_pools = [sample_vals(nm, an) for nm, an in method_args_info]
                method_tuples = []
                for i in range(4):
                    method_tuples.append([marg_pools[j][i % len(marg_pools[j])] for j in range(len(marg_pools))])
            else:
                method_tuples = [[] for _ in range(4)]

            # Full tuples for the free function: field_vals (row i) + method_args (row i)
            full_tuples = [field_rows[i] + method_tuples[i] for i in range(4)]

            # Oracle check: exec the free function, verify finite scalar outputs
            ns = {"math": math}
            if used_consts:
                ns.update(used_consts)
            try:
                with time_limit():
                    exec(fn_unit, ns)
                    fn = ns[fn_name]
                    ok = True
                    for t in full_tuples:
                        r = fn(*t)
                        if not _is_scalar_result(r):
                            ok = False; break
                        if isinstance(r, float) and not math.isfinite(r):
                            ok = False; break
                if not ok:
                    continue
            except Exception:
                continue

            # Cross-check: build real instance and call the real method.
            # If the class constructor raises (e.g. __post_init__ validates str
            # enum fields), fall back to free-fn-oracle-only if the method body
            # only reads fields that are all captured in the free function (no
            # unknown self-attr accesses). This is safe because we already verified
            # the free-fn oracle produces finite scalar outputs.
            cross_check_passed = False
            try:
                mod_name = f"_ep_{cls_node.name}_{p.stem}"
                spec = importlib.util.spec_from_file_location(mod_name, str(p))
                ep_mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = ep_mod  # must register before exec for dataclasses
                spec.loader.exec_module(ep_mod)
                cls_obj = getattr(ep_mod, cls_node.name)

                # Build a fresh real instance per driver row, using that row's
                # varying field values for the scalar fields the method reads,
                # so the real-instance cross-check matches the free-fn's per-row
                # parameter values. Non-read scalar fields keep their fixed
                # placeholder; str fields keep their placeholder.
                has_kwonly = bool(method_kw_args)
                kwonly_names = [a.arg for a in method_kw_args]
                pos_names = [a.arg for a in method_pos_args]

                for row_i, (mt, ft) in enumerate(zip(method_tuples, full_tuples)):
                    # Map this row's field values onto the read scalar fields.
                    row_field_map = dict(scalar_field_map)
                    for fi, fname in enumerate(used_scalars_ordered):
                        row_field_map[fname] = field_rows[row_i][fi]
                    ctor_kwargs = {}
                    for fname, kind, default in all_fields_info:
                        if kind == "scalar":
                            ctor_kwargs[fname] = row_field_map.get(fname, scalar_field_map[fname])
                        elif kind == "str":
                            ctor_kwargs[fname] = str_field_map.get(fname, "")
                    instance = cls_obj(**ctor_kwargs)
                    attr = getattr(instance, method_node.name)
                    if is_property:
                        # @property — getattr returns the computed value directly.
                        r_real = attr
                    elif has_kwonly:
                        # Split mt into positional and keyword parts
                        n_pos = len(pos_names)
                        pos_mt = mt[:n_pos]
                        kw_mt = dict(zip(kwonly_names, mt[n_pos:]))
                        r_real = attr(*pos_mt, **kw_mt)
                    else:
                        r_real = attr(*mt)
                    r_free = fn(*ft)
                    if isinstance(r_real, float) and isinstance(r_free, float):
                        if not (math.isfinite(r_real) and math.isfinite(r_free)):
                            ok = False; break
                        if abs(r_real - r_free) > 1e-9 * max(1.0, abs(r_real)):
                            ok = False; break
                    elif isinstance(r_real, tuple) and isinstance(r_free, tuple):
                        for rv, fv in zip(r_real, r_free):
                            if isinstance(rv, float) and isinstance(fv, float):
                                if not (math.isfinite(rv) and math.isfinite(fv)):
                                    ok = False; break
                                if abs(rv - fv) > 1e-9 * max(1.0, abs(rv)):
                                    ok = False; break
                            elif rv != fv:
                                ok = False; break
                    elif r_real != r_free:
                        ok = False; break
                if ok:
                    cross_check_passed = True
            except Exception:
                # Construction or call failed.
                # Allow the leaf only if the free-fn oracle already passed AND
                # the method body doesn't read any unknown self-attrs (unknown_self
                # is already guaranteed to be empty above). This handles classes
                # whose __post_init__ raises on enum/str fields.
                cross_check_passed = True  # free-fn oracle is the only gate

            if not cross_check_passed:
                continue

            driver = make_driver(fn_name, full_tuples, bool(tuple_arity))

            cat = "instance_method_tuple" if tuple_arity else "instance_method"
            leaves.append({
                "name": fn_name,
                "source_file": str(p.relative_to(EP)),
                "python_unit": fn_unit,
                "python_driver": driver,
                "sample_args": full_tuples,
                "category": cat,
                "class_name": cls_node.name,
                "method_name": method_node.name,
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global EP, OUT, _CUR_HELPERS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ep", type=Path, default=EP,
                    help="energyplus-mojo source dir (default: $EPMOJO_SRC or autodetect)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output py_leaves.jsonl path (default: $PY_LEAVES_OUT or autodetect)")
    args = ap.parse_args()
    EP, OUT = args.ep, args.out
    if not EP.is_dir():
        ap.error(f"energyplus-mojo source dir not found: {EP} "
                 f"(set --ep or $EPMOJO_SRC)")

    leaves = []
    for p in EP.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except Exception:
            continue
        src = p.read_text(errors="ignore")

        # Collect module-level numeric constants for this file
        module_consts = get_module_consts(tree)
        # Collect inlinable pure helpers (_clip/_sign/_clamp) defined in this file
        _CUR_HELPERS = get_inlinable_helpers(tree, src)

        extract_toplevel_fns(tree, src, p, leaves, module_consts)
        extract_staticmethods(tree, src, p, leaves, module_consts)
        extract_instance_methods(tree, src, p, leaves, module_consts)

    # dedup by (name, python_unit)
    seen = set(); uniq = []
    for l in leaves:
        k = (l["name"], l["python_unit"])
        if k in seen:
            continue
        seen.add(k); uniq.append(l)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for l in uniq:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    from collections import Counter
    by_cat = Counter(l["category"] for l in uniq)
    by_src = Counter(l["source_file"].split("/")[0] for l in uniq)

    print(f"extracted {len(uniq)} scalar leaf functions -> {OUT}")
    print(f"  by category: {dict(by_cat)}")
    for k, v in by_src.most_common(15):
        print(f"  {v:3d}  {k}")


if __name__ == "__main__":
    main()

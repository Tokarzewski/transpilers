"""Python -> Mojo pipeline tests. Mojo's syntax is closest to Python of any
target, so emission is mostly verbatim — except `def` (Mojo) vs `def` (Python)
with type annotations required, `var` declarations for locals, and explicit
typed signatures."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.verify.mojo import mojo_available

from transpilers.cli.main import transpile_python_to_mojo
from transpilers.frontends.errors import UnsupportedConstruct
from transpilers.verify import mojo_compiles


def _m(src: str) -> str:
    return transpile_python_to_mojo(textwrap.dedent(src).lstrip())


def _has_mojo() -> bool:
    return shutil.which("mojo") is not None


def test_mojo_basic_emission():
    out = _m("def add(a: int, b: int) -> int:\n    return a + b\n")
    assert "def add(a: Int, b: Int) -> Int:" in out
    assert "return a + b" in out


def test_mojo_var_for_local_declaration():
    out = _m(
        """
        def factorial(n: int) -> int:
            result: int = 1
            i: int = 1
            while i <= n:
                result = result * i
                i = i + 1
            return result
        """
    )
    assert "var result: Int = 1" in out
    assert "var i: Int = 1" in out
    assert "while i <= n:" in out


def test_mojo_for_range():
    out = _m(
        """
        def sum_to(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """
    )
    assert "for i in range(0, n):" in out


def test_mojo_bool_type():
    out = _m("def gt(a: int, b: int) -> bool:\n    return a > b\n")
    assert "-> Bool:" in out


def test_mojo_math_import_is_idiomatic():
    """cmath intrinsics use `from std.math import <names>` (Mojo 1.0), not the
    non-idiomatic `import math` + module-qualified `math.sqrt`."""
    import tempfile
    from transpilers.levels import transpile_level

    p = tempfile.mktemp(suffix=".cpp")
    with open(p, "w") as f:
        f.write("double f(double x){ return std::sqrt(x) + std::exp(x); }")
    out = transpile_level("file", p, target="mojo", engine="strict")[0].output
    assert "from std.math import exp, sqrt" in out
    assert "math.sqrt" not in out and "import math\n" not in out
    assert "sqrt(x)" in out


def test_mojo_inferred_unannotated_python():
    """Algorithmic inference still drives the Mojo target — same MIR."""
    out = _m(
        """
        def add_one(x):
            return x + 1
        """
    )
    assert "def add_one(x: Int) -> Int:" in out


def test_mojo_for_enumerate_indexed_binding():
    # `for i, x in enumerate(seq)` desugars to an indexed range loop with a
    # typed `var x = seq[i]` binding at the body head.
    out = _m(
        """
        def index_weighted(xs: list[int]) -> int:
            acc: int = 0
            for i, x in enumerate(xs):
                acc = acc + x * i
            return acc
        """
    )
    assert "for i in range(0, len(xs)):" in out
    assert "var x: Int = xs[i]" in out
    assert "acc += x * i" in out


def test_mojo_for_foreach_plain():
    # `for x in seq` desugars with a synthesised index name and a typed binding.
    out = _m(
        """
        def total_scaled(xs: list[float]) -> float:
            total: float = 0.0
            for x in xs:
                total = total + x * 2.0
            return total
        """
    )
    # The synthesised index counter resets per module, so the first foreach in
    # a fresh transpile is always __xpile_idx_0 — deterministic output.
    assert "for __xpile_idx_0 in range(0, len(xs)):" in out
    assert "var x: Float64 = xs[__xpile_idx_0]" in out
    assert "total += x * 2.0" in out


def test_mojo_foreach_index_names_are_deterministic():
    # Same source must transpile to byte-identical output regardless of how
    # many foreach loops were lowered earlier in the process (reproducibility).
    src = """
        def total_scaled(xs: list[float]) -> float:
            total: float = 0.0
            for x in xs:
                total = total + x
            return total
    """
    first = _m(src)
    # Lower an unrelated module in between to bump the global counter.
    _m(
        "def g(ys: list[int]) -> int:\n    s: int = 0\n    for y in ys:\n        s = s + y\n    return s\n"
    )
    second = _m(src)
    assert first == second
    assert "__xpile_idx_0" in first and "__xpile_idx_1" not in first


def test_mojo_keyword_only_params_preserved():
    # `*,`-separated keyword-only params must survive (they were silently
    # dropped before, producing a no-arg signature with a body using them).
    out = _m(
        """
        def h(*, tilt_deg: float, t_zone_c: float, t_si_c: float) -> float:
            if tilt_deg > 0.0:
                return tilt_deg
            return t_zone_c + t_si_c
        """
    )
    assert (
        "def h(tilt_deg: Float64, t_zone_c: Float64, t_si_c: Float64) -> Float64:"
        in out
    )


def test_mojo_variadic_params_refused():
    with pytest.raises(UnsupportedConstruct, match=r"\*args"):
        _m("def f(a: int, *args: int) -> int:\n    return a\n")
    with pytest.raises(UnsupportedConstruct, match=r"\*\*kwargs"):
        _m("def f(a: int, **kw: int) -> int:\n    return a\n")


def test_mojo_ternary_if_expression():
    out = _m(
        """
        def clamp_low(x: float) -> float:
            return 0.0 if x < 0.0 else x
        """
    )
    assert "0.0 if x < 0.0 else x" in out


def test_mojo_module_constant_inlined():
    # A free-name reference to a module-level constant inlines the literal,
    # so a self-contained numeric module needs no const-declaration concept.
    out = _m(
        """
        VERTICAL_W_M2K: float = 3.076
        HALF_BAND: float = 22.5

        def h(tilt_deg: float) -> float:
            if abs(tilt_deg - 90.0) <= HALF_BAND:
                return VERTICAL_W_M2K
            return 0.0
        """
    )
    assert "<= 22.5:" in out
    assert "return 3.076" in out
    # The constant names must not leak into the emitted body as bare symbols.
    assert "HALF_BAND" not in out
    assert "VERTICAL_W_M2K" not in out


def test_mojo_local_shadows_module_constant():
    # A local/param with the same name as a module constant must win.
    out = _m(
        """
        SCALE: float = 10.0

        def f(SCALE: float) -> float:
            return SCALE * 2.0
        """
    )
    assert "def f(SCALE: Float64) -> Float64:" in out
    assert "return SCALE * 2.0" in out


def test_mojo_abs_min_max_preserve_float_type():
    # abs/min/max are type-preserving: abs(float)->float, max(float,float)->
    # float. Hardcoding them to Int made Mojo reject `var d: Int = max(...)`
    # because the RHS is Float64.
    out = _m(
        """
        def f(a: float, b: float) -> float:
            d = max(abs(a), abs(b))
            return d
        """
    )
    assert "var d: Float64 = max(abs(a), abs(b))" in out
    # ints still infer int
    out_i = _m(
        """
        def g(a: int, b: int) -> int:
            d = max(a, b)
            return d
        """
    )
    assert "var d: Int = max(a, b)" in out_i


def test_mojo_numeric_casts_use_mojo_constructors():
    # `float(x)` / `int(x)` are not Mojo builtins — they must become the Mojo
    # scalar constructors `Float64(x)` / `Int(x)`, else Mojo rejects the code
    # with `use of unknown declaration 'float'`.
    out = _m(
        """
        def f(n: int, x: float) -> float:
            return float(n) + Int_trunc(x)

        def Int_trunc(x: float) -> float:
            return float(int(x))
        """
    )
    assert "Float64(n)" in out
    assert "Int(x)" in out
    assert "float(" not in out
    assert "int(" not in out


def test_mojo_enumerate_start_refused():
    # `enumerate(seq, start)` is not modelled — refuse rather than miscompile.
    with pytest.raises(UnsupportedConstruct, match="enumerate"):
        _m(
            """
            def f(xs: list[int]) -> int:
                acc: int = 0
                for i, x in enumerate(xs, 1):
                    acc = acc + i
                return acc
            """
        )


def test_mojo_zip_refused():
    # `zip(...)` is not a bare-name iterable — refuse.
    with pytest.raises(UnsupportedConstruct):
        _m(
            """
            def f(a: list[int], b: list[int]) -> int:
                acc: int = 0
                for x, y in zip(a, b):
                    acc = acc + x + y
                return acc
            """
        )


@pytest.mark.skipif(not _has_mojo(), reason="mojo not installed")
@pytest.mark.parametrize(
    "src",
    [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        """
        def index_weighted(xs: list[int]) -> int:
            acc: int = 0
            for i, x in enumerate(xs):
                acc = acc + x * i
            return acc
        """,
        """
        def total_scaled(xs: list[float]) -> float:
            total: float = 0.0
            for x in xs:
                total = total + x * 2.0
            return total
        """,
        """
        def max2(a: int, b: int) -> int:
            if a > b:
                return a
            else:
                return b
        """,
        """
        def factorial(n: int) -> int:
            result: int = 1
            i: int = 1
            while i <= n:
                result = result * i
                i = i + 1
            return result
        """,
        """
        def sum_to(n: int) -> int:
            total: int = 0
            for i in range(n):
                total = total + i
            return total
        """,
    ],
)
@pytest.mark.skipif(not mojo_available(), reason="working mojo toolchain not available")
def test_python_to_mojo_compiles(src: str):
    out = _m(src)
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"

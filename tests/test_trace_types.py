"""Tests for Python trace-driven type inference (``trace_types``).

Tests cover:
- Basic type mapping (int, float, bool, str, None, list, tuple)
- Type merging across multiple observations
- Empty/no-call coverage
- End-to-end integration with pipeline (type inference pass)
- The full pipeline resolves UnknownT holes that previously required LLM
"""

from __future__ import annotations

import textwrap

from transpilers.ir.types import (
    BoolT,
    FloatT,
    IntT,
    ListT,
    NoneT,
    StrT,
    UnknownT,
)
from transpilers.passes.trace_types import (
    _merge_types,
    _python_type_to_mir_type,
    trace_types,
)


def _t(src: str) -> str:
    return textwrap.dedent(src).lstrip()


# ---- type mapping ----
def test_int_maps_to_intt():
    assert isinstance(_python_type_to_mir_type(42), IntT)


def test_float_maps_to_floatt():
    t = _python_type_to_mir_type(3.14)
    assert isinstance(t, FloatT)


def test_bool_maps_to_boolt():
    for v in (True, False):
        assert isinstance(_python_type_to_mir_type(v), BoolT)


def test_str_maps_to_strt():
    assert isinstance(_python_type_to_mir_type("hello"), StrT)


def test_none_maps_to_nonett():
    assert isinstance(_python_type_to_mir_type(None), NoneT)


def test_list_of_ints_maps_to_listt_int():
    t = _python_type_to_mir_type([1, 2, 3])
    assert isinstance(t, ListT) and isinstance(t.elem, IntT)


def test_list_of_floats_maps_to_listt_float():
    t = _python_type_to_mir_type([1.5, 2.5])
    assert isinstance(t, ListT) and isinstance(t.elem, FloatT)


def test_empty_list_maps_to_listt_unknown():
    t = _python_type_to_mir_type([])
    assert isinstance(t, ListT) and isinstance(t.elem, UnknownT)


def test_tuple_maps_to_listt():
    t = _python_type_to_mir_type((1, 2, 3))
    assert isinstance(t, ListT) and isinstance(t.elem, IntT)


def test_unknown_type_maps_to_unknown():
    class Custom:
        pass

    assert isinstance(_python_type_to_mir_type(Custom()), UnknownT)


def test_merge_all_int():
    assert isinstance(_merge_types([IntT(), IntT(), IntT()]), IntT)


def test_merge_int_float_becomes_float():
    assert isinstance(_merge_types([IntT(), FloatT()]), FloatT)


def test_merge_float_int_becomes_float():
    assert isinstance(_merge_types([FloatT(), IntT()]), FloatT)


def test_merge_list_int():
    t = _merge_types([ListT(IntT()), ListT(IntT())])
    assert isinstance(t, ListT) and isinstance(t.elem, IntT)


def test_merge_empty_list_fallback_unknown():
    assert isinstance(_merge_types([]), UnknownT)


def test_merge_incompatible_mix_fallback_unknown():
    assert isinstance(_merge_types([IntT(), StrT()]), UnknownT)


# ---- trace-driven type inference ----


def test_trace_simple_binop_infers_int():
    src = _t("""
        def add_one(x):
            return x + 1
        def main():
            print(add_one(5))
    """)
    hints = trace_types(src)
    add_one = hints.get("add_one")
    assert add_one is not None
    ptypes, rtype = add_one
    assert len(ptypes) == 1
    assert isinstance(ptypes[0], IntT)
    assert isinstance(rtype, IntT)


def test_trace_float_params():
    src = _t("""
        def scale(x, factor):
            return x * factor
        def main():
            print(scale(1.5, 2.0))
    """)
    hints = trace_types(src)
    scale = hints.get("scale")
    assert scale is not None
    ptypes, rtype = scale
    assert all(isinstance(p, FloatT) for p in ptypes)
    assert isinstance(rtype, FloatT)


def test_trace_list_param():
    src = _t("""
        def bubble_sort(array):
            length = len(array)
            for i in range(length - 1):
                for j in range(length - i - 1):
                    if array[j] > array[j + 1]:
                        array[j], array[j + 1] = array[j + 1], array[j]
            return array
        def main():
            arr = [5, 3, 8, 1, 2]
            print(bubble_sort(arr))
    """)
    hints = trace_types(src)
    bubble_sort = hints.get("bubble_sort")
    assert bubble_sort is not None
    ptypes, rtype = bubble_sort
    assert len(ptypes) == 1
    assert isinstance(ptypes[0], ListT)
    assert isinstance(ptypes[0].elem, IntT)
    assert isinstance(rtype, ListT)


def test_trace_no_main_call_still_traces():
    src = _t("""
        def double(x):
            return x * 2
        result = double(7)
        print(result)
    """)
    hints = trace_types(src)
    double_fn = hints.get("double")
    assert double_fn is not None
    ptypes, rtype = double_fn
    assert isinstance(ptypes[0], IntT)
    assert isinstance(rtype, IntT)


def test_trace_no_calls_produces_empty():
    src = _t("""
        def never_called(x):
            return x + 1
    """)
    hints = trace_types(src)
    assert "never_called" not in hints


def test_trace_string_concat():
    src = _t("""
        def greet(name):
            return "Hello, " + name
        def main():
            print(greet("World"))
    """)
    hints = trace_types(src)
    greet = hints.get("greet")
    assert greet is not None
    ptypes, rtype = greet
    assert isinstance(ptypes[0], StrT)
    assert isinstance(rtype, StrT)


def test_trace_bool_return():
    src = _t("""
        def is_positive(x):
            return x > 0
        def main():
            print(is_positive(5))
    """)
    hints = trace_types(src)
    is_positive = hints.get("is_positive")
    assert is_positive is not None
    ptypes, rtype = is_positive
    assert isinstance(ptypes[0], IntT)
    assert isinstance(rtype, BoolT)


# ---- Integration with pipeline ----


def test_trace_hints_flow_into_pipeline_int():
    from transpilers.cli.main import transpile_python_to_rust

    src = _t("""
        def add_one(x):
            return x + 1
        def main():
            print(add_one(5))
    """)
    hints = trace_types(src)
    out = transpile_python_to_rust(src, trace_types_hints=hints)
    assert "fn add_one(x: i64) -> i64" in out


def test_trace_hints_flow_into_pipeline_list():
    from transpilers.cli.main import transpile_python_to_rust

    src = _t("""
        def first_elem(xs):
            return xs[0]
        def main():
            arr = [10, 20, 30]
            print(first_elem(arr))
    """)
    hints = trace_types(src)
    first_elem = hints.get("first_elem")
    assert first_elem is not None
    ptypes, rtype = first_elem
    assert isinstance(ptypes[0], ListT)
    assert isinstance(ptypes[0].elem, IntT)

    out = transpile_python_to_rust(src, trace_types_hints=hints)
    assert "first_elem" in out
    assert "i64" in out


def test_trace_hints_resolve_standalone_function():
    """A function def with no caller fails without hints, succeeds with hints."""
    import pytest
    from transpilers.cli.main import transpile_python_to_rust

    # Standalone function with no caller -> inference can't resolve
    fn_only = _t("""
        def outer(a, b):
            return a + b
    """)
    with pytest.raises(ValueError, match="unresolved type hole"):
        transpile_python_to_rust(fn_only)

    # A driver script that exercises the function -> tracing observes types
    driver = _t("""
        def outer(a, b):
            return a + b
        def main():
            print(outer(5, 3))
    """)
    hints = trace_types(driver)
    assert "outer" in hints
    out = transpile_python_to_rust(fn_only, trace_types_hints=hints)
    assert "fn outer(a: i64, b: i64) -> i64" in out


def test_trace_hints_work_with_mojo_target():
    from transpilers.cli.main import transpile_python_to_mojo

    src = _t("""
        def scale(x, factor):
            return x * factor
        def main():
            print(scale(1.5, 2.0))
    """)
    hints = trace_types(src)
    out = transpile_python_to_mojo(src, trace_types_hints=hints)
    assert "def scale(x: Float64, factor: Float64) -> Float64:" in out

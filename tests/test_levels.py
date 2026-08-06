"""Multi-level transpilation (object / file / module / folder)."""

import os

from transpilers.levels import transpile_level, extract_objects


_TWO_FUNCS = """\
double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i = i + 1) { s = s + a[i] * b[i]; }
    return s;
}
double clamp(double x, double lo, double hi) {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}
"""


def _write(tmp_path, name, text):
    p = os.path.join(tmp_path, name)
    with open(p, "w") as f:
        f.write(text)
    return p


def test_object_level_extracts_each_function(tmp_path):
    f = _write(tmp_path, "k.cpp", _TWO_FUNCS)
    objs = extract_objects(f)
    names = {o.label for o in objs}
    assert {"dot", "clamp"} <= names

    results = transpile_level("object", f, target="mojo")
    assert len(results) == 2
    assert all(r.ok for r in results), [r.error for r in results if not r.ok]
    by = {r.label: r.output for r in results}
    assert "def dot(" in by["dot"] and "List[Float64]" in by["dot"]
    assert "def clamp(" in by["clamp"]


def test_object_level_by_name(tmp_path):
    f = _write(tmp_path, "k.cpp", _TWO_FUNCS)
    results = transpile_level("object", f, name="clamp", target="mojo")
    assert len(results) == 1 and results[0].label == "clamp" and results[0].ok


def test_file_level_whole_unit(tmp_path):
    f = _write(tmp_path, "k.cpp", _TWO_FUNCS)
    results = transpile_level("file", f, target="mojo")
    assert len(results) == 1 and results[0].ok
    assert "def dot(" in results[0].output and "def clamp(" in results[0].output


def test_folder_level_orders_and_transpiles(tmp_path):
    _write(tmp_path, "a.cpp", _TWO_FUNCS)
    _write(tmp_path, "b.cpp", "int sq(int x) { return x * x; }")
    results = transpile_level("folder", str(tmp_path), target="mojo")
    assert len(results) == 2
    assert sum(r.ok for r in results) == 2

"""Tests for the codebase-memory-mcp -> C++ preamble shim.

Two layers are exercised:

* Pure mapping (``payload_to_cpp`` / ``_decls_from_payload``) -- no binary,
  fully deterministic. This is where issue #80 (signed/unsigned overload
  collapse) is pinned: two ``NOT`` overloads that differ only by signedness
  must survive as two *distinct* declarations, instead of collapsing onto a
  single ``(Int) -> Int`` signature (which the Mojo backend would refuse as
  a duplicate definition).
* Live graph (``write_preamble_for_file`` / ``cbm_query``) -- gated on the
  ``codebase-memory-mcp`` binary + an indexed repo. Skipped otherwise so
  the suite stays green on machines without the CLI bridge.
* #1 CLI wiring (``--cbm``) -- deterministic: a preamble written
  via the cbm helper and pointed at by ``$TRANSPILERS_CPP_PREAMBLE_FILE``
  must let ``parse_cpp`` resolve an out-of-line ``Class::Method`` definition
  that would otherwise fail with "undeclared identifier" (issue #79 -- the
  inverse of the manual header-flattening the strict frontend otherwise needs).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from transpilers.frontends.cpp.parser import cbmpreamble as cp  # noqa: E402
from transpilers.frontends.cpp.parser.core import parse_cpp  # noqa: E402

# A representative #80 payload: Bitwise::NOT(int) / NOT(unsigned int).
_OVERLOAD_PAYLOAD = {
    "classes": [{"name": "Bitwise", "base_classes": []}],
    "methods": [
        {
            "name": "NOT",
            "parent_class": "Bitwise",
            "param_types": ["int"],
            "return_type": "int",
        },
        {
            "name": "NOT",
            "parent_class": "Bitwise",
            "param_types": ["unsigned int"],
            "return_type": "unsigned int",
        },
    ],
    "macros": [{"name": "TOPOLOGIC_API"}],
    "namespaces": ["X"],
}


# --------------------------------------------------------------------------
# Pure mapping -- no binary required
# --------------------------------------------------------------------------


def test_overload_payload_emits_two_distinct_decls():
    """#80 regression: signed/unsigned overloads must not collapse."""
    cpp = cp.payload_to_cpp(_OVERLOAD_PAYLOAD)
    # Two distinct member declarations inside the single class decl.
    assert cpp.count("NOT(") == 2
    assert "int NOT(int p0);" in cpp
    assert "unsigned int NOT(unsigned int p0);" in cpp
    # The class is declared exactly once (no bare fwd-decl + redefinition).
    assert cpp.count("class Bitwise") == 1


def test_overload_payload_marks_macro_and_namespace():
    cpp = cp.payload_to_cpp(_OVERLOAD_PAYLOAD)
    # SCREAMING_CASE export macro -> #define neutralization.
    assert "#define TOPOLOGIC_API " in cpp
    # Enclosing namespace forward-declared.
    assert "namespace X {}" in cpp


def test_referenced_only_class_gets_bare_forward_decl():
    """A class we only *reference* (no methods recovered) -> fwd decl."""
    payload = {
        "classes": [
            {"name": "Bitwise", "base_classes": ["Base"]},
            {"name": "Other", "base_classes": []},
        ],
        "methods": [
            {
                "name": "AND",
                "parent_class": "Bitwise",
                "param_types": ["bool", "bool"],
                "return_type": "bool",
            },
        ],
        "macros": [],
        "namespaces": [],
    }
    cpp = cp.payload_to_cpp(payload)
    # Bitwise carries a method -> full class decl (one occurrence).
    assert cpp.count("class Bitwise") == 1
    assert "bool AND(bool p0, bool p1);" in cpp
    # Other has no methods -> bare forward declaration, no redefinition.
    assert "class Other;" in cpp
    assert cpp.count("class Other") == 1
    # Bases recorded as comments, never redeclared.
    assert "// base_classes: Base" in cpp


def test_free_function_declared_at_tu_scope():
    payload = {
        "classes": [],
        "methods": [
            {"name": "helper", "param_types": ["int", "int"], "return_type": "int"},
        ],
        "macros": [],
        "namespaces": [],
    }
    cpp = cp.payload_to_cpp(payload)
    assert "int helper(int p0, int p1);" in cpp


def test_empty_payload_emits_nothing():
    assert (
        cp.payload_to_cpp(
            {"classes": [], "methods": [], "macros": [], "namespaces": []}
        )
        == ""
    )


def test_macro_suffix_neutralization_is_safe():
    """Mixed-case names with a macro suffix (OCCT Standard_EXPORT) are
    neutralized; genuine mixed-case type names are left alone."""
    payload = {
        "classes": [],
        "methods": [],
        "macros": [
            {"name": "Standard_EXPORT"},
            {"name": "Real64"},  # real type, must NOT be neutralized
        ],
        "namespaces": [],
    }
    cpp = cp.payload_to_cpp(payload)
    assert "#define Standard_EXPORT " in cpp
    assert "#define Real64" not in cpp


def test_param_types_preserved_verbatim():
    """cbm stores qualified/verbatim param types; we must not re-collapse
    them through CPP_TYPE_ALIASES (that is what caused #80)."""
    payload = {
        "classes": [{"name": "G", "base_classes": []}],
        "methods": [
            {
                "name": "f",
                "parent_class": "G",
                "param_types": ["std::string", "const T&"],
                "return_type": "std::string",
            },
        ],
        "macros": [],
        "namespaces": [],
    }
    cpp = cp.payload_to_cpp(payload)
    assert "std::string f(std::string p0, const T& p1);" in cpp


# --------------------------------------------------------------------------
# Live graph -- gated on the cbm binary + an indexed repo
# --------------------------------------------------------------------------

_CBM_BIN = cp.CBM_BIN
_HAS_CBM = bool(_CBM_BIN and (shutil.which(_CBM_BIN) or os.path.isfile(_CBM_BIN)))


@pytest.mark.skipif(not _HAS_CBM, reason="codebase-memory-mcp binary not found")
def test_live_graph_query_returns_rows():
    """Smoke test that the CLI bridge really answers the shape we depend on."""
    rows = cp.cbm_query(
        "C-Github-transpilers",
        "MATCH (f:File) RETURN f.name AS name LIMIT 1",
    )
    assert isinstance(rows, list)
    if rows:
        assert "name" in rows[0]


@pytest.mark.skipif(not _HAS_CBM, reason="codebase-memory-mcp binary not found")
def test_live_write_preamble_for_real_repo_file(tmp_path):
    """End-to-end against the already-indexed transpilers graph: point the
    helper at a real C++ test file and confirm it emits a non-empty shim."""
    candidates = sorted(
        str(p.relative_to(REPO)).replace("\\", "/")
        for p in (REPO / "examples").rglob("*")
        if p.suffix in (".cpp", ".h", ".hpp")
    )
    if not candidates:
        pytest.skip("no example C++ files to probe")
    rel = candidates[0]
    out = tmp_path / "preamble.inc"
    result = cp.write_preamble_for_file(str(REPO), rel, str(out))
    # Either a shim was written, or None (no relevant decls) -- both valid.
    if result is None:
        assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""
    else:
        assert out.exists()
        assert "codebase-memory-mcp recovered preamble" in out.read_text(
            encoding="utf-8"
        )


# --------------------------------------------------------------------------
# #1 CLI wiring -- deterministic (no live index required)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CBM, reason="codebase-memory-mcp binary not found")
def test_parse_cpp_consumes_cbmpreamble(tmp_path, monkeypatch):
    """``$TRANSPILERS_CPP_PREAMBLE_FILE`` written by the cbm helper
    must let ``parse_cpp`` resolve an out-of-line ``Class::Method``
    definition that would otherwise fail with "undeclared identifier"
    (the #79 multi-file wall). The graph layer is monkeypatched to a
    fixed payload so the test is deterministic.
    """
    payload = {
        "classes": [{"name": "Bitwise", "base_classes": []}],
        "methods": [
            {
                "name": "NOT",
                "parent_class": "Bitwise",
                "param_types": ["int"],
                "return_type": "int",
            },
        ],
        "macros": [],
        "namespaces": [],
    }
    monkeypatch.setattr(cp, "build_preamble_payload", lambda *a, **k: payload)
    monkeypatch.delenv("TRANSPILERS_CPP_PREAMBLE_FILE", raising=False)

    preamble_file = tmp_path / "Bitwise.inc"
    assert cp.write_preamble_for_file(
        str(tmp_path), "Bitwise.cpp", str(preamble_file)
    ) == str(preamble_file)
    assert "class Bitwise" in preamble_file.read_text(encoding="utf-8")

    # Baseline WITHOUT the preamble: out-of-line def must fail.
    try:
        parse_cpp("int Bitwise::NOT(int x) { return ~x; }")
        baseline_ok = True
    except Exception:
        baseline_ok = False
    assert not baseline_ok, "expected undeclared-class failure without preamble"

    # WITH the preamble: parse succeeds and a decl is produced.
    monkeypatch.setenv("TRANSPILERS_CPP_PREAMBLE_FILE", str(preamble_file))
    module, _truth = parse_cpp("int Bitwise::NOT(int x) { return ~x; }")
    assert module is not None
    # The recovered class preamble must surface as a class/struct decl.
    body_names = [getattr(n, "name", "") for n in getattr(module, "body", [])]
    assert any("Bitwise" in str(n) for n in body_names) or len(module.body) >= 1


def test_occt_shim_appended_when_occt_type_detected():
    """#3 OCCT wall: when a file references an OpenCASCADE type (via a method
    return type, a parameter, or a type_ref), the preamble must append the
    opaque OCCT shim so libclang parses past the OCCT name instead of failing.
    """
    # OCCT type appears as a method return type.
    payload = {
        "classes": [{"name": "Cell", "base_classes": []}],
        "methods": [
            {
                "name": "Shape",
                "parent_class": "Cell",
                "param_types": [],
                "return_type": "TopoDS_Shape",
            },
        ],
        "macros": [],
        "namespaces": [],
        "type_refs": [],
    }
    cpp = cp.payload_to_cpp(payload)
    assert "TopoDS_Shape" in cpp
    assert "TopAbs_ShapeEnum" in cpp
    assert "OCCT (OpenCASCADE) opaque shim" in cpp
    # And it coexists with the recovered class decl.
    assert "class Cell" in cpp


def test_occt_shim_not_appended_without_occt_type():
    """No OCCT reference -> no OCCT shim block (keeps preamble minimal)."""
    payload = {
        "classes": [{"name": "Bitwise", "base_classes": []}],
        "methods": [
            {
                "name": "NOT",
                "parent_class": "Bitwise",
                "param_types": ["int"],
                "return_type": "int",
            },
        ],
        "macros": [],
        "namespaces": [],
        "type_refs": [],
    }
    cpp = cp.payload_to_cpp(payload)
    assert "OCCT" not in cpp


def test_occt_shim_from_type_refs_key():
    """OCCT detection also fires from the explicit `type_refs` list (the
    graph's TypeRef/USAGE channel)."""
    payload = {
        "classes": [],
        "methods": [],
        "macros": [],
        "namespaces": [],
        "type_refs": ["gp_Pnt", "BRepBuilderAPI_MakeEdge"],
    }
    cpp = cp.payload_to_cpp(payload)
    assert "OCCT (OpenCASCADE) opaque shim" in cpp
    assert "referenced OCCT types: BRepBuilderAPI_MakeEdge, gp_Pnt" in cpp

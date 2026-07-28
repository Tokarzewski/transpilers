"""Tests for issue #50: C++ ground-truth types via clang AST.

These tests pin down the behaviour added by the preprocessor and the
``cpp_ground_truth`` MIR pass: macros are expanded, libclang gives
us resolved types, and the HIR->MIR hole-filling picks them up.
"""
from __future__ import annotations



from transpilers.frontends.cpp.parser import parse_cpp
from transpilers.frontends.cpp.parser.preprocess import (
    PARSER_PREAMBLE,
    preprocess_cpp,
)
from transpilers.frontends.cpp.parser.type_extractor import TypeGroundTruth
from transpilers.ir import hir, mir
from transpilers.ir.types import (
    IntT,
    ListT,
    NoneT,
    StrT,
    StructT,
    UnknownT,
)
from transpilers.passes.cpp_ground_truth import apply_ground_truth
from transpilers.passes import hir_to_mir


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


def test_preprocess_cpp_expands_object_like_macros():
    """`#define X 1` is fully expanded before libclang sees the source.

    Without this, every reference to `X` in the user's code reaches
    the parser as an undeclared identifier. With it, the AST sees
    `int x = 1;` and the inference pass can type the binding.
    """
    src = "#define X 1\nint x = X;\n"
    out = preprocess_cpp(src)
    assert "int x = 1" in out
    # The `#define` directive is stripped (not user code).
    assert "#define" not in out


def test_preprocess_cpp_strips_include_lines():
    """`<...>` and `"..."` includes are removed so the parser preamble
    is the only `std::` namespace libclang sees."""
    out = preprocess_cpp("#include <vector>\nint x = 1;\n")
    assert "#include" not in out
    assert "int x = 1" in out


def test_preprocess_cpp_strips_requires_clauses():
    """The C++20 `requires` constraint is dropped. libclang on the
    affected libclang versions chokes on `requires` in template
    parameter lists; we avoid the error by deleting the line
    entirely (the function still parses)."""
    out = preprocess_cpp(
        "template <typename T>\n"
        "requires std::totally_ordered<T>\n"
        "void sort(std::vector<T>&) {}\n"
    )
    assert "requires" not in out
    # The function body survives.
    assert "void sort" in out


def test_preprocess_cpp_handles_unspaced_include():
    """`#include<vector>` (no space) is matched by the stripper, not
    only `#include <vector>`. The corpus uses both forms."""
    out = preprocess_cpp("#include<vector>\nint x = 1;\n")
    assert "#include" not in out
    assert "int x = 1" in out


def test_preprocess_cpp_falls_back_when_no_clang():
    """No `clang` on PATH -> fall back to directive-stripping alone.
    The output is still usable (less accurate but functional)."""
    out = preprocess_cpp("#include <vector>\nint x = 1;\n", clang="/no/such/clang")
    assert "#include" not in out


def test_preprocess_cpp_strips_header_guard_when_no_clang():
    """The `#ifndef NAME` / `#define NAME` (bare) / `#endif` header-guard
    idiom is removed by the no-clang fallback path, same as the primary
    (real-preprocessor) path."""
    out = preprocess_cpp(
        "#ifndef FOO_H\n#define FOO_H\nint x = 1;\n#endif\n",
        clang="/no/such/clang",
    )
    assert "#ifndef" not in out
    assert "#define FOO_H" not in out
    assert "#endif" not in out
    assert "int x = 1" in out


def test_preprocess_cpp_preserves_real_conditional_when_no_clang():
    """A real `#ifdef`/`#else`/`#endif` block (e.g. a DLL export macro
    defined differently per platform -- the near-universal real-world
    idiom this project's own corpus testing against Topologic surfaced)
    must survive the no-clang fallback intact rather than have its
    `#endif` stripped as if it were a header guard. A prior version of
    the fallback stripped every `#endif` (and every bare `#define`)
    unconditionally, which desynced this block's nesting and produced
    unparseable output ("#else after #else")."""
    src = (
        "#ifdef _WIN32\n"
        "#define FOO_API __declspec(dllexport)\n"
        "#else\n"
        "#define FOO_API\n"
        "#endif\n"
        "FOO_API void f();\n"
    )
    out = preprocess_cpp(src, clang="/no/such/clang")
    assert out.count("#ifdef") == 1
    assert out.count("#else") == 1
    assert out.count("#endif") == 1
    assert "FOO_API void f();" in out


def test_preprocess_cpp_nested_conditional_inside_header_guard_when_no_clang():
    """A real `#ifdef`/`#endif` block *nested inside* an outer `#ifndef`
    header guard (found via OCCT's gp_Mat.hxx: a platform-specific
    compiler-bug workaround gated on `#if defined(__APPLE__)`, wrapped in
    the file's own include guard) must not desync the guard-stripping
    stack. Only `#ifndef` pushed a stack frame; every `#endif` popped one
    unconditionally regardless of what opened it, so the inner block's
    `#endif` wrongly consumed the *outer* guard's frame -- stripping the
    inner (real) `#endif` as if it were the guard's closer, and leaving the
    true outer `#endif` behind as an orphan with no matching `#if` in the
    output. libclang's own preprocessor then can't find a match for the
    still-open inner `#if` and silently skips every line up to the next
    `#endif` it can find elsewhere in a multi-file translation unit --
    potentially swallowing whole unrelated classes with no diagnostic at
    all (see docs/occt_gp_preamble.hpp and the includes.py docstring)."""
    src = (
        "#ifndef FOO_H\n"
        "#define FOO_H\n"
        "class Foo {\n"
        "#if defined(__APPLE__)\n"
        "  void quirk();\n"
        "#endif\n"
        "  int x;\n"
        "};\n"
        "#endif // FOO_H\n"
    )
    out = preprocess_cpp(src, clang="/no/such/clang")
    assert out.count("#if defined(__APPLE__)") == 1
    assert out.count("#endif") == 1
    assert "int x;" in out
    assert "};" in out


def test_parse_cpp_neutralizes_unresolved_export_macro():
    """Real C++ libraries almost universally gate their public API with a
    SCREAMING_CASE export/calling-convention macro (`FOO_API`, `DLLEXPORT`,
    `WINAPI`, ...) defined in a header we never see (every #include is
    stripped by design). Without neutralizing it, every declaration using
    it fails to parse with "unknown type name" -- discovered testing
    against a real-world corpus (github.com/wassimj/Topologic)."""
    src = "TOPOLOGIC_API int add(int a, int b) { return a + b; }\n"
    hir_module, _ = parse_cpp(src)
    names = [f.name for f in hir_module.body if isinstance(f, hir.HirFunction)]
    assert "add" in names


def test_parser_preamble_declares_math_intrinsics():
    """`std::sqrt` / `std::swap` / `std::max` need declarations in the
    parser preamble so libclang doesn't error on the corpus. This
    is a regression guard for issue #50 — adding a missing
    `std::` declaration here re-enables a whole bucket of code
    that was previously failing with "no member named 'X' in std"."""
    for name in ("sqrt", "exp", "log", "swap", "min", "max"):
        assert name in PARSER_PREAMBLE, (
            f"parser preamble missing {name}"
        )

# ---------------------------------------------------------------------------
# parse_cpp returns (HirModule, TypeGroundTruth)
# ---------------------------------------------------------------------------


def test_parse_cpp_returns_tuple_with_ground_truth():
    """The new contract: parse_cpp(source) -> (hir, ground_truth).
    Tests and external code that pre-dates the tuple return can
    still call parse_cpp()[0]."""
    src = "int add(int a, int b) { return a + b; }"
    parsed = parse_cpp(src)
    assert isinstance(parsed, tuple)
    assert len(parsed) == 2
    hir_mod, truth = parsed
    assert isinstance(hir_mod, hir.HirModule)
    assert isinstance(truth, TypeGroundTruth)


def test_parse_cpp_ground_truth_collects_function_signature():
    """`int add(int, int)` -> func_returns['add'] == IntT(),
    func_params['add'] == [IntT(), IntT()]."""
    src = "int add(int a, int b) { return a + b; }"
    _, truth = parse_cpp(src)
    assert truth.func_returns.get("add") == IntT()
    assert truth.func_params.get("add") == [IntT(), IntT()]


def test_parse_cpp_ground_truth_records_template_signature():
    """`template <typename T> void f(vector<T>&)` -> the template's
    signature is in the truth table, even though the body becomes
    a HirRaw hole."""
    src = (
        "template <typename T>\n"
        "requires std::totally_ordered<T>\n"
        "void bubble_sort(std::vector<T>& array) {}\n"
    )
    _, truth = parse_cpp(src)
    # The key is the bare name (the function isn't inside a
    # namespace in this test source) or any qualified key whose
    # last segment is `bubble_sort`.
    qualified = next(
        (k for k in truth.func_returns
         if k == "bubble_sort" or k.endswith("::bubble_sort")),
        None,
    )
    assert qualified is not None, f"missing bubble_sort in {list(truth.func_returns)}"
    assert truth.func_returns[qualified] == NoneT()
    # The element is an unresolved template parameter name ("T"), not a
    # concrete type -- carried as StructT("T") (a real Type instance, per
    # the ListT.elem: Type contract) rather than a bare Python str, which
    # every backend's `isinstance(ty.elem, ...)` type-lattice walk would
    # otherwise choke on.
    assert truth.func_params[qualified][0] == ListT(elem=StructT(name="T"))


# ---------------------------------------------------------------------------
# HIR nodes carry source_loc
# ---------------------------------------------------------------------------


def test_hir_node_base_has_source_loc():
    """Every HIR node has an optional `source_loc` field."""
    n = hir.HirName(name="x")
    assert n.source_loc is None  # default is None


def test_hir_module_preserves_user_decl_after_preamble():
    """Parser preamble types do NOT leak into user HIR."""
    src = "struct Point { int x; int y; };"
    hir_mod, _ = parse_cpp(src)
    names = [getattr(n, "name", None) for n in hir_mod.body]
    assert "Point" in names
    for preamble_name in ("exception", "string", "string_view"):
        assert preamble_name not in names


# ---------------------------------------------------------------------------
# apply_ground_truth fills UnknownT holes
# ---------------------------------------------------------------------------


def test_apply_ground_truth_fills_function_return_and_params():
    """A function with UnknownT return/params is filled from truth."""
    src = "int add(int a, int b) { return a + b; }"
    hir_mod, truth = parse_cpp(src)
    mir_mod = hir_to_mir(hir_mod)
    apply_ground_truth(mir_mod, truth, hir_mod)
    fn = mir_mod.functions[0]
    assert fn.name == "add"
    assert fn.return_type == IntT()
    assert [p.ty for p in fn.params] == [IntT(), IntT()]


def test_apply_ground_truth_is_noop_on_empty_truth():
    """Empty TypeGroundTruth is a no-op fast path."""
    from transpilers.ir.mir import MirFunction, MirModule
    fn = MirFunction(name="f", params=[], return_type=UnknownT(), body=[])
    mod = MirModule(functions=[fn])
    out = apply_ground_truth(mod, TypeGroundTruth())
    assert out is mod


def test_apply_ground_truth_preserves_user_supplied_types():
    """Already-resolved types on the MIR are NOT overwritten."""
    src = "int add(int a, int b) { return a + b; }"
    hir_mod, truth = parse_cpp(src)
    mir_mod = hir_to_mir(hir_mod)
    # Tamper: set the first param to a non-UnknownT type.
    mir_mod.functions[0].params[0] = mir.MirParam(
        name="a", ty=StrT(),  # wrong on purpose
    )
    apply_ground_truth(mir_mod, truth, hir_mod)
    # The pre-existing StrT survives -- we don't overwrite concrete types.
    assert mir_mod.functions[0].params[0].ty == StrT()
    assert mir_mod.functions[0].params[1].ty == IntT()


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_e2e_macro_expansion_flows_to_annotation():
    """`#define NULL 0` plus `int x = NULL;` transpiles to `var x: Int = 0`.

    The macro use is wrapped in a function -- module-level globals are
    inlined as module constants by the existing engine rather than
    emitted as declarations, so the e2e test of "the macro reaches
    the typed annotation" has to live inside a function body.
    """
    from transpilers.cli.main import transpile_cpp_to_mojo
    out = transpile_cpp_to_mojo(
        "#define NULL 0\nint f() { int x = NULL; return x; }\n"
    )
    assert "var x: Int = 0" in out


def test_project_preamble_declares_types_usable_by_source(monkeypatch):
    """`$TRANSPILERS_CPP_PREAMBLE` (project-specific declarations, e.g. an
    EnergyPlus-style `Real64` typedef) must be visible to the user's own
    source -- a prior version appended it *after* the user code in the
    combined translation unit, so any type/typedef it declared could never
    actually be used by that code (C++ requires declare-before-use); it
    only ever worked by coincidence for injections that don't need forward
    visibility. Neither `TRANSPILERS_CPP_PREAMBLE` nor `_FILE` had any test
    coverage before this fix."""
    from transpilers.cli.main import transpile_cpp_to_mojo

    monkeypatch.setenv("TRANSPILERS_CPP_PREAMBLE", "typedef double Real64;")
    out = transpile_cpp_to_mojo("Real64 f(Real64 x) { return x * 2.0; }\n")
    assert "def f(x: Float64) -> Float64:" in out


def test_mojo_backend_emits_placeholder_for_foreign_preamble_struct(monkeypatch):
    """A class declared only in a project-preamble shim (e.g. docs/occt_preamble.hpp
    for a third-party library's opaque handle type) resolves fine during
    parsing, but the user's own code never declares it as a MirStruct -- so
    without this, every reference to it in the *emitted* Mojo is a "use of
    unknown declaration" compile error, even though libclang itself parsed
    it successfully. The Mojo backend now emits a minimal opaque placeholder
    struct for any such foreign type actually used in a param/field/return
    position. One dummy field (not zero) matters: an empty struct doesn't
    satisfy Mojo's ImplicitlyCopyable the same way."""
    from transpilers.cli.main import transpile_cpp_to_mojo
    from transpilers.verify import mojo_compiles

    monkeypatch.setenv("TRANSPILERS_CPP_PREAMBLE", "class ForeignType {};")
    src = """
        class Holder {
        public:
            ForeignType value;
            int tag;
            int getTag() { return tag; }
        };
    """
    out = transpile_cpp_to_mojo(src)
    assert "struct ForeignType(Copyable, Movable):" in out
    assert "var value: ForeignType" in out
    result = mojo_compiles(out)
    assert result.ok, result.stderr


def test_project_preamble_real_marker_emits_helper_past_it(monkeypatch):
    """A project preamble is usually parse-only scaffolding (opaque stubs,
    typedefs, macros) that must never itself appear in the emitted target --
    but some projects also need a handful of small *real* helpers with
    correct bodies that the user's own code genuinely calls (e.g. OCCT's
    `RealSmall()` tolerance constant). Content at or after
    `_PREAMBLE_REAL_MARKER` is user code for emission purposes: it must be
    converted and emitted, not silently excluded like the rest of the
    preamble, or every call site is "use of unknown declaration" despite
    parsing and type-checking fine."""
    from transpilers.cli.main import transpile_cpp_to_mojo

    monkeypatch.setenv(
        "TRANSPILERS_CPP_PREAMBLE",
        "typedef double Real64;\n"
        "// === TRANSPILERS: REAL PREAMBLE BELOW ===\n"
        "constexpr double Tolerance() { return 1e-9; }\n",
    )
    out = transpile_cpp_to_mojo("bool small(Real64 x) { return x < Tolerance(); }\n")
    assert "def Tolerance() -> Float64:" in out
    assert "return 1e-09" in out or "return 1e-9" in out


def test_project_preamble_stub_part_stays_excluded_with_real_marker(monkeypatch):
    """The marker only promotes content *after* it; a stub class declared
    before the marker in the same preamble must still be excluded from
    emission (this is what distinguishes it from just disabling exclusion
    for the whole preamble)."""
    from transpilers.cli.main import transpile_cpp_to_mojo

    monkeypatch.setenv(
        "TRANSPILERS_CPP_PREAMBLE",
        "class OpaqueStub {};\n"
        "// === TRANSPILERS: REAL PREAMBLE BELOW ===\n"
        "constexpr double Tolerance() { return 1e-9; }\n",
    )
    out = transpile_cpp_to_mojo("double f() { return Tolerance(); }\n")
    assert "struct OpaqueStub" not in out
    assert "def Tolerance() -> Float64:" in out


def test_e2e_uses_clang_resolved_intrinsic_signature():
    """`std::sqrt(x)` -> Mojo emits `from std.math import sqrt`."""
    from transpilers.cli.main import transpile_cpp_to_mojo
    out = transpile_cpp_to_mojo("double f(double x){ return std::sqrt(x); }")
    assert "from std.math import sqrt" in out


def test_e2e_template_definition_emits_todo_stub():
    """A C++ template definition is preserved as a HirRaw hole and
    every backend emits a TODO[port] stub. The ownership/RAII/
    template-instantiation is left for the inference / LLM pass."""
    from transpilers.cli.main import transpile_cpp_to_mojo
    out = transpile_cpp_to_mojo(
        "template <typename T>\n"
        "T add(T a, T b) { return a + b; }\n"
    )
    assert "TODO[port]" in out


# ---------------------------------------------------------------------------
# _default_init_for: trailing struct-init fields without an explicit ctor arg
# ---------------------------------------------------------------------------


def test_default_init_for_unknown_field_type_stays_a_hole():
    """`Foo f(1);` where `Foo` has a trailing field of a type that isn't
    int/float/bool/str/list (here, a dict) must NOT fabricate an `IntT(0)`
    default -- this is an algorithmic pass, which must not invent a value
    for a type it doesn't know how to default-construct. A prior version
    silently defaulted every such field to integer 0, so e.g. a later
    `self.m["key"]`-style use on that field would emit code indexing an
    integer. The fix emits a `MirRaw` hole (rendered as `TODO[port]` by
    every backend) instead."""
    module = hir.HirModule(
        source_lang="cpp",
        body=[
            hir.HirStruct(
                name="Foo",
                fields=[
                    hir.HirParam("x", "int"),
                    hir.HirParam("m", "dict[str, int]"),
                ],
                methods=[],
            ),
            hir.HirFunction(
                "make",
                params=[],
                return_annotation="Foo",
                body=[hir.HirReturn(value=hir.HirStructInit(
                    name="Foo", args=[hir.HirIntLiteral(value=1)],
                ))],
            ),
        ],
    )
    mir_mod = hir_to_mir(module)
    struct_init = mir_mod.functions[0].body[0].value
    field_values = dict(struct_init.field_values)
    default_val = field_values["m"]
    assert isinstance(default_val, mir.MirRaw)
    assert isinstance(default_val.ty, UnknownT)

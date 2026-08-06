"""Never-refuse C++ -> Python lifter."""

from transpilers.lift import lift_source
from transpilers.levels import transpile_level


def test_lift_struct_to_class():
    out, _ = lift_source("struct P { double x = 1.5; int n = 0; bool ok = false; };")
    assert "class P:" in out
    assert "self.x = 1.5" in out and "self.n = 0" in out and "self.ok = False" in out


def test_lift_for_loop_to_while():
    out, _ = lift_source(
        "double dot(const double* a, int n){ double s=0.0; for(int i=0;i<n;i=i+1){ s=s+a[i]; } return s; }"
    )
    assert "def dot(" in out
    assert "i = 0" in out
    assert "while i < n:" in out
    assert "i = i + 1" in out or "i += 1" in out
    assert "s = s + a[i]" in out


def test_lift_member_chain_recovered_from_tokens():
    # `state.dataCoolTower->GetInputFlag` must lift even though the type isn't
    # declared here (AST name-degraded) — recovered from the token stream.
    out, _ = lift_source(
        "void f(EnergyPlusData &state){ if (state.dataCoolTower->GetInputFlag) { return; } }"
    )
    assert "state.data_cool_tower.get_input_flag" in out


def test_lift_recovers_degraded_method_calls():
    # Method calls through a name-degraded chain (no types) must recover the
    # full member name from tokens, not collapse to `state.(...)`.
    out, _ = lift_source(
        "void f(EnergyPlusData &state, int n){ "
        "  int k = state.dataCoolTower->CoolTowerSys.size(); "
        "  state.dataInput->obj->doThing(n); }"
    )
    assert "state.data_cool_tower.cool_tower_sys.size()" in out
    assert "state.data_input.obj.do_thing(n)" in out


def test_lift_array_member_after_call_recovered():
    # gap 1: ObjexxFCL array-member access `Arr(i).Field` (member AFTER an
    # operator() call) must recover the full chain from tokens, type-less.
    # The 1-based subscript is kept faithfully as a call `arr(i)` (Phase-1).
    out, _ = lift_source(
        "void f(State &state, int i){ "
        "  double t = state.dataZ->zoneHeatBalance(i).MAT; "
        "  if (Zone(i).OutDryBulbTemp > 5.0) return; }"
    )
    assert "t = state.data_z.zone_heat_balance(i).mat" in out
    assert "if zone(i).out_dry_bulb_temp > 5.0:" in out


def test_lift_nested_array_member_chain_recovered():
    # gap 1: chained array-member-after-call `Storage(i).Avail(j)`.
    out, _ = lift_source(
        "void f(State &state, int i, int j){ "
        "  double a = state.dataW->WaterStorage(i).VdotAvailDemand(j); }"
    )
    assert "a = state.data_w.water_storage(i).vdot_avail_demand(j)" in out


def test_lift_namespaced_call_in_condition_recovered():
    # gap 2: UNEXPOSED_EXPR wrapping namespaced calls in conditions.
    # std::abs -> abs (Python builtin); Ns::Func -> snaked func, namespace
    # qualifier dropped. Type-less so the whole comparison degrades to a
    # token stream.
    out, _ = lift_source(
        "double f(double x, const char* a, const char* b){ "
        "  if (std::abs(x) > 1.0) return 1.0; "
        "  while (Util::SameString(a, b)) return 2.0; "
        "  return 0.0; }"
    )
    assert "if abs(x) > 1.0:" in out
    assert "while same_string(a, b):" in out


def test_lift_never_refuses_unknown():
    # A construct the lifter can't handle yet must still produce output + TODO.
    out, _ = lift_source("void g(){ try { foo(); } catch (...) { } }")
    assert "def g(" in out
    assert "TODO[lift]" in out  # didn't crash; stubbed instead


def test_lift_init_list_to_python_list():
    out, _ = lift_source("std::vector<int> f(){ return {1, 2, 3}; }")
    assert "[1, 2, 3]" in out


def test_lift_switch_to_if_elif_else():
    out, _ = lift_source(
        "int f(int k){ int r; switch (k) { "
        "  case 1: r = 10; break; "
        "  case 2: case 3: r = 20; break; "
        "  default: r = 0; } return r; }"
    )
    assert "if k == 1:" in out
    assert "elif k == 2 or k == 3:" in out
    assert "else:" in out
    assert "r = 10" in out and "r = 20" in out and "r = 0" in out
    assert "break" not in out  # switch breaks are dropped, not leaked


def test_lift_if_with_initializer_keeps_condition():
    # C++17 `if (init; cond)` must not read the init as the condition
    # (regression: previously emitted `if None:`).
    out, _ = lift_source(
        "void f(State &state, int n){ "
        "  if (auto x = state.dataX->find(n); x > 0) { return; } }"
    )
    assert "if None:" not in out
    assert "x = state.data_x.find(n)" in out
    assert "if x > 0:" in out


def test_lift_single_return_lambda():
    out, _ = lift_source("void f(){ auto sq = [](double x){ return x * x; }; }")
    assert "lambda x: x * x" in out


def test_lift_do_while_to_while_true_break():
    out, _ = lift_source("void f(int n){ do { n = n - 1; } while (n > 0); }")
    assert "while True:" in out
    assert "n = n - 1" in out
    assert "if not (n > 0):" in out
    assert "break" in out


def _compiles(out):
    import ast

    ast.parse(out)
    return True


def test_lift_keyword_identifier_escaped():
    # C++ identifiers that are Python keywords (`in`, `is`, `class`) must be
    # suffixed `_` so the emitted code is valid Python.
    out, _ = lift_source("void f(){ int in = 0; int is = 1; }")
    assert _compiles(out)
    assert "in_ = 0" in out and "is_ = 1" in out


def test_lift_dynamic_cast_stripped():
    out, _ = lift_source(
        "struct B{}; struct D:B{}; void f(B* b){ auto* d = dynamic_cast<D*>(b); }"
    )
    assert _compiles(out)
    assert "dynamic_cast" not in out and "<" not in out.split("def f")[-1]


def test_lift_assignment_in_condition_hoisted():
    # `if ((x = f()) == 0)` -> hoist `x = f()`, then `if x == 0:`.
    out, _ = lift_source("int g(); void f(){ int x; if ((x = g()) == 0) { return; } }")
    assert _compiles(out)
    assert "x = g()" in out
    assert "if x == 0:" in out


def test_lift_cstyle_cast_in_index_stripped():
    out, _ = lift_source("void f(double* a, double t){ a[(int)t] = 1.0; }")
    assert _compiles(out)
    assert "(int)" not in out
    assert "a[t]" in out


def test_lift_objexx_call_lvalue_to_subscript():
    # ObjexxFCL element assignment `arr(i) = v` (1-based `()` indexing on a
    # name-degraded member) is invalid as a Python call target -> rewrite to a
    # subscript so it's valid Python.
    out, _ = lift_source(
        "void f(State &state, int i){ state.dataX->Arr(i) = 2.0; "
        "  state.dataX->Cnt(i) += 1; }"
    )
    assert _compiles(out)
    assert "state.data_x.arr[i] = 2.0" in out
    assert "state.data_x.cnt[i] += 1" in out


def test_lift_degraded_macro_does_not_emit_unbalanced_garbage():
    # ObjexxFCL-style bounds macros degrade to a BINARY_OPERATOR with a mangled
    # operator; the lifter must emit valid Python (a TODO), never `x ) 0`.
    out, _ = lift_source(
        "#define EP_SIZE_CHECK(a, n) ((void)0)\n"
        "void f(int* poly, int nsides){ EP_SIZE_CHECK(poly, nsides); }"
    )
    assert _compiles(out)
    assert ") 0" not in out and ") None" not in out


def test_lift_objexx_2d_assign_target_recovered():
    # 2D ObjexxFCL element assignment `a(i, j) = v`: structured emit can lose
    # the array name -> must recover a valid target from tokens, never `i = v`.
    out, _ = lift_source(
        "void f(State &state, int k){ state.dataX->a(k - 2, k) = 1.0; }"
    )
    assert _compiles(out)
    assert "= 1.0" in out and "[k - 2, k]" in out


def test_lift_post_increment_on_array_element():
    out, _ = lift_source("void f(State &state, int i){ state.dataX->counts(i)++; }")
    assert _compiles(out)
    assert "state.data_x.counts[i] += 1" in out


def test_lift_operator_method_to_dunder():
    out, _ = lift_source(
        "struct P{ bool operator==(P const& o) const { return true; } "
        "double operator()(int k){ return 0.0; } };"
    )
    assert _compiles(out)
    assert "def __eq__(" in out and "def __call__(" in out
    assert "def operator" not in out


def test_lift_hex_literal_not_corrupted():
    # rstrip of int/float suffixes must not eat hex digits f/F.
    out, _ = lift_source("int f(){ return 0xFF + 0xA0F; }")
    assert _compiles(out)
    assert "0xFF" in out and "0xA0F" in out


def test_lift_prefix_deref_dropped():
    out, _ = lift_source("void f(std::vector<int>* xs){ (*xs).push_back(1); }")
    assert _compiles(out)
    assert "(* " not in out and "(*xs)" not in out


def test_lift_parenthesized_subexpr_keeps_grouping():
    # #58: `(Tamb - Tsurf) * cosTilt` lost its parens (PAREN_EXPR unwrapping)
    # and silently rebound as `tamb - (tsurf * cos_tilt)`.
    out, _ = lift_source(
        "double f(double Tamb, double Tsurf, double cosTilt){ "
        "  double DeltaTempCosTilt = (Tamb - Tsurf) * cosTilt; "
        "  return DeltaTempCosTilt / (Tamb + Tsurf); }"
    )
    assert _compiles(out)
    assert "(tamb - tsurf) * cos_tilt" in out
    assert "delta_temp_cos_tilt / (tamb + tsurf)" in out


def test_lift_precedence_right_operand_keeps_grouping():
    # Right operands at equal precedence only exist via explicit source parens
    # (`a - (b - c)`, `a / (b * c)`) — they must keep them.
    out, _ = lift_source(
        "double f(double a, double b, double c){ return a - (b - c) + a / (b * c); }"
    )
    assert _compiles(out)
    assert "a - (b - c)" in out
    assert "a / (b * c)" in out


def test_lift_unary_minus_on_parenthesized_sum():
    # `-(a + b) * c` must not emit `(-a + b) * c`.
    out, _ = lift_source(
        "double f(double a, double b, double c){ return -(a + b) * c; }"
    )
    assert _compiles(out)
    assert "(-(a + b)) * c" in out


def test_lift_logical_grouping_kept():
    # `(a || b) && c` must not emit `a or b and c` (binds as a or (b and c)).
    out, _ = lift_source("bool f(bool a, bool b, bool c){ return (a || b) && c; }")
    assert _compiles(out)
    assert "(a or b) and c" in out


def test_lift_condition_logical_grouping_kept():
    # Same grouping bug through the if-condition (_hoist_cond rewrite) path.
    out, _ = lift_source(
        "void f(bool a, bool b, bool c){ if ((a || b) && c) { return; } }"
    )
    assert _compiles(out)
    assert "if (a or b) and c:" in out


def test_lift_comparison_child_of_comparison_wrapped():
    # C++ nests comparisons left-to-right; Python CHAINS them. `a < b == c`
    # (C++: `(a < b) == c`) must emit the parens explicitly.
    out, _ = lift_source("bool f(double a, double b, bool c){ return a < b == c; }")
    assert _compiles(out)
    assert "(a < b) == c" in out


def test_levels_lift_engine(tmp_path):
    f = tmp_path / "k.cpp"
    f.write_text("double sq(double x){ return x*x; }")
    results = transpile_level("file", str(f), engine="lift")
    assert len(results) == 1 and results[0].ok
    assert "def sq(" in results[0].output

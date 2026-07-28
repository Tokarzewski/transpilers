"""C++ frontend tests. Validates the third source language flows through the
shared MIR pipeline. Initial C++ subset is deliberately C-like (no classes,
templates, references, namespaces) — those are real C++ features the IR
doesn't model yet."""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import (
    transpile_cpp_to_c,
    transpile_cpp_to_mojo,
    transpile_cpp_to_rust,
    transpile_cpp_to_zig,
)
from transpilers.verify import c_compiles, mojo_compiles, rust_compiles, zig_compiles


def _rust(src: str) -> str:
    return transpile_cpp_to_rust(textwrap.dedent(src).lstrip())


def _zig(src: str) -> str:
    return transpile_cpp_to_zig(textwrap.dedent(src).lstrip())


def _c(src: str) -> str:
    return transpile_cpp_to_c(textwrap.dedent(src).lstrip())


def _mojo(src: str) -> str:
    return transpile_cpp_to_mojo(textwrap.dedent(src).lstrip())


def _has(name: str) -> bool:
    return shutil.which(name) is not None


# ---------- shape ----------

def test_cpp_add_to_rust():
    out = _rust("int add(int a, int b) { return a + b; }")
    assert "fn add(a: i64, b: i64) -> i64" in out


def test_cpp_to_mojo_shape():
    out = _mojo("int add(int a, int b) { return a + b; }")
    assert "def add(a: Int, b: Int) -> Int:" in out
    assert "return a + b" in out


def test_cpp_enum_inlines_to_int_constants():
    # `enum {...}` was previously refused (top-level ENUM_DECL). Each
    # enumerator now becomes a module-level int constant that inlines at use
    # sites, so the construct flows through to every backend.
    src = """
        enum SurfaceClass { Wall = 0, Floor = 1, Roof = 2 };

        int classify(int x) {
            if (x == Wall) return 100;
            if (x == Roof) return 300;
            return 0;
        }
    """
    out = _mojo(src)
    assert "def classify(x: Int) -> Int:" in out
    assert "if x == 0:" in out      # Wall -> 0, inlined
    assert "if x == 2:" in out      # Roof -> 2, inlined
    # same C++ enum flows to other targets from one HIR
    assert "fn classify(x: i64) -> i64" in _rust(src)


def test_cpp_namespace_flattens_to_module_scope():
    # `namespace foo { ... }` members used to be skipped wholesale; they now
    # flatten to module scope and emit.
    out = _mojo(
        """
        namespace hb {
        int classify(int x) { return x + 1; }
        }
        """
    )
    assert "def classify(x: Int) -> Int:" in out


def test_cpp_anonymous_namespace_struct_return_type_resolves():
    # A struct declared inside an anonymous namespace (a common
    # translation-unit-local-helper idiom in real .cxx files) keeps its
    # namespace-qualified spelling from libclang -- literally
    # `(anonymous namespace)::Name` -- when used as a type annotation
    # (e.g. a function's return type). _convert_top_level flattens every
    # namespace, including anonymous ones, registering the struct under
    # its bare name -- so a still-qualified annotation on the return type
    # could never match the registry, leaving it an unresolved type hole
    # (ValueError: unresolved type hole) even though the struct itself
    # converted fine.
    out = _mojo(
        """
        namespace {
        struct Helper {
            int tag;
        };
        Helper makeHelper(int t) {
            Helper h;
            h.tag = t;
            return h;
        }
        }
        int useIt(int t) {
            Helper h = makeHelper(t);
            return h.tag;
        }
        """
    )
    assert "struct Helper" in out
    assert "def makeHelper(t: Int) -> Helper:" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_anonymous_namespace_struct_return_type_compiles():
    out = _mojo(
        """
        namespace {
        struct Helper {
            int tag;
        };
        Helper makeHelper(int t) {
            Helper h;
            h.tag = t;
            return h;
        }
        }
        int useIt(int t) {
            Helper h = makeHelper(t);
            return h.tag;
        }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_typedef_and_using_resolve():
    out = _mojo(
        """
        typedef double Real64;
        using Int32 = int;
        Real64 add(Real64 a, Real64 b) { return a + b; }
        Int32 doubled(Int32 n) { return n * 2; }
        """
    )
    assert "def add(a: Float64, b: Float64) -> Float64:" in out
    assert "def doubled(n: Int) -> Int:" in out


def test_cpp_constructor_to_init():
    out = _mojo(
        """
        struct Vec2 {
            double x;
            double y;
            Vec2(double a, double b) : x(a), y(b) {}
            double norm2() const { return x*x + y*y; }
        };
        """
    )
    assert "def __init__(out self, a: Float64, b: Float64):" in out
    assert "self.x = a" in out
    assert "self.y = b" in out
    # explicit constructor replaces the synthesized fieldwise one
    assert "@fieldwise_init" not in out


def test_cpp_bool_handled():
    out = _rust("bool is_positive(int x) { return x > 0; }")
    assert "fn is_positive(x: i64) -> bool" in out


def test_cpp_for_loop_desugars():
    out = _rust(
        """
        int sum_to(int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total = total + i;
            }
            return total;
        }
        """
    )
    assert "while i < n {" in out
    assert "i += 1;" in out


def test_cpp_std_list_range_for_to_mojo():
    # std::list<T> is a plain opaque shim in the parser preamble (unlike
    # std::vector<T>, which has a full iterator surface) -- so a real-world
    # function range-iterating a std::list (found testing against
    # github.com/wassimj/Topologic) previously failed to even parse with
    # "invalid range expression ... no viable 'begin' function available".
    out = _mojo(
        """
        int sum(const std::list<int>& xs) {
            int total = 0;
            for (const int x : xs) { total = total + x; }
            return total;
        }
        """
    )
    assert "def sum(xs: List[Int]) -> Int:" in out


def test_cpp_std_string_from_char_literal():
    # `std::string("literal")` (a functional-style cast, the common way to
    # construct a std::string from a string literal) previously failed to
    # parse: the preamble's std::string shim declared only a default
    # constructor, so libclang rejected the conversion.
    out = _mojo('std::string greet() { return std::string("hi"); }')
    assert 'return "hi"' in out


def test_cpp_logical_ops_to_mojo():
    out = _mojo("bool both(bool a, bool b) { return a && b; }")
    assert "return a and b" in out


def test_cpp_long_collapses_to_int():
    out = _rust("long sum(long a, long b) { return a + b; }")
    assert "fn sum(a: i64, b: i64) -> i64" in out


def test_cpp_operator_overloads_become_dunders():
    # Member `operator+` / `operator==` / `operator<` rename to the matching
    # Python/Mojo dunder; the backends already emit these as ordinary methods.
    src = """
        struct Vec {
            int x;
            int y;
            Vec(int a, int b) : x(a), y(b) {}
            Vec operator+(Vec o) const { return Vec(x + o.x, y + o.y); }
            bool operator==(Vec o) const { return x == o.x && y == o.y; }
            bool operator<(Vec o) const { return x < o.x; }
        };
    """
    out = _mojo(src)
    assert "def __add__(self, o: Vec) -> Vec:" in out
    assert "def __eq__(self, o: Vec) -> Bool:" in out
    assert "def __lt__(self, o: Vec) -> Bool:" in out
    # same rename flows to other targets from one HIR
    rust = _rust(src)
    assert "fn __add__(&self, o: Vec) -> Vec" in rust


def test_cpp_call_operator_becomes_dunder_call():
    # `operator()` (functor call operator, e.g. a std::sort comparator) was
    # missing from the operator->dunder table, so it fell through to the raw
    # spelling `operator()` -- not a valid Mojo/Python identifier, breaking
    # the emitted struct.
    src = """
        struct Comparator {
            bool operator()(int a, int b) const { return a < b; }
        };
    """
    out = _mojo(src)
    assert "def __call__(self, a: Int, b: Int) -> Bool:" in out
    assert "operator()" not in out


def test_cpp_auto_var_type_inferred():
    # `auto x = expr;` carries no forced annotation, so the IR's own type
    # inference recovers x's type from the RHS: int from int+int, float from
    # the float mix.
    out = _mojo(
        """
        int compute(int a, int b) {
            auto s = a + b;
            auto f = a + 1.5;
            return s;
        }
        """
    )
    assert "var s: Int = a + b" in out
    assert "var f: Float64 = a + 1.5" in out


# ---------- C++ → Mojo compile checks ----------

@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
@pytest.mark.parametrize(
    "src",
    [
        "int add(int a, int b) { return a + b; }",
        """
        int max2(int a, int b) {
            if (a > b) return a;
            else return b;
        }
        """,
        """
        int factorial(int n) {
            int result = 1;
            int i = 1;
            while (i <= n) {
                result = result * i;
                i = i + 1;
            }
            return result;
        }
        """,
        """
        int sum_to(int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total = total + i;
            }
            return total;
        }
        """,
        """
        bool in_range(int x, int lo, int hi) {
            return x >= lo && x <= hi;
        }
        """,
    ],
)
def test_cpp_to_mojo_compiles(src: str):
    out = _mojo(src)
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- Real-world value-type idioms (found via OCCT's gp package) ----------

def test_cpp_forward_declaration_not_double_registered():
    # A forward decl (`class Vec;`) used only by-reference before the real
    # definition, common in headers with circular type relationships, used
    # to also get converted by _convert_top_level (no is_definition() guard)
    # -- producing a second, empty HirStruct alongside the real one. Every
    # backend then emitted the struct twice: a hard "invalid redefinition"
    # compile error, not just a fidelity issue.
    src = """
        class Vec;
        class Ax {
        public:
            Ax(const Vec& v) { }
        };
        class Vec {
        public:
            Vec() { }
        };
    """
    out = _mojo(src)
    assert out.count("struct Vec(") == 1


def test_cpp_copy_ctor_in_var_decl_not_padded():
    # `Vec aCopy = *this;` (copy-initialization, not a call expression) has
    # its own struct-init detection path (_convert_var_decl) separate from
    # _convert_call's -- fixing the copy/move-ctor collapse in one didn't
    # fix the other. Left unfixed, hir_to_mir's trailing-field defaulting
    # padded the "missing" second field with a fabricated 0, and even once
    # padded, the Mojo backend emitted the non-compiling `Vec(self)`.
    src = """
        class Vec {
        public:
            Vec() { }
            Vec(double x, double y) : myX(x), myY(y) { }
            void Scale(double s) { myX = myX * s; myY = myY * s; }
            Vec Scaled(double s) const {
                Vec aCopy = *this;
                aCopy.Scale(s);
                return aCopy;
            }
        private:
            double myX;
            double myY;
        };
    """
    out = _mojo(src)
    assert "Vec(self)" not in out
    assert "var aCopy: Vec = self.copy()" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_copy_ctor_in_var_decl_compiles():
    src = """
        class Vec {
        public:
            Vec() : myX(0), myY(0) { }
            Vec(double x, double y) : myX(x), myY(y) { }
            void Scale(double s) { myX = myX * s; myY = myY * s; }
            Vec Scaled(double s) const {
                Vec aCopy = *this;
                aCopy.Scale(s);
                return aCopy;
            }
        private:
            double myX;
            double myY;
        };
    """
    out = _mojo(src)
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_compound_assign_on_implicit_field():
    # `myX *= s;` inside a method (implicit `this->`, no explicit `self.`)
    # is a MEMBER_REF_EXPR lhs with a compound op -- the assignment-
    # statement converter only special-cased plain `=` for MEMBER_REF_EXPR
    # lhs, so `*=`/`+=`/etc on a bare field fell through to
    # "assignment target MEMBER_REF_EXPR", silently degrading to a dropped
    # `pass # TODO[port]` statement instead of the real mutation.
    out = _mojo(
        """
        class Mat {
        public:
            Mat(double a) : myA(a) { }
            void Scale(double s) { myA *= s; }
        private:
            double myA;
        };
        """
    )
    assert "self.myA = self.myA * s" in out
    assert "TODO[port]" not in out


def test_cpp_compound_assign_operator_overload_dunders():
    # operator+=/-=/*=//=  were missing from the operator-> dunder table
    # (only plain operator() had been added, for a different bug) -- every
    # target emitted the literal invalid identifier `operator+=` etc.
    out = _mojo(
        """
        class Vec {
        public:
            Vec(double x) : myX(x) { }
            Vec& operator+=(const Vec& o) { myX += o.myX; return *this; }
        private:
            double myX;
        };
        """
    )
    assert "def __iadd__" in out
    assert "def operator+=" not in out


def test_cpp_unary_minus_is_neg_not_sub():
    # A member `operator-()` with no explicit params (unary negation) was
    # mapped to the same `__sub__` dunder as binary `operator-(other)` --
    # producing a zero-argument `__sub__`, which every target's own
    # arity-checked dunder machinery rejects (Mojo: "'__sub__' requires 2
    # operands").
    out = _mojo(
        """
        class Vec {
        public:
            Vec(double x) : myX(x) { }
            Vec operator-() const { return Vec(-myX); }
        private:
            double myX;
        };
        """
    )
    assert "def __neg__" in out
    assert "def __sub__" not in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_unary_minus_compiles():
    out = _mojo(
        """
        class Vec {
        public:
            Vec(double x) : myX(x) { }
            Vec operator-() const { return Vec(-myX); }
        private:
            double myX;
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_qualified_static_call_not_treated_as_self_call():
    # An unqualified sibling-method call (`square(x)` inside `cube()`) is
    # deliberately rewritten to `self.square(x)` -- Mojo has no free-
    # function fallback for methods. But `OtherClass::staticMethod()`
    # (explicitly qualified) also resolves to a CXX_METHOD cursor, and
    # without checking for the `::` token, got the identical rewrite:
    # `self.staticMethod()` -- silently calling the *wrong* method (if one
    # of that name even exists on self) instead of the real cross-class one.
    out = _mojo(
        """
        class Tol {
        public:
            static double Small() { return 1e-9; }
        };
        class Foo {
        public:
            bool Check(double a) { return a <= Tol::Small(); }
        };
        """
    )
    assert "self.Small()" not in out


def test_cpp_defaulted_special_member_skipped():
    # `Vec& operator=(const Vec&) = default;` still reports
    # is_definition() == True in libclang (it's compiler-generated, but a
    # real definition) -- without checking is_default_method() too, the
    # engine tried to convert a method with *no body at all*, producing a
    # garbled/empty method instead of skipping it like any other
    # declared-but-not-defined member.
    out = _mojo(
        """
        class Vec {
        public:
            Vec() { }
            Vec(double x) : myX(x) { }
            Vec(const Vec&) = default;
            Vec& operator=(const Vec&) = default;
        private:
            double myX;
        };
        """
    )
    assert "operator=" not in out


def test_cpp_plain_operator_assign_on_implicit_field():
    # `vxdir = theV;` inside a method, where vxdir's type has a user (even
    # `= default`ed) operator=, desugars in the AST to a CALL_EXPR whose
    # first child is the MEMBER_REF_EXPR *being assigned* (`vxdir`), not a
    # normal `object.method(args)` receiver. Treating it as an ordinary
    # method call (the pre-existing MEMBER_REF_EXPR-callee branch) produced
    # nonsense: `self.vxdir(operator=, theV)` -- calling the field itself
    # with the operator-name decl-ref as a bogus first argument.
    out = _mojo(
        """
        class Dir {
        public:
            Dir() { }
            Dir(double x) : myX(x) { }
            Dir& operator=(const Dir&) = default;
        private:
            double myX;
        };
        class Ax {
        public:
            Ax() { }
            void SetVx(Dir theV) { vxdir = theV; }
        private:
            Dir vxdir;
        };
        """
    )
    assert "self.vxdir = theV.copy()" in out
    assert "operator=" not in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_plain_operator_assign_on_implicit_field_compiles():
    out = _mojo(
        """
        class Dir {
        public:
            Dir() : myX(0) { }
            Dir(double x) : myX(x) { }
            Dir& operator=(const Dir&) = default;
        private:
            double myX;
        };
        class Ax {
        public:
            Ax() : vxdir(0) { }
            void SetVx(Dir theV) { vxdir = theV; }
        private:
            Dir vxdir;
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_transitive_mut_self_via_user_method_call():
    # Mojo's `mut self` requirement was only detected for a small hardcoded
    # list of STL-container method names (append/pop/clear/...) calling
    # into a self-rooted receiver. A method that mutates self by calling
    # *another user-defined method* that itself mutates self (`Multiplied()`
    # calling `self.Multiply(x)`) needs `mut self` too, transitively -- this
    # is the standard mutate-in-place / return-a-copy pairing found
    # throughout real value types (OCCT's gp_* package, but not only it),
    # and was previously left as an immutable `self`, which the real Mojo
    # compiler rejects ("invalid use of mutating method on rvalue").
    out = _mojo(
        """
        class Mat {
        public:
            Mat(double a) : myA(a) { }
            void Multiply(double s) { myA *= s; }
            Mat Multiplied(double s) const {
                Mat aCopy = *this;
                aCopy.Multiply(s);
                return aCopy;
            }
        private:
            double myA;
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_mutating_method_call_on_reference_param_compiles():
    # A *parameter* (not `self`) receiving a call to a user-defined mutating
    # method needs `var`/`mut` decoration too, not just self -- previously
    # only a small hardcoded set of STL-container method names (append/...)
    # was recognized for this on params; a user-defined mutator like
    # `XYZ::Add` wasn't, leaving the param an immutable default borrow,
    # which the real Mojo compiler rejects ("invalid use of mutating method
    # on rvalue").
    out = _mojo(
        """
        class XYZ {
        public:
            XYZ(double x) : myX(x) { }
            void Add(double x) { myX += x; }
        private:
            double myX;
        };
        void Transforms(XYZ& theCoord) {
            theCoord.Add(1.0);
        }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_constructor_calling_self_mutating_setter_keeps_out_self():
    # Widening mut-param detection to user-defined methods (previous test)
    # regressed constructors: `self` is always in a method's own param list,
    # so a constructor body calling a self-mutating setter (`self.SetX(...)`)
    # started incorrectly landing `self` in the mut-param set too. Since
    # `lower_params` bakes "var "/"mut " into the parameter NAME STRING, that
    # broke `__init__`'s required `out self` special case ("__init__ method
    # must return Self type with 'out' argument").
    out = _mojo(
        """
        class Pnt {
        public:
            Pnt(double x) : myX(x) { SetX(x + 1.0); }
            void SetX(double x) { myX = x; }
        private:
            double myX;
        };
        """
    )
    assert "def __init__(out self, x: Float64):" in out
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_struct_call_argument_gets_copy_inserted():
    # A struct-typed CALL ARGUMENT (as opposed to a return/assign, already
    # covered) also needs `.copy()` once the receiving parameter is `var`
    # (owned): `aT.Transforms(self.coord)` passes a bare field access into a
    # by-value `var theCoord: XYZ` parameter, which Mojo's `ImplicitlyCopyable`
    # restriction rejects without an explicit `.copy()`.
    out = _mojo(
        """
        class XYZ {
        public:
            XYZ(double x) : myX(x) { }
            void Add(double x) { myX += x; }
        private:
            double myX;
        };
        class Trsf {
        public:
            void Transforms(XYZ theCoord) { theCoord.Add(1.0); }
        };
        class Pnt {
        public:
            Pnt(double x) : coord(x) { }
            void Apply(Trsf& aT) { aT.Transforms(coord); }
        private:
            XYZ coord;
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_struct_reassignment_gets_copy_inserted():
    # A struct-typed REASSIGNMENT (`x = y.field;` where `x` already exists)
    # needs the same `.copy()` insertion as a fresh declaration does -- the
    # reassign branch had no copy-insertion logic at all previously.
    out = _mojo(
        """
        class XYZ {
        public:
            XYZ(double x) : myX(x) { }
        private:
            double myX;
        };
        class Holder {
        public:
            XYZ loc;
        };
        void reassign(Holder& h, Holder& other) {
            XYZ tmp = h.loc;
            tmp = other.loc;
        }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_return_field_of_struct_type_compiles():
    # `return self.field;` where the field's own type is a struct hit the
    # same "cannot be implicitly copied" restriction a bare `return
    # localVar` does -- but the existing fix only checked `node.value.ty`,
    # which infer_types never populates for a MirFieldAccess (no
    # struct-field table available to it), so it silently missed every
    # "return one of my own struct-typed fields" call site.
    out = _mojo(
        """
        class Dir {
        public:
            Dir() : myX(0) { }
            Dir(double x) : myX(x) { }
        private:
            double myX;
        };
        class Ax {
        public:
            Ax() : vxdir(0) { }
            Dir Direction() const { return vxdir; }
        private:
            Dir vxdir;
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_field_of_method_call_result_gets_copy_inserted():
    # `var v: T = a.Method().field;` -- a struct-typed field access rooted
    # in a METHOD CALL's result (not a bare name/nested-field chain) hit
    # the same "cannot be implicitly copied" restriction, but `_resolved_ty`
    # only recursed through MirFieldAccess receivers, never resolving a
    # MirMethodCall's own return type (infer_types never populates one --
    # hir_to_mir doesn't set `ty=` when lowering a HirMethodCall at all).
    # So `a.Direction().coord`'s type came back Unknown and no `.copy()`
    # was inserted for the fresh declaration.
    out = _mojo(
        """
        class XYZ {
        public:
            XYZ(double x) : myX(x) { }
        private:
            double myX;
        };
        class Dir {
        public:
            Dir() : coord(1.0) { }
            XYZ coord;
        };
        class Ax1 {
        public:
            Ax1() : myDir() { }
            Dir Direction() const { return myDir; }
        private:
            Dir myDir;
        };
        void f(Ax1& a) {
            XYZ v = a.Direction().coord;
        }
        """
    )
    assert "a.Direction().coord.copy()" in out
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- C++ → Rust compile check ----------

@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_cpp_to_rust_compiles():
    out = _rust(
        """
        int factorial(int n) {
            int result = 1;
            int i = 1;
            while (i <= n) {
                result = result * i;
                i = i + 1;
            }
            return result;
        }
        """
    )
    result = rust_compiles(out)
    assert result.ok, result.stderr


# ---------- C++ → Zig compile check ----------

@pytest.mark.skipif(not _has("zig"), reason="zig not installed")
def test_cpp_to_zig_compiles():
    out = _zig("int add(int a, int b) { return a + b; }")
    result = zig_compiles(out)
    assert result.ok, result.stderr


# ---------- C++ → C compile check ----------

@pytest.mark.skipif(not _has("cc") and not _has("gcc") and not _has("clang"), reason="no C compiler")
def test_cpp_to_c_compiles():
    """C++ -> C via the shared MIR — the IR is language-agnostic enough that
    this works for the C-compatible subset."""
    out = _c("int add(int a, int b) { return a + b; }")
    assert "int64_t add(int64_t a, int64_t b)" in out
    result = c_compiles(out)
    assert result.ok, result.stderr


# ---------- classes ----------

def test_cpp_class_to_rust_struct_impl():
    out = _rust(
        """
        class Point {
        public:
            int x;
            int y;

            int sum() {
                return this->x + this->y;
            }
        };
        """
    )
    assert "struct Point {" in out
    assert "x: i64," in out
    assert "y: i64," in out
    assert "impl Point {" in out
    assert "fn sum(&self) -> i64" in out
    assert "self.x + self.y" in out


def test_cpp_class_to_mojo_struct():
    out = _mojo(
        """
        class Point {
        public:
            int x;
            int y;

            int sum() {
                return this->x + this->y;
            }
        };
        """
    )
    assert "@fieldwise_init" in out
    assert "struct Point(Copyable, Movable):" in out
    assert "var x: Int" in out
    assert "var y: Int" in out
    assert "def sum(self) -> Int:" in out


def test_cpp_unqualified_sibling_method_call_gets_self():
    # `cube()` calls `square(x)` with no `this->`/qualifier -- valid C++ via
    # implicit member lookup (works even for a *static* sibling method, as
    # here). Every backend's struct methods aren't free functions, so the
    # call must be qualified with `self.` or the emitted target can't reach
    # the method at all (Mojo: "cannot access method directly").
    src = """
        class MathUtil {
        public:
            static int square(int x) { return x * x; }
            static int cube(int x) { return x * square(x); }
        };
    """
    out = _mojo(src)
    assert "self.square(x)" in out
    rust = _rust(src)
    assert "self.square(x)" in rust


_CLASS_SRC = """
class Point {
public:
    int x;
    int y;
    int sum() { return this->x + this->y; }
};
"""


def _to(target: str) -> str:
    from transpilers.cli.main import transpile
    return transpile(textwrap.dedent(_CLASS_SRC).lstrip(), source_lang="cpp", target=target)


def test_cpp_class_to_c_struct():
    out = _to("c")
    assert "typedef struct {" in out
    assert "int64_t x;" in out
    assert "} Point;" in out
    assert "int64_t Point_sum(Point *self)" in out
    assert "self->x + self->y" in out


def test_cpp_class_to_zig_struct():
    out = _to("zig")
    assert "const Point = struct {" in out
    assert "x: i64," in out
    assert "fn sum(self: Point) i64" in out
    assert "self.x + self.y" in out


def test_cpp_class_to_go_struct():
    out = _to("go")
    assert "type Point struct {" in out
    assert "func (self *Point) sum() int64" in out
    assert "self.x + self.y" in out


def test_cpp_class_to_python_class():
    out = _to("python")
    assert "class Point:" in out
    assert "x: int" in out
    assert "def sum(self) -> int:" in out


def test_cpp_class_to_fortran_type():
    out = _to("fortran")
    assert "type :: Point" in out
    assert "integer :: x" in out
    assert "function Point_sum(self) result(result_)" in out
    assert "self%x + self%y" in out


@pytest.mark.skipif(not _has("rustc"), reason="rustc not installed")
def test_cpp_class_compiles_as_rust():
    out = _rust(
        """
        class Point {
        public:
            int x;
            int y;
            int sum() { return this->x + this->y; }
            int scale(int factor) { return this->x * factor + this->y * factor; }
        };
        """
    )
    result = rust_compiles(out)
    assert result.ok, result.stderr


def test_cpp_return_struct_param_is_copy_not_fabricated_ctor():
    # `return v;` where v is a struct-typed parameter is an implicit copy
    # construction -- libclang materializes it as a CALL_EXPR to the
    # struct's name with a single argument (the value being copied). The
    # struct-constructor path mistook this for a *partial fieldwise* ctor
    # call missing its trailing args, padding the missing `y` field with a
    # fabricated 0: `return v;` emitted as `Vec(v, 0)`, silently wrong
    # (and a compile error to boot, since `v` isn't an Int).
    src = """
        struct Vec {
            int x;
            int y;
            Vec(int a, int b) : x(a), y(b) {}
        };
        Vec passthrough(Vec v) { return v; }
    """
    out = _mojo(src)
    assert "Vec(v, 0)" not in out
    assert "def passthrough(v: Vec) -> Vec:" in out
    result = mojo_compiles(out)
    assert result.ok, result.stderr


def test_cpp_return_struct_param_compiles_in_rust_too():
    src = """
        struct Vec {
            int x;
            int y;
            Vec(int a, int b) : x(a), y(b) {}
        };
        Vec passthrough(Vec v) { return v; }
    """
    out = _rust(src)
    result = rust_compiles(out)
    assert result.ok, result.stderr


def test_cpp_ctor_with_fewer_params_than_fields_not_padded():
    # A struct can have MORE fields than its constructor takes explicit
    # params for, when the constructor's member-init list computes some
    # fields from the others (OCCT's `gp_EulerSequence_Parameters`: 6
    # fields, but a 4-param ctor deriving 3 of them from the first param).
    # The struct-init trailing-field-padding logic assumed every arg
    # lines up 1:1 with a field and padded "missing" fields up to the
    # *field* count, fabricating 2 extra args no constructor declares
    # ("no matching function in initialization").
    src = """
        struct Six {
            int a, b, c;
            bool d, e, f;
            Six(int x, bool y, bool z, bool w) : a(x), b(x), c(x), d(y), e(z), f(w) {}
        };
        Six make() { return Six(1, true, false, true); }
    """
    out = _mojo(src)
    assert "Six(1, True, False, True)" in out
    assert "Six(1, True, False, True, False, False)" not in out
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_defaulted_default_ctor_alongside_real_ctor_compiles():
    # `Vec() = default;` alongside another *real*, explicit constructor
    # (`Vec(double, double)`) previously vanished entirely: the explicit
    # constructor made Mojo's `@fieldwise_init` auto-default unavailable
    # (it only synthesizes when a struct has *no* explicit `__init__`), so
    # a 0-arg `Vec()` call site had no matching overload at all ("no
    # matching function in initialization").
    out = _mojo(
        """
        struct Vec {
            double x, y;
            Vec() = default;
            Vec(double a, double b) : x(a), y(b) {}
        };
        Vec origin() { return Vec(); }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- static methods (no implicit self/this) ----------

def test_cpp_static_method_no_self_param():
    # A `static` method has no implicit `this` -- called via
    # `ClassName::method(...)`, no receiver instance at all. Previously
    # every CXX_METHOD unconditionally got an injected `self` param, which
    # is wrong twice over: the real call site has no instance to pass, and
    # Mojo's own static-method convention (`@staticmethod`, no self param)
    # rejects a "self"-first signature with a caller that never supplies one.
    out = _mojo(
        """
        class Tol {
        public:
            static double Small() { return 1e-9; }
        };
        """
    )
    assert "@staticmethod" in out
    assert "def Small() -> Float64:" in out
    assert "def Small(self)" not in out


def test_cpp_qualified_static_call_resolves_to_struct_dot_method():
    # Precision::Angular() (or here, Tol::Small()) called from another
    # class's method must route through the callee's own struct name --
    # `Tol.Small()`, Mojo's real static-call syntax -- not a bare
    # `Small()` free-function reference (Small was never emitted as a
    # free function; that call site would be "unknown declaration") and
    # not `self.Small()` (this is not a same-class call).
    out = _mojo(
        """
        class Tol {
        public:
            static double Small() { return 1e-9; }
        };
        class Foo {
        public:
            bool Check(double a) { return a <= Tol::Small(); }
        };
        """
    )
    assert "self.Small()" not in out
    assert "Tol.Small()" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_qualified_static_call_compiles():
    out = _mojo(
        """
        class Tol {
        public:
            static double Small() { return 1e-9; }
        };
        class Foo {
        public:
            bool Check(double a) { return a <= Tol::Small(); }
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_static_method_signedness_overload_dedup_no_self():
    # dedupe_overloads' rename (issue #80) must not reintroduce a `self`
    # param for a static overload it renames.
    out = _mojo(
        """
        class Bitwise {
        public:
            static int NOT(const int x) { return -x; }
            static unsigned int NOT(const unsigned int x) { return -x; }
        };
        """
    )
    assert "def NOT(x: Int) -> Int:" in out
    assert "def NOT_overload2(x: Int) -> Int:" in out


def test_cpp_unnamed_param_gets_synthesized_name():
    # An unnamed parameter is legal C++ both in a bare declaration and, when
    # genuinely unused, in a definition too (e.g. OCCT's own
    # `void gp_Pnt::DumpJson(Standard_OStream&, int) const`). `cursor.spelling`
    # is "" for it -- previously emitted a blank param name, invalid Mojo.
    out = _mojo(
        """
        double Epsilon(double) { return 1e-16; }
        """
    )
    assert "def Epsilon(_arg0: Float64) -> Float64:" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_unnamed_param_compiles():
    out = _mojo(
        """
        double Epsilon(double) { return 1e-16; }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_unnamed_constructor_and_method_params_get_synthesized_names():
    # Same gap, constructor and method PARM_DECL sites.
    out = _mojo(
        """
        class Foo {
        public:
            Foo(int) {}
            void Bar(int, double) {}
        };
        """
    )
    assert "def __init__(out self, _arg0: Int):" in out
    assert "def Bar(self, _arg0: Int, _arg1: Float64):" in out


def test_cpp_free_operator_overload_gets_sanitized_name():
    # Free (non-member) operator overloads are a common idiom for symmetric
    # binary operators (`gp_Vec operator*(double, const gp_Vec&)`). Call
    # sites desugar straight to a binop (see `_convert_call`'s
    # `_OVERLOAD_BINOPS` handling) -- this function is never invoked by
    # name -- but it still needs a syntactically valid name, and must NOT
    # be the literal dunder (Mojo rejects a global function named
    # `__mul__`: "must be a method, not a global function").
    out = _mojo(
        """
        struct Vec { double x; };
        Vec operator*(double s, const Vec& v) { Vec r; r.x = s * v.x; return r; }
        """
    )
    assert "operator*" not in out
    assert "def __mul__(" not in out
    assert "def _operator__mul__(s: Float64, v: Vec) -> Vec:" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_free_operator_overload_compiles():
    out = _mojo(
        """
        struct Vec { double x; };
        Vec operator*(double s, const Vec& v) { Vec r; r.x = s * v.x; return r; }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_2d_accessor_operator_call_becomes_getitem():
    # A multi-arg `operator()` that returns a REFERENCE (as opposed to a
    # plain value) is the common matrix/grid-class 2D element-accessor
    # idiom (OCCT's gp_Mat/gp_GTrsf/...: `double& operator()(int, int)`),
    # not a generic 0/1-arg functor call. A read call site (`m(1, 2)`)
    # previously emitted a garbled extra `operator()` argument
    # ("m(operator(), 1, 2)", "use of unknown declaration 'operator'").
    out = _mojo(
        """
        struct Mat {
            double v[4];
            double& operator()(int r, int c) { return v[r * 2 + c]; }
        };
        double first(Mat& m) { return m(0, 0); }
        """
    )
    assert "def __getitem__(self, r: Int, c: Int) -> Float64:" in out
    assert "operator()" not in out
    assert "m[0, 0]" in out or "m.__getitem__(0, 0)" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_2d_accessor_operator_call_compiles():
    out = _mojo(
        """
        struct Mat {
            double v[4];
            double& operator()(int r, int c) { return v[r * 2 + c]; }
        };
        double first(Mat& m) { return m(0, 0); }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_2d_accessor_write_use_is_unsupported_hole_not_wrong_code():
    # The reference-returning 2D accessor also supports a WRITE use
    # (`m(1, 1) = v;`, assigning through the returned reference) -- no
    # target modeled here has an analog for that, so it must fall through
    # to a never-refuse hole rather than the previous silent
    # single-index-dropping bug (`self.matrix[1] = ...`, discarding the
    # row index entirely and emitting subtly wrong code).
    out = _mojo(
        """
        struct Mat {
            double v[4];
            double& operator()(int r, int c) { return v[r * 2 + c]; }
            void SetOne(Mat& m) { m(0, 0) = 1.0; }
        };
        """
    )
    assert "TODO[port]" in out


def test_cpp_call_operator_2arg_functor_still_maps_to_call_not_getitem():
    # A VALUE-returning 2-arg functor (e.g. a comparator) must still map to
    # `__call__`, not be misrouted to `__getitem__` by the 2D-accessor fix
    # above -- distinguished by return type (reference vs plain value).
    out = _mojo(
        """
        struct Comparator {
            bool operator()(int a, int b) const { return a < b; }
        };
        """
    )
    assert "def __call__(self, a: Int, b: Int) -> Bool:" in out
    assert "__getitem__" not in out


def test_cpp_macro_constant_in_binop_not_a_todo_hole():
    # `M_PI` (predefined via a `-D` compiler flag for <cmath>/POSIX libm
    # portability -- not standard C++, but ubiquitous in real math-heavy
    # code) used inline in a binary expression hit a libclang tokenizer
    # quirk: a sub-expression cursor whose extent starts exactly at a
    # command-line macro's expansion point returns *no* tokens at all, so
    # `_binop_token` couldn't locate the operator and the whole statement
    # silently became a `TODO[port]` hole -- even though the surrounding
    # statement (and its own tokens) parse just fine.
    out = _mojo(
        """
        double f(double x) { return M_PI - x; }
        """
    )
    assert "TODO[port]" not in out
    assert "3.14159265358979" in out
    assert "def f(x: Float64) -> Float64:" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_macro_constant_in_binop_compiles():
    out = _mojo(
        """
        double f(double x) { return M_PI - x; }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_macro_constant_resolves_to_real_value_not_zero():
    # Beyond just parsing, the literal's own VALUE must be the real constant
    # -- not the previous silent fallback to 0.0 (`_literal_token`'s widened
    # retokenize finds the macro's identifier spelling, not its expanded
    # numeric text, so a naive `float(token.spelling)` would still fail and
    # fall back to a wrong 0.0 even once the binop-token-lookup bug above is
    # fixed). `_PREDEF_FLOAT_MACROS`/`_PREDEF_INT_MACROS` resolve these
    # specific, well-known names to the value the frontend itself defines
    # them as via its own `-D` flags.
    out = _mojo("double area(double r) { return M_PI * r * r; }")
    assert "0.0 * r" not in out
    assert "3.14159265358979" in out


def test_cpp_unary_operator_overload_call_site():
    # `-v` where `v`'s type has a member `operator-()` (unary negation, 0
    # explicit params) arrives at the call site as a CALL_EXPR shaped like
    # the binary-operator-overload case (`_OVERLOAD_BINOPS`), but with only
    # 1 real operand after filtering the operator-ref. That unary case fell
    # through unhandled, reaching the generic call-resolution fallback and
    # emitting garbled code (`v(operator-)`, "unexpected token in
    # expression") instead of a `HirUnaryOp`.
    out = _mojo(
        """
        struct Dir {
            double x;
            Dir(double v) : x(v) {}
            Dir operator-() const { return Dir(-x); }
        };
        Dir negate(Dir d) { return -d; }
        """
    )
    assert "operator-" not in out
    assert "return -d" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_unary_operator_overload_call_site_compiles():
    out = _mojo(
        """
        struct Dir {
            double x;
            Dir(double v) : x(v) {}
            Dir operator-() const { return Dir(-x); }
        };
        Dir negate(Dir d) { return -d; }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_this_deref_assign_becomes_self_reassign_and_mut_self():
    # `(*this) = expr;` (the "replace my whole value" idiom for a
    # mutate-via-copy-assign method, e.g. OCCT's `gp_Quaternion::Multiply`)
    # has no identifier `_decl_name` can extract from a dereference, so it
    # fell through the assignment-target handling entirely and emitted
    # garbled code (`__cpp_overloaded_op__(operator=, ...)`, "unexpected
    # token in expression"). Beyond just parsing, the method also needs
    # `mut self`, since reassigning `self` outright is a mutation just like
    # assigning one of its fields.
    out = _mojo(
        """
        struct Q {
            double x;
            Q(double v) : x(v) {}
            Q Multiplied(const Q& o) const { return Q(x * o.x); }
            void Multiply(const Q& theOther) {
                (*this) = Multiplied(theOther);
            }
        };
        """
    )
    assert "def Multiply(mut self, theOther: Q):" in out
    assert "self = self.Multiplied(theOther.copy())" in out
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_this_vs_addr_of_identity_check_drops_to_false():
    # `this == &theOther` is a raw POINTER-identity comparison (the classic
    # self-assignment/self-comparison guard, e.g. OCCT's
    # `gp_Quaternion::IsEqual`: `if (this == &theOther) return true;`
    # before a real value-comparison fallback) -- not a call to the type's
    # `operator==`. This engine's existing address-of/`this`-to-`self`
    # lowering collapses both sides to plain names, so previously this
    # became a VALUE comparison `self == theOther`, which is wrong (a
    # different check) and can be a compile error outright when the type
    # has no `operator==`, as here.
    out = _mojo(
        """
        struct Q {
            double x;
            Q(double v) : x(v) {}
            bool IsEqual(const Q& theOther) const {
                if (this == &theOther) {
                    return true;
                }
                return x == theOther.x;
            }
        };
        """
    )
    assert "self == theOther" not in out
    assert "if False:" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_this_vs_addr_of_identity_check_compiles():
    out = _mojo(
        """
        struct Q {
            double x;
            Q(double v) : x(v) {}
            bool IsEqual(const Q& theOther) const {
                if (this == &theOther) {
                    return true;
                }
                return x == theOther.x;
            }
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_2d_fixed_array_field_gets_nested_list_annotation():
    # `double myMat[3][3];` (a native 2D fixed-size array field, OCCT's
    # gp_Mat) previously collapsed to a flat `List[Float64]` annotation:
    # the array-type-text heuristic sliced the type spelling at the FIRST
    # `[`, discarding every dimension after the first. Needs one
    # `list[...]` layer per bracket group.
    out = _mojo(
        """
        struct Mat {
            double myMat[2][2];
        };
        """
    )
    assert "var myMat: List[List[Float64]]" in out


def test_cpp_std_swap_desugars_to_temp_swap_not_builtin_call():
    # `std::swap(a, b)` was passed straight through as a call to Mojo's own
    # `swap()` builtin, which rejects two arguments that alias the same
    # underlying container ("allows writing a memory location previously
    # writable through another aliased argument") -- exactly OCCT's
    # `gp_Mat::Transpose` idiom, `std::swap(myMat[0][1], myMat[1][0])`, two
    # elements of the *same* matrix. Desugars to a manual temp-variable
    # swap instead, which sidesteps the aliasing restriction entirely and
    # is correct for every target, not just Mojo.
    out = _mojo(
        """
        struct Mat {
            double myMat[2][2];
            void Transpose() { std::swap(myMat[0][1], myMat[1][0]); }
        };
        """
    )
    assert "swap(" not in out
    assert "__swap_tmp" in out
    assert "def Transpose(mut self):" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_std_swap_on_same_matrix_compiles():
    out = _mojo(
        """
        struct Mat {
            double myMat[2][2];
            void Transpose() { std::swap(myMat[0][1], myMat[1][0]); }
        };
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


def test_cpp_uninitialized_2d_array_local_gets_zero_fill_not_size_literal():
    # `double mymatrix[3][3];` (a native fixed-size array declared with NO
    # initializer at all, OCCT's own `gp_Trsf::InitFromJson` scratch
    # buffer) has its dimension-size expressions (INTEGER_LITERAL cursors
    # for the `3`s) reported as VAR_DECL *children* by libclang -- the
    # generic "last child is the initializer" fallback misread the last
    # dimension-size literal as one, emitting `var mymatrix: List[...] = 3`
    # (a scalar int, not a matrix) instead of a real zero-filled matrix.
    out = _mojo(
        """
        void f() {
            double mymatrix[2][2];
            mymatrix[0][0] = 1.0;
        }
        """
    )
    assert "= 3" not in out
    assert "[[0.0, 0.0], [0.0, 0.0]]" in out


@pytest.mark.skipif(not _has("mojo"), reason="mojo not installed")
def test_cpp_uninitialized_2d_array_local_compiles():
    out = _mojo(
        """
        void f() {
            double mymatrix[2][2];
            mymatrix[0][0] = 1.0;
        }
        """
    )
    result = mojo_compiles(out)
    assert result.ok, f"mojo rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- refusals ----------

def test_cpp_template_preserved_as_raw_hole():
    # Issue #50: templates are now preserved as HirRaw holes (with a
    # TODO[port] stub emitted by every backend), and the *signature*
    # is recorded in the ground truth so callers can pick up the
    # parameter / return types via the AST. We don't refuse the
    # construct outright anymore -- templates are real C++ and the
    # engine should emit *something* rather than aborting.
    out = _mojo("template<typename T> T add(T a, T b) { return a + b; }")
    # The mojo backend should emit a TODO stub for the function body.
    assert "TODO[port]" in out


def test_cpp_class_inheritance_refused():
    with pytest.raises(Exception):
        _rust("class Base { public: int x; }; class D : public Base { public: int y; };")

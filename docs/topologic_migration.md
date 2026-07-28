# Migrating github.com/wassimj/Topologic (issue #79)

Topologic is a real-world CAD/geometry-kernel C++ library built on
OpenCASCADE (OCCT). It's a useful, demanding stress test for the strict
engine: multi-file (`.h`/`.cpp` split), a large third-party dependency, and
real class hierarchies.

## What was blocking it, and what's fixed

1. **Multi-file projects weren't supported at all.** The frontend strips
   every `#include` by design (toolchain-free single-file model), so a
   `.cpp` whose class is declared in a separate `.h` failed immediately
   with "use of undeclared identifier". Fixed by
   `frontends/cpp/parser/includes.py::resolve_local_includes` — opt-in via
   `transpile --include-dir/-I` or `transpile-levels --inc`, transitively
   inlining local headers found on the search path (matching by
   findability, not `"..."` vs `<...>` spelling — Topologic itself uses
   `#include <Bitwise.h>` for its own sibling header).

2. **`_project_preamble()` was silently broken.** The extension point for
   injecting domain-specific declarations (`$TRANSPILERS_CPP_PREAMBLE[_FILE]`
   — originally added for EnergyPlus's `Real64`) appended its content
   *after* the user's source instead of before, so anything it declared
   could never actually be used by that code (C++ requires declare-before-
   use). Fixed by reordering; see `docs/occt_preamble.hpp` for a working
   example.

3. **`docs/occt_preamble.hpp`** — a minimal OCCT shim, usable via the fix
   above. Scope: Topologic's `Utilities.h` bundles its `TOPOLOGIC_API`
   export macro together with an unrelated `TopoDS_Shape`-typed helper, so
   *every* file that only wants the macro also inherits an OCCT dependency
   it doesn't otherwise touch. This shim declares just enough
   (`TopoDS_Shape`, `TopAbs_ShapeEnum`) to get past that specific coupling
   — it is not a real OCCT binding and won't get a file whose own logic
   calls deep into the OCCT API (`BRepBuilderAPI_*`, `Geom_*`, `gp_*`, ...)
   to a compiling target; that would need a real Mojo-side OCCT binding,
   which doesn't exist.

4. **Two general (non-Topologic-specific) bugs** these changes surfaced
   once parsing got far enough to reach emission and real compilation
   (validated against the actual Mojo 1.0.0b2 compiler):
   - An unqualified call to a sibling method (`square(x)` inside `cube()`,
     valid C++ via implicit member lookup, including for static methods)
     was emitted as a free function call instead of `self.square(x)` —
     every backend's methods live on a struct, not as free functions.
   - `operator()` (the functor call operator) was missing from the
     operator→dunder table, so it emitted the literal invalid identifier
     `operator()` instead of `__call__`.

5. **`dedupe_overloads` pass (issue #80).** Two C++ overloads differing
   only by signedness (`int` vs `unsigned int`) collapse to the same target
   scalar type, so both emitted a method with an *identical* signature in
   the same struct — a guaranteed duplicate-definition compile error in
   Mojo/Rust/Zig (found via `Bitwise::NOT(int)` / `Bitwise::NOT(unsigned
   int)`). Renamed deterministically after type inference; use
   `--fidelity idiomatic` when verifying such a file, since the rename is
   itself a (intentional) structural-fidelity divergence under the default
   `structural` gate.

6. A stray-comma bug in the Rust backend's struct emission for a class with
   *zero* fields (e.g. a static-method-only utility class like `Bitwise`
   itself) — `struct Empty {\n,\n}` is a syntax error, not just cosmetic.

7. **The Mojo backend now auto-emits a minimal opaque placeholder struct**
   for any foreign type the parser resolved (e.g. via the OCCT preamble
   shim above) but the user's own code never declared — otherwise every
   reference to it in the *emitted* Mojo is "use of unknown declaration",
   even though libclang parsed it fine. Verified working for the realistic
   case of storing/passing an opaque value through without calling methods
   on it directly (a struct field of the foreign type, e.g.).

## Honest current state: 0/51 `TopologicCore` files transpile-and-compile

Every one of the 51 files in `TopologicCore/src/` still fails end-to-end
(`transpile ... --fidelity idiomatic --verify`), even with every fix above
and the OCCT shim. But *why* they fail changed meaningfully, and the
remaining wall is now precisely characterized rather than "everything dies
at parse":

- **~45 files**: still fail at `libclang parse errors` — they transitively
  pull in an OCCT header the shim doesn't cover (only `TopoDS_Shape` /
  `TopAbs_ShapeEnum` are declared; the corpus uses 111 distinct OCCT
  headers, ~400 uses of `TopoDS_Shape` alone).
- **`Bitwise.cpp`**: now parses *completely* (including `Utilities.h`'s
  `OcctShapeComparator`) and reaches real Mojo compilation, failing only on
  `rkOcctShape.TShape().operator->()` — the OCCT shim's `TopoDS_Shape` type
  resolves for parsing, but its placeholder struct has no `TShape()` method
  and `operator->()` has no Mojo/Python equivalent at all (there's no
  transparent-dereference operator to map it to). Getting *this* to compile
  needs per-method stubbing on every opaque OCCT type actually called into,
  not just the type names — a materially bigger, unbounded-looking task
  (every new file can call a new method on a new OCCT type) rather than a
  fixed one, so it's left as future work rather than chased further here.
- **5 files** (`IntAttribute`, `DoubleAttribute`, `ListAttribute`,
  `Geometry`, `Surface`): now get past the OCCT wall entirely and hit a
  *different*, pre-existing, unrelated engine limitation —
  `class member TYPEDEF_DECL` (a nested type alias inside a class body,
  e.g. `typedef std::shared_ptr<IntAttribute> Ptr;`, isn't modeled).

None of this is specific to Topologic in the way it first looked: getting a
real-world OCCT-based C++ project to a *compiling* Mojo target output would
need a real OCCT binding for Mojo (which doesn't exist) or a purpose-built,
much larger opaque-type shim covering real method surfaces for dozens of
OCCT classes — a project on the scale of the engine itself, not a bug-
fixing pass. `transpilers.lift` (the never-refuse C++→Python engine)
already handles this gracefully today by degrading unresolvable constructs
to `# TODO[lift]` stubs, but only targets Python, not Mojo.

## What generalizes beyond Topologic

Everything in this session except the OCCT shim itself is a real, general
engine fix, each validated against the real Mojo 1.0.0b2 / rustc / zig /
gfortran compilers (now installed in dev/CI environments that have them) —
multi-file header resolution, the `_project_preamble()` ordering fix, the
sibling-method-call and copy-constructor bugs, `operator()`, the overload
dedup pass, the Rust empty-struct fix, and the foreign-struct placeholder
all apply to *any* C++ input, not just this corpus.

## Follow-up: a *real* binding for OCCT's `gp` package (issue #79 continued)

Point 3 above notes that `docs/occt_preamble.hpp` only stubs a couple of
opaque OCCT types to get *past* Topologic's coupling to them — it doesn't
attempt to transpile any real OCCT logic. As a narrower follow-on (per
explicit user direction: start with just `gp_Pnt`/`gp_Vec`/`gp_Dir` rather
than the whole inheritance-free `gp` package), this session went further:
`docs/occt_gp_preamble.hpp` targets actually transpiling `gp`'s own real
math logic to Mojo, since `gp` (unlike the rest of OCCT) has no
`Handle(T)`/`Standard_Transient` reference counting and no inheritance —
exactly the strict engine's current C++ subset.

**Current state: `gp_Pnt.cxx` (plus its real transitive `gp`-package
dependencies — gp_XYZ, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Mat, and
their 2D counterparts, all pulled in by `gp_Pnt.cxx`'s own real `#include`
list) now parses and converts to HIR with *zero* errors, and gets
substantially further into real Mojo compilation than before — but is not
yet a clean `--verify` pass.** Getting there surfaced eight distinct
general engine bugs, none Topologic- or OCCT-specific:

1. **A header-guard corruption bug in `preprocess.py`'s no-clang fallback**
   (`_strip_header_guard`): it tracked stack depth only for `#ifndef`
   opens, but popped on *any* `#endif` regardless of what opened it. A
   real `#ifdef`/`#endif` block nested inside an outer `#ifndef` guard (a
   platform-specific compiler-bug workaround, found in `gp_Mat.hxx`) wrongly
   consumed the outer guard's stack frame, stripping the *inner* `#endif`
   as if it were the guard's closer and leaving the *real* outer `#endif`
   behind as an orphan. libclang's own preprocessor then silently skipped
   every line between the still-open inner `#if` and the next `#endif` it
   could find — potentially thousands of lines, with zero diagnostics,
   silently discarding real class definitions. This took the most
   debugging effort in the session to isolate (compiler agreed the code
   parsed with no errors *and* produced no AST nodes for the missing
   classes — the two facts only made sense once the "silently-skipped
   inactive branch" explanation was found).
2. **`resolve_local_includes` couldn't represent circular textual
   includes** (`gp_Dir.hxx` forward-declares `gp_Ax2d` and only
   `#include`s it at the *bottom* of the file, after its own class body —
   `gp_Ax2d.hxx` does the mirror-image thing for `gp_Dir`). The original
   hoist-dependencies-before-dependents model can't hoist both before each
   other. Rewritten to expand each `#include` strictly in place (matching
   real preprocessor/header-guard semantics: a header's second inclusion
   anywhere is simply dropped) instead of pulling full dependency text to
   the front.
3. **`parse_cpp` relied on clang's default `-ferror-limit=20`**, so a
   large multi-file translation unit could blow past the limit before the
   macro-like-unknown-type retry loop ever saw every distinct macro name,
   or before `_check_diagnostics`'s final error report reflected more than
   an arbitrary prefix of the real problem set. Added `-ferror-limit=0`.
4. **`std::hash<T>`/`std::equal_to<T>` had no primary template
   declared** in the parser preamble, so the near-universal "make my type
   usable as an unordered_map/set key" idiom (`template<> struct
   hash<gp_Pnt> {...}`) failed to parse at all ("explicit specialization of
   undeclared template"). Added primary templates; the specialization
   itself is recognized (by its telltale bare-`TYPE_REF`-as-first-child
   shape, which no ordinary class produces) and skipped rather than
   transpiled — it's STL-interop plumbing for the type above it, not that
   type's own logic, and commonly uses constructs (`union`) outside the
   modeled subset anyway.
5. **`_convert_top_level` never checked `is_definition()` before
   converting a `CLASS_DECL`/`STRUCT_DECL`.** A forward declaration
   (`class gp_XYZ;`, used purely for by-reference/pointer parameters before
   the real definition) got converted into a second, *empty* `HirStruct`
   alongside the real one — every backend then emitted the struct twice, a
   hard "invalid redefinition" compile error, not just a fidelity
   divergence.
6. **The copy/move-constructor collapse fix from the Topologic session
   (`return v;` → plain `v`, not a fabricated `Vec(v, 0)`) only covered
   `_convert_call`'s call-expression path — `_convert_var_decl` (local
   variable declarations, `Vec aCopy = *this;`) has its own, separate
   struct-init detection with the identical bug.** Applied the same fix
   there.
7. **Several class-body/statement shapes had no handling at all**, each
   real and common in value-type-heavy C++, none OCCT-specific:
   `friend class X;` (a pure access-control declaration, safe to drop —
   adds no field/method), a templated *method* on an otherwise-concrete
   class (`template<class T> void GetMat4(T&)` — dropped, since the IR
   doesn't model templates and the struct's other concrete members are
   unaffected), a class-nested `enum class D {...}` (hoisted to the same
   module-level int constants a top-level enum already gets — references
   resolve by bare enumerator name regardless of nesting), an out-of-line
   *constructor* definition (`Class::Class(...) {...}`, previously only
   out-of-line *methods* were attached back to their struct), and a
   compound assignment on an implicitly-`this`-qualified field (`myA *=
   s;` inside a method — only plain `=` was special-cased for a
   `MEMBER_REF_EXPR` LHS, so `*=`/`+=`/etc on a bare field silently
   degraded to a dropped `pass # TODO[port]`).
8. **A `= default`ed special member (`Vec& operator=(const Vec&) =
   default;`) still reports `is_definition() == True` in libclang** despite
   having no explicit body — converting it produced a garbled, bodiless
   method instead of being skipped like any other declared-but-not-defined
   member.
9. **A CALL_EXPR shape unique to overloaded assignment operators**: when a
   field is assigned without explicit `this->` and its type has a
   user-defined (even `=default`ed) `operator=`, C++ desugars `vxdir =
   theV;` to `vxdir.operator=(theV)`, which libclang represents as a
   CALL_EXPR whose *first child is the field being assigned*, not an
   ordinary method receiver. The existing "MEMBER_REF_EXPR callee ⇒ method
   call" branch misread this as `self.vxdir(operator=, theV)` — calling
   the field itself with the operator-name decl-ref as a bogus argument.
   Fixed by recognizing `operator=`/the existing compound-assignment
   overload names up front and desugaring to a plain (or augmented)
   `HirFieldAssign`/`HirAssign`, mirroring the equivalent statement-level
   fix in point 7.
10. **A qualified cross-class static call (`Precision::Angular()`) was
    misdetected as an unqualified same-class self-call.** The Topologic
    session's fix for `square(x)` (implicit `this->` lookup inside another
    method of the same class) checked only whether the callee resolved to
    a `CXX_METHOD` cursor — true for *any* method call libclang can
    resolve, qualified or not. A real cross-class static call got rewritten
    to `self.Angular()`, silently invoking the wrong method (if one of that
    name even exists on self) instead of the real one — a correctness bug,
    not just a compile failure. Fixed by checking the callee's own tokens
    for a `::` (present for a qualified reference, absent for the
    unqualified case the original fix targeted).
11. **Two more operator/mutability bugs in `_method_name` and the Mojo
    backend**, both independent of the parser fixes above:
    `+`/`-` are ambiguous between unary and binary overloads (`operator-()`
    with no explicit params is negation; `operator-(other)` is
    subtraction) — both mapped to `__sub__`, producing a zero-argument
    `__sub__` every target's own arity check rejects. Fixed by checking
    parameter count. Separately, Mojo's `mut self` requirement was only
    detected via a small hardcoded STL-container method-name list; a
    method that mutates self *by calling another user-defined method that
    itself mutates self* (`Multiplied()` calling `self.Multiply(x)` —
    exactly the mutate-in-place/return-a-copy pairing OCCT's `gp_*` types
    use throughout for every arithmetic operation) was left as immutable
    `self`, which the real compiler rejects. Fixed with a module-wide,
    fixed-point closure over method names that mutate their receiver
    (conservative by design: it matches by bare method name, not
    per-struct, so at worst it marks an unrelated same-named method `mut`
    too — harmless, since Mojo permits that).
12. Two Mojo-only copy-insertion gaps, both the same underlying restriction
    (`return`/assign of a bare struct-typed reference needs `.copy()`) just
    missed for **fields** specifically: `return self.field` and `var x =
    self.field` weren't covered because `infer_types` never resolves a
    `MirFieldAccess`'s own type (no struct-field table available to it) —
    the existing check only looked at the pre-resolved `.ty`. Added a
    lowering-time struct-field-type lookup as a fallback, plus a
    `lower_field_assign` override in the Mojo backend (`self.other = ...`
    had no copy-insertion at all — the shared base implementation is
    correct for Rust/Zig, which don't need it, but Mojo does).

## Follow-up 2: closing both structural gaps above (`--include-impls`, static methods)

The two "structural gaps" the first follow-up left as future work — out-of-
line `.cxx`-only methods, and qualified static calls to a preamble-stubbed
type — are now both solved, generally, not just for `gp`. Along the way this
surfaced four more real bugs, one of them a genuine correctness-affecting
performance cliff, none OCCT-specific.

1. **New `resolve_local_includes(..., include_impls=True)` / CLI
   `--include-impls`.** For every locally-resolved header, also pulls in its
   sibling `.cxx`/`.cpp` implementation file (same stem), so a method
   declared in a header but defined out-of-line in its own `.cxx` — fine
   for a real build (the body resolves at link time) — has its real body
   present in the one amalgamated translation unit this engine builds.
   Implemented as a strict **second pass**, only after every header is
   fully resolved, not interleaved into the header expansion at each
   header's first encounter: a `.cxx` file routinely `#include`s more
   headers than its own class's declarations strictly need (it has real
   method bodies to compile, not just signatures), which introduces
   completeness cycles the pure header graph never had to solve (e.g.
   `gp_Mat.cxx` needing `gp_XYZ` complete for its own body, while
   `gp_XYZ`'s header-side completion is still mid-stack because a
   different branch is what's currently pulling `gp_XYZ.hxx` in).
   Resolving all headers first sidesteps this for free.
2. **`_build_provenance_map` was O(mir_nodes × hir_nodes)**: it looked up
   each MIR/LIR node's matching HIR provenance with a linear re-scan of
   every already-recorded provenance, instead of an O(1) dict lookup. Fine
   for a hand-written algorithm slice (dozens of nodes), catastrophic for a
   real amalgamated translation unit — `--include-impls` on `gp_Pnt.cxx`
   turned a ~2 second transpile into a multi-minute hang. Fixed with a
   `hir_id -> HirProvenance` index built once. This is a general
   correctness-of-performance bug, not specific to C++ or to
   `--include-impls` — any sufficiently large single-file input would
   eventually have hit it.
3. **A struct declared inside an anonymous namespace** (a common
   translation-unit-local-helper idiom — `gp_Quaternion.cxx` has one for
   Euler-angle decomposition) **kept its namespace-qualified spelling**
   (literally `(anonymous namespace)::Name`, libclang's own spelling for
   it) when used as a type annotation, even though `_convert_top_level`
   flattens every namespace and registers the struct under its bare name.
   A still-qualified return-type annotation could never match that
   registry entry, leaving it an unresolved type hole (a hard `ValueError`
   crash, not a graceful degradation) despite the struct itself converting
   fine. Fixed by stripping any `::`-qualified prefix off a `RECORD`-kind
   type's spelling before use.
4. **No support for C++ `static` methods at all.** Every `CXX_METHOD`
   unconditionally got an implicit `self` param injected, wrong for a
   `static` method twice over: the real call site (`ClassName::method(...)`)
   has no instance to pass, and Mojo's own static-method convention
   (`@staticmethod`, no `self` parameter) rejects a `self`-first signature
   nothing ever supplies a receiver for. Added `HirFunction.is_static` /
   `MirFunction.is_static` (parser sets it from `cursor.is_static_method()`,
   no `self` param injected when true), a Mojo `@staticmethod` decorator,
   and routed the earlier qualified-cross-class-static-call fix (which
   previously fell through to an unresolved bare-name call) to build a
   `HirMethodCall` whose receiver is the callee's own struct name — Mojo's
   real static-call syntax is `StructName.method(...)`, so the struct name
   doubles as a valid "receiver" expression even though it names a type,
   not a value. This is what makes `Precision::Angular()` /
   `gp::Resolution()` resolve correctly now instead of falling through.
5. **The project-preamble exclusion boundary couldn't express "some of this
   preamble should be emitted for real."** A preamble is usually parse-only
   scaffolding (opaque stubs, typedefs, macros) that must never appear in
   the output, but `gp`'s own real headers call real preamble-provided
   helpers (`RealSmall()`, `Epsilon()`, `Precision::Angular()`) that
   genuinely need bodies in the emitted target too, or every call site is
   "use of unknown declaration" despite parsing and type-checking fine.
   Added a literal marker line (`_PREAMBLE_REAL_MARKER`) splitting a
   preamble into an excluded "stub" part and an included "real" part.
   Getting the ordering right took a second attempt: the real part has to
   land in the final translation unit *after* the generic std:: shim
   (`PARSER_PREAMBLE`, itself added later, inside `preprocess_cpp`) and
   immediately before the user's own source — not where it textually sits
   in the preamble file, right before `PARSER_PREAMBLE` — so `parse_cpp`
   now splits the preamble and prepends the real part directly onto
   *source* rather than trying to express a non-contiguous exclusion region
   with a single line-number threshold.

**Result: `gp_Pnt.cxx` (via `--include-impls`, pulling in ~15 of the `gp`
package's `.cxx` files transitively) now goes from timing out entirely to
transpiling in ~2 seconds and getting substantially further into real Mojo
compilation** — the "has no attribute" and "unknown declaration" error
classes that motivated this follow-up are gone. What's left is a longer
tail of narrower, more specific issues scattered across peripheral parts of
the package (`gp_Mat`/`gp_Mat2d`'s in-place `swap()` on nested SIMD
indexing, a handful of Mojo aliasing-exclusivity errors on self-referential
calls, an anonymous-namespace helper struct's constructor call in
`gp_Quaternion.cxx` receiving more arguments than it declares, and a
transitive-mutability gap for *parameters* mirroring the one already fixed
for `self`) — each looks like a real, fixable, general bug on inspection,
but none are blocking in the way the two structural gaps were, and chasing
all of them was out of scope for this pass.

## Follow-up 3: closing the parameter-mutability and call-argument-copy gaps, plus three general C++ frontend bugs

Picking up the "transitive-mutability gap for parameters" and the two
narrower error categories left at the end of Follow-up 2, six real bugs came
out of continuing to chase `gp_Pnt.cxx --include-impls --verify`'s remaining
errors down to (near) zero. None are OCCT-specific; all are general C++
idioms this pass just hadn't exercised yet.

1. **Parameter mutability only recognized a small hardcoded list of STL
   method names, not user-defined mutating methods.** `_params_reassigned`
   (`mir_to_mojo_lir.py`) decided whether a parameter needs `var`/`mut`
   decoration by checking if a mutating method gets called on it, but only
   recognized `append`/`push_back`/etc. — a user-defined mutator like
   `gp_XYZ::Add` wasn't recognized at all, leaving the parameter an
   immutable default borrow, which the real Mojo compiler rejects
   ("invalid use of mutating method on rvalue"). Fixed by adding
   `_compute_mir_mutating_method_names`, a MIR-level fixed-point closure
   mirroring the existing LIR-level one in `backends/mojo/emit.py` (needed
   one IR tier earlier because `lower_params` runs during MIR→LIR lowering,
   before the LIR-level closure would exist).

2. **That fix regressed constructors.** Widening the mutating-method set to
   include user-defined methods meant `self` — always present in every
   method's own `param_names`, including `__init__`'s — now frequently
   matched too, since a constructor's own body commonly calls a
   self-mutating setter. `lower_params` bakes the `"var "` prefix directly
   into the parameter *name string*, so `__init__`'s special-cased "self
   gets `out`, not `var`" check (which compares against the literal string
   `"self"`) silently stopped matching, falling through to an invalid `var
   self: T` signature ("`__init__` method must return Self type with 'out'
   argument"). Fixed by excluding `"self"` from `param_names` at the top of
   `_params_reassigned` — self's mutability is handled entirely by a
   separate, already-correct mechanism in `emit.py`.

3. **`ImplicitlyCopyable` also blocks struct-typed CALL ARGUMENTS, not just
   return/assign/field-assign.** Once parameter mutability started
   resolving correctly, call sites like `aT.Transforms(self.coord)` (passing
   a bare struct field into a `var`-decorated, i.e. owned/consuming,
   parameter) started hitting the same "cannot be implicitly copied"
   restriction previously fixed only for return/assign/field-assign
   positions. Fixed with a new `lower_arg` hook (default in
   `_mir_lower_base.py`, overridden in the Mojo lowering) representing "a
   value passed *by value* into a call" — distinct from a receiver
   expression, which is borrowed, not consumed — wired into constructor
   field-inits, method-call args, plain-call args, and the `push_back`
   special case. Also caught the same gap in *reassignment* (`x = y.field`
   where `x` already existed) — the reassign branch of `lower_assign` had
   no copy-insertion logic at all, only the fresh-declaration branch did.
   Scoped to `StructT` only (verified empirically that bare `String`/`List`/
   `Dict` call arguments do *not* need `.copy()`, unlike their
   return/assign positions).

4. **Unnamed parameters produced a blank Mojo parameter name.**
   `cursor.spelling` is `""` for a C++ `PARM_DECL` with no identifier —
   legal both in a declaration and, when the parameter is genuinely unused,
   in a definition (OCCT's own `void gp_Pnt::DumpJson(Standard_OStream&,
   int) const` omits its unused second parameter's name). Emitted
   `def Epsilon(: Float64) -> Float64:` — invalid Mojo. Fixed by a
   `_param_name` helper that synthesizes `_argN` when the spelling is
   empty, applied at all three `HirParam` sites (constructor, method, free
   function).

5. **Free (non-member) operator overloads emitted the literal invalid
   `operator*` token as their name.** `gp_Vec operator*(double, const
   gp_Vec&)` — the standard idiom for a symmetric binary operator — is
   converted by `_convert_function`, which never applied the
   operator→dunder mapping `_convert_method` already used
   (`_method_name`/now `_operator_name`). Renaming it to the literal dunder
   isn't right either, though: nothing calls this function by name (a call
   site like `2.0 * v` desugars straight to a binop, see
   `_OVERLOAD_BINOPS`), and Mojo (like every target here) rejects a global
   function named `__mul__` outright ("must be a method, not a global
   function"). Fixed by applying the same dunder mapping and then
   prefixing away from the reserved dunder spelling (`__mul__` →
   `_operator__mul__`) for free functions specifically.

6. **The 2D matrix-element accessor idiom (`double& operator()(int, int)`)
   wasn't handled at all**, at either the class-definition or call-site
   level — assessed and deliberately deferred at the end of Follow-up 2 as
   a bigger feature. A *read* use (`M(1, 1)`) produced a garbled extra
   argument (`M(operator(), 1, 1)`, "use of unknown declaration
   'operator'"); a *write* use (`aMat(1, 1) = v;`) fell into the existing
   `operator[]` single-index assignment path (built for 1-D subscripts) and
   silently dropped the row index, producing wrong code that also didn't
   compile ("expression must be mutable in assignment"). Both are real,
   common idioms — not OCCT-specific — found via `gp_Mat`/`gp_GTrsf`/
   `gp_Mat2d`/`gp_GTrsf2d`. Fixed the *read* half generically: a multi-arg
   `operator()` that **returns a reference** (as opposed to a 0/1-arg
   functor call, or a value-returning multi-arg functor like a
   `std::sort` comparator — distinguished by checking `result_type.kind`
   for `LVALUEREFERENCE`/`RVALUEREFERENCE`) maps to `__getitem__`, both at
   the class-definition side (`_operator_name`) and the call-site side
   (`_convert_call`). The *write* half is left as a genuinely unsupported
   construct (a never-refuse `TODO[port]` hole): the idiom relies on
   returning a mutable C++ reference for the caller to assign through, and
   none of these value-oriented targets model that, so there's no
   `__setitem__` counterpart to pair it with.

**Result:** re-running `gp_Pnt.cxx --include-impls --verify` after all six
fixes drops the error count from ~136 individual compiler diagnostics to
~72, and eliminates every category these fixes targeted outright: "expected
argument name", "expected '(' for argument list", "use of unknown
declaration 'operator'", "'__mul__' must be a method, not a global
function", and "expression must be mutable in assignment" (18 occurrences)
are all gone. What's left (`no matching function in initialization` — the
pre-existing anonymous-namespace helper-struct arity mismatch in
`gp_Quaternion.cxx`; `__todo_port__`/unknown-declaration for a handful of
OCCT-specific APIs like `TCollection_AsciiString` this pilot's preamble
doesn't stub; a SIMD-`swap()`-specific mutability case; a few `List[Float64]`
literal-conversion mismatches; `gp_Quaternion` missing `__eq__`) is either
already-documented and deprioritized, or narrow enough to not be worth
chasing in this pass.

## Follow-up 4: the `gp_EulerSequence_Parameters` arity mismatch, and a defaulted-ctor gap

The `no matching function in initialization` bucket (28 of the ~72
remaining errors — the single largest category left) turned out to be one
general bug, not an OCCT-specific one, and chasing it down surfaced a
second general bug in the same area.

1. **`HirStructInit`'s trailing-field-padding logic assumed a struct's
   positional ctor args always line up 1:1 with its fields.** That's true
   for the common per-field aggregate constructor (`gp_XYZ(x, y, z)`: 3
   params, 3 fields), but `gp_EulerSequence_Parameters` has a real,
   explicit constructor with a member-init list that *computes* 3 of its 6
   fields (`i`, `j`, `k`) from a single `theAx1` parameter — 4 declared
   params, 6 fields. Every real call site in the OCCT source passes exactly
   4 args (`Params(1, F, F, T)`), but the padding logic — which exists to
   handle genuine C++ partial aggregate init — treated the 2 "missing"
   fields as needing a fabricated default, emitting a 6-arg call
   (`gp_EulerSequence_Parameters(1, F, F, T, False, False)`) against a
   4-param constructor. Fixed by tracking each struct's explicit `__init__`
   parameter types (`_STRUCT_CTOR_PARAM_TYPES` in `hir_to_mir.py`,
   populated in `_lower_struct` right after its methods are lowered) and
   padding to *that* arity instead of the field count whenever the struct
   has a real constructor — a struct with no explicit constructor (the
   aggregate case) is unaffected, since there's nothing to key off besides
   field count.

2. **A `= default` 0-arg constructor alongside another real, explicit
   constructor left the class with no way to construct it with 0 args.**
   `_convert_class` already skipped `= default` members (nothing to
   convert), relying on Mojo's `@fieldwise_init` to auto-synthesize a usable
   default constructor — but `@fieldwise_init` is only emitted when the
   struct has *no* explicit `__init__` at all (see `emit.py`), and OCCT's
   `gp_Vec` declares both `gp_Vec() = default;` *and* `gp_Vec(const gp_XYZ&)`
   / `gp_Vec(double, double, double)`. `@fieldwise_init` gets dropped for
   the two real constructors, `= default` gets silently skipped, and
   `gp_Vec()` (used throughout the package to construct a zero vector) has
   no matching overload at all. Fixed by detecting this specific
   combination (`is_default_constructor()` among an otherwise-skipped
   defaulted member, plus at least one other real `__init__` ending up on
   the same class) and synthesizing an explicit `__init__(out self):` that
   value-initializes every field — recursing into a struct-typed field's
   own 0-arg construction, mirroring what C++'s defaulted constructor
   actually does for a member with its own default constructor.

**Result:** `no matching function in initialization` drops from 28 to 0.
As a side effect (the previously-broken `translateEulerSequence` and its
callers now type-check further), `__todo_port__`/unknown-declaration count
also drops, from 20 to 10. Total error count for `gp_Pnt.cxx
--include-impls --verify` is now ~35, down from ~136 at the start of
Follow-up 3.

## Follow-up 5: a libclang tokenizer quirk around `-D`-defined macro constants

The next-largest bucket, `use of unknown declaration 'anAng'` (plus a good
chunk of the generic `__todo_port__` count), traced to a single libclang
tokenizer quirk affecting every `-D`-predefined constant this frontend
injects for stdlib portability (`M_PI`, `NULL`, `EXIT_FAILURE`, `INT_MIN`,
...) whenever one is used inline in a binary or unary expression.

`const double anAng = M_PI - Angle(theOther);` parses to a perfectly
ordinary AST (`VAR_DECL` → `BINARY_OPERATOR` → `[FLOATING_LITERAL,
CALL_EXPR]`) — but `_binop_token` (which finds the actual operator symbol
by slicing `cursor.get_tokens()` between the two operand extents, since
libclang's bindings expose no direct `.operator` accessor) got an *empty*
token list for the `BINARY_OPERATOR` cursor. libclang's tokenizer can
return nothing for a sub-expression extent that starts or ends exactly at
a command-line macro's expansion point — even though tokenizing the
*enclosing* `VAR_DECL`/`DECL_STMT` (whose extent starts safely before the
macro) works fine and includes the real `M_PI` / `-` tokens. `_binop_token`
raised `UnsupportedConstruct`, turning the *entire* declaration statement
into a `TODO[port]` hole — and every later reference to `anAng` in the same
function then became "use of unknown declaration", since the variable was
never actually declared.

Fixed with `_tokens_for` (`tokens.py`): try the cursor's own tokens first;
if empty, retokenize a *widened* range (from column 1 of the cursor's
first line to the start of the line after its last) and let the existing
location-based filtering in `_binop_token`/`_unary_token` pick out the
right token — widening the search window can't return a *wrong* token,
only find one that direct tokenization missed, since callers still match
by exact source position.

That alone fixes the "which operator" problem, but not the literal's own
*value*: the token `_tokens_for` recovers for `M_PI` is still the literal
macro identifier text ("M_PI"), never libclang's expanded numeric text —
there's no `evaluate()`-style API in this libclang binding version to ask
for the constant's semantic value directly. Since these are constants the
frontend itself defines via its own `-D` flags, `_PREDEF_INT_MACROS`/
`_PREDEF_FLOAT_MACROS` (`core.py`) map the well-known names (`M_PI`, `NULL`,
`EXIT_FAILURE`, `INT_MIN`, ...) to the same values already passed on the
command line, so `_literal_token` resolves them exactly instead of the
previous silent (and explicitly pre-existing, documented) fallback to a
wrong `0`/`0.0`.

**Result:** `use of unknown declaration 'anAng'` and the remaining
`__todo_port__` count both drop to 0. Total error count for `gp_Pnt.cxx
--include-impls --verify` is now ~21, down from ~35 at the start of this
follow-up and ~136 at the start of Follow-up 3.

## Follow-up 6: unary operator-overload calls, and `(*this) = expr` self-reassignment

The `unexpected token in expression` bucket (3 of the ~21 remaining errors)
turned out to be two distinct, general C++ call-site gaps, both variations
on "a `CALL_EXPR` shaped like an overloaded-operator call that the existing
per-shape handling didn't recognize."

1. **A unary member operator overload's call site was never handled**,
   only its definition was. `-v` where `v`'s type declares a member
   `operator-()` (unary negation, 0 explicit params — already correctly
   mapped to `__neg__` at the class-definition side, see Follow-up 3's
   `_operator_name`) arrives at the call site as the same `CALL_EXPR`
   shape the binary-operator-overload handling (`_OVERLOAD_BINOPS`)
   expects, except with only 1 real operand after filtering the
   operator-ref — that comment literally said "unary forms fall through",
   and they did, straight into the generic call-resolution fallback, which
   has no notion of a bare operator-ref child and emitted garbled code
   (`vydir(operator-)`, matching OCCT's own `vydir = -vydir;` idiom).
   Fixed by handling the 1-operand case for `+`/`-` (the only two tokens
   valid as unary operators in this set) with a plain `HirUnaryOp`.

2. **`(*this) = expr;` — the "replace my whole value via copy-assign"
   idiom (OCCT's `gp_Quaternion::Multiply`: `(*this) = Multiplied(theOther);`
   ) — has no identifier `_decl_name` can extract from a dereference**, so
   it fell through the existing `operator=` call-site handling (built for a
   bare-name or `obj.field` target) entirely, reaching the same generic
   call-resolution fallback and emitting a raw, un-parseable `operator=`
   token reference. Fixed by recognizing `*this`/`(*this)` (however
   parenthesized/unexposed-wrapped; `_is_this_deref` in `tokens.py`) as the
   target "self", producing a plain `self = expr` reassignment. That alone
   wasn't enough for a *correct* compile, though: reassigning `self`
   outright needs `mut self` in Mojo exactly like assigning one of its
   fields does, but neither the LIR-level `_mutates_self` (`emit.py`) nor
   its MIR-level counterpart (`_mir_mutates_self`, needed one IR tier
   earlier for parameter-mutability decisions — see Follow-up 3) checked
   for a whole-`self` reassignment, only field/subscript assigns and
   mutating method calls. Fixed both to also recognize a reassignment whose
   target is literally `"self"`.

**Result:** `unexpected token in expression` drops to 0. Total error count
for `gp_Pnt.cxx --include-impls --verify` is now ~18, down from ~21 at the
start of this follow-up and ~136 at the start of Follow-up 3.

## Follow-up 7: `this == &other` is a pointer check, not a value comparison

The last single-instance error, `'gp_Quaternion' does not implement the
'__eq__' method`, traced to a real semantic bug rather than a missing
feature: `if (this == &theOther) { return true; }` (OCCT's
`gp_Quaternion::IsEqual`'s classic self-assignment/self-comparison guard,
before falling back to a real value comparison) is a raw **pointer-identity**
comparison in C++ (`this` and `&theOther` are both `gp_Quaternion*`) — not a
call to the type's `operator==`. But this frontend's existing address-of/
`this`-to-`self` lowering (`&x` drops the indirection to just `x`; `this`
becomes `self`) collapses both sides to plain names before the comparison
is even built, so it silently became a **value** comparison,
`self == theOther` — a different check, and a compile error outright for
any type (like `gp_Quaternion`) with no `operator==` defined.

This engine doesn't model pointer/reference identity at all (parameters
are handled by value/borrow, never tracked aliases), so there's no direct
translation available. But every real-world instance of this idiom is
followed by a full value-comparison fallback that already computes the
correct result regardless of whether the identity fast path is taken —
fixed by detecting the `this == &X` / `&X == this` shape directly on the
cursors (`_is_this_expr`/`_is_addr_of_expr` in `core.py`, checked before
either side is converted) and dropping it to a boolean literal (`False`
for `==`, `True` for `!=`): the identity fast path is simply never taken,
correctness is preserved by the fallback that already exists, and only a
micro-optimization is lost.

**Result:** `'gp_Quaternion' does not implement the '__eq__' method`
disappears from `gp_Pnt.cxx --include-impls --verify`'s error list.

## Follow-up 8: multi-dimensional array types, and `std::swap` on the same container

The largest remaining bucket, `invalid use of mutating method on rvalue of
type 'Float64'` (6 of the ~18 errors left, all from `gp_Mat::Transpose`'s
`std::swap(myMat[0][1], myMat[1][0])`), was previously scoped and
deliberately deferred as "SIMD-`swap()`-specific mutability" in Follow-up
2. Chasing it down surfaced two distinct, general bugs.

1. **Multi-dimensional native array types collapsed to a single flat
   `list[T]`, silently dropping every dimension after the first.**
   `_type_text`'s array-type-text heuristic (`types.py`) detects an array
   spelling by slicing at the *first* `[` (`"double[3][3]"` → head
   `"double"`), which correctly identifies the element type but then wraps
   it in exactly *one* `list[...]` layer regardless of how many bracket
   groups followed — so OCCT's `gp_Mat::myMat` (`double myMat[3][3];`, a
   real 3×3 matrix) got the field annotation `List[Float64]` instead of
   `List[List[Float64]]`. Ordinary `self.myMat[i][j]` reads/writes still
   happened to compile regardless (Mojo doesn't strictly enforce the
   stated `var` annotation against the initializer's own inferred type),
   which is why this had stayed hidden — it only surfaced where the
   *declared* element type mattered for resolving a generic call's type
   parameter (see point 2). Fixed by wrapping in one `list[...]` layer per
   bracket group (`cleaned.count("[")`) instead of exactly one.

2. **`std::swap(a, b)` was passed straight through as a call to Mojo's own
   `swap()` builtin**, which — once point 1 correctly resolved `self.myMat[0]`
   as `List[Float64]` rather than a bare `Float64` — rejects two arguments
   that alias the same underlying container outright: "argument of 'swap'
   call allows writing a memory location previously writable through
   another aliased argument". This isn't a Mojo quirk to route around with
   more type-inference precision — it's Mojo's aliasing-exclusivity model
   correctly refusing two simultaneous mutable borrows into the same
   value, and `std::swap(myMat[0][1], myMat[1][0])` (two elements of the
   *same* matrix) does exactly that. No amount of correct typing fixes a
   call shaped this way; the call itself has to change. Fixed by
   desugaring `std::swap(a, b)` (detected via `cursor.referenced.
   semantic_parent.spelling == "std"`, not just the bare spelling `"swap"`,
   since that's also an unrelated, real OCCT method name elsewhere) directly
   into a manual temp-variable swap (`tmp = a; a = b; b = tmp;`) at the C++
   frontend level — this sidesteps Mojo's aliasing check entirely (no two
   arguments ever passed to the same call) and is correct for every target,
   not a Mojo-specific workaround.

**Result:** `invalid use of mutating method on rvalue of type 'Float64'`
drops to 0. Total error count for `gp_Pnt.cxx --include-impls --verify` is
now ~11, down from ~18 at the start of this follow-up and ~136 at the
start of Follow-up 3.

## Follow-up 9: an uninitialized fixed-size array's libclang children are its bounds, not its value

The `cannot implicitly convert 'IntLiteral[N]' value to 'List[...]'`
errors (3 of the ~11 remaining) traced to a libclang AST quirk specific to
declaring a native fixed-size array with **no initializer at all**
(OCCT's `gp_Trsf::InitFromJson`: `double mymatrix[3][3];`, a scratch buffer
filled in by the following lines, not initialized up front). Such a
`VAR_DECL` still reports two `INTEGER_LITERAL` *children* — but they're the
array's own **dimension-size expressions** (`3`, `3`), not an initializer;
there is no `=` or `{` anywhere in the declaration. `_convert_var_decl`'s
generic "the last child is the initializer" fallback doesn't know that and
misread the trailing dimension-size literal as one, emitting `var
mymatrix: List[List[Float64]] = 3` — a bare integer assigned to a matrix-
typed variable.

Fixed by checking for a real initializer *first* (an `=` or `{` token
anywhere in the declaration's own tokens) whenever the type is one of the
array kinds, and when there isn't one, zero-filling a nested list literal
that matches the array's actual shape (`_default_array_value`, recursing
through `ci.Type.element_count` for each dimension) instead of consulting
the children at all.

**Result:** `cannot implicitly convert 'IntLiteral[N]' value to
'List[...]'` drops to 0. Total error count for `gp_Pnt.cxx --include-impls
--verify` is now ~8, down from ~11 at the start of this follow-up and ~136
at the start of Follow-up 3.

## Follow-up 10: `_resolved_ty` never resolved a method call's own return type

The last single-instance `ImplicitlyCopyable` error, `var XYZ: gp_XYZ =
A1.Direction().coord`, traced to a gap in the copy-insertion type
resolution added back in Follow-up 3. `_resolved_ty` (`mir_to_mojo_lir.py`)
recurses through a `MirFieldAccess`'s receiver to resolve a field's type
when `infer_types` didn't fill one in — but it only handled a *field*
receiver, never a *method call* receiver: `A1.Direction().coord`'s outer
field access recurses into `A1.Direction()` (a `MirMethodCall`), and
`_resolved_ty` had no case for that shape at all, since `hir_to_mir` never
sets `ty=` on a `MirMethodCall` to begin with (no method-signature table
available to it, mirroring the field case). The chain came back
`UnknownT()`, no `.copy()` got inserted, and `var XYZ: gp_XYZ = ...`
handed a bare reference into a fresh `gp_XYZ`-typed declaration.

Fixed by adding a `_method_return_types` table (struct name → method name
→ declared return type, populated in `lower_module` alongside the existing
`_field_types`) and a matching case in `_resolved_ty`: a `MirMethodCall`
resolves its receiver's type first, then looks up the callee method's
return type on that struct — the same one-level-at-a-time recursion
`_resolved_ty` already uses for field access, extended to cover a
method-call link in the same receiver chain.

**Result:** `value of type 'gp_XYZ' cannot be implicitly copied` drops to
0. Total error count for `gp_Pnt.cxx --include-impls --verify` is now ~7,
down from ~8 at the start of this follow-up and ~136 at the start of
Follow-up 3. What remains (`TCollection_AsciiString`, an OCCT-specific
external type this pilot's preamble doesn't stub; a
pointer-to-contiguous-fields idiom in `gp_XYZ::GetData`/`ChangeData`
returning `&x` as if it were `double*` into a 3-element buffer, with no
faithful representation in a value-oriented target without unsafe raw
pointer modeling) is narrower, more OCCT/idiom-specific, and a reasonable
stopping point for this pass.

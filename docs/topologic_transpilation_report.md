# Topologic → Mojo / Python-lift transpilation test (issue #79)

Test run against the local Topologic C++ source at
`C:/Users/amd/Documents/GitHub/Topologic` (the user's own clone; not cloned
again). Engine under test: the `transpilers` repo on branch
`issue-79-topologic-mojo`. Pure-Python pipeline, no Mojo toolchain installed
on this host, so the compile stage is **SKIP / unverified** (auto-skip, as
documented in `README.md`).

## Subset chosen

Topologic is a real-world CAD/geometry-kernel library on top of OpenCASCADE
(OCCT). `TopologicCore/src/` has 42 `.cpp` files (~11k LOC) plus headers; the
god-classes `Topology.cpp` (3.5k LOC) and `Graph.cpp` (2.3k LOC) are
outliers. Rather than try the whole tree in one shot, this run used a
representative 17-file subset spanning the code's structure:

- **Leaf utilities / value wrappers** — `Bitwise.cpp`, `Dictionary.cpp`,
  `Context.cpp`, `DoubleAttribute.cpp`, `IntAttribute.cpp`,
  `StringAttribute.cpp`, `ListAttribute.cpp` (small, self-contained logic).
- **Geometry base + OCCT-typed leaves** — `Geometry.cpp` (abstract base,
  `shared_ptr` typedef), `Line.cpp`, `Surface.cpp` (`Handle(Geom_*)`).
- **One class per level of the topology hierarchy** — `Vertex.cpp`,
  `Edge.cpp`, `Wire.cpp`, `Face.cpp`, `Shell.cpp`, `Cell.cpp`,
  `Cluster.cpp`.

This mirrors the `topologic_migration.md` characterisation: the codebase is
uniformly multi-file (`.h`/`.cpp` split), uniformly `namespace TopologicCore`,
uniformly `std::shared_ptr`-heavy, and — critically — *every* header pulls in
`Utilities.h`, which bundles the `TOPOLOGIC_API` export macro together with an
unrelated `TopoDS_Shape`-typed comparator, so even a "leaf" file inherits an
OCCT dependency it never otherwise touches.

## Strict engine (C++ → Mojo)

Method: `resolve_local_includes(file, include_dirs=[TopologicCore/include])`
then `transpile(src, source_lang="cpp", target="mojo")`, matching the
`transpile-levels --level folder --inc ...` path.

Result, **without** any OCCT shim:

- **Emitted: 0 / 17. Failed: 17 / 17.**
- Failure buckets (all 17): **`occt-external-type`** — `unknown type name
  'TopoDS_Shape'` / `unknown type name 'TopAbs_ShapeEnum'` (transitively
  inherited via `Utilities.h`). Secondary errors only appear past that
  (`no member named 'begin' in 'TopologicCore::Dictionary'`, `no member named
  'invalid_argument' in namespace 'std'`).

Result, **with** `docs/occt_preamble.hpp` supplied via
`TRANSPILERS_CPP_PREAMBLE_FILE` (the shim that declares `TopoDS_Shape` /
`TopAbs_ShapeEnum` as opaque types):

- **`Bitwise.cpp` emits cleanly** (1891 bytes of Mojo) — it parses completely
  and reaches emission, the same result documented in
  `topologic_migration.md`.
- **5 files fail on `class member TYPEDEF_DECL`** — `IntAttribute`,
  `DoubleAttribute`, `ListAttribute`, `Geometry`, `Surface`. A nested type
  alias (`typedef std::shared_ptr<X> Ptr;` inside a class body) is not yet
  modeled by the frontend. This is the *next* wall after OCCT, and it is a
  general engine gap, not a Topologic quirk.
- **`Vertex.cpp`** fails on `no member named 'runtime_error' in namespace
  'std'` (a `std::` exception type missing from the parser preamble).
- **`Line.cpp` / other geometry-typed files** fail on `unknown type name
  'Geom_Line'` etc. — the shim only covers the `TopoDS_*` macro-coupling
  pattern, not the ~111 distinct OCCT headers the corpus actually uses.

## Lift engine (C++ → Python)

Method: `lift_source(resolve_local_includes(file, inc=[...]), inc=[...])`,
the never-refuse whole-file C++→Python path.

Result: **2518 nodes, 14 `TODO[lift]` stubs → 99.4% mechanical.**

Per-file stub counts (all but two files are 100% mechanical):

| file | nodes | TODO | mechanical |
|---|---|---|---|
| Bitwise / DoubleAttribute / IntAttribute / StringAttribute / ListAttribute / Context / Geometry / Line / Surface | — | 0 | 100% |
| Dictionary.cpp | 150 | 1 | 99.3% |
| Vertex.cpp | 248 | 1 | 99.6% |
| Edge.cpp | 342 | 3 | 99.1% |
| Wire.cpp | 211 | 2 | 99.1% |
| Face.cpp | 478 | 1 | 99.8% |
| Shell.cpp | 206 | 1 | 99.5% |
| Cell.cpp | 395 | 2 | 99.5% |
| Cluster.cpp | 292 | 3 | 99.0% |

**The single dominant stubbed construct is `CXX_TRY_STMT`** — OCCT's pervasive
`try { ... } catch (Standard_Failure) { ... }` exception blocks. The lift
engine does not translate try/catch and emits `pass  # TODO[lift]:
CXX_TRY_STMT :: ?` for each one. Every non-trivial OCCT-calling file (Vertex
and up the hierarchy) carries these; the leaf utility/value files have none.

## Engine fix made in this branch

Running the pure-Python C++ frontend on this Windows host surfaced a
cross-platform bug that was blocking **100% of C++ inputs** (not just
Topologic):

- `src/transpilers/frontends/cpp/parser/preprocess.py`'s `PARSER_PREAMBLE`
  hardcoded `size_t` and the `operator new`/`delete` declarations as
  `unsigned long`. On Windows libclang's `size_t` is `unsigned long long`
  (LLP64), so the redeclaration of `operator new` collided and libclang
  emitted two severity-3 (fatal) diagnostics for **every** translation unit —
  `_check_diagnostics` then rejected everything with "libclang parse errors".
- Fix: declare them with the compiler's own `__SIZE_TYPE__` macro, which
  resolves to the correct `size_t` on any target.

Validation: before the fix, the C++ test files produced **68 failures**;
after, **5 failures** — and all 5 remaining are environmental (4 require the
`mojo` binary, 1 requires a `clang` binary), not regressions. The change is
scoped to the parser preamble and is covered by the existing C++ suite.

## Top blockers to real C++ → Mojo of Topologic

1. **OpenCASCADE (OCCT) is not available as a Mojo target binding.** The
   corpus uses ~111 distinct OCCT headers and ~400 uses of `TopoDS_Shape`
   alone. `docs/occt_preamble.hpp` stubs only the two types involved in
   Topologic's macro-coupling, so files whose *own* logic calls the OCCT API
   (`Geom_*`, `BRepBuilderAPI_*`, `gp_*`, ...) cannot compile. This is the
   dominant, unbounded blocker — it needs a real OCCT→Mojo binding, a project
   on the scale of the engine itself, not a bug fix.
2. **`class member TYPEDEF_DECL`** (nested `typedef std::shared_ptr<T> Ptr;`)
   is unmodeled — blocks 5 of 17 subset files even past the OCCT shim. A
   general, bounded fix.
3. **Missing `std::` exception types** in the parser preamble
   (`std::runtime_error`, `std::invalid_argument`, ...) — a small, bounded
   fix (add them to `PARSER_PREAMBLE`).
4. **OCCT try/catch (`CXX_TRY_STMT`)** is a hard lift-engine stub and,
   implicitly, an unmodeled construct in the strict path too.
5. **Compile stage unverified on this host** (no `mojo` installed); the
   emit-level results above are the strict engine's ceiling here.

## Files

- `docs/topologic_transpilation_report.md` — this report.
- `docs/topologic_sweep.json` — machine-readable sweep output.
- `scripts/sweep_topologic.py` — the diagnostic sweep driver (subset, both
  engines, bucket classification); accepts an optional path arg / `$TOPOLOGIC_ROOT`.
- `src/transpilers/frontends/cpp/parser/preprocess.py` — the `size_t` /
  `operator new` preamble fix.

## Candidate follow-up issues

- **Bounded:** model `class member TYPEDEF_DECL` (nested type alias) in the
  C++ frontend.
- **Bounded:** add common `std::` exception classes to `PARSER_PREAMBLE`.
- **Bounded:** strict-engine handling (or explicit `TODO[port]` hole) for
  `CXX_TRY_STMT` / catch of OCCT `Standard_Failure`.
- **Large / out of scope for a bug-fix pass:** a real OCCT (`gp`-package
  first, per prior direction) Mojo binding; see the extended analysis in
  `docs/topologic_migration.md`.

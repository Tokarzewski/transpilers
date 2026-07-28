// Preamble for transpiling OCCT's `gp` package (basic geometric primitives:
// gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, ...) — see docs/topologic_migration.md.
//
// Unlike docs/occt_preamble.hpp (which stubs a couple of OCCT types as
// opaque placeholders just to get *past* them), this preamble targets
// actually transpiling the gp package's own real logic: it has zero
// inheritance and no Handle(T)/Standard_Transient reference counting, just
// plain value types with real math method bodies — exactly what the
// strict engine's current C++ subset can handle.
//
// Usage:
//     TRANSPILERS_CPP_PREAMBLE_FILE=docs/occt_gp_preamble.hpp \
//         transpile gp_Pnt.cxx --source cpp --target mojo \
//         -I path/to/occt/src/FoundationClasses/TKMath/gp --include-impls --verify
//
// DEFINE_STANDARD_ALLOC is OCCT's custom-allocator operator new/delete
// macro (memory-pool bookkeeping) -- semantically irrelevant to a target
// with its own memory model, and safe to expand to nothing: it only ever
// adds member operators, never changes a class's field layout or method
// set. Defining it empty here relies on the *real* preprocessor macro
// expansion (preprocess.py's `clang -E` step), not the parser preamble --
// this text runs before that step, same as any other -D.
#define DEFINE_STANDARD_ALLOC

typedef int Standard_Integer;
typedef double Standard_Real;
typedef bool Standard_Boolean;
typedef const char* Standard_CString;
typedef unsigned int Standard_ExtCharacter;

// Only used by Dump/InitFromJson (serialization) methods, not core
// geometry -- opaque is enough since we don't attempt to transpile those.
class Standard_OStream {};
class Standard_SStream {};
// Same rationale: a different OCCT module's string type and dump-parsing
// helper, both only ever referenced inside InitFromJson bodies (see the
// OCCT_DUMP_*/OCCT_INIT_VECTOR_CLASS no-op macros below for the rest of
// that surface). A trivial stub is enough for these call sites to
// type-check; the values are never used for anything real downstream.
class TCollection_AsciiString
{
public:
  TCollection_AsciiString() {}
  TCollection_AsciiString(Standard_CString) {}
};
class Standard_Dump
{
public:
  static TCollection_AsciiString Text(const Standard_SStream&) { return TCollection_AsciiString(); }
};

// Runtime-assertion macros (raise a C++ exception if CONDITION holds).
// Upstream OCCT itself already expands these to nothing in a release
// (No_Exception / No_Standard_*) build -- see e.g. Standard_OutOfRange.hxx.
// Exceptions aren't part of this engine's modeled C++ subset regardless, so
// dropping the check entirely (rather than, say, keeping the condition
// as a dead `if`) matches release-build OCCT semantics most closely.
#define Standard_OutOfRange_Raise_if(CONDITION, MESSAGE)
#define Standard_ConstructionError_Raise_if(CONDITION, MESSAGE)
#define Standard_DomainError_Raise_if(CONDITION, MESSAGE)
#define Standard_NullValue_Raise_if(CONDITION, MESSAGE)
#define Standard_NoSuchObject_Raise_if(CONDITION, MESSAGE)
#define Standard_DivideByZero_Raise_if(CONDITION, MESSAGE)

// `DEFINE_STANDARD_EXCEPTION(Name, Base)` is OCCT's macro for generating an
// exception class's full boilerplate (constructors, RTTI registration,
// Raise() helpers, ...) -- irrelevant here since exceptions aren't part of
// this engine's modeled C++ subset (see the _Raise_if macros above, which
// already drop the `throw` sites this would otherwise construct for). A
// trivial empty class is enough for the type to exist syntactically
// wherever it's named (e.g. gp_Vec.hxx declares `gp_VectorWithNullMagnitude`
// this way, purely so its own release-build-dead Raise_if macro type-checks).
// Deliberately *not* modeled as `class C1 : public C2 {}`: the strict
// engine doesn't support C++ inheritance yet, and every one of these is
// dead code once the corresponding _Raise_if macro is a no-op -- the real
// base-class relationship is never actually exercised.
#define DEFINE_STANDARD_EXCEPTION(C1, C2) class C1 {};
// gp_VectorWithNullMagnitude.hxx (like other real OCCT headers this
// preamble's -I inlines) defines its own _Raise_if macro conditionally on
// whether `No_Exception` is set, exactly matching upstream's own
// release-build toggle -- setting it here gets us that no-op for free,
// consistently, without needing to chase down and override each such
// macro name by hand as new ones are inlined.
#define No_Exception

// The `_Raise_if` macros above cover conditionally-thrown exceptions;
// gp's .cxx implementation files (pulled in via --include-impls) also
// throw a couple of these directly and unconditionally on a genuine error
// path (e.g. `gp_GTrsf::SetForm` on a null determinant) rather than
// through a Raise_if macro, so the type still needs to exist even though
// exceptions themselves are out of scope. A constructor accepting the
// same message argument these call sites pass is enough for `throw
// Standard_ConstructionError("...")` to type-check.
class Standard_ConstructionError
{
public:
  Standard_ConstructionError(Standard_CString) {}
};

// `OCCT_DUMP_VECTOR_CLASS(stream, name, n, ...)` formats a DumpJson() body,
// and `OCCT_INIT_VECTOR_CLASS(text, name, pos, n, ...)` parses one back in
// InitFromJson(); serialization isn't attempted (see the
// Standard_OStream/SStream note above), so no-ops keep their call sites
// syntactically valid without needing to model JSON I/O at all. Because a
// no-op macro's arguments are discarded by the preprocessor before any
// semantic check, this also means whatever real (undeclared) types appear
// only inside these calls -- e.g. `Standard_Dump::Text(...)` -- never need
// stubbing themselves.
#define OCCT_DUMP_VECTOR_CLASS(...)
#define OCCT_INIT_VECTOR_CLASS(...)
#define OCCT_DUMP_FIELD_VALUE_NUMERICAL(...)
#define OCCT_DUMP_CLASS_BEGIN(...)
#define OCCT_DUMP_FIELD_VALUES_DUMPED(...)
#define OCCT_DUMP_BASE_CLASS(...)
#define OCCT_INIT_FIELD_VALUE_INTEGER(...)
#define OCCT_INIT_FIELD_VALUE_REAL(...)

// Method-template converter to NCollection's (separate-module) 4x4 matrix
// type (`gp_Trsf::GetMat4`); opaque is enough since we never call it or
// inspect its fields, only need the name to resolve for the declaration to
// parse -- same rationale as the Mojo backend's auto-emitted foreign-struct
// placeholders (see docs/topologic_migration.md).
template <typename T> class NCollection_Mat4 {};

// === TRANSPILERS: REAL PREAMBLE BELOW ===
// Everything above this marker is parse-only scaffolding (opaque stubs,
// typedefs, no-op macros) that must never itself appear in the emitted
// target -- see core.py's _PREAMBLE_REAL_MARKER. Everything below it is a
// small set of real, correct helpers that gp's real headers/impls actually
// call at real call sites (not just inside a now-no-op macro), so it has
// to be converted and emitted like ordinary user code, or every one of
// those call sites is "use of unknown declaration" in the output despite
// parsing and type-checking fine.

// Small tolerance/parity helpers, real bodies (the exact tolerance values
// don't affect whether this transpiles, only whether a downstream
// behavioral comparison against real OCCT would match).
constexpr double RealSmall() { return 1e-16; }
constexpr double Epsilon(double) { return 1e-16; }
constexpr bool IsOdd(int theValue) { return (theValue % 2) != 0; }
constexpr bool IsEven(int theValue) { return (theValue % 2) == 0; }

// `Precision.hxx` (a different OCCT module) provides tolerance constants
// used by-value throughout `gp`, always via a qualified static call
// (`Precision::Angular()`) -- the real values are the canonical OCCT
// defaults (see TKMath/Precision/Precision.cxx) since these DO affect
// behavioral verification of the transpiled arithmetic, unlike the
// don't-care tolerance helpers above.
class Precision
{
public:
  static constexpr double Confusion() { return 1e-7; }
  static constexpr double Angular() { return 1e-12; }
};

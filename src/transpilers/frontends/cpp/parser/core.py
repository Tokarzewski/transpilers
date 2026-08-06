"""C++ source -> HIR via libclang.

Initial subset (deliberately C-like C++):
  - free functions with primitive params (int/long/short/char/bool/float/double/void)
  - return, if/else, while, range-based for (C-style with ++)
  - declarations and assignments
  - binary / comparison / logical ops, unary not/neg
  - integer literals, names, calls

Out of scope for the initial slice: classes, templates, references, namespaces,
auto, `std::` types. These are real C++ — the IR doesn't model them yet and we
refuse rather than emit broken code.

Operator extraction uses tokens (not libclang's BinaryOperator extension), so
this works against any reasonably recent libclang.
"""

from __future__ import annotations

import os as _os
import re as _re

import clang.cindex as ci

from transpilers.ir import hir

from .errors import UnsupportedConstruct
from .libclang_config import *  # noqa: F401,F403  (_system_include_args, _host_triple, ...)
from .types import *  # noqa: F401,F403  (CPP type aliases + _type_text)
from .tokens import *  # noqa: F401,F403  (op sets + token/loc helpers)
from .simd import _lift_simd_intrinsic
from .preprocess import preprocess_cpp
from .type_extractor import TypeGroundTruth

CursorKind = ci.CursorKind

_TypeKind = ci.TypeKind
_INTEGRAL_KINDS = frozenset(
    getattr(_TypeKind, k)
    for k in (
        "BOOL",
        "CHAR_U",
        "UCHAR",
        "CHAR16",
        "CHAR32",
        "USHORT",
        "UINT",
        "ULONG",
        "ULONGLONG",
        "UINT128",
        "CHAR_S",
        "SCHAR",
        "WCHAR",
        "SHORT",
        "INT",
        "LONG",
        "LONGLONG",
        "INT128",
        "ENUM",
    )
    if getattr(_TypeKind, k, None) is not None
)
_FLOATING_KINDS = frozenset(
    getattr(_TypeKind, k)
    for k in ("FLOAT", "DOUBLE", "LONGDOUBLE", "FLOAT128")
    if getattr(_TypeKind, k, None) is not None
)


def _is_integral_type(t) -> bool:
    try:
        return t.get_canonical().kind in _INTEGRAL_KINDS
    except Exception:
        return False


def _is_floating_type(t) -> bool:
    try:
        return t.get_canonical().kind in _FLOATING_KINDS
    except Exception:
        return False


def _operand_is_floating(cursor) -> bool:
    # clang inserts implicit-cast UNEXPOSED/PAREN wrappers whose type already
    # reflects the destination (e.g. `int`), so the cast operand's own type
    # lies. Drill through transparent wrappers to the real value's type.
    cur = cursor
    for _ in range(8):
        if _is_floating_type(cur.type):
            return True
        if cur.kind not in (CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR):
            break
        kids = [k for k in cur.get_children() if k.kind != CursorKind.TYPE_REF]
        if not kids:
            break
        cur = kids[-1]
    return _is_floating_type(cur.type)


_POSTFIX_EFFECT_STACK: list[list["hir.HirNode"]] = []


def _push_postfix_frame() -> None:
    _POSTFIX_EFFECT_STACK.append([])


def _pop_postfix_frame() -> "list[hir.HirNode]":
    return _POSTFIX_EFFECT_STACK.pop() if _POSTFIX_EFFECT_STACK else []


def _record_postfix_effect(node: "hir.HirNode") -> None:
    if _POSTFIX_EFFECT_STACK:
        _POSTFIX_EFFECT_STACK[-1].append(node)


INPUT_NAME = "input.cpp"

# Prepended preamble so single-function snippets extracted for migration parse
# without their original #includes. These declarations let libclang resolve
# common <cmath> calls, <cstdint> types, and min/max; the top-level loop skips
# TYPEDEF_DECL and non-definition FUNCTION_DECL, so they never reach the output.
_PREAMBLE = """
typedef signed char int8_t; typedef short int16_t; typedef int int32_t;
typedef long long int64_t; typedef unsigned char uint8_t;
typedef unsigned short uint16_t; typedef unsigned int uint32_t;
typedef unsigned long long uint64_t; typedef unsigned long long size_t;
typedef long ssize_t; typedef long ptrdiff_t;
double exp(double); double log(double); double log10(double); double log2(double);
double sqrt(double); double cbrt(double); double pow(double, double);
double sin(double); double cos(double); double tan(double);
double asin(double); double acos(double); double atan(double); double atan2(double, double);
double sinh(double); double cosh(double); double tanh(double);
double fabs(double); double ceil(double); double floor(double); double round(double);
double trunc(double); double fmod(double, double); double hypot(double, double);
double fmin(double, double); double fmax(double, double); double fma(double, double, double);
double max(double, double); double min(double, double);
int max(int, int); int min(int, int); int abs(int); long labs(long);
double clamp(double, double, double); int clamp(int, int, int);
namespace std {
}
"""


def _project_preamble() -> str:
    """Project-specific declarations to prepend (e.g. EnergyPlus `Real64`), so
    domain typedefs resolve without baking them into the general core preamble.
      $TRANSPILERS_CPP_PREAMBLE       — inline declaration text
      $TRANSPILERS_CPP_PREAMBLE_FILE  — path to a declarations file
    """
    inline = _os.environ.get("TRANSPILERS_CPP_PREAMBLE", "")
    path = _os.environ.get("TRANSPILERS_CPP_PREAMBLE_FILE", "")
    if path and _os.path.isfile(path):
        try:
            inline += "\n" + open(path, encoding="utf-8").read()
        except OSError:
            pass
    return ("\n" + inline + "\n") if inline else ""


# A project preamble is usually *parse-only* scaffolding (opaque stub
# classes, typedefs, macros) that must exist for libclang to get past a
# type it can't see the real declaration of, but that should never itself
# appear in the emitted target -- e.g. `class Standard_OStream {};` is
# never meant to become a real Mojo struct. But some projects also need a
# handful of small *real* helpers with correct bodies (e.g. OCCT's
# `RealSmall()`/`Epsilon()`/`IsOdd()` tolerance/parity functions, called
# for real from headers/impls the -I resolver pulls in) -- these DO need
# to be emitted, or every call site is "use of unknown declaration" in the
# output even though the call parsed and type-checked fine. A preamble
# marks the split with this literal line; everything at or after it is
# treated as ordinary user code (converted and emitted normally) instead
# of being excluded like the rest of the preamble.
_PREAMBLE_REAL_MARKER = "// === TRANSPILERS: REAL PREAMBLE BELOW ==="


def _split_project_preamble() -> tuple[str, str]:
    """Split ``_project_preamble()`` into ``(stub_part, real_part)`` at
    ``_PREAMBLE_REAL_MARKER``. ``real_part`` is ``""`` if the preamble has
    no marker (the common case -- most project preambles are parse-only).

    ``real_part`` must land in the final translation unit *after*
    ``PARSER_PREAMBLE`` (the generic std:: shim, also excluded, and itself
    only added later inside ``preprocess_cpp``) and immediately before the
    user's own source -- not where it textually sits in the preamble file,
    right before ``PARSER_PREAMBLE``. ``parse_cpp`` prepends it directly to
    *source* for exactly this reason; see the call site there.
    """
    project = _project_preamble()
    marker_at = project.find(_PREAMBLE_REAL_MARKER)
    if marker_at == -1:
        return project, ""
    return project[:marker_at], project[marker_at:]


# Number of lines occupied by everything that must stay excluded from user
# HIR: the parse-only project preamble stub part, plus PARSER_PREAMBLE
# (added after it -- see _split_project_preamble and the parse_cpp call
# site). Cursors with ``location.line <= this`` come from that excluded
# region; cursors after it (the preamble's own real part, if any, followed
# by the user's actual source) are ordinary user code. Cursors at exactly
# this line come from the leading newline of the user source, which is
# harmless to skip.
def _compute_user_first_line() -> int:
    from .preprocess import PARSER_PREAMBLE

    stub_part, _real_part = _split_project_preamble()
    return len((stub_part + PARSER_PREAMBLE).splitlines())


_UNKNOWN_TYPE_NAME_RE = _re.compile(r"unknown type name '([A-Za-z_][A-Za-z0-9_]*)'")
_SCREAMING_MACRO_RE = _re.compile(r"^[A-Z][A-Z0-9_]*$")
# Some real-world projects (e.g. OCCT's `Standard_EXPORT`) name their export
# macro with a mixed-case prefix, only the *suffix* is the SCREAMING-CASE
# convention. A real type name ending in one of these is essentially
# inconceivable, so matching on suffix alone is still safe.
_MACRO_SUFFIX_RE = _re.compile(r"_(EXPORT|IMPORT|API|DLL|DECL)$")


def _macro_like_unknown_types(tu: ci.TranslationUnit) -> set[str]:
    """Names from "unknown type name '...'" diagnostics that look like an
    export / calling-convention macro (``FOO_API``, ``FOO_EXPORT``,
    ``DLLEXPORT``, ``WINAPI``, ``Standard_EXPORT``, ...) rather than a real
    missing type.

    Real-world C++ libraries almost universally gate their public API with
    a macro like this, defined in a header we never see (every #include is
    stripped before libclang runs, by design -- see ``preprocess.py``). Left
    alone, every function/class declaration in the file fails to parse on a
    token that carries no type information at all. SCREAMING_CASE (or a
    SCREAMING_CASE suffix on an otherwise-mixed-case name) is a near-
    universal C/C++ convention for macros and essentially never used for a
    real type name, so treating it as an empty ``-D`` define is safe: it can
    only remove a no-op qualifier, never rename or reinterpret real user
    code.
    """
    names: set[str] = set()
    for d in tu.diagnostics:
        if d.severity < ci.Diagnostic.Error:
            continue
        m = _UNKNOWN_TYPE_NAME_RE.search(d.spelling)
        if not m:
            continue
        name = m.group(1)
        if _SCREAMING_MACRO_RE.match(name) or _MACRO_SUFFIX_RE.search(name):
            names.add(name)
    return names


def parse_cpp(source: str):
    """Parse C++ source to HIR.

    Returns ``(hir_module, ground_truth)`` where *ground_truth* is a
    ``TypeGroundTruth`` populated from the same clang AST. The HIR
    carries ``source_loc`` fields on each node so a later pass
    (``transpilers.passes.cpp_ground_truth``) can match each HIR
    declaration to its concrete type. For backward compatibility
    callers that ignore the second value keep working; the legacy
    single-return shape is preserved as a private ``_parse_cpp_legacy``
    and any ``parse_cpp(source)[0]`` usage in the tests continues to
    pass.

    The pipeline threads the ground truth through to
    ``infer_types`` (see ``run_stages``); the legacy single-return
    behaviour is opt-in via the keyword arg below, but **new** code
    should always unpack the tuple.
    """
    index = ci.Index.create()
    # Pre-define the most common stdlib constants so LeetCode/competitive
    # files that use `NULL` / `INT_MIN` without including <cstddef>/<climits>
    # still parse. The values are the canonical platform-32-bit limits;
    # downstream type inference treats these as plain int constants.
    predefs = [
        "-DNULL=0",
        "-DINT_MIN=(-2147483647 - 1)",
        "-DINT_MAX=2147483647",
        "-DLONG_MIN=(-9223372036854775807L - 1)",
        "-DLONG_MAX=9223372036854775807L",
        "-DLLONG_MIN=(-9223372036854775807LL - 1)",
        "-DLLONG_MAX=9223372036854775807LL",
        "-DUINT_MAX=4294967295U",
        "-DEXIT_SUCCESS=0",
        "-DEXIT_FAILURE=1",
        # <cmath>/math.h's M_* constants are POSIX, not standard C++, but
        # ubiquitous in real-world math-heavy code (graphics, CAD, physics)
        # that assumes a POSIX libm rather than including them explicitly.
        "-DM_PI=3.14159265358979323846",
        "-DM_PI_2=1.57079632679489661923",
        "-DM_PI_4=0.78539816339744830962",
        "-DM_2_PI=0.63661977236758134308",
        "-DM_E=2.71828182845904523536",
        "-DM_SQRT2=1.41421356237309504880",
        "-DM_SQRT1_2=0.70710678118654752440",
        # <cfloat>/float.h, same rationale as the M_* constants above.
        "-DDBL_EPSILON=2.2204460492503131e-16",
        "-DFLT_EPSILON=1.19209290e-7F",
    ]
    triple_args = []
    triple = _host_triple()
    if triple:
        triple_args = [f"--target={triple}"]
    # C++20 is required to make `requires std::totally_ordered<T>` parse;
    # the preprocessor strips those clauses for the older libclang
    # behaviour, but the parser still uses c++20 so language features
    # like designated initializers work end-to-end.
    # Clang's default -ferror-limit=20 silently stops parsing once that many
    # diagnostics accumulate -- fine for small algorithm-corpus files, but a
    # real multi-file translation unit (thousands of inlined header lines,
    # see includes.py) can blow past 20 diagnostics before even reaching the
    # user's own code. Truncated parsing means truncated diagnostics too: the
    # macro-like-unknown-type retry loop below only ever sees whatever fit in
    # the first 20, so a file needing 5 distinct macro fixes could get stuck
    # rediscovering the same one or two forever, and _check_diagnostics's
    # final error report reflects an arbitrary prefix, not the real picture.
    parse_args = (
        ["-std=c++20", "-x", "c++", "-nostdinc++", "-ferror-limit=0"]
        + triple_args
        + predefs
    )
    # Project-specific declarations (EnergyPlus-style Real64) layer on
    # top of the parser preamble from preprocess_cpp. Both are no-ops
    # when empty. Must come BEFORE the user source (declare-before-use --
    # a prior version appended this after preprocess_cpp(source), i.e.
    # after the user's own code, so a type/typedef declared here could
    # never actually be used by that code; only worked for cases that
    # never exercised it, e.g. macro-only injections that the real `-E`
    # step expands positionally regardless of preamble placement).
    # The preamble's stub part goes before preprocess_cpp (ahead of even
    # PARSER_PREAMBLE, which preprocess_cpp adds internally) exactly as
    # before. Its *real* part (see _split_project_preamble/
    # _PREAMBLE_REAL_MARKER) instead goes directly onto the front of
    # *source*, so it lands after PARSER_PREAMBLE and immediately before
    # the user's own code -- matching _compute_user_first_line()'s
    # boundary, which only excludes stub_part + PARSER_PREAMBLE. Real
    # C++ code with no directives of its own to strip, so prepending it
    # pre-preprocessing (rather than post-, like stub_part) is harmless.
    _stub_preamble, _real_preamble = _split_project_preamble()
    preprocessed = _stub_preamble + preprocess_cpp(_real_preamble + source)
    # Retry with unresolved export/calling-convention macros (TOPOLOGIC_API,
    # DLLEXPORT, WINAPI, ...) neutralized -- see _macro_like_unknown_types.
    # Bounded by the number of distinct macro names actually seen, so this
    # always terminates; real parse errors just surface on the final attempt.
    defined_macros: set[str] = set()
    tu = None
    for _attempt in range(4):
        tu = index.parse(
            INPUT_NAME,
            args=parse_args + [f"-D{name}=" for name in sorted(defined_macros)],
            unsaved_files=[(INPUT_NAME, preprocessed)],
            options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
        new_macros = _macro_like_unknown_types(tu) - defined_macros
        if not new_macros:
            break
        defined_macros |= new_macros
    _check_diagnostics(tu)
    _KNOWN_STRUCT_NAMES.clear()
    # First pass: record struct/class names (recursing into namespaces) so
    # VAR_DECLs later resolve to HirStructInit instead of integer-default.
    _register_struct_names(tu.cursor.get_children())
    body: list[hir.HirNode] = []
    for c in tu.cursor.get_children():
        _convert_top_level(c, body)
    # Build the ground truth from the same TU. Only decls from the
    # user's input.cpp are recorded, so the table is small and stable
    # across calls.
    truth = TypeGroundTruth.from_tu(tu, only_input=True)
    return hir.HirModule(source_lang="cpp", body=body), truth


# Backward-compatibility shim: code that still calls ``parse_cpp(src)``
# expecting a single ``HirModule`` gets a transparent wrapper that
# discards the ground truth. New code should always unpack the tuple.
def _parse_cpp_legacy(source: str) -> hir.HirModule:
    return parse_cpp(source)[0]


def _register_struct_names(cursors) -> None:
    for c in cursors:
        if not _from_input(c):
            continue
        if c.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            _KNOWN_STRUCT_NAMES.add(c.spelling)
        elif c.kind == CursorKind.NAMESPACE:
            _register_struct_names(c.get_children())


def _convert_top_level(c: ci.Cursor, body: list[hir.HirNode]) -> None:
    """Dispatch one translation-unit- or namespace-level declaration into
    `body`. Namespaces are flattened — Mojo has no namespace, so members are
    emitted at module scope (name collisions across namespaces are a caller
    risk until qualified-name mangling lands)."""
    if not _from_input(c):
        return
    if c.kind == CursorKind.NAMESPACE:
        for inner in c.get_children():
            _convert_top_level(inner, body)
        return
    if c.kind == CursorKind.FUNCTION_DECL and c.is_definition():
        body.append(_set_loc(c, _convert_function(c)))
        return
    if (
        c.kind in (CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR)
        and c.is_default_method()
    ):
        # `= default` (in-class or, rarer, out-of-line `Class::Class(...) =
        # default;`): no explicit body to convert -- see the identical skip
        # in _convert_class's in-class handling for why this can't just be
        # covered by the `is_definition()` checks below (a defaulted special
        # member still reports is_definition() == True).
        return
    if c.kind == CursorKind.CXX_METHOD and c.is_definition():
        # Out-of-line member definition `Real64 Class::method(...) {...}`.
        # If its class is present in this TU, attach it to that struct as a real
        # method (with `self`), so `this->field` resolves. Otherwise (standalone
        # extraction) fall back to a free function — scalar methods use only their
        # explicit params; a `this->member` ref then becomes an unresolved name.
        cls = c.semantic_parent.spelling if c.semantic_parent else None
        for node in body:
            if isinstance(node, hir.HirStruct) and node.name == cls:
                node.methods.append(_convert_method(c, struct_name=cls))
                return
        body.append(_set_loc(c, _convert_function(c)))
        return
    if c.kind == CursorKind.CONSTRUCTOR and c.is_definition():
        # Out-of-line constructor `Class::Class(...) {...}` -- same
        # in-class-vs-declared-only split real headers use for constructors
        # as for regular methods above (a `Class(...);` prototype in the
        # class body, with the body defined later in the same header).
        cls = c.semantic_parent.spelling if c.semantic_parent else None
        for node in body:
            if isinstance(node, hir.HirStruct) and node.name == cls:
                node.methods.append(_convert_constructor(c, struct_name=cls))
                return
        return
    if c.kind == CursorKind.FUNCTION_TEMPLATE and c.is_definition():
        # The IR doesn't model templates, so the function body becomes a
        # HirRaw hole. The signature (param / return types) is still
        # recorded in the ground truth at the function's source location,
        # so a downstream instantiation at a call site can pick up the
        # real types.
        body.append(hir.HirRaw(snippet=_cursor_snippet(c)))
        return
    if c.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
        if not c.is_definition():
            # A forward declaration (`class gp_XYZ;`), not a definition. Real
            # multi-file/multi-header input (unlike the single hand-written
            # algorithm-corpus files this frontend originally targeted)
            # routinely forward-declares a type long before its real
            # definition -- e.g. so an earlier header can take it by
            # reference/pointer. Converting it too would append a second,
            # empty HirStruct for the same name alongside the real one,
            # which downstream passes have no way to merge: the emitted
            # target ends up with two conflicting `struct gp_XYZ` definitions
            # (a hard compile error on every backend that has one).
            return
        # An explicit full specialization (`template<> struct hash<T> {...}`,
        # the standard idiom for making a user type usable as an
        # unordered_map/unordered_set key) surfaces as a normal STRUCT_DECL
        # named after the *primary* template, whose first child is a bare
        # TYPE_REF to the specialized argument -- a shape no ordinary class
        # produces (a base-class reference lives inside a CXX_BASE_SPECIFIER,
        # never as a direct child). It's STL-interop plumbing for the type
        # above it, not that type's own logic, and often uses constructs
        # (e.g. `union`) outside this engine's modeled subset regardless; skip
        # it rather than emitting a same-named, disconnected struct.
        first = next(c.get_children(), None)
        if first is not None and first.kind == CursorKind.TYPE_REF:
            return
        # A class-nested enum (e.g. OCCT's `enum class D { X, Y, Z, ... }`
        # inside gp_Dir, used for constructing standard-axis directions)
        # hoists to the same module-level int constants a top-level enum
        # gets (_convert_enum below) -- references resolve by the bare
        # enumerator name regardless of nesting (DECL_REF_EXPR only ever
        # carries the unqualified spelling, see the DECL_REF_EXPR case in
        # _convert_expr), so a nested enum needs no different treatment
        # than a global one already gets. _convert_class skips ENUM_DECL
        # children since they're handled here instead.
        for member in c.get_children():
            if member.kind == CursorKind.ENUM_DECL:
                body.extend(_convert_enum(member))
        body.append(_set_loc(c, _convert_class(c)))
        return
    if c.kind == CursorKind.FUNCTION_DECL:
        return
    if c.kind == CursorKind.VAR_DECL:
        # Top-level globals (e.g. `const double KELVIN = 273.15`).
        # Translate to module-level assignments so functions can reference them.
        try:
            body.append(_convert_var_decl(c))
        except UnsupportedConstruct:
            pass
        return
    if c.kind == CursorKind.ENUM_DECL:
        # `enum { A = 0, B = 1 }` → one module-level int constant per
        # enumerator. Module-constant inlining (hir_to_mir) substitutes each
        # name with its value at use sites, so no enum IR node is needed and
        # the result flows to every backend.
        body.extend(_convert_enum(c))
        return
    if c.kind in (
        CursorKind.INCLUSION_DIRECTIVE,
        CursorKind.MACRO_DEFINITION,
        CursorKind.MACRO_INSTANTIATION,
        CursorKind.USING_DIRECTIVE,
        CursorKind.USING_DECLARATION,
        CursorKind.TYPEDEF_DECL,
        CursorKind.TYPE_ALIAS_DECL,  # `using Real64 = double;`
        # The parser preamble declares several `template <...> class X {}`
        # shadow declarations for the `std::` namespace. These have
        # `CLASS_TEMPLATE` / `FUNCTION_TEMPLATE` kinds; we want to skip
        # them at the top level (they only exist so the *user*'s code
        # can refer to e.g. `std::vector<T>`). `FUNCTION_TEMPLATE`
        # *definitions* are handled above (become HirRaw holes).
        CursorKind.CLASS_TEMPLATE,
        CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
        CursorKind.FUNCTION_TEMPLATE,
    ):
        # Type-alias declarations carry no runtime content; uses resolve via
        # libclang type canonicalization (`Real64` -> `double` -> Float64).
        return
    raise UnsupportedConstruct(f"top-level {c.kind.name}")


def _convert_enum(cursor: ci.Cursor) -> list[hir.HirNode]:
    """C-style `enum { A = 0, B = 1 }` → one module-level int constant per
    enumerator (libclang resolves implicit values via `enum_value`). These
    feed the module-constant table; references inline to their literal value
    on every target, so a dedicated enum IR node isn't required for the
    named-int case."""
    out: list[hir.HirNode] = []
    for c in cursor.get_children():
        if c.kind == ci.CursorKind.ENUM_CONSTANT_DECL:
            out.append(
                hir.HirAssign(
                    target=c.spelling,
                    value=hir.HirIntLiteral(value=int(c.enum_value)),
                    annotation="int",
                )
            )
    return out


def _convert_class(cursor: ci.Cursor) -> hir.HirStruct:
    """Subset: public fields (`int x;`), public methods. No constructors,
    destructors, inheritance, templates, operator overloads, references.
    Members under `private:` / `protected:` are refused so we don't silently
    expose internal API."""
    name = cursor.spelling
    fields: list[hir.HirParam] = []
    methods: list[hir.HirFunction] = []
    has_defaulted_default_ctor = False
    for c in cursor.get_children():
        if c.kind == ci.CursorKind.CXX_ACCESS_SPEC_DECL:
            # `public:` / `private:` / `protected:` markers. Only public is
            # currently supported.
            continue
        if c.kind == ci.CursorKind.FIELD_DECL:
            fields.append(hir.HirParam(name=c.spelling, annotation=_type_text(c.type)))
            continue
        if (
            c.kind == ci.CursorKind.CXX_METHOD
            and c.is_definition()
            and not c.is_default_method()
        ):
            methods.append(_convert_method(c, struct_name=name))
            continue
        if (
            c.kind == ci.CursorKind.CONSTRUCTOR
            and c.is_definition()
            and not c.is_default_method()
        ):
            methods.append(_convert_constructor(c, struct_name=name))
            continue
        if (
            c.kind == ci.CursorKind.CONSTRUCTOR
            and c.is_default_method()
            and c.is_default_constructor()
        ):
            # `gp_Vec() = default;` (or the implicit compiler-generated
            # equivalent) alongside other real, explicit constructors on the
            # same class (e.g. gp_Vec also declares `gp_Vec(double, double,
            # double)`). Mojo's `@fieldwise_init` synthesizes a usable
            # default ctor only when the struct has *no* explicit `__init__`
            # at all (see `_class_conformances` in backends/mojo/emit.py) --
            # once any real constructor is emitted, that auto-generation is
            # dropped, so a defaulted 0-arg ctor needs its own explicit
            # `__init__` synthesized too, or `gp_Vec()` call sites have no
            # matching overload ("no matching function in initialization").
            # Deferred until the whole class is scanned (see below the loop)
            # since whether it's actually needed depends on whether any
            # *other* constructor ends up defined on this same class.
            has_defaulted_default_ctor = True
            continue
        if c.kind in (
            ci.CursorKind.CXX_METHOD,
            ci.CursorKind.CONSTRUCTOR,
            ci.CursorKind.DESTRUCTOR,
        ):
            # Declared-but-not-defined methods/ctors/dtors, and other
            # `= default` ones (no explicit body to convert -- e.g. `Dir&
            # operator=(const Dir&) = default;`, extremely common on
            # OCCT-style value types): skip silently rather than raising,
            # matching real compiler-generated behavior. `is_default_method()`
            # still reports `is_definition() == True`, so without excluding
            # it above too, this branch would never even see it.
            continue
        if c.kind == ci.CursorKind.CXX_BASE_SPECIFIER:
            raise UnsupportedConstruct(
                f"C++ inheritance for class {name!r} not yet supported"
            )
        if c.kind == ci.CursorKind.FRIEND_DECL:
            # `friend class X;` / `friend void f(...);` only grants another
            # class/function access to this one's private members -- a
            # C++ access-control concept with no analog in any target this
            # engine emits. It adds no field, method, or behavior to the
            # struct itself, so it's safe to drop entirely rather than
            # refuse the whole class over it.
            continue
        if c.kind == ci.CursorKind.ENUM_DECL:
            # Hoisted to module-level constants by the caller (see the
            # ENUM_DECL scan in _convert_top_level) before _convert_class
            # ever runs; nothing left to do here.
            continue
        if c.kind == ci.CursorKind.FUNCTION_TEMPLATE:
            # A templated method (e.g. `template <class T> void
            # GetMat4(T&)`), as opposed to the class itself being a
            # template. The IR doesn't model templates; unlike a top-level
            # function template (kept as a HirRaw hole so a call site can
            # still instantiate it), a member template has no standalone
            # meaning outside its class, and dropping it leaves the
            # struct's other concrete fields and methods untouched.
            continue
        raise UnsupportedConstruct(f"class member {c.kind.name}")
    if has_defaulted_default_ctor and any(m.name == "__init__" for m in methods):
        methods.append(_synthesized_default_ctor(fields, struct_name=name))
    return hir.HirStruct(name=name, fields=fields, methods=methods)


def _default_value_for_annotation(annotation: str) -> hir.HirNode:
    """A best-effort zero/empty value for *annotation*, for synthesizing a
    defaulted 0-arg constructor's field-init body (see
    `_synthesized_default_ctor`). A struct-typed field recurses into that
    struct's own 0-arg construction (mirroring real C++ value-init
    semantics for a member with its own default ctor); anything else falls
    back to a scalar zero rather than refusing the whole class over it."""
    if annotation in _KNOWN_STRUCT_NAMES:
        return hir.HirStructInit(name=annotation, args=[])
    if annotation == "bool":
        return hir.HirBoolLiteral(value=False)
    if annotation in ("float", "double"):
        return hir.HirFloatLiteral(value=0.0)
    if annotation == "str":
        return hir.HirStringLiteral(value="")
    return hir.HirIntLiteral(value=0)


def _synthesized_default_ctor(
    fields: list[hir.HirParam], *, struct_name: str
) -> hir.HirFunction:
    """A `__init__(out self):` that zero/value-initializes every field, for
    a class with an explicitly-defaulted (`= default`) 0-arg constructor
    alongside other real, explicit constructors. Mojo's `@fieldwise_init`
    only auto-synthesizes a usable default ctor when a struct has *no*
    explicit `__init__` at all; once any other constructor is emitted, a
    defaulted 0-arg ctor needs this explicit counterpart too, or a 0-arg
    construction call site (`gp_Vec()`) has no matching overload."""
    body: list[hir.HirNode] = [
        hir.HirFieldAssign(
            obj=hir.HirName(name="self"),
            field=f.name,
            value=_default_value_for_annotation(f.annotation),
        )
        for f in fields
    ]
    return hir.HirFunction(
        name="__init__",
        params=[hir.HirParam(name="self", annotation=struct_name)],
        return_annotation="None",
        body=body,
    )


def _param_name(cursor: ci.Cursor, index: int) -> str:
    """A PARM_DECL's identifier, or a synthesized placeholder if it has
    none. C++ allows an unnamed parameter both in a declaration and, when
    the parameter is genuinely unused, in a definition too (e.g. OCCT's own
    `void gp_Pnt::DumpJson(Standard_OStream&, int) const`, whose second
    parameter is never referenced in the body). `cursor.spelling` is `""`
    in that case; every target needs *some* identifier there."""
    return cursor.spelling or f"_arg{index}"


def _convert_constructor(cursor: ci.Cursor, *, struct_name: str) -> hir.HirFunction:
    """C++ constructor → Mojo `__init__(out self, ...)`. The member-init list
    (`: x(a), y(b)`) appears in the AST as alternating MEMBER_REF / init-expr
    children before the body; each pair becomes a `self.<field> = <expr>`
    assignment, followed by the constructor body's statements."""
    params: list[hir.HirParam] = [hir.HirParam(name="self", annotation=struct_name)]
    field_inits: list[hir.HirNode] = []
    body_stmts: list[hir.HirNode] = []
    kids = list(cursor.get_children())
    i = 0
    pidx = 0
    while i < len(kids):
        c = kids[i]
        if c.kind == CursorKind.PARM_DECL:
            params.append(
                hir.HirParam(name=_param_name(c, pidx), annotation=_type_text(c.type))
            )
            pidx += 1
        elif c.kind == CursorKind.MEMBER_REF and i + 1 < len(kids):
            field_inits.append(
                hir.HirFieldAssign(
                    obj=hir.HirName(name="self"),
                    field=c.spelling,
                    value=_convert_expr(kids[i + 1]),
                )
            )
            i += 2
            continue
        elif c.kind == CursorKind.COMPOUND_STMT:
            body_stmts = _convert_compound(c)
        i += 1
    return hir.HirFunction(
        name="__init__",
        params=params,
        return_annotation="None",  # constructors return nothing -> NoneT
        body=field_inits + body_stmts,
    )


# C++ operator token → Python/Mojo dunder method name. Used to rename member
# `operator+` etc. into the dunder the backends already emit as methods.
# Note: `operator+` / `operator-` are spelling-identical for unary and binary
# forms; we map to the binary dunder (the common case for member arithmetic).
_OPERATOR_DUNDERS: dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "%": "__mod__",
    "==": "__eq__",
    "!=": "__ne__",
    "<": "__lt__",
    ">": "__gt__",
    "<=": "__le__",
    ">=": "__ge__",
    "&": "__and__",
    "|": "__or__",
    "^": "__xor__",
    "<<": "__lshift__",
    ">>": "__rshift__",
    "[]": "__getitem__",
    "()": "__call__",
    # Compound-assignment overloads (`gp_Mat::operator+=`, common on any
    # value-type class with in-place mutation, not just OCCT). Missing from
    # this table has the same failure mode operator() did before it was
    # added: the literal invalid identifier `operator+=` gets emitted
    # instead of the target's dunder.
    "+=": "__iadd__",
    "-=": "__isub__",
    "*=": "__imul__",
    "/=": "__itruediv__",
    "%=": "__imod__",
    "&=": "__iand__",
    "|=": "__ior__",
    "^=": "__ixor__",
    "<<=": "__ilshift__",
    ">>=": "__irshift__",
}


_UNARY_OPERATOR_DUNDERS: dict[str, str] = {
    "-": "__neg__",
    "+": "__pos__",
}


def _operator_name(cursor: ci.Cursor, *, unary_param_count: int) -> str:
    """Map an operator-overload to its dunder; otherwise the raw spelling.
    `operator+` → `__add__`, `operator==` → `__eq__`, etc. An operator token
    we don't recognize keeps the raw spelling (no refusal — same as before).

    `+`/`-` are ambiguous: unary negation/plus take one fewer explicit
    parameter than the binary form of the same token (a member
    `operator-()` has zero explicit params — just the implicit `self`;
    a free `operator-(x)` has one). *unary_param_count* is that
    threshold, so this same logic serves both member and free-function
    callers. Same token, different arity, different dunder in every
    target (`__neg__` takes no argument; `__sub__` requires one). Mapping
    both to `__sub__` produces a nonsensical zero-arg `__sub__`, which every
    target's own dunder-arity checker (Mojo's included) rejects outright.
    """
    spelling = cursor.spelling
    if spelling.startswith("operator"):
        token = spelling[len("operator") :].strip()
        n_params = sum(
            1 for c in cursor.get_children() if c.kind == ci.CursorKind.PARM_DECL
        )
        if token in ("-", "+") and n_params == unary_param_count:
            return _UNARY_OPERATOR_DUNDERS[token]
        if (
            token == "()"
            and n_params >= 2
            and cursor.result_type.kind
            in (ci.TypeKind.LVALUEREFERENCE, ci.TypeKind.RVALUEREFERENCE)
        ):
            # A multi-arg call operator RETURNING A REFERENCE is never a
            # generic functor invocation (a functor/predicate -- e.g. a
            # `std::sort` comparator -- returns a plain value like `bool`,
            # by value): it's the common matrix/grid-class 2D element
            # accessor idiom (`T& operator()(int row, int col)`, e.g.
            # OCCT's gp_Mat/gp_GTrsf/gp_Mat2d/gp_GTrsf2d), where the
            # reference return exists precisely so a write use
            # (`M(1, 1) = v`) can assign through it. Map to `__getitem__`
            # so a read call site (`M(1, 1)`, handled in `_convert_call`)
            # becomes `M[1, 1]`'s dunder. There's no `__setitem__`
            # counterpart emitted for this: the reference-return-as-lvalue
            # idiom has no analog in any of these value-oriented targets,
            # so a write use is left as a genuinely unsupported construct
            # (see `_convert_assignment_stmt`) rather than paired with a
            # setter whose signature couldn't match the source body anyway.
            return "__getitem__"
        if token in _OPERATOR_DUNDERS:
            return _OPERATOR_DUNDERS[token]
    return spelling


def _method_name(cursor: ci.Cursor) -> str:
    """Map an operator-overload member to its dunder; otherwise the spelling.
    See `_operator_name` — a member's implicit `self` means its unary forms
    take zero explicit parameters."""
    return _operator_name(cursor, unary_param_count=0)


def _convert_method(cursor: ci.Cursor, *, struct_name: str) -> hir.HirFunction:
    # A `static` method has no implicit `this`/`self` -- it's called via
    # `ClassName::method(...)`, no receiver instance at all (e.g. OCCT's
    # `gp::Resolution()` / `Precision::Angular()` tolerance helpers).
    # Injecting a `self` param for one anyway would be wrong twice over: the
    # target emits it as `@staticmethod` (no self param at all — Mojo
    # rejects a bodyless-of-self instance method with "self argument must
    # be present"), and a real call site never has an instance to pass.
    is_static = cursor.is_static_method()
    params: list[hir.HirParam] = (
        [] if is_static else [hir.HirParam(name="self", annotation=struct_name)]
    )
    body: list[hir.HirNode] = []
    pidx = 0
    for c in cursor.get_children():
        if c.kind == CursorKind.PARM_DECL:
            params.append(
                hir.HirParam(name=_param_name(c, pidx), annotation=_type_text(c.type))
            )
            pidx += 1
        elif c.kind == CursorKind.COMPOUND_STMT:
            body = _convert_compound(c)
    return hir.HirFunction(
        name=_method_name(cursor),
        params=params,
        return_annotation=_type_text(cursor.result_type),
        body=body,
        is_static=is_static,
    )


def _check_diagnostics(tu: ci.TranslationUnit) -> None:
    fatal = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if fatal:
        msgs = "\n".join(f"  {d.spelling}" for d in fatal[:5])
        raise UnsupportedConstruct(f"libclang parse errors:\n{msgs}")


def _from_input(cursor: ci.Cursor) -> bool:
    """True for cursors that come from the user code (not the parser
    preamble). The preamble is the first ``PARSER_PREAMBLE`` +
    ``_project_preamble`` lines of the unsaved file; anything at or
    before that line offset is the preamble and is not user code."""
    if cursor.location.file is None:
        return False
    if cursor.location.file.name != INPUT_NAME:
        return False
    try:
        first = _compute_user_first_line()
    except Exception:
        return True
    return cursor.location.line > first


def _cursor_loc(cursor: ci.Cursor) -> str | None:
    """Canonical ``file:line:col`` source-location string for *cursor*.

    Returns ``None`` for cursors that don't have a usable file location
    (built-in macros, implicit nodes) so the HIR field stays ``None`` and
    the ground-truth pass simply doesn't look them up.
    """
    try:
        loc = cursor.location
    except Exception:
        return None
    f = loc.file
    if f is None:
        return None
    return f"{f.name}:{loc.line}:{loc.column}"


def _set_loc(cursor: ci.Cursor, node: "hir.HirNode") -> "hir.HirNode":
    """Stamp *node*'s ``source_loc`` with *cursor*'s file:line:col.

    Used by the dispatchers (``_convert_top_level`` and the
    per-statement / per-expression converters) so every HIR node
    carries a back-reference to the libclang cursor it came from. The
    C++ ground-truth pass at issue #50 uses those locations to look up
    concrete types from the AST.
    """
    if node is None:
        return node
    loc = _cursor_loc(cursor)
    if loc is not None:
        try:
            object.__setattr__(node, "source_loc", loc)
        except Exception:
            # HIR node might be frozen (dataclass with frozen=True);
            # silently skip rather than break the whole conversion.
            pass
    return node


def _convert_function(cursor: ci.Cursor) -> hir.HirFunction:
    params: list[hir.HirParam] = []
    body: list[hir.HirNode] = []
    idx = 0
    for c in cursor.get_children():
        if c.kind == CursorKind.PARM_DECL:
            params.append(
                hir.HirParam(name=_param_name(c, idx), annotation=_type_text(c.type))
            )
            idx += 1
        elif c.kind == CursorKind.COMPOUND_STMT:
            body = _convert_compound(c)
    # A free (non-member) operator overload -- e.g. `gp_Vec operator*(double,
    # const gp_Vec&)`, the common idiom for a symmetric binary operator --
    # is just as real a C++ construct as a member operator, but the free
    # function itself is never invoked by name: a call site like `2.0 * v`
    # is desugared straight to a binop by `_convert_call`/`_convert_binop`
    # (see the CALL_EXPR 'operator*' handling above), never a call to this
    # function. Naming it here only has to keep it a syntactically valid
    # identifier instead of the literal invalid `operator*` token -- it must
    # NOT be the actual dunder, though: Mojo (like every target here)
    # requires a `__mul__`-etc-named callable to be a struct method, and
    # rejects a global function with that exact name outright ("must be a
    # method, not a global function"). Since nothing calls this function by
    # name anyway, prefix away from the reserved dunder spelling.
    # Unary here means one explicit parameter (no implicit `self` to absorb
    # the operand), unlike a member's zero-param unary form.
    name = _operator_name(cursor, unary_param_count=1)
    if name.startswith("__") and name.endswith("__"):
        name = f"_operator{name}"
    return hir.HirFunction(
        name=name,
        params=params,
        return_annotation=_type_text(cursor.result_type),
        body=body,
    )


def _convert_compound(cursor: ci.Cursor) -> list[hir.HirNode]:
    out: list[hir.HirNode] = []
    for c in cursor.get_children():
        out.extend(_convert_stmt(c))
    return out


def _cursor_snippet(cursor: ci.Cursor) -> str:
    """Best-effort original source text for a cursor, for a HirRaw hole.

    Joins the cursor's tokens (whitespace-insensitive but faithful enough to
    show what was skipped). Falls back to the cursor kind name when no tokens
    are available (e.g. macro-expanded or implicit nodes)."""
    try:
        toks = [t.spelling for t in cursor.get_tokens()]
    except Exception:  # pragma: no cover - defensive
        toks = []
    text = " ".join(toks).strip()
    if not text:
        return f"<{cursor.kind.name}>"
    # Cap very large snippets so a single unsupported block doesn't bloat output.
    if len(text) > 400:
        text = text[:400] + " ..."
    return text


def _convert_stmt(cursor: ci.Cursor) -> list[hir.HirNode]:
    """Never-refuse statement conversion: a caught UnsupportedConstruct is
    turned into a HirRaw hole carrying the source snippet, so one unsupported
    statement no longer aborts the whole function body."""
    _push_postfix_frame()
    try:
        result = _convert_stmt_inner(cursor)
    except UnsupportedConstruct:
        result = [hir.HirRaw(snippet=_cursor_snippet(cursor))]
    finally:
        effects = _pop_postfix_frame()
    return result + effects


def _convert_stmt_inner(cursor: ci.Cursor) -> list[hir.HirNode]:
    kind = cursor.kind
    if kind == CursorKind.RETURN_STMT:
        kids = list(cursor.get_children())
        if not kids:
            return [hir.HirReturn(value=None)]
        child = _strip_unexposed(kids[0])
        # `return (lhs = rhs)` — assignment-in-return. Desugar to assign + return lhs.
        if child.kind == CursorKind.BINARY_OPERATOR:
            try:
                op = _binop_token(child)
            except UnsupportedConstruct:
                op = ""
            if op == "=":
                bk = list(child.get_children())
                try:
                    assign = _convert_assignment_stmt(child)
                    return_val = _lhs_as_subscript_or_name(bk[0])
                    return [assign, hir.HirReturn(value=return_val)]
                except UnsupportedConstruct:
                    pass
        return [hir.HirReturn(value=_convert_expr_stmt(kids[0]))]
    if kind == CursorKind.DECL_STMT:
        out: list[hir.HirNode] = []
        for c in cursor.get_children():
            if c.kind == CursorKind.VAR_DECL:
                out.append(_convert_var_decl(c))
            else:
                raise UnsupportedConstruct(f"decl {c.kind.name}")
        return out
    if kind == CursorKind.IF_STMT:
        return [_convert_if(cursor)]
    if kind == CursorKind.WHILE_STMT:
        return [_convert_while(cursor)]
    if kind == CursorKind.FOR_STMT:
        return _convert_for(cursor)
    if kind == CursorKind.CXX_FOR_RANGE_STMT:
        return _convert_for_range(cursor)
    if kind == CursorKind.COMPOUND_STMT:
        return _convert_compound(cursor)
    if kind in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
        return [_convert_assignment_stmt(cursor)]
    if kind == CursorKind.UNARY_OPERATOR:
        return [_convert_unary_stmt(cursor)]
    if kind == CursorKind.CALL_EXPR:
        swapped = _convert_std_swap_stmt(cursor)
        if swapped is not None:
            return swapped
        return [_convert_expr_stmt(cursor)]
    if kind == CursorKind.BREAK_STMT:
        return [hir.HirBreak()]
    if kind == CursorKind.CONTINUE_STMT:
        return [hir.HirContinue()]
    if kind == CursorKind.NULL_STMT:
        return []
    if kind == CursorKind.SWITCH_STMT:
        return _convert_switch(cursor)
    if kind == CursorKind.UNEXPOSED_EXPR:
        # libclang sometimes wraps real statements (especially assignments
        # that involve operator overloads) in UNEXPOSED. Drill in.
        kids = list(cursor.get_children())
        if len(kids) == 1:
            return _convert_stmt_inner(kids[0])
    raise UnsupportedConstruct(f"stmt {kind.name}")


_ARRAY_TYPE_KINDS = (
    ci.TypeKind.CONSTANTARRAY,
    ci.TypeKind.INCOMPLETEARRAY,
    ci.TypeKind.VARIABLEARRAY,
)


def _default_array_value(t: ci.Type) -> hir.HirNode:
    """A zero-filled nested list literal matching a native fixed-size
    array type's shape (`double[3][3]` -> a 3x3 list of `0.0`), for a
    VAR_DECL with no real initializer -- see `_convert_var_decl`'s
    no-initializer array handling."""
    if t.kind in _ARRAY_TYPE_KINDS:
        count = t.element_count if t.kind == ci.TypeKind.CONSTANTARRAY else 0
        return hir.HirList(
            elements=[_default_array_value(t.element_type) for _ in range(count)]
        )
    ann = _type_text(t)
    if ann in _KNOWN_STRUCT_NAMES:
        return hir.HirStructInit(name=ann, args=[])
    if ann == "bool":
        return hir.HirBoolLiteral(value=False)
    if ann == "float":
        return hir.HirFloatLiteral(value=0.0)
    if ann == "str":
        return hir.HirStringLiteral(value="")
    return hir.HirIntLiteral(value=0)


def _convert_var_decl(cursor: ci.Cursor) -> hir.HirNode:
    annotation = _type_text(cursor.type)
    if cursor.type.kind in _ARRAY_TYPE_KINDS and not any(
        t.spelling in ("=", "{") for t in cursor.get_tokens()
    ):
        # A native fixed-size array declared with NO initializer at all
        # (`double mymatrix[3][3];`, OCCT's own `gp_Trsf::InitFromJson`
        # scratch buffer) has its dimension-size expressions (INTEGER_LITERAL
        # cursors for the `3`s) reported as VAR_DECL *children* by libclang --
        # there's no actual initializer here, but the generic `kids[-1]`
        # fallback below would misread the last dimension-size literal as
        # one, emitting e.g. `var mymatrix: List[List[Float64]] = 3`. Zero-
        # fill a nested list literal matching the array's real shape instead.
        return hir.HirAssign(
            target=cursor.spelling,
            value=_default_array_value(cursor.type),
            annotation=annotation,
        )
    kids = list(cursor.get_children())
    if annotation in _KNOWN_STRUCT_NAMES:
        # `Point p;` (no kids) → zero-init StructInit.
        # `Point p(1, 2);` → libclang emits a CALL_EXPR with the ctor args.
        # `Point p{1, 2};` (uniform init) → INIT_LIST_EXPR.
        if not kids:
            init: hir.HirNode = hir.HirStructInit(name=annotation, args=[])
        else:
            ctor = kids[-1]
            # Genuine constructor: brace-init `{...}` or an explicit `Type(...)`
            # call. An `auto`-typed var whose initializer is an expression that
            # *returns* the struct (operator call `a+b`, factory `makeVec()`) must
            # NOT be re-read as a constructor — convert it as an expression.
            is_ctor = ctor.kind == ci.CursorKind.INIT_LIST_EXPR or (
                ctor.kind == ci.CursorKind.CALL_EXPR
                and (ctor.spelling or "") == annotation
            )
            if is_ctor:
                real = [
                    c for c in ctor.get_children() if c.kind != ci.CursorKind.TYPE_REF
                ]
                # Copy/move ctor (`Mat aCopy = *this;` / `Mat aCopy(other);`):
                # a single arg whose own type is this same struct is a
                # whole-value copy, not a partial fieldwise init. Collapse to
                # just the argument, mirroring the identical fix in
                # _convert_call's _KNOWN_STRUCT_NAMES branch (this is a
                # separate code path for local-variable declarations, so the
                # same libclang shape needs the same handling here too).
                # Without it, hir_to_mir's trailing-field defaulting padded
                # the "missing" fields, and even once every field was
                # supplied, the Mojo backend emitted `Mat(self)` -- which
                # fails to compile for any struct with its own declared
                # constructors (no @fieldwise_init-synthesized one accepts a
                # same-type argument).
                if len(real) == 1:
                    arg_ty = (
                        (real[0].type.spelling or "")
                        .replace("const ", "")
                        .rstrip("&")
                        .strip()
                    )
                    if arg_ty == annotation:
                        init = _convert_expr(real[0])
                        return hir.HirAssign(
                            target=cursor.spelling, value=init, annotation=annotation
                        )
                args: list[hir.HirNode] = []
                for c in ctor.get_children():
                    if c.kind == ci.CursorKind.TYPE_REF:
                        continue
                    if c.kind == ci.CursorKind.UNEXPOSED_EXPR:
                        inner = list(c.get_children())
                        if len(inner) == 1:
                            args.append(_convert_expr(inner[0]))
                            continue
                    args.append(_convert_expr(c))
                init = hir.HirStructInit(name=annotation, args=args)
            else:
                init = _convert_expr(ctor)
        return hir.HirAssign(target=cursor.spelling, value=init, annotation=annotation)
    init = _convert_expr(kids[-1]) if kids else hir.HirIntLiteral(value=0)
    # `auto x = expr;` — drop the (deduced) annotation so the IR's own type
    # inference (hir_to_mir / infer_types) recovers x's type from the RHS,
    # rather than baking in clang's deduction. The struct branch above already
    # handled `auto p = Point(...)` via the deduced name, so by here `auto`
    # only covers scalar/expression initializers.
    if cursor.type.kind == ci.TypeKind.AUTO:
        annotation = None
    return hir.HirAssign(target=cursor.spelling, value=init, annotation=annotation)


_KNOWN_STRUCT_NAMES: set[str] = set()


def _convert_if(cursor: ci.Cursor) -> hir.HirNode:
    kids = list(cursor.get_children())
    if len(kids) < 2:
        raise UnsupportedConstruct("malformed if")
    cond = _convert_expr(kids[0])
    body = _convert_stmt(kids[1])
    orelse: list[hir.HirNode] = []
    if len(kids) >= 3:
        orelse = _convert_stmt(kids[2])
    return hir.HirIf(test=cond, body=body, orelse=orelse)


def _convert_while(cursor: ci.Cursor) -> hir.HirNode:
    kids = list(cursor.get_children())
    test_cursor = kids[0]
    body = _convert_stmt(kids[1]) if len(kids) > 1 else []
    # `while (t--)` competitive-programming idiom: post-decrement on a name
    # means "test t != 0 then decrement". Desugar to a clean form rather
    # than refusing (which #4 made the default for postfix in expr context).
    stripped = _strip_unexposed(test_cursor)
    if stripped.kind == CursorKind.UNARY_OPERATOR and _unary_token(stripped) in (
        "--",
        "++",
    ):
        op = _unary_token(stripped)
        inner_kids = list(stripped.get_children())
        if inner_kids and inner_kids[0].kind == CursorKind.DECL_REF_EXPR:
            name = inner_kids[0].spelling
            cond = hir.HirCompare(
                op="!=", left=hir.HirName(name=name), right=hir.HirIntLiteral(value=0)
            )
            step = "-" if op == "--" else "+"
            update = hir.HirAssign(
                target=name,
                value=hir.HirIntLiteral(value=1),
                annotation=None,
                augmented_op=step,
            )
            return hir.HirWhile(test=cond, body=[update, *body])
    cond = _convert_expr(test_cursor)
    return hir.HirWhile(test=cond, body=body)


def _convert_switch(cursor: ci.Cursor) -> list[hir.HirNode]:
    """`switch (x) { case A: ...; break; case B: ...; ... default: ...; }`
    desugars to a chain of `if/elif/else`. Fall-through is *not* modelled —
    each case body terminates at its break (we strip those during conversion).
    Cases without an explicit break would semantically need fall-through
    support, but that's out of scope for the initial subset."""
    kids = list(cursor.get_children())
    if len(kids) < 2:
        raise UnsupportedConstruct("malformed switch")
    test_expr = _convert_expr(kids[0])
    body_cursor = kids[1]
    cases: list[tuple[hir.HirNode, list[hir.HirNode]]] = []
    default_body: list[hir.HirNode] = []

    body_kids = (
        list(body_cursor.get_children())
        if body_cursor.kind == CursorKind.COMPOUND_STMT
        else [body_cursor]
    )
    current_value: hir.HirNode | None = None
    current_body: list[hir.HirNode] = []
    is_default = False

    def flush() -> None:
        nonlocal current_value, current_body, is_default
        if is_default:
            default_body.extend(current_body)
        elif current_value is not None:
            cases.append((current_value, current_body))
        current_value = None
        current_body = []
        is_default = False

    for c in body_kids:
        if c.kind == CursorKind.CASE_STMT:
            flush()
            case_kids = list(c.get_children())
            if case_kids:
                current_value = _convert_expr(case_kids[0])
                # The case's statement child(ren) follow the value.
                for inner in case_kids[1:]:
                    current_body.extend(_convert_stmt(inner))
        elif c.kind == CursorKind.DEFAULT_STMT:
            flush()
            is_default = True
            for inner in c.get_children():
                current_body.extend(_convert_stmt(inner))
        else:
            current_body.extend(_convert_stmt(c))
    flush()

    # Strip trailing HirBreak from each case body — switch semantics.
    def _strip_break(stmts: list[hir.HirNode]) -> list[hir.HirNode]:
        return [s for s in stmts if not isinstance(s, hir.HirBreak)]

    # Build nested if/elif/else from the cases.
    chain: list[hir.HirNode] = _strip_break(default_body)
    for value, body in reversed(cases):
        chain = [
            hir.HirIf(
                test=hir.HirCompare(op="==", left=test_expr, right=value),
                body=_strip_break(body),
                orelse=chain,
            )
        ]
    return chain


def _convert_for_range(cursor: ci.Cursor) -> list[hir.HirNode]:
    """`for (auto x : xs)` → indexed loop with x = xs[i] inside.

    libclang exposes the children as: VAR_DECL (loop var), iterable expr,
    then COMPOUND_STMT body (or a bare statement when no braces).
    Desugars to `for i in range(len(xs)): x = xs[i]; <body>`.
    """
    kids = list(cursor.get_children())
    var_decl = next((c for c in kids if c.kind == CursorKind.VAR_DECL), None)
    body_cursor = next((c for c in kids if c.kind == CursorKind.COMPOUND_STMT), None)
    # Iterable: first non-VAR_DECL, non-COMPOUND_STMT, non-NULL_STMT child.
    iter_cursor = next(
        (
            c
            for c in kids
            if c.kind
            not in (CursorKind.VAR_DECL, CursorKind.COMPOUND_STMT, CursorKind.NULL_STMT)
        ),
        None,
    )
    # Single-statement body (no braces): the last child that isn't the loop
    # variable or the iterable.
    if body_cursor is None and iter_cursor is not None:
        candidates = [
            c
            for c in kids
            if c.kind not in (CursorKind.VAR_DECL, CursorKind.NULL_STMT)
            and c is not iter_cursor
        ]
        body_cursor = candidates[-1] if candidates else None

    if var_decl is None or iter_cursor is None or body_cursor is None:
        raise UnsupportedConstruct("malformed range-for")

    iter_expr = _convert_expr(iter_cursor)
    var_name = var_decl.spelling
    idx_name = "__xpile_idx"

    if body_cursor.kind == CursorKind.COMPOUND_STMT:
        body = _convert_compound(body_cursor)
    else:
        body = _convert_stmt(body_cursor)

    bind = hir.HirAssign(
        target=var_name,
        value=hir.HirSubscript(value=iter_expr, index=hir.HirName(name=idx_name)),
        annotation=None,
    )
    loop = hir.HirFor(
        target=idx_name,
        iter=hir.HirCall(
            func="range",
            args=[
                hir.HirIntLiteral(value=0),
                hir.HirCall(func="len", args=[iter_expr]),
            ],
        ),
        body=[bind, *body],
    )
    return [loop]


def _convert_for(cursor: ci.Cursor) -> list[hir.HirNode]:
    """C-style for desugars at the frontend: init; while(cond) { body; step; }."""
    kids = list(cursor.get_children())
    # libclang's FOR_STMT children layout, in order:
    #   init (DECL_STMT or expression, may be omitted),
    #   cond (expression, may be omitted),
    #   step (expression, may be omitted),
    #   body (statement).
    # Missing parts simply aren't present; we use heuristics to detect.
    *headers, body_node = kids if kids else (None,)
    init_part: ci.Cursor | None = None
    cond_part: ci.Cursor | None = None
    step_part: ci.Cursor | None = None
    if len(headers) == 3:
        init_part, cond_part, step_part = headers
    elif len(headers) == 2:
        init_part, cond_part = headers
    elif len(headers) == 1:
        cond_part = headers[0]
    out: list[hir.HirNode] = []
    if init_part is not None:
        out.extend(_convert_stmt(init_part))
    cond = (
        _convert_expr(cond_part)
        if cond_part is not None
        else hir.HirBoolLiteral(value=True)
    )
    inner = _convert_stmt(body_node)
    if step_part is not None:
        inner.extend(_convert_stmt(step_part))
    out.append(hir.HirWhile(test=cond, body=inner))
    return out


def _convert_expr_stmt(cursor: ci.Cursor) -> hir.HirNode:
    """Never-refuse expression conversion at a statement boundary.

    Used where an expression is lowered in statement position (a bare
    expression statement, a return value). A caught UnsupportedConstruct
    becomes a HirRaw hole instead of aborting the enclosing function. Internal
    recursive expression conversion still uses `_convert_expr` (which raises),
    so the existing per-node fallbacks that probe alternative children keep
    working."""
    try:
        return _convert_expr(cursor)
    except UnsupportedConstruct:
        return hir.HirRaw(snippet=_cursor_snippet(cursor))


# Mirrors the numeric-valued `-D` predefines in `parse_cpp`'s `predefs`
# list (NULL, M_PI, ...) -- kept in sync with that list by hand. An
# INTEGER_LITERAL/FLOATING_LITERAL cursor for one of these command-line
# `-D`-defined constants loses its own token text under a libclang
# tokenizer quirk at macro-expansion boundaries (see `_tokens_for`'s and
# `_literal_token`'s docstrings); the token *found* via the widened
# retokenize fallback is still just the macro's identifier spelling
# ("M_PI"), never its expanded numeric text, so `float()`/`int()` on it
# would still fail. Resolve these specific, well-known names to the value
# we ourselves defined them as, rather than silently falling back to 0.
_PREDEF_INT_MACROS: dict[str, int] = {
    "NULL": 0,
    "EXIT_SUCCESS": 0,
    "EXIT_FAILURE": 1,
    "INT_MIN": -2147483648,
    "INT_MAX": 2147483647,
    "UINT_MAX": 4294967295,
    "LONG_MIN": -9223372036854775808,
    "LONG_MAX": 9223372036854775807,
    "LLONG_MIN": -9223372036854775808,
    "LLONG_MAX": 9223372036854775807,
}

_PREDEF_FLOAT_MACROS: dict[str, float] = {
    "M_PI": 3.14159265358979323846,
    "M_PI_2": 1.57079632679489661923,
    "M_PI_4": 0.78539816339744830962,
    "M_2_PI": 0.63661977236758134308,
    "M_E": 2.71828182845904523536,
    "M_SQRT2": 1.41421356237309504880,
    "M_SQRT1_2": 0.70710678118654752440,
    "DBL_EPSILON": 2.2204460492503131e-16,
    "FLT_EPSILON": 1.19209290e-7,
}


def _convert_expr(cursor: ci.Cursor) -> hir.HirNode:
    kind = cursor.kind
    if kind == CursorKind.INTEGER_LITERAL:
        token = _literal_token(cursor)
        if token is None:
            return hir.HirIntLiteral(value=0)
        if token.spelling in _PREDEF_INT_MACROS:
            return hir.HirIntLiteral(value=_PREDEF_INT_MACROS[token.spelling])
        try:
            return hir.HirIntLiteral(value=int(token.spelling.rstrip("uUlL"), 0))
        except ValueError:
            # Some other macro-expanded literal (`_literal_token`'s widened
            # retokenize found the macro's own identifier spelling, not its
            # expanded numeric text -- we don't have the actual constant
            # value available for anything outside `_PREDEF_INT_MACROS`
            # without evaluating). Fall back to 0 so the file still parses;
            # a real lossy compromise for an unrecognized macro constant.
            return hir.HirIntLiteral(value=0)
    if kind == CursorKind.FLOATING_LITERAL:
        token = _literal_token(cursor)
        if token is None:
            return hir.HirFloatLiteral(value=0.0)
        if token.spelling in _PREDEF_FLOAT_MACROS:
            return hir.HirFloatLiteral(value=_PREDEF_FLOAT_MACROS[token.spelling])
        try:
            return hir.HirFloatLiteral(value=float(token.spelling.rstrip("fFlL")))
        except ValueError:
            return hir.HirFloatLiteral(value=0.0)
    if kind == CursorKind.CXX_BOOL_LITERAL_EXPR:
        token = next(cursor.get_tokens(), None)
        return hir.HirBoolLiteral(value=token is not None and token.spelling == "true")
    if kind == CursorKind.CHARACTER_LITERAL:
        token = next(cursor.get_tokens(), None)
        if token is None:
            return hir.HirIntLiteral(value=0)
        spelling = token.spelling
        if len(spelling) >= 3 and spelling.startswith("'") and spelling.endswith("'"):
            inner = spelling[1:-1]
            # Common escapes only — the rest stay as their literal byte.
            escape_map = {
                "\\n": 10,
                "\\t": 9,
                "\\r": 13,
                "\\0": 0,
                "\\'": 39,
                "\\\\": 92,
            }
            if inner in escape_map:
                return hir.HirIntLiteral(value=escape_map[inner])
            if len(inner) == 1:
                return hir.HirIntLiteral(value=ord(inner))
        return hir.HirIntLiteral(value=0)
    if kind == CursorKind.STRING_LITERAL:
        token = next(cursor.get_tokens(), None)
        if token is None:
            raise UnsupportedConstruct("string literal without tokens")
        return hir.HirStringLiteral(value=token.spelling[1:-1])
    if kind == CursorKind.DECL_REF_EXPR:
        return hir.HirName(name=cursor.spelling)
    if kind == CursorKind.UNEXPOSED_EXPR or kind == CursorKind.PAREN_EXPR:
        # Pass through to the single child — libclang often wraps real exprs.
        kids = list(cursor.get_children())
        if len(kids) == 1:
            return _convert_expr(kids[0])
    if kind == CursorKind.BINARY_OPERATOR:
        return _convert_binop(cursor)
    if kind == CursorKind.UNARY_OPERATOR:
        return _convert_unary_expr(cursor)
    if kind == CursorKind.MEMBER_REF_EXPR:
        # `obj.field` or `this->field` — the LHS object is the only child.
        kids = list(cursor.get_children())
        # Inside a method body, libclang sometimes elides the implicit `this`;
        # in that case there's no child and we treat as a self-reference.
        if not kids:
            return hir.HirFieldAccess(
                value=hir.HirName(name="self"), field=cursor.spelling
            )
        return hir.HirFieldAccess(value=_convert_expr(kids[0]), field=cursor.spelling)
    if kind == CursorKind.CXX_THIS_EXPR:
        # `this` → `self` in our HIR vocabulary; the method-conversion sets
        # the first parameter to `self`.
        return hir.HirName(name="self")
    if kind == CursorKind.CALL_EXPR:
        return _convert_call(cursor)
    if kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
        # `arr[index]` — two children: the array expression and the index.
        kids = list(cursor.get_children())
        if len(kids) == 2:
            return hir.HirSubscript(
                value=_convert_expr(kids[0]), index=_convert_expr(kids[1])
            )
    if kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
        raise UnsupportedConstruct("compound assignment as expression")
    if kind in (CursorKind.GNU_NULL_EXPR, CursorKind.CXX_NULL_PTR_LITERAL_EXPR):
        # `NULL` / `nullptr` — lower as a zero literal. Lossy (no real null
        # type), but suffices for comparisons like `p == NULL`.
        return hir.HirIntLiteral(value=0)
    if kind in (
        CursorKind.CSTYLE_CAST_EXPR,
        CursorKind.CXX_STATIC_CAST_EXPR,
        CursorKind.CXX_FUNCTIONAL_CAST_EXPR,
    ):
        # `(T)x` / `static_cast<T>(x)` / `T(x)`.
        #
        # A float->int cast in C truncates toward zero; dropping it would
        # silently change semantics (e.g. `(int)(t/5.0)` becomes a float).
        # When the destination is integral and the operand is floating, wrap
        # the value in `Int(...)` (Mojo's Int(Float) truncates toward zero,
        # matching C). All other casts are dropped — type inference / target
        # coercion handle widening.
        kids = list(cursor.get_children())
        inner = None
        for c in kids[::-1]:
            if c.kind != CursorKind.TYPE_REF:
                inner = c
                break
        if inner is None and kids:
            inner = kids[-1]
        if inner is None:
            return hir.HirIntLiteral(value=0)
        value = _convert_expr(inner)
        if _is_integral_type(cursor.type) and _operand_is_floating(inner):
            return hir.HirCall(func="Int", args=[value])
        return value
    if kind == CursorKind.CONDITIONAL_OPERATOR:
        # `cond ? a : b` — three children. Lower as a __ternary__ builtin
        # call so each backend folds it into a target-native if-expression.
        kids = list(cursor.get_children())
        if len(kids) == 3:
            return hir.HirCall(
                func="__ternary__",
                args=[
                    _convert_expr(kids[0]),
                    _convert_expr(kids[1]),
                    _convert_expr(kids[2]),
                ],
            )
    if kind == CursorKind.INIT_LIST_EXPR:
        # `{1, 2, 3}` brace-init at expression position — emit as a list literal.
        return hir.HirList(elements=[_convert_expr(c) for c in cursor.get_children()])
    if kind in (CursorKind.CXX_UNARY_EXPR,):
        # `sizeof(T)` / `alignof(T)` — emit as a constant placeholder (real
        # value isn't available without type-size context).
        return hir.HirIntLiteral(value=0)
    if kind == CursorKind.CXX_NEW_EXPR:
        # `new T(args)` — drop the heap allocation, yield T-constructed value.
        # Best-effort: emit the call-style construction, lossy on ownership.
        kids = list(cursor.get_children())
        for c in kids[::-1]:
            if c.kind != CursorKind.TYPE_REF:
                try:
                    return _convert_expr(c)
                except UnsupportedConstruct:
                    continue
        return hir.HirIntLiteral(value=0)
    if kind in (CursorKind.TYPE_REF, CursorKind.TEMPLATE_REF, CursorKind.NAMESPACE_REF):
        # A bare type reference at expression position — usually a sizeof
        # argument or template parameter leak. Yield 0; the surrounding
        # context handles the semantics.
        return hir.HirIntLiteral(value=0)
    if kind == CursorKind.UNEXPOSED_EXPR:
        # Multi-child UNEXPOSED nodes (e.g. operator-overload call wrapper)
        # — drill into the last non-trivial child.
        kids = list(cursor.get_children())
        for c in kids[::-1]:
            if c.kind != CursorKind.TYPE_REF:
                try:
                    return _convert_expr(c)
                except UnsupportedConstruct:
                    continue
        # Empty UNEXPOSED (or all children failed): yield 0 so the call site
        # parses. Lossy but matches our other I/O-strip behavior.
        return hir.HirIntLiteral(value=0)
    raise UnsupportedConstruct(f"expr {kind.name}")


def _strip_paren_unexposed(cursor: ci.Cursor) -> ci.Cursor:
    while cursor.kind in (CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR):
        kids = list(cursor.get_children())
        if len(kids) != 1:
            return cursor
        cursor = kids[0]
    return cursor


def _is_this_expr(cursor: ci.Cursor) -> bool:
    return _strip_paren_unexposed(cursor).kind == CursorKind.CXX_THIS_EXPR


def _is_addr_of_expr(cursor: ci.Cursor) -> bool:
    c = _strip_paren_unexposed(cursor)
    if c.kind != CursorKind.UNARY_OPERATOR or len(list(c.get_children())) != 1:
        return False
    try:
        return _unary_token(c) == "&"
    except UnsupportedConstruct:
        return False


def _convert_binop(cursor: ci.Cursor) -> hir.HirNode:
    op = _binop_token(cursor)
    kids = list(cursor.get_children())
    if op in ("==", "!=") and (
        (_is_this_expr(kids[0]) and _is_addr_of_expr(kids[1]))
        or (_is_this_expr(kids[1]) and _is_addr_of_expr(kids[0]))
    ):
        # `this == &theOther` / `&theOther == this` -- a pointer-identity
        # self-aliasing check (the classic self-assignment/self-comparison
        # guard idiom, e.g. OCCT's `gp_Quaternion::IsEqual`: `if (this ==
        # &theOther) return true;` before falling back to a real value
        # comparison). This engine doesn't model pointer/reference identity
        # -- parameters are handled by value/borrow, not tracked aliases --
        # and after the existing address-of/`this`-to-`self` lossy
        # lowering, both sides collapse to plain names, so naively emitting
        # a VALUE comparison here isn't just imprecise but can be a compile
        # error outright when the type has no `operator==` (as here).
        # Every real-world instance of this idiom is followed by a full
        # value-comparison fallback that computes the correct result
        # regardless of whether the identity fast path is taken, so
        # dropping the fast path (never true) preserves correctness at the
        # cost of a micro-optimization.
        return hir.HirBoolLiteral(value=(op == "!="))
    left = _convert_expr(kids[0])
    right = _convert_expr(kids[1])
    if op in COMPARE_OPS:
        return hir.HirCompare(op=op, left=left, right=right)
    if op in LOGICAL_OPS:
        return hir.HirBoolOp(op="and" if op == "&&" else "or", left=left, right=right)
    if op in ARITH_OPS:
        return hir.HirBinOp(op=op, left=left, right=right)
    if op in ("&", "|", "^", "<<", ">>"):
        return hir.HirBinOp(op=op, left=left, right=right)
    raise UnsupportedConstruct(f"binary op {op!r}")


def _convert_assignment_stmt(cursor: ci.Cursor) -> hir.HirNode:
    """A BINARY_OPERATOR (or COMPOUND_ASSIGNMENT_OPERATOR) at statement position
    using `=` / `+=` / `-=` / etc. Field assignments (`obj.field = v`)
    branch to HirFieldAssign; plain identifier assignments emit HirAssign."""
    op = _binop_token(cursor)
    if op not in ASSIGN_OPS:
        raise UnsupportedConstruct(f"expression statement with op {op!r}")
    kids = list(cursor.get_children())
    lhs = kids[0]
    rhs = _convert_expr(kids[1])
    if lhs.kind == CursorKind.MEMBER_REF_EXPR:
        lhs_kids = list(lhs.get_children())
        obj = _convert_expr(lhs_kids[0]) if lhs_kids else hir.HirName(name="self")
        if op == "=":
            value = rhs
        else:  # compound: obj.field op= v -> obj.field = obj.field op v
            value = hir.HirBinOp(
                op=op[:-1],
                left=hir.HirFieldAccess(value=obj, field=lhs.spelling),
                right=rhs,
            )
        return hir.HirFieldAssign(obj=obj, field=lhs.spelling, value=value)
    # Subscript assign, plain (`arr[i] = v`) or compound (`arr[i] += v`).
    # ARRAY_SUBSCRIPT for native arrays; CALL_EXPR for the std::vector operator[]
    # overload (children may be [obj, operator[]-ref, index]).
    sub = None
    if lhs.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
        lk = list(lhs.get_children())
        if len(lk) == 2:
            sub = (_convert_expr(lk[0]), _convert_expr(lk[1]))
    elif lhs.kind == CursorKind.CALL_EXPR and lhs.spelling != "operator()":
        # The `lhs.spelling != "operator()"` guard excludes the 2D
        # element-accessor idiom (`aMat(1, 1) = v`, see `_operator_name`'s
        # `__getitem__` mapping and the read-side handling in
        # `_convert_call`): that C++ method returns a mutable reference for
        # the caller to assign *through*, which none of these targets
        # model, so there's no `__setitem__` counterpart to route a write
        # to. Falling through to the `UnsupportedConstruct` below (a
        # never-refuse hole) is more honest than reusing this 1-D
        # `(lk[0], lk[-1])` reduction, which would silently drop the row
        # index and emit a wrong single-index subscript assign.
        lk = [
            c
            for c in lhs.get_children()
            if c.kind not in (CursorKind.TYPE_REF, CursorKind.NAMESPACE_REF)
            and not (
                (c.spelling or "").startswith("operator")
                and c.kind != CursorKind.CALL_EXPR
            )
        ]
        if len(lk) >= 2:
            sub = (_convert_expr(lk[0]), _convert_expr(lk[-1]))
    if sub is not None:
        obj, index = sub
        if op == "=":
            value = rhs
        else:  # compound: arr[i] op= v  ->  arr[i] = arr[i] op v
            value = hir.HirBinOp(
                op=op[:-1], left=hir.HirSubscript(value=obj, index=index), right=rhs
            )
        return hir.HirSubscriptAssign(obj=obj, index=index, value=value)
    target = _decl_name(lhs)
    if target is None:
        raise UnsupportedConstruct(f"assignment target {lhs.kind.name}")
    aug = None if op == "=" else op[:-1]
    return hir.HirAssign(target=target, value=rhs, annotation=None, augmented_op=aug)


def _convert_unary_expr(cursor: ci.Cursor) -> hir.HirNode:
    op = _unary_token(cursor)
    kids = list(cursor.get_children())
    if op == "!":
        return hir.HirUnaryOp(op="not", operand=_convert_expr(kids[0]))
    if op == "-":
        return hir.HirUnaryOp(op="-", operand=_convert_expr(kids[0]))
    if op == "+":
        return _convert_expr(kids[0])
    if op in ("&", "*"):
        # Address-of / dereference — drop the indirection. Lossy but lets
        # I/O-heavy corpus parse.
        return _convert_expr(kids[0])
    if op == "~":
        return hir.HirUnaryOp(op="~", operand=_convert_expr(kids[0]))
    if op in ("++", "--"):
        # Post-increment/decrement in expression position (e.g. `s[l++]`):
        # return the PRE-increment value and defer the side effect to a
        # statement appended after the enclosing expression-statement.
        # _convert_stmt drains the postfix-effect stack after each statement.
        target_name = _decl_name(kids[0])
        operand = _convert_expr(kids[0])
        if target_name is not None:
            sign = "+" if op == "++" else "-"
            _record_postfix_effect(
                hir.HirAssign(
                    target=target_name,
                    value=hir.HirIntLiteral(value=1),
                    annotation=None,
                    augmented_op=sign,
                )
            )
            return operand
        # subscript postfix: `arr[i]++` / `m[k]++` -> defer `arr[i] = arr[i] + 1`
        if isinstance(operand, hir.HirSubscript):
            sign = "+" if op == "++" else "-"
            _record_postfix_effect(
                hir.HirSubscriptAssign(
                    obj=operand.value,
                    index=operand.index,
                    value=hir.HirBinOp(
                        op=sign, left=operand, right=hir.HirIntLiteral(value=1)
                    ),
                )
            )
            return operand
        raise UnsupportedConstruct(f"postfix {op!r} on complex expression")
    raise UnsupportedConstruct(f"unary op {op!r} as expression")


def _convert_unary_stmt(cursor: ci.Cursor) -> hir.HirNode:
    """`i++` / `i--` as a statement."""
    op = _unary_token(cursor)
    if op not in ("++", "--"):
        raise UnsupportedConstruct(f"unary stmt {op!r}")
    kids = list(cursor.get_children())
    target = _decl_name(kids[0])
    sign = "+" if op == "++" else "-"
    if target is not None:
        return hir.HirAssign(
            target=target,
            value=hir.HirIntLiteral(value=1),
            annotation=None,
            augmented_op=sign,
        )
    # `arr[i]++` / `m[k]++` as a statement -> arr[i] = arr[i] + 1
    operand = _convert_expr(kids[0])
    if isinstance(operand, hir.HirSubscript):
        return hir.HirSubscriptAssign(
            obj=operand.value,
            index=operand.index,
            value=hir.HirBinOp(op=sign, left=operand, right=hir.HirIntLiteral(value=1)),
        )
    raise UnsupportedConstruct(f"++/-- on {kids[0].kind.name}")


_TUPLE_CONSTRUCTORS = frozenset({"tuple", "pair", "make_pair", "make_tuple"})

_LIST_CONSTRUCTORS = frozenset({"vector", "array", "deque", "list"})

_OVERLOAD_BINOPS = frozenset(
    {
        "operator+",
        "operator-",
        "operator*",
        "operator/",
        "operator%",
        "operator==",
        "operator!=",
        "operator<",
        "operator>",
        "operator<=",
        "operator>=",
        "operator&&",
        "operator||",
    }
)

_OVERLOAD_AUGOPS = {
    "operator+=": "+",
    "operator-=": "-",
    "operator*=": "*",
    "operator/=": "/",
    "operator%=": "%",
}


def _lhs_as_subscript_or_name(lhs: ci.Cursor) -> hir.HirNode:
    """Convert an assignment LHS cursor to an expression for use in `return lhs`."""
    lhs = _strip_unexposed(lhs)
    if lhs.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
        lk = list(lhs.get_children())
        if len(lk) == 2:
            return hir.HirSubscript(
                value=_convert_expr(lk[0]), index=_convert_expr(lk[1])
            )
    if lhs.kind == CursorKind.CALL_EXPR:
        # operator[] overload: obj = first child, index = last meaningful arg
        lk = list(lhs.get_children())
        # Filter out TYPE_REF / operator-ref cursors; keep object + index
        args = [
            c
            for c in lk
            if c.kind not in (CursorKind.TYPE_REF, CursorKind.NAMESPACE_REF)
        ]
        # First is typically the object; last meaningful arg is the index.
        # Filter the operator[] callee ref (any cursor kind) by spelling.
        meaningful = [
            a
            for a in args
            if not (
                (a.spelling or "").startswith("operator")
                and a.kind != CursorKind.CALL_EXPR
            )
        ]
        if len(meaningful) >= 2:
            return hir.HirSubscript(
                value=_convert_expr(meaningful[0]), index=_convert_expr(meaningful[-1])
            )
    if lhs.kind == CursorKind.MEMBER_REF_EXPR:
        lk = list(lhs.get_children())
        obj = _convert_expr(lk[0]) if lk else hir.HirName(name="self")
        return hir.HirFieldAccess(value=obj, field=lhs.spelling)
    return _convert_expr(lhs)


def _assign_to(target: hir.HirNode, value: hir.HirNode) -> hir.HirNode:
    """Build the right *Assign statement for a target expression shape
    produced by `_lhs_as_subscript_or_name` (subscript / field / plain
    name), for `_convert_std_swap_stmt`'s manual swap desugaring."""
    if isinstance(target, hir.HirSubscript):
        return hir.HirSubscriptAssign(obj=target.value, index=target.index, value=value)
    if isinstance(target, hir.HirFieldAccess):
        return hir.HirFieldAssign(obj=target.value, field=target.field, value=value)
    if isinstance(target, hir.HirName):
        return hir.HirAssign(target=target.name, value=value, annotation=None)
    raise UnsupportedConstruct(f"swap target shape {type(target).__name__}")


def _convert_std_swap_stmt(cursor: ci.Cursor) -> list[hir.HirNode] | None:
    """`std::swap(a, b);` as a statement -> `tmp = a; a = b; b = tmp;`.

    Not just a style choice: Mojo's own `swap()` builtin rejects two
    arguments that alias the same underlying container ("argument ...
    allows writing a memory location previously writable through another
    aliased argument") -- exactly OCCT's `gp_Mat::Transpose` idiom,
    `std::swap(myMat[0][1], myMat[1][0])`, two elements of the *same*
    matrix. A manual temp-variable swap sidesteps that aliasing-exclusivity
    restriction entirely (no two arguments passed to any single call), and
    is correct universally -- not a Mojo-specific workaround.

    Returns None (not applicable) for anything that isn't a genuine
    `std::swap(a, b)` two-argument call, so the caller falls back to
    ordinary call-expression handling.
    """
    if cursor.spelling != "swap":
        return None
    ref = cursor.referenced
    if (
        ref is None
        or ref.semantic_parent is None
        or ref.semantic_parent.spelling != "std"
    ):
        return None
    kids = [
        k
        for k in cursor.get_children()
        if k.kind
        not in (CursorKind.NAMESPACE_REF, CursorKind.TYPE_REF, CursorKind.TEMPLATE_REF)
        and (k.spelling or "") != "swap"
    ]
    if len(kids) != 2:
        return None
    a = _lhs_as_subscript_or_name(kids[0])
    b = _lhs_as_subscript_or_name(kids[1])
    tmp = "__swap_tmp"
    return [
        hir.HirAssign(target=tmp, value=a, annotation=None),
        _assign_to(a, b),
        _assign_to(b, hir.HirName(name=tmp)),
    ]


def _iter_index(node: "hir.HirNode"):
    """Interpret a vector iterator expr as (container, index): c.begin() -> (c,0),
    c.end() -> (c, len(c)), c.begin()+k -> (c, k). Returns None otherwise."""
    if (
        isinstance(node, hir.HirMethodCall)
        and node.method in ("begin", "end")
        and not node.args
    ):
        idx = (
            hir.HirIntLiteral(value=0)
            if node.method == "begin"
            else hir.HirCall(func="len", args=[node.receiver])
        )
        return (node.receiver, idx)
    if (
        isinstance(node, hir.HirBinOp)
        and node.op == "+"
        and isinstance(node.left, hir.HirMethodCall)
        and node.left.method == "begin"
    ):
        return (node.left.receiver, node.right)
    return None


def _convert_call(cursor: ci.Cursor) -> hir.HirNode:
    kids = list(cursor.get_children())

    # Overloaded binary operator `a + b` arrives as CALL_EXPR 'operator+' with
    # kids [lhs, operator-ref, rhs]. Map to a real binop/compare so Mojo's struct
    # operator methods (__add__/__eq__/...) drive it.
    if cursor.spelling in _OVERLOAD_BINOPS:
        op = cursor.spelling[len("operator") :]
        operands = [k for k in kids if (k.spelling or "") != cursor.spelling]
        if len(operands) == 2:
            lhs = _convert_expr(operands[0])
            rhs = _convert_expr(operands[1])
            if op in ("==", "!=", "<", ">", "<=", ">="):
                return hir.HirCompare(op=op, left=lhs, right=rhs)
            if op in ("&&", "||"):
                return hir.HirBoolOp(
                    op="and" if op == "&&" else "or", left=lhs, right=rhs
                )
            return hir.HirBinOp(op=op, left=lhs, right=rhs)
        # Unary form of the same token (1 operand after filtering, e.g.
        # `-vydir` calling a member `gp_Dir::operator-()` with no explicit
        # params): only `+`/`-` are valid unary operators among this set.
        # Without this, the call fell through to the generic call-resolution
        # path at the bottom of this function, which has no notion of a
        # bare operator-ref child and emitted garbled code (a spurious
        # `operator-` call embedded as an argument, e.g. `vydir(operator-)`,
        # "unexpected token in expression").
        if len(operands) == 1 and op in ("+", "-"):
            return hir.HirUnaryOp(op=op, operand=_convert_expr(operands[0]))

    # Overloaded assignment (`r = x` / `r += x`, e.g. std::string concat, or
    # any user struct with an `operator=`/`operator+=`) arrives as CALL_EXPR
    # 'operator=' / 'operator+='. Desugar to a plain (or augmented) assign.
    # The lhs is the *first real operand*, not necessarily a bare name: a
    # field assigned without explicit `this->` (`vxdir = theV;` inside a
    # method, `vxdir`'s type having a user `operator=`) arrives with a
    # MEMBER_REF_EXPR lhs, which _decl_name (bare DeclRefExpr only) can't
    # name — falling through here previously misread the MEMBER_REF_EXPR as
    # an ordinary call *receiver* (`self.vxdir(operator=, theV)`, nonsense:
    # `vxdir` isn't callable and `operator=`/theV aren't its arguments).
    if cursor.spelling == "operator=" or cursor.spelling in _OVERLOAD_AUGOPS:
        operands = [k for k in kids if (k.spelling or "") != cursor.spelling]
        if len(operands) == 2:
            lhs, rhs = operands
            aug = _OVERLOAD_AUGOPS.get(cursor.spelling)
            if lhs.kind == CursorKind.MEMBER_REF_EXPR:
                lhs_kids = list(lhs.get_children())
                obj = (
                    _convert_expr(lhs_kids[0]) if lhs_kids else hir.HirName(name="self")
                )
                value = _convert_expr(rhs)
                if aug is not None:
                    value = hir.HirBinOp(
                        op=aug,
                        left=hir.HirFieldAccess(value=obj, field=lhs.spelling),
                        right=value,
                    )
                return hir.HirFieldAssign(obj=obj, field=lhs.spelling, value=value)
            target = _decl_name(lhs) or ("self" if _is_this_deref(lhs) else None)
            if target is not None:
                return hir.HirAssign(
                    target=target,
                    value=_convert_expr(rhs),
                    annotation=None,
                    augmented_op=aug,
                )

    # Struct constructor call `Vec2(a, b)` — including the implicit ctor libclang
    # materializes for brace-init `return {a, b};` when the type is fully known
    # (CALL_EXPR whose spelling is a known struct name). Emit a StructInit.
    if cursor.spelling in _KNOWN_STRUCT_NAMES:
        real = [c for c in kids if c.kind != CursorKind.TYPE_REF]
        # Copy/move ctor (`Vec(other)` -- including the implicit one libclang
        # materializes for `return v;` / `Vec v2 = v1;`): a single arg whose
        # own type is this same struct is a whole-value copy, not a partial
        # fieldwise init. Collapse to just the argument (mirrors the
        # copy/move-ctor handling for vector<T>/string/map below). Without
        # this, hir_to_mir's trailing-field defaulting padded the "missing"
        # fields with a fabricated value, e.g. `return v;` -> `Vec(v, 0)`.
        if len(real) == 1:
            arg_ty = (
                (real[0].type.spelling or "").replace("const ", "").rstrip("&").strip()
            )
            if arg_ty == cursor.spelling:
                return _convert_expr(real[0])
        args = [_convert_expr(c) for c in real]
        return hir.HirStructInit(name=cursor.spelling, args=args)

    # std::vector<T> sized constructor: (n) or (n, fill). Emit a marker the Mojo
    # lowering turns into `[fill] * n` using the declared element type. The empty
    # `vector<T> v;` default-ctor has no kids and is handled below as a placeholder.
    if cursor.spelling == "vector":
        real = [
            k
            for k in kids
            if k.kind
            not in (
                CursorKind.TYPE_REF,
                CursorKind.TEMPLATE_REF,
                CursorKind.NAMESPACE_REF,
            )
        ]
        # Brace-init `vector<T>{a, b}` / `return {a, b}` is an ELEMENT list, not a
        # sized ctor (identical cursor shape — distinguish by `{` in the tokens).
        if "{" in [t.spelling for t in cursor.get_tokens()]:
            if not real:  # `{}` -> typed empty (lower_* knows the type)
                return hir.HirCall(func="__cpp_overloaded_op__", args=[])
            return hir.HirList(elements=[_convert_expr(k) for k in real])
        # copy/move ctor: vector<T>(otherVector) -> just the argument (Mojo copies
        # on assign/return). Distinguish from the sized ctor by the arg type.
        if len(real) == 1 and "vector" in (real[0].type.spelling or ""):
            return _convert_expr(real[0])
        if (
            len(real) == 2
        ):  # iterator-range ctor: vector<T>(c.begin()[+a], c.end()|c.begin()+b)
            a_hir, b_hir = _convert_expr(real[0]), _convert_expr(real[1])
            ia, ib = _iter_index(a_hir), _iter_index(b_hir)
            if ia and ib:
                return hir.HirCall(func="__vector_slice__", args=[ia[0], ia[1], ib[1]])
        if real:  # sized ctor: (n) or (n, fill)
            return hir.HirCall(
                func="__vector_fill__", args=[_convert_expr(k) for k in real]
            )

    # map/set copy/move ctor (e.g. `return freq;` for a map return) -> the arg;
    # the empty default-ctor (no args) falls through to the placeholder below.
    if cursor.spelling in ("map", "unordered_map", "_mapbase", "set", "unordered_set"):
        real = [
            k
            for k in kids
            if k.kind
            not in (
                CursorKind.TYPE_REF,
                CursorKind.TEMPLATE_REF,
                CursorKind.NAMESPACE_REF,
            )
        ]
        if len(real) == 1 and any(
            s in (real[0].type.spelling or "") for s in ("map", "set")
        ):
            return _convert_expr(real[0])

    # std::string copy / construct-from-value (`return r;`, `string(other)`,
    # `string("lit")`) -> the single argument; lower_return adds `.copy()` for a
    # bare String name. The 0-arg default ctor falls through to empty-string below.
    if cursor.spelling in ("string", "basic_string"):
        real = [
            k
            for k in kids
            if k.kind
            not in (
                CursorKind.TYPE_REF,
                CursorKind.TEMPLATE_REF,
                CursorKind.NAMESPACE_REF,
            )
        ]
        if len(real) == 1:
            return _convert_expr(real[0])

    # const `vec[i]` READS come through the operator[] overload as a CALL_EXPR
    # (non-const reads are ARRAY_SUBSCRIPT_EXPR, handled elsewhere). Mirror the
    # assign-side handling and lower to a subscript.
    if cursor.spelling == "operator[]" or any(
        k.kind == CursorKind.DECL_REF_EXPR and "operator[]" in (k.spelling or "")
        for k in kids
    ):
        meaningful = [
            c
            for c in kids
            if c.kind not in (CursorKind.TYPE_REF, CursorKind.NAMESPACE_REF)
            and not (
                (c.spelling or "").startswith("operator")
                and c.kind != CursorKind.CALL_EXPR
            )
        ]
        if len(meaningful) >= 2:
            return hir.HirSubscript(
                value=_convert_expr(meaningful[0]), index=_convert_expr(meaningful[-1])
            )

    # 2D element-accessor idiom (`aMat(1, 1)`, `theMat(row, col)`): a common
    # matrix/grid class's `T& operator()(int, int)` overload, arriving as a
    # CALL_EXPR with the same [obj, operator-ref, arg...] shape operator[]
    # has above (as opposed to `_method_name`/`__call__`'s ordinary 0/1-arg
    # functor invocation). Mirrors the class-definition side's mapping of a
    # 2+-arg `operator()` to `__getitem__` (see `_operator_name`). A write
    # use (`aMat(1, 1) = v`) is handled -- or rather, deliberately left
    # unsupported -- separately in `_convert_assignment_stmt`. Gated on the
    # callee actually returning a reference (matching `_operator_name`'s
    # class-definition-side check) so an ordinary value-returning 2-arg
    # functor call (a `bool operator()(int, int) const` comparator) isn't
    # misrouted to a `__getitem__` the class was never given.
    if cursor.spelling == "operator()":
        op_ref = next(
            (
                k
                for k in kids
                if (k.spelling or "").startswith("operator")
                and k.kind != CursorKind.CALL_EXPR
            ),
            None,
        )
        referenced = op_ref.referenced if op_ref is not None else None
        if referenced is not None and referenced.result_type.kind in (
            ci.TypeKind.LVALUEREFERENCE,
            ci.TypeKind.RVALUEREFERENCE,
        ):
            meaningful = [
                c
                for c in kids
                if c.kind not in (CursorKind.TYPE_REF, CursorKind.NAMESPACE_REF)
                and not (
                    (c.spelling or "").startswith("operator")
                    and c.kind != CursorKind.CALL_EXPR
                )
            ]
            if len(meaningful) >= 3:
                return hir.HirMethodCall(
                    receiver=_convert_expr(meaningful[0]),
                    method="__getitem__",
                    args=[_convert_expr(a) for a in meaningful[1:]],
                )

    # Detect tuple/pair constructor: cursor.spelling is the type name ('tuple',
    # 'pair', etc.) and the first child is NOT a callee reference but an argument.
    if cursor.spelling in _TUPLE_CONSTRUCTORS and kids:
        real = [
            k
            for k in kids
            if k.kind
            not in (
                CursorKind.TYPE_REF,
                CursorKind.TEMPLATE_REF,
                CursorKind.NAMESPACE_REF,
            )
        ]
        # brace-init `{a, b}` (tokens contain `{`) is always an element list, even
        # when the first element is a variable (DECL_REF). Otherwise fall back to
        # the "first child isn't a callee ref" heuristic for make_pair/make_tuple.
        if "{" in [t.spelling for t in cursor.get_tokens()]:
            return hir.HirCall(
                func="tuple",
                args=[hir.HirList(elements=[_convert_expr(c) for c in real])],
            )
        first = _strip_unexposed(kids[0])
        if first.kind not in (CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR):
            elements = [_convert_expr(c) for c in kids]
            return hir.HirCall(func="tuple", args=[hir.HirList(elements=elements)])

    if cursor.spelling in _LIST_CONSTRUCTORS and kids:
        first = _strip_unexposed(kids[0])
        if first.kind not in (CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR):
            elements = [_convert_expr(c) for c in kids]
            return hir.HirList(elements=elements)

    if not kids:
        # libclang's CALL_EXPR exposes no children for operator-overload
        # calls like `cin >> n` / `cout << x`. Lose the I/O side effect
        # and emit a synthetic no-op call — competitive-programming code
        # is the main culprit and we care about the algorithm body, not
        # the I/O wrapping.
        return hir.HirCall(func="__cpp_overloaded_op__", args=[])
    callee = kids[0]
    # `obj.method(args)` arrives as CALL_EXPR whose callee is MEMBER_REF_EXPR.
    if callee.kind == CursorKind.MEMBER_REF_EXPR:
        callee_kids = list(callee.get_children())
        receiver = (
            _convert_expr(callee_kids[0]) if callee_kids else hir.HirName(name="self")
        )
        args = [_convert_expr(a) for a in kids[1:]]
        return hir.HirMethodCall(receiver=receiver, method=callee.spelling, args=args)
    # libclang sometimes wraps the callee in UNEXPOSED — drill in.
    while callee.kind == CursorKind.UNEXPOSED_EXPR:
        inner = list(callee.get_children())
        if not inner:
            break
        callee = inner[0]
    name = _decl_name(callee) or callee.spelling
    if not name:
        # Operator overload or similar opaque callee — emit a placeholder
        # so the surrounding code parses (same rationale as the zero-kids
        # case above).
        return hir.HirCall(
            func="__cpp_overloaded_op__", args=[_convert_expr(a) for a in kids[1:]]
        )
    args = [_convert_expr(a) for a in kids[1:]]
    # Unqualified call to another method of the same class (`square(x)` inside
    # `cube()`, relying on implicit `this->`/no-qualifier lookup — including
    # calls to a *static* sibling method, which C++ also permits unqualified).
    # libclang still resolves the callee to a CXX_METHOD even with no explicit
    # receiver; every backend's struct methods take an explicit `self`
    # (mirroring how an explicit `this->square(x)` / `self.square(x)` lowers
    # here — see the MEMBER_REF_EXPR branch above), so without this the
    # emitted call has no way to reach the method at all in a target where
    # methods aren't free functions (Mojo: "cannot access method directly").
    referenced = callee.referenced
    # A *qualified* static call to another class's method (`Precision::
    # Angular()`) also resolves `referenced.kind` to CXX_METHOD -- libclang
    # doesn't distinguish "found via unqualified lookup" from "found via
    # explicit ClassName::" in the cursor kind, only in its own token
    # spelling. Without this check, a real cross-class static call like
    # `Precision::Angular()` got silently rewritten to `self.Angular()`,
    # which is not just unsupported but *wrong* — a different method on
    # (usually) an unrelated struct, if it exists on self at all.
    qualified = any(t.spelling == "::" for t in callee.get_tokens())
    if (
        referenced is not None
        and referenced.kind == CursorKind.CXX_METHOD
        and not qualified
    ):
        return hir.HirMethodCall(
            receiver=hir.HirName(name="self"), method=name, args=args
        )
    # A qualified call to another class's *static* method (`Precision::
    # Angular()`, `gp::Resolution()`) — every target here maps a static
    # method to a receiver-less callable reached via the struct's own name
    # (Mojo: `@staticmethod` + `StructName.method(...)`), so the struct
    # name doubles as a valid "receiver" expression for a HirMethodCall
    # despite naming a type, not a value.
    if (
        referenced is not None
        and referenced.kind == CursorKind.CXX_METHOD
        and qualified
        and referenced.is_static_method()
        and referenced.semantic_parent is not None
    ):
        struct_name = referenced.semantic_parent.spelling
        return hir.HirMethodCall(
            receiver=hir.HirName(name=struct_name), method=name, args=args
        )
    # SIMD intrinsic lifting: turn Intel `_mm*` calls into semantic HIR
    # operations on SIMD-typed values. Mojo will emit idiomatic `a + b`
    # on SIMD types; other targets fall back to the original call form.
    lifted = _lift_simd_intrinsic(name, args)
    if lifted is not None:
        return lifted
    return hir.HirCall(func=name, args=args)

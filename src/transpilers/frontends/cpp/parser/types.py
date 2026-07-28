"""C++ type spelling -> HIR type-text mapping (leaf utility)."""
from __future__ import annotations

import clang.cindex as ci

from .errors import UnsupportedConstruct

# Internal shim used by the nested ``std::vector<std::vector<T>>``
# branch in ``_type_text``: feeding the inner spelling back through
# the alias logic without constructing a real libclang ``Type``.
# ``ci.Type`` has no public constructor, so we wrap the spelling in
# a stand-in that exposes just the attribute the recursion needs.
class _TypeShim:
    __slots__ = ("spelling",)

    def __init__(self, spelling: str) -> None:
        self.spelling = spelling


CPP_TYPE_ALIASES: dict[str, str] = {
    # Integer family — collapse all width/signedness variants onto `int`.
    "int": "int",
    "signed": "int",
    "signed int": "int",
    "unsigned": "int",
    "unsigned int": "int",
    "long": "int",
    "signed long": "int",
    "unsigned long": "int",
    "long long": "int",
    "signed long long": "int",
    "unsigned long long": "int",
    "short": "int",
    "signed short": "int",
    "unsigned short": "int",
    "char": "int",
    "signed char": "int",
    "unsigned char": "int",
    # stdint-style names that show up after `#include <cstdint>`.
    "int8_t": "int",
    "int16_t": "int",
    "int32_t": "int",
    "int64_t": "int",
    "uint8_t": "int",
    "uint16_t": "int",
    "uint32_t": "int",
    "uint64_t": "int",
    "size_t": "int",
    "std::size_t": "int",
    "ssize_t": "int",
    "ptrdiff_t": "int",
    "std::ptrdiff_t": "int",
    # Floating-point.
    "float": "float",
    "double": "float",
    "long double": "float",
    # Booleans / void.
    "bool": "bool",
    "_Bool": "bool",
    "void": "None",
    # String shapes — collapse onto our StrT.
    "char *": "str",
    "const char *": "str",
    "std::string": "str",
    "std::string_view": "str",
    "string_view": "str",
    "basic_string": "str",
    "basic_string_view": "str",
    "std::basic_string": "str",
    "std::basic_string_view": "str",
}

SIMD_TYPE_ALIASES: dict[str, str] = {
    "__m256d": "simd[float, 4]",
    "__m256":  "simd[float, 8]",
    "__m256i": "simd[int, 8]",
    "__m128d": "simd[float, 2]",
    "__m128":  "simd[float, 4]",
    "__m128i": "simd[int, 4]",
    "__m512d": "simd[float, 8]",
    "__m512":  "simd[float, 16]",
    "__m512i": "simd[int, 16]",
}

_VECTOR_ELEM_ALIASES = {
    "int": "int", "long": "int", "long long": "int", "short": "int",
    "unsigned": "int", "unsigned int": "int", "unsigned long": "int",
    "char": "int", "unsigned char": "int", "signed char": "int",
    "size_t": "int", "ptrdiff_t": "int",
    "float": "float", "double": "float", "long double": "float",
    "bool": "bool",
    "string": "str", "std::string": "str",
}

def _type_text(t: ci.Type) -> str:
    spelling = t.spelling
    # Strip cv-qualifiers and references for the alias lookup.
    cleaned = spelling.replace("const ", "").replace("volatile ", "").strip()
    if cleaned in CPP_TYPE_ALIASES:
        return CPP_TYPE_ALIASES[cleaned]
    if cleaned in SIMD_TYPE_ALIASES:
        return SIMD_TYPE_ALIASES[cleaned]
    # std::string family → str.
    if cleaned in ("string", "std::string", "std::basic_string<char>"):
        return "str"
    # std::vector<T> → list[T]. Recursive in case T is itself a template
    # (e.g. `std::vector<std::vector<int>>` -> `list[list[int]]`).
    if cleaned.startswith(("vector<", "std::vector<")) and cleaned.endswith(">"):
        inner = cleaned.split("<", 1)[1][:-1].strip()
        # Inner can be another vector: recurse the spelling, not the
        # alias table, so ``std::vector<std::vector<int>>`` becomes
        # ``list[list[int]]`` rather than ``list[std::vector<int>]``
        # (which the backends don't understand).
        if inner.startswith(("vector<", "std::vector<")) and inner.endswith(">"):
            inner = _type_text(
                # The libclang Type wrapper is annoying to construct
                # here, so we recurse by feeding the cleaned spelling
                # back through the alias logic via a tiny shim.
                _TypeShim(spelling=inner)
            )
        else:
            inner = _VECTOR_ELEM_ALIASES.get(inner, inner)
        return f"list[{inner}]"
    # std::list<T> / std::stack<T> / std::queue<T> -> list[T] (Mojo List)
    for _pre in ("list<", "std::list<", "stack<", "std::stack<", "queue<", "std::queue<"):
        if cleaned.startswith(_pre) and cleaned.endswith(">"):
            inner = cleaned.split("<", 1)[1][:-1].strip()
            return f"list[{_VECTOR_ELEM_ALIASES.get(inner, inner)}]"
    # std::tuple<...> / std::pair<A,B> -> tuple[...] (Mojo tuple)
    for _pre in ("tuple<", "std::tuple<", "pair<", "std::pair<"):
        if cleaned.startswith(_pre) and cleaned.endswith(">"):
            inner = cleaned.split("<", 1)[1][:-1].strip()
            parts, depth, last = [], 0, 0
            for i, ch in enumerate(inner):
                if ch in "[<":
                    depth += 1
                elif ch in "]>":
                    depth -= 1
                elif ch == "," and depth == 0:
                    parts.append(inner[last:i].strip())
                    last = i + 1
            parts.append(inner[last:].strip())
            parts = [_VECTOR_ELEM_ALIASES.get(p, p) for p in parts]
            return f"tuple[{', '.join(parts)}]"
    # std::unordered_map<K,V> / std::map<K,V> -> dict[K, V]
    for _pre in ("unordered_map<", "std::unordered_map<", "map<", "std::map<"):
        if cleaned.startswith(_pre) and cleaned.endswith(">"):
            inner = cleaned.split("<", 1)[1][:-1].strip()
            depth = 0
            for i, ch in enumerate(inner):  # top-level comma between K and V
                if ch in "[<":
                    depth += 1
                elif ch in "]>":
                    depth -= 1
                elif ch == "," and depth == 0:
                    k = _VECTOR_ELEM_ALIASES.get(inner[:i].strip(), inner[:i].strip())
                    v = _VECTOR_ELEM_ALIASES.get(inner[i + 1:].strip(), inner[i + 1:].strip())
                    return f"dict[{k}, {v}]"
            break
    # Array types — `int[]`, `int[10]`, `int[n]`. Drop the size(s); carry
    # the element type as a list. Multi-dimensional (`double[3][3]`) needs
    # one `list[...]` layer per bracket group -- slicing at just the first
    # `[` previously collapsed every dimension onto a single flat
    # `list[float]`, silently discarding the second (and any further)
    # dimension (OCCT's `gp_Mat::myMat`: `double myMat[3][3];`).
    if "[" in cleaned and cleaned.endswith("]"):
        head = cleaned[: cleaned.index("[")].strip()
        if head in _VECTOR_ELEM_ALIASES:
            result = _VECTOR_ELEM_ALIASES[head]
            for _ in range(cleaned.count("[")):
                result = f"list[{result}]"
            return result
    # Array libclang kinds.
    if t.kind in (ci.TypeKind.CONSTANTARRAY, ci.TypeKind.INCOMPLETEARRAY, ci.TypeKind.VARIABLEARRAY):
        try:
            return f"list[{_type_text(t.element_type)}]"
        except UnsupportedConstruct:
            return "list[int]"
    # Best-effort fallback: collapse on the canonical kind.
    kind = t.kind
    INTEGER_KINDS = {
        ci.TypeKind.INT, ci.TypeKind.LONG, ci.TypeKind.LONGLONG,
        ci.TypeKind.SHORT, ci.TypeKind.SCHAR, ci.TypeKind.UCHAR,
        ci.TypeKind.CHAR_S, ci.TypeKind.CHAR_U,
        ci.TypeKind.UINT, ci.TypeKind.ULONG, ci.TypeKind.ULONGLONG, ci.TypeKind.USHORT,
    }
    if kind in INTEGER_KINDS:
        return "int"
    if kind in (ci.TypeKind.FLOAT, ci.TypeKind.DOUBLE, ci.TypeKind.LONGDOUBLE):
        return "float"
    if kind == ci.TypeKind.BOOL:
        return "bool"
    if kind == ci.TypeKind.VOID:
        return "None"
    # Struct/class types: pass the bare name through so HIR→MIR resolves it
    # against the struct registry (HirStruct names land in StructT(name)).
    # Exception: template-shaped names (`std::vector<int>`) collapse to
    # `list[T]` because libclang surfaces instantiated templates as RECORD
    # kinds whose spelling is the original `vector<...>` text. Without
    # this branch a function like
    #   void rotate(std::vector<std::vector<int>>& m)
    # would emit `std::vector<std::vector<int>>` as a HIR annotation
    # and the backends (which only know `list[T]`) would refuse it.
    if kind == ci.TypeKind.RECORD:
        if cleaned.startswith(("vector<", "std::vector<")) and cleaned.endswith(">"):
            inner = cleaned.split("<", 1)[1][:-1].strip()
            return f"list[{_VECTOR_ELEM_ALIASES.get(inner, inner)}]"
        # A struct declared inside a namespace (including an anonymous one,
        # spelled literally `(anonymous namespace)::Name` by libclang --
        # e.g. a translation-unit-local helper struct in a real-world .cxx)
        # keeps its namespace-qualified spelling here, but _convert_top_level
        # flattens every namespace (Mojo has no namespace, so members land
        # at module scope under their bare name). Left qualified, this
        # struct's name would never match the registry HirStruct entries
        # are keyed by, leaving every reference to it an unresolved type
        # hole even though the struct itself converted fine.
        if "::" in cleaned:
            return cleaned.rsplit("::", 1)[1]
        return cleaned
    # Raw pointers — in C-style C++ these are almost always buffers, indexed
    # like arrays (`p[i]`). Model as an indexable `list[elem]` so subscripting
    # lowers correctly (Mojo `List[T]`, Rust `Vec<T>`, ...). `char*` is already
    # handled as `str` above. We don't model pointer lifetimes/ownership.
    if kind == ci.TypeKind.POINTER:
        try:
            return f"list[{_type_text(t.get_pointee())}]"
        except UnsupportedConstruct:
            return "list[int]"
    # References are pass-by-reference *scalars* (`int& x` used as `x`, not
    # `x[i]`) — collapse onto the pointee's type.
    if kind in (ci.TypeKind.LVALUEREFERENCE, ci.TypeKind.RVALUEREFERENCE):
        try:
            return _type_text(t.get_pointee())
        except UnsupportedConstruct:
            return "int"
    # User-defined typedefs (`typedef long long siz`) — resolve through
    # the canonical type so aliases for primitive types still translate.
    if kind == ci.TypeKind.TYPEDEF:
        canonical = t.get_canonical()
        if canonical.kind != kind:
            try:
                return _type_text(canonical)
            except UnsupportedConstruct:
                pass
        return "int"
    # `auto x = ...` — libclang exposes the deduced type via the canonical
    # form. Recurse so we get the real type.
    if kind in (ci.TypeKind.AUTO, ci.TypeKind.ELABORATED, ci.TypeKind.UNEXPOSED):
        canonical = t.get_canonical()
        if canonical.kind != kind:  # avoid infinite recursion
            try:
                return _type_text(canonical)
            except UnsupportedConstruct:
                pass
        return "int"
    raise UnsupportedConstruct(f"C++ type {spelling!r} (kind={kind.name})")


__all__ = ['CPP_TYPE_ALIASES', 'SIMD_TYPE_ALIASES', '_VECTOR_ELEM_ALIASES', '_type_text']

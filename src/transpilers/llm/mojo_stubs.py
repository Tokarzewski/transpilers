"""Mojo stdlib semantics as Python stubs for LLM prompt injection.

This module provides Python-compatible stub definitions that mirror Mojo's
standard library types and functions. These stubs serve two purposes:

1. **LLM prompt injection**: When translating C++ → Mojo, inject the relevant
   stubs into the system prompt so the LLM knows the correct Mojo API surface.

2. **Type checking / IDE support**: The stubs are valid Python (using typing
   annotations) so editors can provide minimal autocomplete.

Usage:
    from transpilers.llm.mojo_stubs import get_stubs_for_apis, MOJO_STDLIB_STUBS

    # Get stubs relevant to a set of C++ APIs
    context = get_stubs_for_apis(["std::vector", "std::sort", "std::string"])
    # Inject context into LLM system prompt

    # Or use the full stub block
    print(MOJO_STDLIB_STUBS)

Mojo version targeted: 24.x (MAX Platform)
Last reviewed: 2025-05
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Mojo stdlib stub definitions (as Python type stubs)
# ---------------------------------------------------------------------------

MOJO_STDLIB_STUBS = '''\
# Mojo Standard Library — Python-syntax stubs
# These show the API surface for LLM context injection.
# Actual Mojo syntax differs (fn/struct/var, ownership, etc.)

# --- Collections ---
struct List[T]:
    """Dynamic array, equivalent to C++ std::vector<T>."""
    fn __init__(inout self): ...
    fn append(inout self, value: T): ...
    fn pop(inout self) -> T: ...
    fn __len__(self) -> Int: ...
    fn __getitem__(self, idx: Int) -> T: ...
    fn __setitem__(inout self, idx: Int, val: T): ...
    fn clear(inout self): ...
    fn resize(inout self, n: Int): ...

struct Dict[K, V]:
    """Hash map, equivalent to C++ std::unordered_map<K,V>."""
    fn __init__(inout self): ...
    fn __setitem__(inout self, key: K, val: V): ...
    fn __getitem__(self, key: K) -> V: ...
    fn __contains__(self, key: K) -> Bool: ...
    fn get(self, key: K, default: V) -> V: ...
    fn keys(self) -> List[K]: ...
    fn values(self) -> List[V]: ...

struct Set[T]:
    """Hash set, equivalent to C++ std::set<T>."""
    fn __init__(inout self): ...
    fn add(inout self, val: T): ...
    fn remove(inout self, val: T): ...
    fn __contains__(self, val: T) -> Bool: ...
    fn __len__(self) -> Int: ...

struct Optional[T]:
    """Optional value, equivalent to C++ std::optional<T>."""
    fn __init__(inout self): ...
    fn value(self) -> T: ...
    fn has_value(self) -> Bool: ...

# --- Strings ---
struct String:
    """Owning UTF-8 string, equivalent to C++ std::string."""
    fn __init__(inout self, s: StringLiteral): ...
    fn __add__(self, other: String) -> String: ...
    fn __len__(self) -> Int: ...
    fn find(self, sub: String) -> Int: ...
    fn split(self, sep: String) -> List[String]: ...
    fn strip(self) -> String: ...
    fn upper(self) -> String: ...
    fn lower(self) -> String: ...
    fn startswith(self, prefix: String) -> Bool: ...
    fn endswith(self, suffix: String) -> Bool: ...

# --- Math ---
fn abs[T: Numeric](x: T) -> T: ...
fn max[T: Comparable](a: T, b: T) -> T: ...
fn min[T: Comparable](a: T, b: T) -> T: ...
fn pow(base: Float64, exp: Float64) -> Float64: ...
fn sqrt(x: Float64) -> Float64: ...
fn floor(x: Float64) -> Float64: ...
fn ceil(x: Float64) -> Float64: ...
fn round(x: Float64) -> Float64: ...

# math module
struct math:
    @staticmethod fn sqrt(x: Float64) -> Float64: ...
    @staticmethod fn pow(x: Float64, y: Float64) -> Float64: ...
    @staticmethod fn log(x: Float64) -> Float64: ...
    @staticmethod fn exp(x: Float64) -> Float64: ...
    @staticmethod fn sin(x: Float64) -> Float64: ...
    @staticmethod fn cos(x: Float64) -> Float64: ...
    @staticmethod fn pi() -> Float64: ...

# --- I/O ---
fn print(msg: String): ...
fn print(value: Int): ...
fn print(value: Float64): ...
fn input(prompt: String) -> String: ...

# --- Numeric types ---
alias Int = Int64    # Mojo's default Int is pointer-sized (usually 64-bit)
alias Float = Float64
alias Bool = Bool

# Explicit-width integers
alias Int8   = SIMD[DType.int8, 1]
alias Int16  = SIMD[DType.int16, 1]
alias Int32  = SIMD[DType.int32, 1]
alias Int64  = SIMD[DType.int64, 1]
alias UInt8  = SIMD[DType.uint8, 1]
alias UInt16 = SIMD[DType.uint16, 1]
alias UInt32 = SIMD[DType.uint32, 1]
alias UInt64 = SIMD[DType.uint64, 1]
alias Float32 = SIMD[DType.float32, 1]
alias Float64 = SIMD[DType.float64, 1]

# --- Ownership / memory ---
struct Owned[T]:
    """Uniquely-owned heap value, analogous to C++ std::unique_ptr<T>."""
    fn __init__(inout self, val: T): ...
    fn __del__(owned self): ...

struct Ref[T, lifetime: Lifetime]:
    """Borrowed reference (no heap alloc), analogous to C++ const T&."""
    ...

# --- Algorithms ---
fn sort[T: Comparable](inout list: List[T]): ...
fn sort[T](inout list: List[T], key: fn(T) -> T): ...
fn reversed[T](list: List[T]) -> List[T]: ...
fn enumerate[T](list: List[T]) -> List[Tuple[Int, T]]: ...
fn zip[A, B](a: List[A], b: List[B]) -> List[Tuple[A, B]]: ...

# --- Error handling ---
struct Error:
    fn __init__(inout self, msg: String): ...

fn raise_error(e: Error): ...
'''

# ---------------------------------------------------------------------------
# Per-API stub fragments for targeted injection
# ---------------------------------------------------------------------------

# Maps C++ API names → list of stub categories to inject
_CPP_TO_STUB_CATEGORIES: dict[str, list[str]] = {
    "std::vector": ["List"],
    "std::unordered_map": ["Dict"],
    "std::map": ["Dict"],
    "std::set": ["Set"],
    "std::string": ["String"],
    "std::optional": ["Optional"],
    "std::unique_ptr": ["Owned"],
    "std::shared_ptr": ["Owned"],
    "std::sort": ["sort", "List"],
    "std::max": ["max"],
    "std::min": ["min"],
    "std::abs": ["abs"],
    "std::sqrt": ["math"],
    "std::pow": ["math"],
    "std::floor": ["math"],
    "std::ceil": ["math"],
    "printf": ["print"],
    "std::cout": ["print"],
    "std::cin": ["input"],
    "std::pair": ["Tuple"],
    "std::tuple": ["Tuple"],
}

# Stub fragments by category (short forms for prompt injection)
STUB_FRAGMENTS: dict[str, str] = {
    "List": "struct List[T]: fn append(inout self, value: T): ... fn __len__(self) -> Int: ... fn __getitem__(self, idx: Int) -> T: ...",
    "Dict": "struct Dict[K, V]: fn __setitem__(inout self, key: K, val: V): ... fn __getitem__(self, key: K) -> V: ... fn __contains__(self, key: K) -> Bool: ...",
    "Set": "struct Set[T]: fn add(inout self, val: T): ... fn __contains__(self, val: T) -> Bool: ...",
    "String": "struct String: fn __add__(self, other: String) -> String: ... fn __len__(self) -> Int: ... fn find(self, sub: String) -> Int: ...",
    "Optional": "struct Optional[T]: fn value(self) -> T: ... fn has_value(self) -> Bool: ...",
    "Owned": "struct Owned[T]: # uniquely-owned, like std::unique_ptr; auto-freed when out of scope",
    "sort": "fn sort[T: Comparable](inout list: List[T]): ...",
    "max": "fn max[T: Comparable](a: T, b: T) -> T: ...",
    "min": "fn min[T: Comparable](a: T, b: T) -> T: ...",
    "abs": "fn abs[T: Numeric](x: T) -> T: ...",
    "math": "struct math:\n    @staticmethod fn sqrt(x: Float64) -> Float64: ...\n    @staticmethod fn pow(x: Float64, y: Float64) -> Float64: ...",
    "print": "fn print(msg: String): ...\nfn print(value: Int): ...",
    "input": "fn input(prompt: String) -> String: ...",
    "Tuple": "# Mojo tuples: Tuple[A, B] — use (a, b) literal syntax",
}


def get_stubs_for_apis(cpp_apis: list[str]) -> str:
    """Return a minimal Mojo stub block covering the given C++ API names.

    Args:
        cpp_apis: List of C++ API names like ["std::vector", "std::sort"].

    Returns:
        A string of Mojo stubs to inject into an LLM prompt.
    """
    categories: set[str] = set()
    for api in cpp_apis:
        cats = _CPP_TO_STUB_CATEGORIES.get(api.strip(), [])
        categories.update(cats)

    if not categories:
        return ""

    lines = ["# Mojo stdlib API reference (for translation context):"]
    for cat in sorted(categories):
        frag = STUB_FRAGMENTS.get(cat, "")
        if frag:
            lines.append(frag)
    return "\n".join(lines)


def get_full_stubs() -> str:
    """Return the complete Mojo stdlib stub block."""
    return MOJO_STDLIB_STUBS


def build_prompt_context(cpp_apis: list[str], *, full: bool = False) -> str:
    """Build a prompt context block with Mojo stubs for the given C++ APIs.

    Args:
        cpp_apis: C++ API names encountered in the source.
        full:     If True, return the full stub block regardless of APIs.

    Returns:
        Formatted string ready to prepend to an LLM translation prompt.
    """
    if full:
        return f"## Mojo stdlib reference\n\n```\n{MOJO_STDLIB_STUBS}\n```\n"
    stubs = get_stubs_for_apis(cpp_apis)
    if not stubs:
        return ""
    return f"## Mojo stdlib reference (relevant APIs)\n\n```\n{stubs}\n```\n"


if __name__ == "__main__":
    import sys

    apis = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else ["std::vector", "std::sort", "std::string"]
    )
    print(build_prompt_context(apis))

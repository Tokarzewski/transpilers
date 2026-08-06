"""C++ preprocessor front-end (issue #50).

Wraps the host ``clang -E`` so we expand real macros (``INT_MIN``, project
``#define``s, conditional ``#ifdef`` blocks) and strip ``#include`` lines
*before* libclang sees the source. This means:

* macro-expanded tokens reach the parser as real identifiers / literals,
  so a downstream inference pass can reason about their types;
* C++20 ``requires std::foo<T>`` constraints -- which libclang cannot
  parse inside template parameter lists in our setup -- become harmless
  blank lines that we delete;
* the parser never trips over missing system headers (``<vector>``,
  ``<concepts>``, etc.) because the preamble defines a small ``std``
  namespace as opaque class templates.

Returned text is the *user* code, with macros expanded and includes/guards
removed, suitable for handing to libclang as ``unsaved_files`` content. The
string is byte-stable for the same input (clang's line-marker output is
deterministic), so callers can use it as a cache key if they want to.

Usage::

    from transpilers.frontends.cpp.parser.preprocess import preprocess_cpp
    preprocessed = preprocess_cpp("int x = 1;")
    # -> "namespace std { ... }\nint x = 1;"
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

# Minimal std:: namespace shadow so the user's templates can refer to
# common types (std::vector<T>, std::totally_ordered<T>, std::swap, ...)
# without dragging in a full libcxx. We declare them as opaque class
# templates; the backends treat any of these names as list[T] / str / etc.
# via the existing _type_text aliases.
#
# This is the *parser* preamble -- it is *not* user code. The HIR never
# sees it: only top-level decls from the original file are kept. We add
# it to the preprocessed source so libclang can build a valid AST.
PARSER_PREAMBLE: Final[str] = """
namespace std {
    template <typename T> class vector {
    public:
        vector(); vector(int); vector(int, const T&);
        vector(const T*, const T*); vector(T*, T*);   // iterator-range ctor
        T& operator[](int);
        const T& operator[](int) const;   // const-vector indexing (very common)
        T& at(int); const T& at(int) const;
        unsigned long size() const;
        bool empty() const;
        void push_back(const T&);
        void pop_back(); void clear();
        void resize(int); void resize(int, const T&);
        T& back(); T& front();
        T* begin(); T* end();
        const T* begin() const; const T* end() const;
    };
    template <typename T, typename U> T exchange(T&, U&&);
    template <typename T> void swap(T&, T&);
    template <typename T> struct totally_ordered { static const bool value = true; };
    template <typename T> struct equality_comparable { static const bool value = true; };
    template <typename T, typename Alloc = int> class list {
    public:
        list();
        void push_back(const T&); void push_front(const T&);
        void pop_back(); void pop_front(); void clear();
        bool empty() const; unsigned long size() const;
        T& front(); const T& front() const;
        T& back(); const T& back() const;
        T* begin(); T* end();
        const T* begin() const; const T* end() const;
    };
    template <typename K, typename V> struct pair { K first; V second; };
    template <typename... T> struct tuple { tuple(); tuple(T...); };
    template <typename K, typename V> class _mapbase {
    public:
        _mapbase();
        V& operator[](const K&);
        const V& operator[](const K&) const;
        V& at(const K&); const V& at(const K&) const;
        unsigned long size() const; bool empty() const;
        unsigned long count(const K&) const;
        void clear();
    };
    template <typename K, typename V> class unordered_map : public _mapbase<K, V> {};
    template <typename K, typename V> class map : public _mapbase<K, V> {};
    template <typename K, typename Cmp = int> class set {};
    template <typename K, typename Cmp = int> class unordered_set {};
    // Primary templates only -- real-world types routinely specialize these
    // (`template<> struct hash<MyType> {...};`) to make themselves usable as
    // unordered_map/unordered_set keys. Without a declared primary template,
    // libclang rejects the specialization outright ("explicit specialization
    // of undeclared template"), which used to fail parsing for any file with
    // this near-universal STL-interop idiom, unrelated to the file's own
    // logic (see e.g. OCCT's gp_Pnt.hxx).
    template <typename T> struct hash { unsigned long operator()(const T&) const; };
    template <typename T> struct equal_to { bool operator()(const T&, const T&) const; };
    template <class T, class Container = int, class Cmp = int> class priority_queue {};
    template <class T> class queue { public:
        queue(); void push(const T&); void pop();
        T& front(); const T& front() const; T& back(); const T& back() const;
        bool empty() const; unsigned long size() const;
    };
    template <class T> class stack { public:
        stack(); void push(const T&); void pop();
        T& top(); const T& top() const; bool empty() const; unsigned long size() const;
    };
    template <class T> class deque {};
    template <typename R, typename... A> class function {};
    class exception {};
    using size_t = unsigned long;
    using ptrdiff_t = long;
    class string { public:
        string(); string(const char*); string(const string&);
        unsigned long size() const; unsigned long length() const; bool empty() const;
        char& operator[](unsigned long); const char& operator[](unsigned long) const;
        char& at(unsigned long); const char& at(unsigned long) const;
        const char* begin() const; const char* end() const;
        void push_back(char); void clear();
        string operator+(const string&) const; string operator+(const char*) const;
        string& operator+=(const string&); string& operator+=(const char*); string& operator+=(char);
    };
    string operator+(const char*, const string&);
    class string_view {};
    template <typename T> T numeric_limits_min() { return T(); }
    template <typename T> T numeric_limits_max() { return T(); }
    template <typename T> class numeric_limits { public: static T min(); static T max(); };
    class underflow_error {};
    template <bool B, typename T = void> struct enable_if {};
    template <typename T> T min(const T&, const T&);
    template <typename T> T max(const T&, const T&);
    template <typename T> class initializer_list { public:
        const T* begin() const; const T* end() const; unsigned long size() const; };
    template <typename T> T min(initializer_list<T>);
    template <typename T> T max(initializer_list<T>);
    string to_string(int); string to_string(long); string to_string(long long);
    string to_string(unsigned); string to_string(unsigned long);
    string to_string(float); string to_string(double);
    // Math intrinsics -- the existing strict engine (cmath
    // detection) routes these to `from math import <name>` in
    // Mojo / Rust. The original frontend declared them at TU
    // scope (see `_PREAMBLE` in core.py); when the C++ source
    // already wraps them in `std::`, we need a `std::`-qualified
    // declaration too, otherwise libclang errors out.
    double sqrt(double); double exp(double); double log(double);
    double log10(double); double log2(double); double cbrt(double);
    double pow(double, double);
    double sin(double); double cos(double); double tan(double);
    double asin(double); double acos(double); double atan(double);
    double atan2(double, double);
    double sinh(double); double cosh(double); double tanh(double);
    double fabs(double); double ceil(double); double floor(double);
    double round(double); double trunc(double);
    double fmod(double, double); double hypot(double, double);
    double fmin(double, double); double fmax(double, double);
    double fma(double, double, double); double copysign(double, double);
    double expm1(double); double log1p(double); double erf(double);
    int abs(int); long labs(long);
    double clamp(double, double, double); int clamp(int, int, int);
    // iostream-shaped IO. The strict engine doesn't model stream
    // operators (>> / <<); calls to cin/cout become HirRaw holes
    // that every backend emits as TODO[port] stubs. We declare
    // them as opaque class types with the operator overloads so
    // libclang's template-based type resolution doesn't fail.
    class istream { public: istream& operator>>(int&); istream& operator>>(long&);
                            istream& operator>>(double&); istream& operator>>(char&);
                            istream& operator>>(char*); istream& operator>>(void*); };
    class ostream { public: ostream& operator<<(int); ostream& operator<<(long);
                            ostream& operator<<(double); ostream& operator<<(char);
                            ostream& operator<<(const char*);
                            ostream& operator<<(ostream& (*)(ostream&)); };
    istream& cin = *(istream*)0;
    ostream& cout = *(ostream*)0;
    ostream& cerr = *(ostream*)0;
    ostream& clog = *(ostream*)0;
    ostream& endl(ostream&);
    // C stdio -- used by competitive-programming examples.
    int scanf(const char*, ...);
    int printf(const char*, ...);
    int sscanf(const char*, const char*, ...);
    int sprintf(char*, const char*, ...);
    // Common algorithm-shaped templates that the backends already
    // know about (mostly as `list[T]` aliases). Declared as opaque
    // templates so libclang can resolve them when the user code
    // calls them with concrete types.
    template <typename It, typename T> void fill(It, It, const T&);
    template <typename It, typename T> It find(It, It, const T&);
    template <typename It, typename T> It remove(It, It, const T&);
    template <typename It, typename T> T accumulate(It, It, T);
    template <typename It> void sort(It, It);
    template <typename It, typename Cmp> void sort(It, It, Cmp);
    template <typename It> It unique(It, It);
    template <typename It, typename Pred> It unique(It, It, Pred);
    template <typename It, typename Out> Out copy(It, It, Out);
    template <typename It, typename Out> Out move(It, It, Out);
    // Smart pointers -- still leave the ownership lowering to
    // the inference / LLM pass; we just need the type to exist
    // so the AST can hold a constructor call.
    template <typename T> class shared_ptr { public: shared_ptr(T*); T* operator->(); T& operator*(); };
    template <typename T> class unique_ptr { public: unique_ptr(T*); T* operator->(); T& operator*(); };
    // Cast helpers.
    template <typename T, typename U> T* static_cast_helper(U*);
    template <typename T, typename U> T* dynamic_cast_helper(U*);
}
void* operator new(unsigned long, void*);
void* operator new(unsigned long);
void operator delete(void*);

// TU-scope `using` declarations so user code that writes bare
// `vector<int>` (instead of `std::vector<int>`) still resolves
// against the parser preamble. Mirrors the older `using` block in
// the original core._PREAMBLE.
using std::vector;
using std::list;
using std::pair;
using std::tuple;
using std::map;
using std::unordered_map;
using std::set;
using std::unordered_set;
using std::queue;
using std::stack;
using std::deque;
using std::priority_queue;
using std::string;
using std::string_view;
using std::function;
using std::shared_ptr;
using std::unique_ptr;
using std::ptrdiff_t;
using std::exception;
using std::numeric_limits;
using std::underflow_error;
"""

# Line markers clang emits (e.g. '# 42 "foo.cpp" 2'). Strip them so
# the byte stream we feed to libclang is just user code + the preamble.
_LINEMARKER_RE = re.compile(r"^#\s*\d+\s+\"[^\"]+\".*$", re.MULTILINE)
# `include` may or may not have a space between the directive and
# the header (`#include<string>` vs `#include <string>`). The
# `\s*` after `include` is the only thing keeping both forms working.
_INCLUDE_RE = re.compile(r"^#\s*include\s*[<\"][^\"<>]+[>\"]\s*$", re.MULTILINE)
_IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+(\S+)\s*$")
_DEFINE_GUARD_RE = re.compile(r"^\s*#\s*define\s+(\S+)\s*$")
_ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
# Any other conditional-opening directive. Needed so `#endif` always pops the
# frame it actually closes -- see the note in _strip_header_guard about why
# only tracking #ifndef on the stack is unsound.
_OTHER_IF_OPEN_RE = re.compile(r"^\s*#\s*(?:if|ifdef)\b")
# C++20 `requires` constraints that confuse our libclang + template
# combination (the parser confuses `void` with a type-cast). Match
# both the standalone form (`requires std::foo<T>` on its own line) and
# the inline form (`requires std::foo<T> void name(...)`).
_REQUIRES_LINE_RE = re.compile(r"^\s*requires\s+std::\S.*$", re.MULTILINE)
_REQUIRES_INLINE_RE = re.compile(r"requires\s+std::\w+<[^>]*>\s*")


def _strip_header_guard(text: str) -> str:
    """Remove the `#ifndef NAME` / `#define NAME` (bare) / matching `#endif`
    header-guard idiom, leaving every other conditional directive
    (`#ifdef`, `#if defined(...)`, `#else`, and their `#endif`) untouched.

    This has to track nesting depth rather than regex-match `#endif`/bare
    `#define` globally: real-world headers routinely wrap feature/export
    macros in `#ifdef X ... #else ... #endif` (e.g. a DLL-export macro
    defined differently per platform). A prior version of this function
    stripped *every* `#endif` and *every* bare `#define` line unconditionally,
    which silently deleted the `#endif` closing such a block while leaving
    its `#if`/`#else` behind -- corrupting the directive nesting so badly
    that libclang's own preprocessor (which handles `#ifdef`/`#else`/`#endif`
    natively and would have gotten this right on its own) then rejected the
    file with "#else after #else" / "unterminated conditional directive".
    Only a *paired* `#ifndef`+bare-`#define`+`#endif` is guaranteed to be a
    no-op guard; only that pairing is safe to remove.

    The stack must track *every* open `#if`/`#ifdef`/`#ifndef`, not just
    `#ifndef`: a real header commonly nests an unrelated conditional (e.g. a
    platform-specific compiler-bug workaround gated on
    `#if defined(__APPLE__)`) inside its own outer include-guard. If only
    `#ifndef` pushed a stack frame, that inner `#if`'s `#endif` would pop the
    *outer* guard's frame instead of its own (since every `#endif` popped
    unconditionally) -- stripping the inner `#endif` as if it were the outer
    guard's closer, and leaving the real outer guard's `#endif` behind as an
    orphan. libclang's own preprocessor then can't find a match for the
    still-open inner `#if` and treats it as false (on any platform other
    than the one it's guarding), silently skipping every line up to the next
    `#endif` it can find -- typically some unrelated header's guard closer
    thousands of lines later, swallowing whole classes with no diagnostic at
    all. Pushing a frame for every conditional keeps `#endif` matched to the
    directive it actually closes, guard or not.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    # Each stack entry is True if this #if-family level is a stripped guard.
    guard_stack: list[bool] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m_ifndef = _IFNDEF_RE.match(line)
        if m_ifndef:
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            m_define = _DEFINE_GUARD_RE.match(lines[j]) if j < n else None
            if m_define and m_define.group(1) == m_ifndef.group(1):
                guard_stack.append(True)
                i = j + 1  # drop both the #ifndef and the #define line
                continue
            guard_stack.append(False)
            out.append(line)
            i += 1
            continue
        if _OTHER_IF_OPEN_RE.match(line):
            guard_stack.append(False)
            out.append(line)
            i += 1
            continue
        if _ENDIF_RE.match(line):
            is_guard = guard_stack.pop() if guard_stack else False
            if not is_guard:
                out.append(line)
            i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def _strip_directives(text: str) -> str:
    """Remove cpp directives libclang does not need: linemarkers, includes,
    header guards, and C++20 `requires` clauses. Everything else
    (including user `#define`s that did not fire, blank lines, and real
    `#ifdef`/`#if`/`#else`/`#endif` conditionals) is preserved -- libclang's
    own preprocessor (invoked when we hand it this text) evaluates those
    correctly on its own."""
    out = _LINEMARKER_RE.sub("", text)
    out = _INCLUDE_RE.sub("", out)
    out = _strip_header_guard(out)
    out = _REQUIRES_LINE_RE.sub("", out)
    out = _REQUIRES_INLINE_RE.sub("", out)
    return out


def _clang_executable() -> str | None:
    """Locate the host `clang` binary. Returns None if no clang is found;
    callers that need strict preprocessor behaviour should treat that as
    a hard failure."""
    found = shutil.which("clang++") or shutil.which("clang")
    if found:
        return found
    for c in ("/usr/bin/clang++", "/usr/bin/clang"):
        if Path(c).is_file():
            return c
    return None


def preprocess_cpp(
    source: str,
    *,
    clang: str | None = None,
    std: str = "c++20",
    timeout: float = 15.0,
) -> str:
    """Run the real preprocessor on *source* and return the cleaned result.

    The cleaned result is *macro-expanded user code* -- directives stripped,
    `#include`s removed, `requires` clauses deleted, and the parser preamble
    prepended. The output is ready to hand to libclang.

    On any failure (no clang, timeout, non-zero exit) we fall back to the
    raw *source* with the same directive-stripping applied. This is
    deliberately lenient: the parser is a best-effort tool, and the failure
    mode for "no clang" should be the same as the failure mode for "we
    could not preprocess".
    """
    if not source:
        return PARSER_PREAMBLE

    if clang is None:
        clang = _clang_executable()
    if clang is None:
        return PARSER_PREAMBLE + _strip_directives(source)

    try:
        proc = subprocess.run(
            [clang, "-E", f"-std={std}", "-x", "c++", "-nostdinc++", "-"],
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError, subprocess.TimeoutExpired, OSError:
        return PARSER_PREAMBLE + _strip_directives(source)

    if proc.returncode != 0 or not proc.stdout:
        return PARSER_PREAMBLE + _strip_directives(source)

    return PARSER_PREAMBLE + _strip_directives(proc.stdout)


__all__ = ["PARSER_PREAMBLE", "preprocess_cpp"]

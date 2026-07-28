"""Local (project-relative) `#include "X.h"` resolution for multi-file C++.

The strict engine's frontend is designed around single self-contained
translation units: ``preprocess.py`` strips every ``#include`` line
unconditionally (angle *and* quoted) so libclang never trips over a missing
system header. That works well for algorithm-corpus slices, but a real
multi-file C++ project splits declarations (``.h``) from definitions
(``.cpp``) across files -- feeding just the ``.cpp`` in means the class the
out-of-line methods belong to is never declared, and every method fails to
parse with "use of undeclared identifier".

``resolve_local_includes`` closes that specific gap: given an entry-point
file, it transitively inlines every ``#include`` it can actually find on
the search path -- base classes, shared typedefs, sibling declarations --
so the combined text is a self-contained translation unit before it ever
reaches ``preprocess_cpp``. An include that can't be found on the search
path is left completely alone; it's still stripped by the existing
``preprocess.py`` pipeline exactly as before (its real declarations were
never available toolchain-free anyway).

Whether a header is "local" is decided by *resolvability on the search
path*, not by ``"..."`` vs ``<...>`` spelling: real build systems routinely
put a project's own include directory on the ``-I`` search path, so a
project's own headers often use angle brackets too (e.g. Topologic's
`#include <Bitwise.h>` for its own sibling header) -- the angle/quote
distinction is only a *search-order* hint in the C++ standard, not a
local/system classification, and treating it as one would silently skip
exactly the includes this function exists to resolve.

This does not attempt to be a real preprocessor: no macro-guarded
conditional ``#include``, no distinct ``-I``/``-isystem`` search order
semantics, no handling of the same logical header appearing under two
different relative spellings. It solves the common case -- "this project's
own headers, once each" -- which is what unblocks real multi-file corpora
like a CAD kernel's core library.

One real-preprocessor behavior it *does* have to replicate: header guards
(``#ifndef FOO_H`` / ``#define FOO_H``, or ``#pragma once``). Real C++
headers routinely have circular *textual* includes -- A.h forward-declares
B and includes B.h only at the *bottom* of the file (for inline method
bodies that need B complete), while B.h includes A.h at its *top* (for the
reverse reason). A real preprocessor handles this by expanding each
``#include`` in place and skipping a header's body entirely on re-entry
(the guard macro is already defined) -- so B.h's inclusion from A.h "just"
resumes A.h's own text afterward, by which point B is a complete type. A
naive "hoist every dependency's full text before the file that wants it"
model (this module's original implementation) cannot represent that: it
would need to hoist *both* A.h and B.h before each other, which is
impossible, and ends up parking one of them too late, leaving the other
looking at an incomplete type. Inlining strictly in place (like a real
preprocessor) sidesteps the paradox for free: on re-entry the include line
just disappears, exactly as ``#pragma once`` would make it, and whichever
of A/B's own class body was already mid-expansion simply continues right
where its ``#include`` was, now with the other one fully defined above it.
"""
from __future__ import annotations

import re
from pathlib import Path

_INCLUDE_RE = re.compile(r'^[ \t]*#\s*include\s*[<"]([^">]+)[>"][^\n]*', re.MULTILINE)

_HEADER_SUFFIXES = (".hxx", ".hpp", ".h", ".hh")
_IMPL_SUFFIXES = (".cxx", ".cpp", ".cc", ".c")


def _find_header(name: str, requesting_dir: Path, include_dirs: list[Path]) -> Path | None:
    """Search alongside the including file first (the standard's own rule
    for ``"..."`` includes -- harmless to apply to ``<...>`` too here, since
    we only fall back to it after nothing already resolved), then each
    explicit search directory, in order."""
    candidate = requesting_dir / name
    if candidate.is_file():
        return candidate
    for d in include_dirs:
        candidate = d / name
        if candidate.is_file():
            return candidate
    return None


def _find_impl(header_path: Path, include_dirs: list[Path]) -> Path | None:
    """A header's sibling implementation file (same stem, a `.cxx`/`.cpp`/
    `.cc`/`.c` suffix), searched alongside the header first, then each
    search directory. Only headers have one to find; a `.cxx`/`.cpp` file
    passed here (e.g. the entry point itself, or one already pulled in as
    another header's impl) has no further impl of its own."""
    if header_path.suffix.lower() not in _HEADER_SUFFIXES:
        return None
    stem = header_path.stem
    for d in (header_path.parent, *include_dirs):
        for suffix in _IMPL_SUFFIXES:
            candidate = d / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def resolve_local_includes(
    path: str | Path,
    include_dirs: list[str | Path] | None = None,
    *,
    include_impls: bool = False,
) -> str:
    """Return *path*'s content with every transitively-reachable local
    header expanded in place, exactly where its ``#include`` line was.

    Headers are deduplicated by resolved absolute path -- like a
    ``#pragma once``/include-guard, a header's body is only expanded on its
    *first* encounter; every later ``#include`` of the same file (including
    ones reached via a dependency cycle) is simply dropped, which both
    terminates cycles and matches real preprocessor semantics for the
    common "A and B each need the other complete for inline bodies, but
    only forward-declare each other for the rest" idiom. An include not
    found on the search path (typically because it actually names a
    vendored third-party header) is left as-is; the existing preprocessing
    pipeline strips the directive line either way.

    ``include_impls`` (opt-in, default off so existing `-I`-only callers are
    unaffected): also pull in each inlined header's sibling `.cxx`/`.cpp`
    implementation file, if one exists on the search path. A real multi-file
    project routinely declares a method in a header (`Standard_EXPORT double
    Angle(...) const;`) and defines it out-of-line in that header's own
    `.cxx` -- fine for a real build (declaration is enough at compile time,
    the body is resolved at link time), but this engine emits one
    amalgamated translation unit with no separate link step, so calling
    into such a method from another *inlined* method reports "has no
    attribute" unless that `.cxx`'s real body is part of the same unit.

    Impls are appended as a distinct *second pass*, strictly after every
    header has been fully resolved -- not interleaved into the header
    expansion at the point each header is first reached. A `.cxx` file
    routinely `#include`s more headers than its own class's declarations
    strictly need (it has real method bodies to compile, not just
    signatures), which can introduce completeness cycles the pure header
    graph never had to solve: e.g. gp_Mat.cxx needing gp_XYZ complete for
    its own body, while gp_XYZ's header-side completion is still mid-stack
    because gp_Dir.hxx (reached from a different branch) is what's
    currently pulling gp_XYZ.hxx in. Resolving all headers first, with
    impls only appended once that settles, sidesteps this: by the time any
    `.cxx` body is emitted, every locally-known type is already complete,
    so there's nothing left for a `.cxx`'s own broader include list to
    race against. The entry file's own `.cxx` doesn't get "re-found" as
    its own header's impl (it's already in `seen` from the very first
    `expand()` call), so this can't loop back on the entry point itself.
    """
    entry = Path(path).resolve()
    dirs = [Path(d).resolve() for d in (include_dirs or [])]
    seen: set[Path] = set()
    visited_order: list[Path] = []

    def expand(file_path: Path) -> str:
        seen.add(file_path.resolve())
        visited_order.append(file_path)
        text = file_path.read_text(encoding="utf-8", errors="replace")

        def _replace(m: re.Match[str]) -> str:
            name = m.group(1)
            dep = _find_header(name, file_path.parent, dirs)
            if dep is None:
                return m.group(0)
            if dep.resolve() in seen:
                return ""
            return expand(dep)

        return _INCLUDE_RE.sub(_replace, text)

    out = expand(entry)
    if not include_impls:
        return out

    # Second pass: for every header resolved above (including ones an
    # impl's own broader #include list discovers along the way -- hence
    # the index-based loop over a list still being appended to, a
    # fixed-point walk rather than a single fixed-size iteration),
    # append its sibling impl once, in discovery order.
    impl_done: set[Path] = set()
    impl_parts: list[str] = []
    i = 0
    while i < len(visited_order):
        candidate = visited_order[i]
        i += 1
        rp = candidate.resolve()
        if rp in impl_done:
            continue
        impl_done.add(rp)
        impl = _find_impl(candidate, dirs)
        if impl is not None and impl.resolve() not in seen:
            impl_parts.append(expand(impl))
    return out + "\n" + "\n".join(impl_parts)


__all__ = ["resolve_local_includes"]

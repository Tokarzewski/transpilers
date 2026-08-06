"""libclang discovery + toolchain include-path configuration (leaf utility)."""

from __future__ import annotations

import glob as _glob
import os as _os

import clang.cindex as ci


def _configure_libclang() -> None:
    """Point the clang bindings at an available libclang shared library.

    Cross-platform. Priority order:
      1. ``$TRANSPILERS_LIBCLANG`` explicit override (any OS).
      2. Windows: newer libclang.dll from LLVM-MinGW / official LLVM (the PyPI
         ``libclang`` wheel caps at 18.1.1 — too old for libc++ 22 headers).
      3. POSIX: the bundled PyPI ``libclang`` wheel (clang/native/libclang.{so,dylib}),
         then common system locations. Without this the bindings only find an
         unversioned ``libclang.so`` on the loader path, which most distros
         don't ship — so the C++ frontend would fail to import on Linux.

    If nothing is found we leave the bindings at their default and let the first
    ``Index.create()`` raise a clear LibclangError.
    """
    override = _os.environ.get("TRANSPILERS_LIBCLANG")
    candidates: list[str] = [override] if override else []
    if _os.name == "nt":
        local_app = _os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.extend(
                _glob.glob(
                    _os.path.join(
                        local_app,
                        "Microsoft",
                        "WinGet",
                        "Packages",
                        "MartinStorsjo.LLVM-MinGW.*",
                        "llvm-mingw-*",
                        "bin",
                        "libclang.dll",
                    )
                )
            )
        candidates.append(r"C:\Program Files\LLVM\bin\libclang.dll")
    else:
        # System locations first: like the PyPI `libclang` wheel capping at
        # 18.1.1 on Windows (see module docstring), it's also stale on Linux
        # distros that ship a newer `clang` bindings package - an apt-installed
        # libclang matching the current LLVM release is more likely correct
        # than the bundled wheel, so it's tried first rather than last.
        candidates.extend(sorted(_glob.glob("/usr/lib/llvm-*/lib/libclang.so*")))
        candidates.extend(sorted(_glob.glob("/usr/lib/llvm-*/lib/libclang-*.so*")))
        candidates.extend(sorted(_glob.glob("/usr/lib/x86_64-linux-gnu/libclang*.so*")))
        candidates.extend(
            [
                "/opt/homebrew/opt/llvm/lib/libclang.dylib",
                "/usr/local/opt/llvm/lib/libclang.dylib",
            ]
        )
        try:  # bundled PyPI libclang wheel (last resort)
            import clang as _clang

            _native = _os.path.join(_os.path.dirname(_clang.__file__), "native")
            candidates.extend(
                sorted(_glob.glob(_os.path.join(_native, "libclang*.so*")))
            )
            candidates.extend(
                sorted(_glob.glob(_os.path.join(_native, "libclang*.dylib")))
            )
        except Exception:
            pass
    for p in candidates:
        if p and _os.path.isfile(p):
            try:
                ci.Config.set_library_file(p)
            except Exception:
                pass
            return


_SYSTEM_INCLUDE_CACHE: list[str] | None = None

_HOST_TRIPLE_CACHE: str | None = None


def _host_triple() -> str | None:
    """Default target triple of the located clang. We pass this to libclang
    so its built-in triple (which may differ — official LLVM on Windows
    defaults to ``...msvc`` and would error on libc++ headers from a
    MinGW install) matches the headers we're feeding it."""
    global _HOST_TRIPLE_CACHE
    if _HOST_TRIPLE_CACHE is not None:
        return _HOST_TRIPLE_CACHE or None
    import subprocess

    clang = _find_clang()
    if not clang:
        _HOST_TRIPLE_CACHE = ""
        return None
    try:
        out = subprocess.run(
            [clang, "-print-target-triple"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        triple = out.stdout.strip()
    except Exception:
        triple = ""
    _HOST_TRIPLE_CACHE = triple
    return triple or None


def _looks_like_path(s: str) -> bool:
    """Match an absolute path on either POSIX (`/...`) or Windows (`C:/`,
    `C:\\`). Reject lines like ``End of search list.`` or framework
    directory annotations."""
    if not s:
        return False
    if s.startswith("/"):
        return True
    # Windows drive letter, e.g. "C:\foo" or "C:/foo"
    return len(s) >= 3 and s[1] == ":" and s[2] in ("/", "\\")


def _find_clang() -> str | None:
    """Locate a clang/clang++ binary. Prefers PATH, then well-known Windows
    install locations dropped by winget (LLVM-MinGW, official LLVM) so that
    a fresh install works without the user editing PATH."""
    import glob
    import os
    import shutil

    found = shutil.which("clang++") or shutil.which("clang")
    if found:
        return found

    candidates: list[str] = []
    env_override = os.environ.get("TRANSPILERS_CLANG")
    if env_override:
        candidates.append(env_override)

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            # LLVM-MinGW (winget MartinStorsjo.LLVM-MinGW.{UCRT,MSVCRT})
            candidates.extend(
                glob.glob(
                    os.path.join(
                        local_app,
                        "Microsoft",
                        "WinGet",
                        "Packages",
                        "MartinStorsjo.LLVM-MinGW.*",
                        "llvm-mingw-*",
                        "bin",
                        "clang++.exe",
                    )
                )
            )
        # Official LLVM (winget LLVM.LLVM)
        candidates.append(r"C:\Program Files\LLVM\bin\clang++.exe")

    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _system_include_args() -> list[str]:
    """Ask the host `clang` for its system header search paths and return
    them as `-isystem` args. libclang's bundled headers (the `clang/<ver>/
    include/` directory) ship with `stddef.h`, `stdint.h`, etc. — adding
    them lets the corpus's `#include <stddef.h>` actually resolve."""
    global _SYSTEM_INCLUDE_CACHE
    if _SYSTEM_INCLUDE_CACHE is not None:
        return _SYSTEM_INCLUDE_CACHE
    import subprocess

    clang = _find_clang()
    if not clang:
        _SYSTEM_INCLUDE_CACHE = []
        return []
    try:
        out = subprocess.run(
            [clang, "-E", "-x", "c++", "-v", "-"],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        _SYSTEM_INCLUDE_CACHE = []
        return []
    # Output between "#include <...> search starts here:" and "End of search list."
    lines = out.stderr.splitlines()
    paths: list[str] = []
    in_block = False
    for line in lines:
        if "search starts here" in line:
            in_block = True
            continue
        if "End of search list" in line:
            break
        if in_block:
            p = line.strip()
            # macOS prints framework dirs like `/path (framework directory)` —
            # split on whitespace to pull out the leading path.
            head = p.split()[0] if p else ""
            if _looks_like_path(head):
                paths.append(head)
    args: list[str] = []
    for p in paths:
        args.extend(["-isystem", p])
    _SYSTEM_INCLUDE_CACHE = args
    return args


# Configure libclang at import time.
_configure_libclang()

__all__ = [
    "_configure_libclang",
    "_SYSTEM_INCLUDE_CACHE",
    "_HOST_TRIPLE_CACHE",
    "_host_triple",
    "_looks_like_path",
    "_find_clang",
    "_system_include_args",
]

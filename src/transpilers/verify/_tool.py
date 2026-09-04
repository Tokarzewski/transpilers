"""Shared toolchain-path resolution for the native-compiler verify gates.

On Windows, ``shutil.which()`` can find ``.BAT``/``.CMD`` shims (e.g. a pixi
``mojo.bat`` launcher) that a bare-name ``subprocess`` call cannot launch —
``CreateProcess`` only auto-appends ``.exe``. Resolving to the absolute path
``which()`` returns fixes the mismatch (``CreateProcess`` runs ``.BAT`` files
through ``cmd.exe`` when given the full name) and is a no-op on POSIX.
"""

from __future__ import annotations

import shutil


def resolve_tool(name: str) -> str:
    """Absolute path for *name* via PATH, or *name* unchanged if not found.

    Callers already handle a missing tool (their ``*_available()`` check or
    the not-found result path); this only normalizes the launch path so a
    tool that ``which()`` *can* see is also launchable.
    """
    return shutil.which(name) or name

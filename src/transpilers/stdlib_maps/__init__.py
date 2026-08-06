"""Source-stdlib -> target-stdlib mapping tables.

Data, not code. LLM-proposed mappings get promoted into these TOML files once
a human approves — this is the self-improving loop that bends the system from
LLM-heavy toward algorithmic-heavy over time.

Auto-generated mappings are produced by running::

    python scripts/gen_stdlib_maps.py

which writes ``auto_generated.yaml`` alongside this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_MAPS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_stdlib_maps(path: Path | str | None = None) -> dict[str, dict[str, list[str]]]:
    """Load the auto-generated stdlib mapping tables from YAML.

    Returns a dict with up to two top-level keys:

    .. code-block:: python

        {
            "cpp_to_python": {"std::sort": ["sorted", "list.sort"], ...},
            "cpp_to_mojo":   {"std::vector": ["List[T]"], ...},
        }

    Args:
        path: Path to the YAML file to load.  Defaults to
              ``stdlib_maps/auto_generated.yaml`` inside this package.
              Pass an explicit path to load a different file (useful for
              testing).

    Returns:
        A dict of mapping tables, or an empty dict if the file does not
        exist or cannot be parsed.

    Example::

        from transpilers.stdlib_maps import load_stdlib_maps

        maps = load_stdlib_maps()
        py_equiv = maps.get("cpp_to_python", {}).get("std::sort", [])
        # ["sorted", "list.sort"]
    """
    yaml_path = Path(path) if path is not None else _MAPS_DIR / "auto_generated.yaml"

    if not yaml_path.exists():
        return {}

    text = yaml_path.read_text(encoding="utf-8")

    # Try PyYAML first (optional dependency).
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return _normalise(data)
        return {}
    except ImportError:
        pass  # fall through to the hand-rolled parser

    return _parse_yaml_fallback(text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(data: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Ensure all values are ``dict[str, list[str]]``."""
    result: dict[str, dict[str, list[str]]] = {}
    for section, mapping in data.items():
        if not isinstance(mapping, dict):
            continue
        clean: dict[str, list[str]] = {}
        for k, v in mapping.items():
            if isinstance(v, list):
                clean[str(k)] = [str(x) for x in v]
            elif isinstance(v, str):
                clean[str(k)] = [v]
        result[str(section)] = clean
    return result


def _parse_yaml_fallback(text: str) -> dict[str, dict[str, list[str]]]:
    """Minimal hand-rolled YAML parser for the specific format gen_stdlib_maps.py produces.

    Handles only:
      - Top-level keys ending with ``:``.
      - Entries of the form ``  "key": ["val1", "val2"]``.
      - Comment lines starting with ``#``.
    """
    import re

    result: dict[str, dict[str, list[str]]] = {}
    current_section: str | None = None

    # Pattern: optional whitespace, quoted key, colon, inline list of quoted strings.
    entry_re = re.compile(
        r'^\s+"([^"]+)":\s*\[([^\]]*)\]',
    )
    section_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Skip blank lines and comments.
        if not line or line.lstrip().startswith("#"):
            continue

        # Section header?
        sec_m = section_re.match(line)
        if sec_m:
            current_section = sec_m.group(1)
            result.setdefault(current_section, {})
            continue

        # Entry?
        if current_section is None:
            continue

        ent_m = entry_re.match(line)
        if ent_m:
            key = ent_m.group(1)
            raw_list = ent_m.group(2)
            # Extract quoted strings from the inline list.
            values = re.findall(r'"([^"]*)"', raw_list)
            result[current_section][key] = values

    return result

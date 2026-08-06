"""Zim-based RAG for offline documentation lookup during transpilation.

When the LLM encounters an unknown C++ API call (e.g. `std::transform`,
`boost::filesystem::path`), this module searches local Kiwix .zim archives
at D:\\zim to find the equivalent Python/Mojo API and inject the doc snippet
into the LLM prompt.

Install libzim: pip install libzim

Relevant archives at D:\\zim:
  - devdocs_en_cpp_*.zim       — C++ standard library reference
  - docs.python.org_*.zim      — Python stdlib reference
  - mojolang.org_*.zim         — Mojo language reference
  - devdocs_en_gcc_*.zim       — GCC/compiler reference

Usage:
    from transpilers.rag import ZimRag, lookup_api

    rag = ZimRag(zim_dir="D:/zim")
    result = rag.lookup("std::sort", source_lang="cpp", target_lang="python")
    print(result.snippet)   # "sorted(iterable, *, key=None, reverse=False)"
    print(result.url)       # Article URL for citation

    # Or as a simple function call:
    snippet = lookup_api("unordered_map", source_lang="cpp", target_lang="python")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


ZIM_DIR_DEFAULT = Path(os.environ.get("ZIM_DIR", r"D:\zim"))

# Map source API patterns → search terms in the target language doc
CPP_TO_PYTHON_HINTS: dict[str, list[str]] = {
    "std::sort": ["sorted", "list.sort"],
    "std::vector": ["list", "array"],
    "std::unordered_map": ["dict", "defaultdict"],
    "std::map": ["dict", "SortedDict"],
    "std::string": ["str", "string"],
    "std::cout": ["print"],
    "std::cin": ["input"],
    "std::max": ["max"],
    "std::min": ["min"],
    "std::abs": ["abs", "math.fabs"],
    "std::sqrt": ["math.sqrt"],
    "std::pow": ["pow", "math.pow"],
    "std::floor": ["math.floor"],
    "std::ceil": ["math.ceil"],
    "std::round": ["round", "math.round"],
    "std::find": ["str.find", "list.index"],
    "std::transform": ["map", "list comprehension"],
    "std::accumulate": ["sum", "functools.reduce"],
    "std::fill": ["list comprehension", "itertools.repeat"],
    "std::copy": ["list.copy", "copy.copy"],
    "std::set": ["set"],
    "std::deque": ["collections.deque"],
    "std::queue": ["queue.Queue"],
    "std::stack": ["list"],
    "std::pair": ["tuple"],
    "std::tuple": ["tuple"],
    "std::optional": ["None", "Optional"],
    "std::shared_ptr": ["object reference"],
    "std::unique_ptr": ["object reference", "contextlib"],
    "std::move": ["(no equivalent — Python uses references)"],
    "printf": ["print", "f-string"],
    "sprintf": ["f-string", "str.format"],
    "malloc": ["list", "bytearray"],
    "free": ["(automatic — Python GC)"],
    "memcpy": ["copy", "slice assignment"],
    "strlen": ["len"],
    "strcmp": ["== operator"],
    "strcpy": ["= assignment"],
}

CPP_TO_MOJO_HINTS: dict[str, list[str]] = {
    "std::vector": ["List[T]"],
    "std::unordered_map": ["Dict[K, V]"],
    "std::map": ["Dict[K, V]"],
    "std::string": ["String"],
    "std::sort": ["sort()"],
    "std::max": ["max()"],
    "std::min": ["min()"],
    "std::abs": ["abs()"],
    "std::sqrt": ["math.sqrt()"],
    "std::optional": ["Optional[T]"],
    "std::pair": ["Tuple[A, B]"],
    "std::tuple": ["Tuple[...]"],
    "std::shared_ptr": ["Reference types in Mojo use ownership"],
    "std::unique_ptr": ["Owned[T]"],
    "printf": ["print()"],
}


@dataclass
class DocResult:
    """Result of a documentation lookup."""

    query: str
    found: bool
    snippet: str = ""
    url: str = ""
    target_api: str = ""
    source: str = ""  # "zim" | "hint_table" | "not_found"


def find_zim_files(zim_dir: Path | str = ZIM_DIR_DEFAULT) -> list[Path]:
    """Return all .zim files in the given directory, sorted by modification time."""
    d = Path(zim_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.zim"), key=lambda f: f.stat().st_mtime, reverse=True)


def _lang_zim_pattern(lang: str) -> str:
    """Return glob pattern for zim files relevant to a target language."""
    patterns = {
        "python": "docs.python.org",
        "mojo": "mojolang.org",
        "cpp": "devdocs_en_cpp",
        "rust": "doc.rust-lang.org",
        "go": "pkg.go.dev",
    }
    return patterns.get(lang, lang)


class ZimRag:
    """Offline documentation lookup over Kiwix .zim archives.

    Args:
        zim_dir: Directory containing .zim files. Defaults to D:\\zim
                 or the ZIM_DIR environment variable.

    Example:
        rag = ZimRag()
        r = rag.lookup("std::vector", source_lang="cpp", target_lang="python")
        if r.found:
            print(r.snippet)
    """

    def __init__(self, zim_dir: Path | str = ZIM_DIR_DEFAULT) -> None:
        self.zim_dir = Path(zim_dir)
        self._zim_cache: dict[str, object] = {}  # path -> libzim.Archive

    def _open_zim(self, path: Path) -> object | None:
        """Lazy-open a zim archive. Returns None if libzim not installed."""
        key = str(path)
        if key in self._zim_cache:
            return self._zim_cache[key]
        try:
            import libzim  # type: ignore

            archive = libzim.Archive(str(path))
            self._zim_cache[key] = archive
            return archive
        except ImportError:
            return None
        except Exception:
            return None

    def _search_zim(
        self, archive: object, query: str, max_results: int = 3
    ) -> list[str]:
        """Search a zim archive, return list of article snippets."""
        try:
            import libzim  # type: ignore

            searcher = libzim.Searcher(archive)
            results = searcher.search(libzim.Query().set_query(query))
            snippets = []
            for i, entry in enumerate(results):
                if i >= max_results:
                    break
                try:
                    item = entry.get_item()
                    content = bytes(item.content).decode("utf-8", errors="replace")
                    # Strip HTML tags for snippet
                    clean = re.sub(r"<[^>]+>", " ", content)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    snippets.append(clean[:500])
                except Exception:
                    pass
            return snippets
        except Exception:
            return []

    def lookup(
        self,
        api_name: str,
        *,
        source_lang: str = "cpp",
        target_lang: str = "python",
    ) -> DocResult:
        """Look up the target-language equivalent of a source API name.

        First tries the hint table (fast, no I/O), then searches .zim files.
        """
        # 1. Check hint table
        hints_map = (
            CPP_TO_PYTHON_HINTS if target_lang == "python" else CPP_TO_MOJO_HINTS
        )
        normalized = api_name.strip()
        for pattern, targets in hints_map.items():
            if pattern in normalized or normalized in pattern:
                snippet = f"{normalized} → {', '.join(targets)}"
                return DocResult(
                    query=api_name,
                    found=True,
                    snippet=snippet,
                    target_api=targets[0],
                    source="hint_table",
                )

        # 2. Search zim files
        target_pattern = _lang_zim_pattern(target_lang)
        zim_files = find_zim_files(self.zim_dir)
        target_zims = [f for f in zim_files if target_pattern in f.name.lower()]

        for zim_path in target_zims[:2]:  # search top-2 most recent zims
            archive = self._open_zim(zim_path)
            if archive is None:
                break
            snippets = self._search_zim(archive, normalized)
            if snippets:
                return DocResult(
                    query=api_name,
                    found=True,
                    snippet=snippets[0],
                    url=f"zim://{zim_path.name}/{normalized}",
                    source="zim",
                )

        return DocResult(query=api_name, found=False, source="not_found")

    def build_prompt_injection(
        self, unknown_apis: list[str], *, target_lang: str = "python"
    ) -> str:
        """Return a doc-context block to prepend to an LLM translation prompt.

        Args:
            unknown_apis: List of C++ API names encountered during parsing.
            target_lang: Target language for lookup.

        Returns:
            Formatted string ready to inject into an LLM prompt.
        """
        results = [self.lookup(api, target_lang=target_lang) for api in unknown_apis]
        found = [r for r in results if r.found]
        if not found:
            return ""
        lines = ["## API equivalence reference (from offline docs)\n"]
        for r in found:
            lines.append(f"- **{r.query}** → `{r.target_api or r.snippet[:80]}`")
        lines.append("")
        return "\n".join(lines)


# Module-level convenience function
_default_rag: ZimRag | None = None


def lookup_api(
    api_name: str,
    *,
    source_lang: str = "cpp",
    target_lang: str = "python",
    zim_dir: Path | str = ZIM_DIR_DEFAULT,
) -> DocResult:
    """Module-level convenience wrapper around ZimRag.lookup()."""
    global _default_rag
    if _default_rag is None or str(_default_rag.zim_dir) != str(zim_dir):
        _default_rag = ZimRag(zim_dir=zim_dir)
    return _default_rag.lookup(
        api_name, source_lang=source_lang, target_lang=target_lang
    )


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "std::vector"
    target = sys.argv[2] if len(sys.argv) > 2 else "python"
    result = lookup_api(query, target_lang=target)
    print(f"Query   : {result.query}")
    print(f"Found   : {result.found}")
    print(f"Source  : {result.source}")
    print(f"Snippet : {result.snippet}")
    if result.url:
        print(f"URL     : {result.url}")

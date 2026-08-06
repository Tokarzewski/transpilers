"""String support across targets.

Strings are the first place targets meaningfully diverge. Rust emits
`format!(...)` for concat and `String::from(...)` for literals; Zig emits
literals natively but refuses concat (runtime concat requires an allocator —
surfacing the gap rather than emitting broken code).
"""

from __future__ import annotations

import shutil
import textwrap

import pytest

from transpilers.cli.main import transpile_python_to_rust, transpile_python_to_zig
from transpilers.verify import rust_compiles, zig_compiles


def _rust(src: str) -> str:
    return transpile_python_to_rust(textwrap.dedent(src).lstrip())


def _zig(src: str) -> str:
    return transpile_python_to_zig(textwrap.dedent(src).lstrip())


# ---------- Rust ----------


def test_rust_string_literal_is_owned():
    out = _rust(
        """
        def shout() -> str:
            return "loud"
        """
    )
    assert "fn shout() -> String" in out
    assert 'String::from("loud")' in out


def test_rust_string_concat_uses_format():
    out = _rust(
        """
        def greet(name: str) -> str:
            return "hello, " + name
        """
    )
    assert "format!(" in out
    assert 'String::from("hello, ")' in out


def test_rust_concat_chain_flattens():
    out = _rust(
        """
        def triple(s: str) -> str:
            return s + s + s
        """
    )
    assert 'format!("{}{}{}", s, s, s)' in out


def test_rust_str_inferred_from_concat_with_literal():
    """`s + "x"` anchors `s: str` even without annotation."""
    out = _rust(
        """
        def tag(s):
            return s + "!"
        """
    )
    assert "fn tag(s: String) -> String" in out


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc not installed")
@pytest.mark.parametrize(
    "src",
    [
        """
        def shout() -> str:
            return "loud"
        """,
        """
        def greet(name: str) -> str:
            return "hello, " + name
        """,
        """
        def triple(s: str) -> str:
            return s + s + s
        """,
    ],
)
def test_rust_strings_compile(src: str):
    out = _rust(src)
    result = rust_compiles(out)
    assert result.ok, f"rustc rejected:\n{out}\n\nstderr:\n{result.stderr}"


# ---------- Zig ----------


def test_zig_string_literal_native():
    out = _zig(
        """
        def shout() -> str:
            return "loud"
        """
    )
    assert "fn shout() []const u8" in out
    assert 'return "loud";' in out


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not installed")
def test_zig_string_literal_compiles():
    out = _zig(
        """
        def shout() -> str:
            return "loud"
        """
    )
    result = zig_compiles(out)
    assert result.ok, result.stderr


def test_zig_refuses_concat_clearly():
    """Surface the gap, don't emit broken code."""
    with pytest.raises(NotImplementedError, match="string concatenation in Zig"):
        _zig(
            """
            def greet(name: str) -> str:
                return "hello, " + name
            """
        )

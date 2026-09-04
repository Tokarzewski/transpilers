"""Every native-compiler verifier must bound its `subprocess.run` call with a
timeout. Without one, a pathological emitted program (or a hung toolchain)
hangs the calling process indefinitely -- a real DoS vector given the API
server (`transpilers.api.server`) is network-facing with auth opt-in rather
than opt-out. Mocked so this runs without any compiler actually installed."""

from __future__ import annotations

import contextlib
import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize(
    "module_name, func_name, source, needs_which",
    [
        ("transpilers.verify.rust", "rust_compiles", "fn f() {}", False),
        ("transpilers.verify.c", "c_compiles", "int f(void) { return 0; }", True),
        ("transpilers.verify.mojo", "mojo_compiles", "def f():\n    pass\n", True),
        (
            "transpilers.verify.fortran",
            "fortran_compiles",
            "subroutine f()\nend subroutine f\n",
            True,
        ),
    ],
)
def test_verifier_subprocess_call_has_timeout(
    module_name, func_name, source, needs_which
):
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)

    fake_result = MagicMock(returncode=0, stderr="")
    with contextlib.ExitStack() as stack:
        mock_run = stack.enter_context(
            patch(f"{module_name}.subprocess.run", return_value=fake_result)
        )
        if needs_which:
            # Gates resolve launch paths through _tool.resolve_tool, so the
            # PATH lookup happens there — mock it at the source.
            stack.enter_context(
                patch(
                    "transpilers.verify._tool.shutil.which",
                    return_value="/usr/bin/fake-tool",
                )
            )
        if module_name == "transpilers.verify.mojo":
            # mojo_available() probes the tool once (cached); the which-mock
            # above is not enough — force the availability check to pass.
            stack.enter_context(
                patch(f"{module_name}.mojo_available", return_value=True)
            )
        fn(source)

    assert mock_run.called, f"{func_name} never invoked subprocess.run"
    _, kwargs = mock_run.call_args
    assert "timeout" in kwargs, (
        f"{func_name}'s subprocess.run call has no timeout= bound"
    )
    assert isinstance(kwargs["timeout"], (int, float)) and kwargs["timeout"] > 0

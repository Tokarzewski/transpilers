"""Wall-clock guard for in-process `exec()`/function calls on untrusted-shaped
source (behavioral + SMT/sampling verifiers). Without this, a pathological
input (infinite loop, runaway recursion) hangs the calling process forever --
there is no subprocess boundary to kill here, unlike the native-compiler
verifiers. Same SIGALRM pattern already used in
`scripts/sft/build_algorithm_pairs.py`; shared here for the two in-tree
verifiers that also `exec()` arbitrary source.
"""

from __future__ import annotations

import contextlib
import signal


class ExecTimeout(BaseException):
    """Derives from BaseException, not Exception, so callers' broad
    ``except Exception`` (used to capture source/target runtime behavior as a
    divergence, not a crash) cannot swallow the alarm and let a hung call
    loop forever."""


@contextlib.contextmanager
def time_limit(seconds: float):
    """No-op on platforms without SIGALRM (e.g. Windows) or when disabled."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise(signum, frame):  # noqa: ARG001
        raise ExecTimeout()

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

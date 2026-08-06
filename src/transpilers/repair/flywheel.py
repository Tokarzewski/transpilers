"""Flywheel recorder for the verification-driven repair loop (issue #47).

Every verified repair the loop produces is also the highest-value, hardest
SFT example (issue #51) — frontier-tier answers to cases the local model
missed. This module is the writer side of that pipeline: a thin append-only
JSONL store that the SFT promotion step reads back.

Schema stability matters: downstream tools (issue #51) consume this file
directly. Adding fields is safe; renaming requires a migration. See
:data:`FlywheelRecord` for the field list.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from transpilers.llm.client import ModelTier
from transpilers.repair.signal import RepairSignal

# Default path for the flywheel JSONL log. The loop appends a record per
# verified pass and the SFT pipeline (issue #51) reads it back. The default
# lives next to the LLM cache so the whole "hard-case" pipeline is colocated.
DEFAULT_FLYWHEEL_PATH = (
    Path(__file__).resolve().parent.parent / "llm" / "cache" / "flywheel.jsonl"
)


@dataclass
class FlywheelRecord:
    """One verified repair → candidate SFT data.

    Schema is consumed by issue #51's promotion step. Fields are stable —
    adding new ones is fine, renaming requires a migration.
    """

    source_lang: str
    target: str
    source: str
    broken_attempts: list[dict[str, Any]] = field(default_factory=list)
    fixed_code: str = ""
    fixing_tier: str = ""
    fixing_signal_kind: str = ""
    fixing_signal_bucket: str = ""
    fixing_signal_diagnostic: str = ""
    fix_prompt: str = ""
    timestamp: float = field(default_factory=lambda: time.time())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Render as a single JSONL line (compact, no trailing newline)."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlywheelRecord":
        """Inverse of :meth:`to_jsonl` — robust to extra keys."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


class Flywheel:
    """Append-only JSONL writer for verified repairs.

    The loop calls :meth:`record` once per verified pass. The writer
    intentionally *does not* lock the file — the loop is single-threaded
    per process and the worst-case conflict is a partial trailing line
    that the next consumer can detect and skip.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        also_stdout: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else DEFAULT_FLYWHEEL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.also_stdout = also_stdout
        self._written = 0

    @property
    def count(self) -> int:
        """Number of records written by *this* instance (not the file)."""
        return self._written

    def record(
        self,
        *,
        source_lang: str,
        target: str,
        source: str,
        broken_attempts: list[dict[str, Any]],
        fixed_code: str,
        fixing_tier: ModelTier | str,
        fixing_signal: RepairSignal,
        fix_prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> FlywheelRecord:
        """Persist a verified repair. Returns the record actually written."""
        rec = FlywheelRecord(
            source_lang=source_lang,
            target=target,
            source=source,
            broken_attempts=list(broken_attempts),
            fixed_code=fixed_code,
            fixing_tier=fixing_tier.value
            if isinstance(fixing_tier, ModelTier)
            else str(fixing_tier),
            fixing_signal_kind=fixing_signal.kind,
            fixing_signal_bucket=fixing_signal.bucket,
            fixing_signal_diagnostic=fixing_signal.diagnostic,
            fix_prompt=fix_prompt,
            extra=dict(extra or {}),
        )
        self._append_line(rec.to_jsonl())
        self._written += 1
        return rec

    def _append_line(self, line: str) -> None:
        # We open in append mode every time so a long-running daemon that
        # imports the module does not pin the file descriptor and block the
        # SFT promotion step from reading it.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
        if self.also_stdout:
            print(line)


def read_flywheel(
    path: Path | str | None = None,
    *,
    skip_malformed: bool = True,
) -> Iterator[FlywheelRecord]:
    """Stream records from the flywheel log. Used by the SFT promotion step."""
    p = Path(path) if path is not None else DEFAULT_FLYWHEEL_PATH
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                if skip_malformed:
                    continue
                raise
            yield FlywheelRecord.from_dict(d)


def merge_dedup(
    records: Iterable[FlywheelRecord],
    *,
    key_fields: tuple[str, ...] = ("source", "fixed_code"),
) -> list[FlywheelRecord]:
    """De-duplicate a stream of records by the joined key fields.

    Issue #51's promotion step is the canonical consumer: it wants one
    record per (source, fixed_target) pair, not a flood of identical
    repairs from repeated re-runs.
    """
    seen: set[tuple] = set()
    out: list[FlywheelRecord] = []
    for r in records:
        key = tuple(getattr(r, f) for f in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

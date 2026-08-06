"""Repair-outcome data model + persistent tracker (issue #51, the data flywheel).

Every behaviorally-verified repair in the pipeline lands here as a
:class:`RepairOutcome` record: *what failed, who/what fixed it, and how it
ranks in the algorithmic-vs-LLM ratio*. The :class:`RepairTracker` writes
records to a JSONL log so the loop is reproducible and auditable.

The metric we care about
------------------------
For each unit (source, target) the pipeline produces one of four end-states:

* ``algorithmic``     - the algorithmic transpiler emitted compile-clean code
                        with no LLM call at any point.
* ``rule``            - a deterministic rule (e.g. :mod:`scripts.sft.mojo_repair`)
                        repaired a mechanical defect with no LLM call.
* ``llm``             - an LLM (Claude / Qwen / etc.) was invoked to repair
                        or fill a type hole.
* ``unrepaired``      - even after LLM repair attempts the unit does not pass
                        the verify gate.

Of these, ``llm`` is the *costly* bucket. The flywheel's north-star metric is
``llm_fraction = llm / (algorithmic + rule + llm)`` - the share of *passing*
units that needed an LLM. As verified repairs are promoted to
``stdlib_maps/`` and the SFT corpus, the algorithmic emitter / rule patches
should be able to handle more of these cases unaided, and the fraction should
fall monotonically.

The promotion priority is the opposite: when a unit can ONLY be made to pass
by an LLM (no algorithmic emitter, no rule fix), it is the highest-priority
candidate to feed back into the algorithmic side. See
:mod:`scripts.sft.promote_repair`.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Default location: one persistent log, append-only, gitignored.
DEFAULT_LOG_PATH = Path(
    os.environ.get("REPAIR_OUTCOMES_LOG", "data/repair_outcomes.jsonl")
)

# Snapshot of the aggregated ratio: git-tracked for the trend line.
DEFAULT_METRICS_PATH = Path(
    os.environ.get("FLYWHEEL_METRICS_PATH", "data/flywheel_metrics.json")
)


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VALID_VERDICTS = ("algorithmic", "rule", "llm", "unrepaired")


def _coerce_verdict(v: str) -> str:
    """Normalise free-form verdict strings into the canonical 4-value set."""
    v = (v or "").strip().lower()
    if v in VALID_VERDICTS:
        return v
    if v in ("none", "skip", "skipped"):
        return "unrepaired"
    if v in ("ok", "pass", "passed"):
        return "algorithmic"
    return "unrepaired"


# ---------------------------------------------------------------------------
# The unit-of-work record
# ---------------------------------------------------------------------------


@dataclass
class RepairOutcome:
    """One verified-or-not unit, with the path it took through the pipeline.

    Designed to be cheap to emit (one dict write per unit) and cheap to roll
    up (counters over the ``verdict`` field give the LLM fraction directly).
    """

    # --- identity (what unit) -------------------------------------------------
    source_lang: str
    target: str
    fingerprint: str  # sha256[:16] of the source code (caller-computed)
    source_id: str = ""  # free-form: file path, repo/fn, or record id
    construct: str = ""  # offending construct / symbol / type hole, if known
    bucket: str = ""  # taxonomy bucket (parse / unresolved-symbol / ...)

    # --- outcome (who fixed it) ----------------------------------------------
    # One of: "algorithmic" | "rule" | "llm" | "unrepaired".
    # ``verdict == "unrepaired"`` is the only terminal state with no LLM that
    # didn't help; the other three all passed the verify gate.
    verdict: str = "unrepaired"

    # --- accounting -----------------------------------------------------------
    n_llm_calls: int = 0  # how many LLM round-trips were needed (0, 1, 2, ...)
    n_rule_passes: int = 0  # how many deterministic rule patches were applied
    n_repair_passes: int = 0  # total repair iterations (LLM + rule combined)
    wallclock_ms: int = 0  # end-to-end time for the unit, in milliseconds

    # --- bookkeeping ----------------------------------------------------------
    ts: float = field(default_factory=time.time)
    notes: str = ""  # free-form: prompt id, cache key, model, etc.

    # --- convenience predicates ----------------------------------------------
    @property
    def is_pass(self) -> bool:
        """Did the unit make it through the verify gate?"""
        return _coerce_verdict(self.verdict) in ("algorithmic", "rule", "llm")

    @property
    def is_algorithmic_or_rule(self) -> bool:
        """Did we solve it WITHOUT a single LLM call? (the metric numerator)"""
        return _coerce_verdict(self.verdict) in ("algorithmic", "rule")

    @property
    def is_frontier_only(self) -> bool:
        """Frontier-only: an LLM was the only thing that got us to PASS.

        This is the highest-priority case for promotion back into the
        algorithmic side - see :mod:`scripts.sft.promote_repair`.
        """
        v = _coerce_verdict(self.verdict)
        return v == "llm" and self.n_rule_passes == 0

    # --- serialisation -------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "RepairOutcome":
        d = json.loads(line)
        # Tolerate older records that lacked a ts field.
        d.setdefault("ts", 0.0)
        d.setdefault("notes", "")
        return cls(**d)


# ---------------------------------------------------------------------------
# Persistent tracker
# ---------------------------------------------------------------------------


class RepairTracker:
    """Append-only JSONL log of repair outcomes + ratio aggregator.

    Usage::

        tracker = RepairTracker()                    # writes to data/repair_outcomes.jsonl
        tracker.record(RepairOutcome(
            source_lang="cpp", target="mojo",
            fingerprint=fp, verdict="llm", n_llm_calls=1,
        ))
        tracker.flush_metrics()                      # refresh data/flywheel_metrics.json
        snap = tracker.aggregate()                   # in-memory rollup
        print(snap["llm_fraction"], snap["trend"])

    The class is deliberately *not* thread-safe; per-thread instances are
    fine because each ``record()`` is a single fwrite+flush, and the
    aggregator reads from the on-disk log when the in-memory cache is empty.
    """

    def __init__(
        self,
        log_path: Path | str | None = None,
        metrics_path: Path | str | None = None,
        *,
        create_dirs: bool = True,
    ) -> None:
        self.log_path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
        self.metrics_path = (
            Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH
        )
        if create_dirs:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        # Append-only; never re-write history.
        self._fh = self.log_path.open("a", encoding="utf-8")

    # -----------------------------------------------------------------------
    # Writing
    # -----------------------------------------------------------------------

    def record(self, outcome: RepairOutcome) -> None:
        """Append one outcome to the log. Normalises verdict first."""
        outcome.verdict = _coerce_verdict(outcome.verdict)
        self._fh.write(outcome.to_json() + "\n")
        self._fh.flush()

    def record_many(self, outcomes: Iterable[RepairOutcome]) -> int:
        n = 0
        for o in outcomes:
            self.record(o)
            n += 1
        return n

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "RepairTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------------

    def iter_log(self) -> Iterator[RepairOutcome]:
        """Yield every outcome in the on-disk log (one JSON per line)."""
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield RepairOutcome.from_json(line)
            except json.JSONDecodeError, TypeError, KeyError:
                # Skip corrupted lines rather than abort the whole rollup.
                continue

    def aggregate(self) -> dict[str, Any]:
        """Compute the algorithmic-vs-LLM ratio + a per-source/target breakdown.

        Returned keys (stable - these are the keys ``flywheel_run.py`` and the
        dashboard use):

        * ``total``             - total outcomes counted
        * ``passed``            - outcomes with ``is_pass == True``
        * ``by_verdict``        - ``Counter({algorithmic, rule, llm, unrepaired})``
        * ``llm_fraction``      - ``llm / passed``  (0.0 if passed == 0)
        * ``algorithmic_fraction`` - ``(algorithmic+rule) / passed``
        * ``by_source_lang``    - ``dict[source_lang, dict[verdict, n]]``
        * ``by_target``         - ``dict[target, dict[verdict, n]]``
        * ``frontier_only``     - count of LLM-only passes (promotion candidates)
        * ``n_llm_calls``       - total LLM round-trips across the log
        * ``trend``             - list of {ts, llm_fraction, total} sorted by ts;
                                  this is the line the dashboard plots
        """
        by_verdict: Counter[str] = Counter()
        by_source: dict[str, Counter[str]] = {}
        by_target: dict[str, Counter[str]] = {}
        frontier_only = 0
        n_llm = 0
        # For the trend: bin the log into windows of 50 outcomes.
        trend_window = 50
        trend: list[dict[str, Any]] = []
        seen = 0
        bin_seen = 0
        bin_llm = 0
        bin_passed = 0
        bin_first_ts: float | None = None

        def _flush_bin() -> None:
            nonlocal bin_seen, bin_llm, bin_passed, bin_first_ts
            if bin_seen == 0:
                return
            trend.append(
                {
                    "ts": bin_first_ts or 0.0,
                    "total": bin_seen,
                    "passed": bin_passed,
                    "llm": bin_llm,
                    "llm_fraction": (bin_llm / bin_passed) if bin_passed else 0.0,
                }
            )
            bin_seen = bin_llm = bin_passed = 0
            bin_first_ts = None

        for o in self.iter_log():
            seen += 1
            bin_seen += 1
            if bin_first_ts is None:
                bin_first_ts = o.ts
            v = _coerce_verdict(o.verdict)
            by_verdict[v] += 1
            by_source.setdefault(o.source_lang, Counter())[v] += 1
            by_target.setdefault(o.target, Counter())[v] += 1
            if o.is_pass:
                bin_passed += 1
            if v == "llm":
                bin_llm += 1
                n_llm += o.n_llm_calls
                if o.is_frontier_only:
                    frontier_only += 1
            if bin_seen >= trend_window:
                _flush_bin()
        _flush_bin()  # final partial window

        passed = sum(by_verdict[v] for v in ("algorithmic", "rule", "llm"))
        llm_count = by_verdict["llm"]
        llm_fraction = (llm_count / passed) if passed else 0.0
        alg_count = by_verdict["algorithmic"] + by_verdict["rule"]
        return {
            "total": seen,
            "passed": passed,
            "by_verdict": dict(by_verdict),
            "llm_fraction": llm_fraction,
            "algorithmic_fraction": (alg_count / passed) if passed else 0.0,
            "by_source_lang": {k: dict(v) for k, v in by_source.items()},
            "by_target": {k: dict(v) for k, v in by_target.items()},
            "frontier_only": frontier_only,
            "n_llm_calls": n_llm,
            "trend": trend,
        }

    # -----------------------------------------------------------------------
    # Snapshot to disk
    # -----------------------------------------------------------------------

    def flush_metrics(self) -> dict[str, Any]:
        """Write the current aggregate to ``metrics_path`` and return it.

        The on-disk file is small (one dict) and is the canonical place
        downstream tooling (the dashboard, the LLaMA-Factory mix weight
        scheduler, the README badge) reads from.
        """
        snap = self.aggregate()
        snap["updated_at"] = time.time()
        snap["log_path"] = str(self.log_path)
        self.metrics_path.write_text(
            json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return snap


# ---------------------------------------------------------------------------
# Convenience: one-liner rollup (used by the dashboard, no class state)
# ---------------------------------------------------------------------------


def rollup(
    log_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
) -> dict[str, Any]:
    """Read the log, write the metrics file, return the snapshot.

    Equivalent to ``RepairTracker(...).flush_metrics()`` but does not keep
    the file handle open - useful from one-shot scripts and CI.
    """
    with RepairTracker(log_path=log_path, metrics_path=metrics_path) as t:
        return t.flush_metrics()

"""Cached, typed-hole LLM client.

Three rules this module enforces:
  1. LLMs operate on typed holes, never free-form text. Every call declares an
     expected response shape and a validator.
  2. Results are cached by hash(prompt + model + temperature) so transpilation
     is reproducible and bills don't recur on re-runs.
  3. Validators reject malformed output before it touches the IR.

The Anthropic call is intentionally lazy-imported so the module is usable in
test/CI without API keys; algorithmic-only slices of the pipeline must not
require LLM credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, TypeVar


T = TypeVar("T")

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Tier model (issue #47)
# ---------------------------------------------------------------------------


class ModelTier(Enum):
    """Escalation order for the verification-driven repair loop (issue #47).

    Tiers are tried in order; the loop escalates to the next tier only when
    the previous tier's answer failed verification. ``CACHED`` is essentially
    free (a cache hit on a previously-verified prompt is the cheapest possible
    answer and the loop should prefer it).
    """

    CACHED = "cached"
    LOCAL_FINETUNED = "local_finetuned"
    FRONTIER = "frontier"

    @classmethod
    def ordered(cls) -> tuple["ModelTier", ...]:
        """The escalation order, low → high cost."""
        return (cls.CACHED, cls.LOCAL_FINETUNED, cls.FRONTIER)

    def __lt__(self, other: "ModelTier") -> bool:  # type: ignore[override]
        order = self.ordered()
        return order.index(self) < order.index(other)


@dataclass
class TierConfig:
    """Per-tier wiring.

    A tier is "available" when its model name is known *and* its backend has
    credentials. ``CACHED`` is always available (cache hit → free answer;
    cache miss → fall through to the next tier).
    """

    tier: ModelTier
    model: str
    backend: str = ""  # "openai" | "anthropic" | "" (use env-derived default)
    base_url: str = ""  # openai-compatible endpoint (vLLM, TGI)
    api_key: str = ""  # optional override
    extra: dict = field(default_factory=dict)

    def available(self) -> bool:
        """True when this tier has both a model and an auth path."""
        if self.tier == ModelTier.CACHED:
            return True
        if not self.model:
            return False
        backend = (self.backend or _default_backend()).lower()
        if backend == "anthropic":
            return bool(self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if backend == "openai":
            return bool(
                self.api_key
                or os.environ.get("OPENAI_API_KEY")
                or self.base_url
                or os.environ.get("OPENAI_BASE_URL")
            )
        # Unknown backend — let the call site decide; treat as available if
        # the model is set so test fakes can inject a callable.
        return bool(self.model)


def _default_backend() -> str:
    if os.environ.get("TRANSPILER_LLM_BACKEND"):
        return os.environ.get("TRANSPILER_LLM_BACKEND", "").lower()
    if os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    return "anthropic"


@dataclass
class TypedHole(Generic[T]):
    """A request the LLM is allowed to fill.

    `kind` keys the prompt template; `context` is serialized into the prompt;
    `validate` rejects malformed output. The hole's type T is what the
    validated answer deserializes to.
    """

    kind: str
    context: dict
    validate: Callable[[str], T]


class LlmClient:
    def __init__(
        self, model: str | None = None, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> None:
        # Model resolves from (explicit arg) > TRANSPILER_LLM_MODEL env > default.
        # For a self-hosted vLLM server this is the served name, e.g.
        # "Qwen2.5-Coder-3B-Instruct".
        self.model = model or os.environ.get("TRANSPILER_LLM_MODEL", DEFAULT_MODEL)
        # OPENAI_BASE_URL may be a single endpoint or a comma-separated POOL of
        # OpenAI-compatible servers (e.g. NPU + iGPU + CPU). Holes round-robin
        # across the pool so all available hardware shares the load.
        self._base_urls = [
            u.strip().rstrip("/")
            for u in os.environ.get("OPENAI_BASE_URL", "").split(",")
            if u.strip()
        ]
        self._rr = 0
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fill(self, hole: TypedHole[T], *, temperature: float = 0.0) -> T:
        prompt = self._render(hole)
        cache_key = self._cache_key(prompt, temperature)
        cached = self._cache_get(cache_key)
        raw = cached if cached is not None else self._call(prompt, temperature)
        if cached is None:
            self._cache_put(cache_key, raw)
        return hole.validate(raw)

    def _render(self, hole: TypedHole) -> str:
        template_path = Path(__file__).parent / "prompts" / f"{hole.kind}.md"
        template = template_path.read_text() if template_path.exists() else "{context}"
        return template.replace("{context}", json.dumps(hole.context, indent=2))

    def _cache_key(self, prompt: str, temperature: float) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(str(temperature).encode())
        h.update(prompt.encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> str | None:
        f = self.cache_dir / f"{key}.txt"
        return f.read_text() if f.exists() else None

    def _cache_put(self, key: str, value: str) -> None:
        (self.cache_dir / f"{key}.txt").write_text(value)

    def _call(self, prompt: str, temperature: float) -> str:
        # Backend select: TRANSPILER_LLM_BACKEND=openai|anthropic, else inferred
        # (openai when OPENAI_BASE_URL is set, otherwise anthropic). This lets a
        # self-hosted model (e.g. Qwen2.5-Coder-3B via vLLM) serve the holes
        # instead of a hosted API, at on-prem cost.
        backend = os.environ.get("TRANSPILER_LLM_BACKEND", "").lower()
        if not backend:
            backend = "openai" if os.environ.get("OPENAI_BASE_URL") else "anthropic"
        if backend == "openai":
            return self._call_openai(prompt, temperature)
        return self._call_anthropic(prompt, temperature)

    def _call_openai(self, prompt: str, temperature: float) -> str:
        """OpenAI-compatible chat endpoint — works with a self-hosted vLLM
        server (`vllm serve <model>`), HF TGI, or the OpenAI API. Configure with
        OPENAI_BASE_URL (e.g. ``http://<host>:8000/v1``) and OPENAI_API_KEY
        (vLLM accepts any token — use ``EMPTY`` if the server is unsecured)."""
        if not self._base_urls:
            raise RuntimeError(
                "OpenAI-compatible LLM backend selected but OPENAI_BASE_URL is not set "
                "(e.g. http://<host>:8000/v1 for a vLLM server; comma-separate for a pool)."
            )
        # Round-robin across the endpoint pool — successive holes hit different
        # devices (NPU / iGPU / CPU), spreading load across the hardware.
        base_url = self._base_urls[self._rr % len(self._base_urls)]
        self._rr += 1
        # Stdlib-only POST to /chat/completions — no `openai` SDK dependency, so
        # any OpenAI-compatible server (vLLM, FastFlowLM/Lemonade, TGI, OpenAI)
        # works out of the box.
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 1024,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"] or ""

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLM hole reached and ANTHROPIC_API_KEY not set. Either annotate the source "
                "to keep the pipeline algorithmic, set the key, or point at a self-hosted model "
                "via TRANSPILER_LLM_BACKEND=openai + OPENAI_BASE_URL."
            )
        from anthropic import Anthropic

        client = Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Tiered client (issue #47)
# ---------------------------------------------------------------------------


class _NoCacheHit(Exception):
    """Internal: no tier ≤ max_tier has a cached answer for this hole."""


class _StubTier:
    """Test-only tier: a callable producing a raw string, with an in-memory
    cache so test harnesses can simulate cache hits/elevation patterns."""

    def __init__(
        self, fn: Callable[[str, float], str], cached: dict[str, str] | None = None
    ) -> None:
        self.fn = fn
        self.cached: dict[str, str] = cached if cached is not None else {}


def _hole_prompt(hole: "TypedHole | str") -> str:
    """Best-effort prompt extraction for stubs that only care about the
    rendered string (cache lookups)."""
    if isinstance(hole, str):
        return hole
    return LlmClient()._render(hole)


def _client_from_config(cfg: TierConfig, cache_dir: Path) -> LlmClient:
    """Materialise a single-tier ``LlmClient`` from a ``TierConfig``."""
    # We override the env-derived default backend by pre-setting the env
    # vars that ``LlmClient._call`` consults. This is intentional: the
    # TieredLlmClient treats each tier as fully independent, and the LlmClient
    # singleton reads its backend from env on every call.
    prior_backend = os.environ.get("TRANSPILER_LLM_BACKEND")
    prior_base_url = os.environ.get("OPENAI_BASE_URL")
    prior_api_key = os.environ.get("OPENAI_API_KEY")
    try:
        if cfg.backend:
            os.environ["TRANSPILER_LLM_BACKEND"] = cfg.backend
        if cfg.base_url:
            os.environ["OPENAI_BASE_URL"] = cfg.base_url
        if cfg.api_key:
            os.environ["OPENAI_API_KEY"] = cfg.api_key
        return LlmClient(model=cfg.model, cache_dir=cache_dir)
    finally:
        # Restore (or unset) so we don't leak tier-local state into the
        # surrounding process.
        for var, prior in (
            ("TRANSPILER_LLM_BACKEND", prior_backend),
            ("OPENAI_BASE_URL", prior_base_url),
            ("OPENAI_API_KEY", prior_api_key),
        ):
            if prior is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prior


class TieredLlmClient:
    """Tier-aware wrapper around one ``LlmClient`` per escalation tier.

    The repair loop (issue #47) escalates ``CACHED → LOCAL_FINETUNED →
    FRONTIER``; the tiered client:

    * **CACHED** — short-circuits to a free cache hit when a previous run
      already produced a verified answer for the same prompt. Tries each
      configured tier's on-disk cache. Cache miss → return ``None`` and let
      the loop escalate.
    * **LOCAL_FINETUNED** — uses the self-hosted Qwen2.5-Coder (the LoRA
      adapter from ``data/sft/cpp_mojo/``) served via vLLM. Cheap, on-prem,
      and the loop's preferred *non-cache* tier.
    * **FRONTIER** — falls back to the Anthropic API (or whatever
      ``TierConfig.backend`` says) for the hard cases the local model misses.

    The tiered client shares the on-disk cache layout with ``LlmClient`` (the
    cache key already includes the model name, so a frontier response and a
    local-fine-tuned response live at distinct paths and can both be
    retrieved independently).

    Each tier can be:

    * a configured ``TierConfig`` (will materialise an ``LlmClient``),
    * an existing ``LlmClient`` (used as-is), or
    * a callable ``(prompt, temperature) -> str`` for tests (wrapped in a
      ``_StubTier`` that also carries an in-memory cache so tests can
      pre-seed the cache and assert hit/elevation behavior).

    A tier that is missing or whose config is not ``available()`` is treated
    as a permanent miss: the loop simply escalates past it.
    """

    def __init__(
        self,
        tiers: "dict[ModelTier, TierConfig | LlmClient | Callable[[str, float], str] | None] | None" = None,
        *,
        default_cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.tiers: dict[ModelTier, object] = {}
        for tier in ModelTier.ordered():
            cfg = (tiers or {}).get(tier)
            # CACHED is special: when not configured, materialise a default
            # disk-backed LlmClient so the loop's pin-to-CACHED path has
            # somewhere to land. This is the issue's "verified answer
            # replay" mechanism: every successful repair writes to CACHED
            # and the next run reuses it for free.
            if cfg is None and tier == ModelTier.CACHED:
                cfg = LlmClient(
                    model=f"__tiered_cached__{DEFAULT_MODEL}",
                    cache_dir=default_cache_dir,
                )
            self.tiers[tier] = _materialise_tier(cfg, default_cache_dir)

    # ---- public API ----

    def available_tiers(self) -> list[ModelTier]:
        """Tiers that are configured, in escalation order."""
        return [t for t in ModelTier.ordered() if self.tiers.get(t) is not None]

    def fill(
        self,
        hole: "TypedHole[T]",
        *,
        max_tier: ModelTier = ModelTier.FRONTIER,
        temperature: float = 0.0,
    ) -> "tuple[T, ModelTier]":
        """Fill *hole* at the first tier ≤ *max_tier* that has a cached answer.

        Returns ``(value, tier_used)``. **Does not call any LLM** — CACHED is
        the only path that *only* reads the cache. If no tier has a hit, the
        loop should escalate via :meth:`call`.
        """
        for tier in ModelTier.ordered():
            if tier > max_tier:
                break
            raw = self._cached_read_for_hole(tier, hole, temperature)
            if raw is None:
                continue
            try:
                return hole.validate(raw), tier
            except Exception:
                # Malformed cached value — drop and escalate.
                continue
        raise _NoCacheHit()

    def call(
        self,
        hole: "TypedHole[T] | str",
        *,
        tier: ModelTier,
        temperature: float = 0.0,
    ) -> str:
        """Call *tier* with the given hole (or pre-rendered prompt) and return
        the **raw** string the LLM produced. Caching is delegated to the
        underlying ``LlmClient`` when possible.

        If *hole* is a string it is treated as a pre-rendered prompt (no
        template lookup). Validation is the caller's responsibility.
        """
        target = self.tiers.get(tier)
        if target is None:
            raise RuntimeError(
                f"tier {tier.value!r} is not configured on this TieredLlmClient"
            )
        if isinstance(target, _StubTier):
            prompt = _hole_prompt(hole)  # type: ignore[arg-type]
            return target.fn(prompt, temperature)
        assert isinstance(target, LlmClient)
        if isinstance(hole, str):
            key = target._cache_key(hole, temperature)
            cached = target._cache_get(key)
            if cached is not None:
                return cached
            raw = target._call(hole, temperature)
            target._cache_put(key, raw)
            return raw
        return target.fill(hole, temperature=temperature)  # type: ignore[arg-type]

    # ---- cache helpers (for the repair loop) ----

    def cache_lookup(
        self,
        prompt: str,
        *,
        tier: ModelTier,
        temperature: float = 0.0,
    ) -> str | None:
        """Peek at the cache for a (tier, prompt) pair."""
        target = self.tiers.get(tier)
        if target is None:
            return None
        if isinstance(target, _StubTier):
            return target.cached.get(prompt)
        assert isinstance(target, LlmClient)
        return target._cache_get(target._cache_key(prompt, temperature))

    def cache_store(
        self,
        prompt: str,
        raw: str,
        *,
        tier: ModelTier,
        temperature: float = 0.0,
    ) -> None:
        """Force-store a verified answer in *tier*'s cache so the next run
        picks it up for free (issue #47: feeds the flywheel)."""
        target = self.tiers.get(tier)
        if target is None:
            return
        if isinstance(target, _StubTier):
            target.cached[prompt] = raw
            return
        assert isinstance(target, LlmClient)
        target._cache_put(target._cache_key(prompt, temperature), raw)

    # ---- internals ----

    def _cached_read_for_hole(
        self,
        tier: ModelTier,
        hole: "TypedHole[T] | str",
        temperature: float,
    ) -> str | None:
        target = self.tiers.get(tier)
        if target is None:
            return None
        if isinstance(target, _StubTier):
            prompt = _hole_prompt(hole)  # type: ignore[arg-type]
            return target.cached.get(prompt)
        assert isinstance(target, LlmClient)
        if isinstance(hole, str):
            return target._cache_get(target._cache_key(hole, temperature))
        return target._cache_get(target._cache_key(target._render(hole), temperature))


def _materialise_tier(
    cfg: "TierConfig | LlmClient | Callable[[str, float], str] | None",
    default_cache_dir: Path,
) -> object:
    """Coerce the per-tier constructor argument into one of:
    * ``_StubTier`` for a callable,
    * ``LlmClient`` for a ``TierConfig`` or an existing client.
    Returns ``None`` for ``None`` and for unavailable configs.

    Special case: when *cfg* is ``None`` and the caller is
    :class:`TieredLlmClient` constructing the **CACHED** tier, we
    materialise a default disk-backed :class:`LlmClient` so that the
    pin-to-CACHED path (issue #47) has somewhere to land. Non-CACHED
    tiers with a ``None`` config are returned as ``None`` (the loop
    escalates past them).
    """
    if cfg is None:
        return None
    if callable(cfg) and not isinstance(cfg, (LlmClient, TierConfig)):
        return _StubTier(cfg)  # type: ignore[arg-type]
    if isinstance(cfg, LlmClient):
        return cfg
    if isinstance(cfg, TierConfig):
        if not cfg.available():
            return None
        return _client_from_config(cfg, default_cache_dir)
    raise TypeError(
        f"tier value must be TierConfig, LlmClient, callable, or None; got {type(cfg).__name__}"
    )

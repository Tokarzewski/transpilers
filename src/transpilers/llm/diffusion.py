"""DiffusionGemma client — inference via diffusion sampling (not autoregressive).

DiffusionGemma (google/diffusion-gemma-26b-it) is a 26B MoE (3.8B active) model
that uses *text diffusion* instead of standard token-by-token autoregressive
decoding.  It generates 256-token *canvases* in parallel and iteratively
denoises them, achieving up to 4× faster inference on dedicated GPUs.

**Backends:**

*   **llama.cpp** (diffusion-aware branch) — the primary local-inference path;
    requires ``llama-diffusion-cli`` built from the ``gh pr 24423`` branch.
*   **vLLM** (OpenAI-compatible API) — for cloud / server deployment; vLLM
    exposes a ``/chat/completions`` endpoint with diffusion-specific parameters
    passed as extra_body fields.
*   **Unsloth Studio** — for interactive inference and fine-tuning via the
    Unsloth web UI.

**Hardware requirements (4-bit quantized):**

| Precision | Min VRAM |
|-----------|----------|
| Q4_K_M    | 18 GB    |
| Q5_K_M    | 20 GB    |
| Q8_0      | 28 GB    |
| BF16/FP16 | 52 GB    |

Usage::

    from transpilers.llm.diffusion import DiffusionGemmaClient

    client = DiffusionGemmaClient(backend="llamacpp", model="path/to/model.gguf")
    reply = client.generate("Translate C++ to Mojo:\n\n```cpp\nint add(int a, int b) { return a + b; }\n```")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFFUSION_MODEL_ID = "google/diffusion-gemma-26b-it"
"""HuggingFace model ID for the instruct-tuned DiffusionGemma."""

DIFFUSION_MODEL_GGUF_REPO = "unsloth/diffusiongemma-26B-A4B-it-GGUF"
"""HF repo hosting pre-quantized GGUF files for DiffusionGemma."""

DEFAULT_CANVAS_LEN = 256
"""Tokens per diffusion canvas (fixed by the model architecture)."""

DEFAULT_MAX_DENOISING_STEPS = 48
"""Maximum EB sampler denoising steps (fewer = faster, more = quality)."""

EB_SAMPLER_DEFAULTS = {
    "temperature_start": 0.8,
    "temperature_end": 0.4,
    "entropy_bound": 0.1,
    "confidence": 0.005,
    "max_steps": DEFAULT_MAX_DENOISING_STEPS,
    "adaptive_stopping": True,
}
"""Recommended entropy-bounded sampler defaults (from Unsloth docs)."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiffusionSamplerConfig:
    """Parameters for the entropy-bounded diffusion sampler.

    These control the denoising process — *not* the autoregressive
    temperature/top-p/top-k that standard LLMs use.
    """

    temperature_start: float = 0.8
    """Initial denoising temperature (linearly decayed to temperature_end)."""

    temperature_end: float = 0.4
    """Final denoising temperature."""

    entropy_bound: float = 0.1
    """Mutual-information bound for token selection.

    At each denoising step, the sampler selects tokens whose entropy is below
    this threshold.  Non-selected tokens are re-noised before the next step.
    """

    confidence: float = 0.005
    """Adaptive-stopping entropy threshold.

    When the average canvas entropy drops below this value, generation stops
    early (if adaptive_stopping is enabled).
    """

    max_steps: int = DEFAULT_MAX_DENOISING_STEPS
    """Maximum number of denoising steps (cap)."""

    adaptive_stopping: bool = True
    """Stop early when all tokens are confident (reduces latency)."""


@dataclass
class DiffusionGenerationConfig:
    """Top-level generation config for DiffusionGemma."""

    max_new_tokens: int = 1024
    """Approximate total tokens to generate (rounded up to canvas boundaries)."""

    canvas_len: int = DEFAULT_CANVAS_LEN
    """Tokens per diffusion canvas (must be 256 for the base model)."""

    sampler: DiffusionSamplerConfig = field(default_factory=DiffusionSamplerConfig)
    """Diffusion sampler parameters."""

    thinking: bool = False
    """Enable Gemma 4-style thinking mode (adds ``<|think|>`` token)."""

    system_prompt: str = ""
    """Optional system prompt (applied as a prefix)."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DiffusionGemmaClient:
    """Client for DiffusionGemma inference across multiple backends.

    Args:
        backend: One of ``"llamacpp"``, ``"vllm"``, or ``"unsloth"``.
        model: Model identifier (GGUF path for llama.cpp, HF model ID for vLLM).
        vllm_base_url: Base URL for vLLM OpenAI-compatible API
            (e.g. ``http://localhost:8000/v1``).  Required when ``backend="vllm"``.
        llamacpp_binary: Path to the ``llama-diffusion-cli`` binary.
            Auto-detected from ``PATH`` when not set.
        config: Default generation configuration.
    """

    def __init__(
        self,
        backend: Literal["llamacpp", "vllm", "unsloth"] = "llamacpp",
        model: str = "",
        *,
        vllm_base_url: str = "",
        llamacpp_binary: str = "",
        config: DiffusionGenerationConfig | None = None,
    ) -> None:
        self.backend = backend
        self.model = model or _default_model(backend)
        self.config = config or DiffusionGenerationConfig()

        if backend == "vllm":
            self._vllm_base_url = vllm_base_url or os.environ.get(
                "DIFFUSION_VLLM_BASE_URL", ""
            )
            if not self._vllm_base_url:
                raise RuntimeError(
                    "vLLM backend selected but no base URL provided. "
                    "Set DIFFUSION_VLLM_BASE_URL env var or pass vllm_base_url."
                )
        elif backend == "llamacpp":
            self._llamacpp_binary = (
                llamacpp_binary
                or os.environ.get("LLAMA_DIFFUSION_CLI")
                or _find_llamacpp_binary()
            )
            if not self._llamacpp_binary:
                raise RuntimeError(
                    "llama.cpp backend selected but `llama-diffusion-cli` not found "
                    "in PATH.  Build from https://github.com/ggml-org/llama.cpp "
                    "(gh pr checkout 24423) or set LLAMA_DIFFUSION_CLI env var."
                )
        elif backend == "unsloth":
            self._check_unsloth()

    # ---- Public API ----

    def generate(
        self,
        prompt: str,
        *,
        config: DiffusionGenerationConfig | None = None,
    ) -> str:
        """Generate text using DiffusionGemma.

        Args:
            prompt: Input prompt (user message).
            config: Optional per-call generation config override.

        Returns:
            Generated text (with thinking channel stripped by default).
        """
        cfg = config or self.config
        formatted = self._format_prompt(prompt, cfg)

        if self.backend == "llamacpp":
            return self._generate_llamacpp(formatted, cfg)
        elif self.backend == "vllm":
            return self._generate_vllm(formatted, cfg)
        elif self.backend == "unsloth":
            return self._generate_unsloth(formatted, cfg)
        raise ValueError(f"Unknown backend: {self.backend}")

    def generate_stream(
        self,
        prompt: str,
        *,
        config: DiffusionGenerationConfig | None = None,
    ):
        """Stream generated tokens as they are produced.

        Yields:
            str: Token fragments.
        """
        cfg = config or self.config
        formatted = self._format_prompt(prompt, cfg)

        if self.backend == "llamacpp":
            yield from self._generate_llamacpp_stream(formatted, cfg)
        elif self.backend == "vllm":
            yield from self._generate_vllm_stream(formatted, cfg)
        else:
            raise NotImplementedError(
                "Streaming is only supported for llamacpp and vllm backends."
            )

    # ---- Prompt formatting ----

    def _format_prompt(self, prompt: str, cfg: DiffusionGenerationConfig) -> str:
        """Apply system prompt and thinking-token preamble."""
        parts: list[str] = []
        if cfg.system_prompt:
            parts.append(cfg.system_prompt.strip())
        if cfg.thinking:
            parts.append("<|think|>")
        parts.append(prompt.strip())
        return "\n\n".join(parts)

    # ---- Backend: llama.cpp ----

    def _llamacpp_args(self, cfg: DiffusionGenerationConfig) -> list[str]:
        """Build llama-diffusion-cli argument list."""
        sampler = cfg.sampler
        args = [
            self._llamacpp_binary,
            "-m",
            self.model,
            "-ngl",
            "99",  # offload all layers to GPU
            "-cnv",  # multi-turn conversation mode
            "-n",
            str(cfg.max_new_tokens),
            "--diffusion-eb-max-steps",
            str(sampler.max_steps),
            "--diffusion-eb-t-max",
            str(sampler.temperature_start),
            "--diffusion-eb-t-min",
            str(sampler.temperature_end),
            "--diffusion-eb-entropy-bound",
            str(sampler.entropy_bound),
            "--diffusion-eb-confidence",
            str(sampler.confidence),
        ]
        if not sampler.adaptive_stopping:
            args.append("--no-diffusion-eb-adaptive")
        if cfg.thinking:
            args.extend(["--diffusion-thinking", "1"])
        return args

    def _generate_llamacpp(self, prompt: str, cfg: DiffusionGenerationConfig) -> str:
        """Run llama-diffusion-cli with the given prompt and return output."""
        args = self._llamacpp_args(cfg)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            result = subprocess.run(
                [*args, "--file", prompt_file],
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            raise RuntimeError("llama-diffusion-cli timed out after 300s")
        finally:
            Path(prompt_file).unlink(missing_ok=True)

        return self._strip_thinking(output)

    def _generate_llamacpp_stream(self, prompt: str, cfg: DiffusionGenerationConfig):
        """Stream tokens from llama-diffusion-cli."""
        args = self._llamacpp_args(cfg)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            proc = subprocess.Popen(
                [*args, "--file", prompt_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                yield line
        finally:
            Path(prompt_file).unlink(missing_ok=True)

    # ---- Backend: vLLM (OpenAI-compatible API) ----

    def _generate_vllm(self, prompt: str, cfg: DiffusionGenerationConfig) -> str:
        """Call vLLM's ``/chat/completions`` endpoint with diffusion params."""
        import urllib.request

        sampler = cfg.sampler
        body = {
            "model": self.model,
            "max_tokens": cfg.max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
            # Diffusion-specific parameters passed to the vLLM backend.
            "extra_body": {
                "diffusion_canvas_len": cfg.canvas_len,
                "diffusion_max_denoising_steps": sampler.max_steps,
                "diffusion_temperature_start": sampler.temperature_start,
                "diffusion_temperature_end": sampler.temperature_end,
                "diffusion_entropy_bound": sampler.entropy_bound,
                "diffusion_adaptive_stopping": sampler.adaptive_stopping,
            },
        }

        base_url = self._vllm_base_url.rstrip("/")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": (f"Bearer {os.environ.get('HF_TOKEN', 'EMPTY')}"),
            },
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
        raw = data["choices"][0]["message"]["content"] or ""
        return self._strip_thinking(raw)

    def _generate_vllm_stream(self, prompt: str, cfg: DiffusionGenerationConfig):
        """Stream from vLLM SSE endpoint."""
        import urllib.request

        sampler = cfg.sampler
        body = {
            "model": self.model,
            "max_tokens": cfg.max_new_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "extra_body": {
                "diffusion_canvas_len": cfg.canvas_len,
                "diffusion_max_denoising_steps": sampler.max_steps,
                "diffusion_temperature_start": sampler.temperature_start,
                "diffusion_temperature_end": sampler.temperature_end,
                "diffusion_entropy_bound": sampler.entropy_bound,
                "diffusion_adaptive_stopping": sampler.adaptive_stopping,
            },
        }

        base_url = self._vllm_base_url.rstrip("/")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": (f"Bearer {os.environ.get('HF_TOKEN', 'EMPTY')}"),
            },
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue

    # ---- Backend: Unsloth ----

    def _generate_unsloth(self, prompt: str, cfg: DiffusionGenerationConfig) -> str:
        """Generate using Unsloth's in-memory FastLanguageModel.

        .. note::
            This path requires the model to have been loaded into memory
            externally (e.g. via ``unsloth.FastLanguageModel.from_pretrained``).
            For one-shot inference, use the llama.cpp or vLLM backends instead.
        """
        try:
            from unsloth import FastLanguageModel  # noqa: F401,F811
        except ImportError:
            raise RuntimeError(
                "Unsloth backend requires the `unsloth` package. "
                "Install via `pip install unsloth`."
            )

        model, tokenizer = self._get_unsloth_model()
        formatted = self._format_prompt(prompt, cfg)
        inputs = tokenizer([formatted], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            diffusion_canvas_len=cfg.canvas_len,
            diffusion_denoising_steps=cfg.sampler.max_steps,
            diffusion_temperature_start=cfg.sampler.temperature_start,
            diffusion_temperature_end=cfg.sampler.temperature_end,
            diffusion_entropy_bound=cfg.sampler.entropy_bound,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the input prefix from the output.
        input_len = len(
            tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        )
        result = decoded[input_len:].strip()
        return self._strip_thinking(result)

    def _get_unsloth_model(self):
        """Lazy-load and cache the model for the Unsloth backend."""
        attr = "_unsloth_cache"
        cached: tuple | None = getattr(self, attr, None)
        if cached is not None:
            return cached
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise RuntimeError("Unsloth is required for the unsloth backend.")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model,
            max_seq_length=4096,
            dtype=None,  # auto-detect
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        setattr(self, attr, (model, tokenizer))
        return model, tokenizer

    _unsloth_cache: tuple | None = None

    # ---- Helpers ----

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove the ``<|think|>`` / thinking channel from output.

        When thinking mode is enabled, the model may emit::

            <|channel>thought ... <channel|> final answer

        This helper strips the thought channel and returns only the final answer.
        """
        text = re.sub(
            r"<\|channel\>thought.*?<channel\|\>",
            "",
            text,
            flags=re.DOTALL,
        )
        text = (
            text.replace("<|think|>", "")
            .replace("<|channel>", "")
            .replace("<channel|>", "")
        )
        return text.strip()

    @staticmethod
    def _check_unsloth() -> None:
        """Verify that the ``unsloth`` package is installed."""
        try:
            import unsloth  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Unsloth backend requires `unsloth`.  Install with:\n"
                "    pip install unsloth\n"
                "See https://unsloth.ai/docs/models/diffusiongemma"
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _default_model(backend: str) -> str:
    """Return the default model identifier for *backend*."""
    if backend == "llamacpp":
        candidates = [
            Path.home() / ".cache/llama.cpp/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
            Path.home() / ".cache/huggingface/hub/diffusiongemma.gguf",
            Path.cwd() / "models/diffusiongemma.gguf",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return (
            "unsloth/diffusiongemma-26B-A4B-it-GGUF/"
            "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
        )
    return DIFFUSION_MODEL_ID


def _find_llamacpp_binary() -> str:
    """Search ``PATH`` for the ``llama-diffusion-cli`` binary."""
    for name in ("llama-diffusion-cli", "llama-diffusion"):
        for dirpath in os.environ.get("PATH", "").split(os.pathsep):
            path = Path(dirpath) / name
            if path.exists() and os.access(str(path), os.X_OK):
                return str(path)
    return ""


def download_gguf(
    quant: str = "Q4_K_M",
    *,
    cache_dir: Path = Path.home() / ".cache/llama.cpp",
) -> Path:
    """Download a DiffusionGemma GGUF file from Hugging Face.

    Args:
        quant: Quantization type (``Q4_K_M``, ``Q5_K_M``, ``Q8_0``, etc.).
        cache_dir: Local cache directory.

    Returns:
        Path to the downloaded GGUF file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "Downloading GGUFs requires `huggingface_hub`.  Install with:\n"
            "    pip install huggingface_hub hf_transfer"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"diffusiongemma-26B-A4B-it-{quant}.gguf"
    repo_id = DIFFUSION_MODEL_GGUF_REPO
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
    )


def make_diffusion_client(
    backend: str = "llamacpp",
    model: str = "",
    **kwargs,
) -> DiffusionGemmaClient:
    """Factory function that creates a :class:`DiffusionGemmaClient`.

    Args:
        backend: ``"llamacpp"`` | ``"vllm"`` | ``"unsloth"``
        model: Model identifier.
        **kwargs: Passed to :class:`DiffusionGemmaClient`.

    Returns:
        Configured client.
    """
    return DiffusionGemmaClient(backend=backend, model=model, **kwargs)  # type: ignore[arg-type]

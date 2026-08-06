# transpilers

Hybrid algorithmic + LLM source-to-source transpiler. **N-to-M** across many
languages through a single shared IR pipeline.

**11 frontends** — Python, C, C++, Java, C#, TypeScript, JavaScript, Fortran,
Go, Visual Basic, and Assembly (via PyGhidra → C-like pseudocode).
**7 targets** — Rust, Zig, C, Mojo, Go, Python, Fortran.

## Why hybrid

Pure-algorithmic transpilers can't infer types or map idioms; pure-LLM ones
(TransCoder et al.) produce unverifiable output. So we split the work:

| Algorithmic | LLM |
|-------------|-----|
| Parse, symbol resolution, dataflow, emission, verification | Type-hole filling, idiom mapping, naming, comments |

Three rules the codebase enforces:

1. **LLMs operate on typed holes, never free-form text** — every call declares a
   response shape and a validator.
2. **Every LLM output is verified** — parse, compile, and behaviorally (tests).
3. **All LLM calls are cached** by `hash(prompt + model + temperature)`, so the
   pipeline is reproducible.

Promoted LLM answers land in `stdlib_maps/` as data, so the system bends from
LLM-heavy toward algorithmic-heavy over time.

## Architecture: three-tier IR

```
Frontend  →  HIR (source-faithful)  →  MIR (normalized, typed)  →  LIR (target-shaped)  →  Backend  →  verify
```

The key invariant: **HIR→MIR may not invent types it doesn't know.** Missing
information becomes an `UnknownT` hole that a later pass — algorithmic inference
first, LLM fallback — must explicitly fill. The pipeline refuses constructs it
can't model rather than emit wrong code.

A second engine, the **never-refuse lift** (`transpilers.lift`), does whole-file
C++ → Python 1:1, emitting `# TODO[lift]` stubs for gaps (~96% mechanical on
EnergyPlus). Run any granularity — object · file · module · folder/repo — with
`transpile-levels --engine {strict|lift}`.

## Status

Verified end-to-end (compile + run + output match) on `examples/algorithms/`
(18 Python files): **Rust, Go, Mojo, Python, Fortran, C all 18/18; Zig 17/18**
(needs ArrayList growth for `sieve.py`).

Real-world corpus (`examples/samples/`, transpile-only → Rust): JavaScript
26/27, C++ 78/170, Python 32/57, Java 27/74, C 20/70, Go 1/9.

The end-to-end Mojo and C++ numbers above are *compile-and-run* results that
require those toolchains to be installed. The engine itself (parse → IR →
emit, type inference, the verifiers, and all pure-Python paths) is exercised by
the pytest suite, which runs without any target toolchain. Mojo target
verification is gated on a working `mojo` install; where it is absent, the
matrix reports those cells as `SKIP` rather than `PASS`, so don't read the
Mojo figures as reproducible in a toolchain-less CI.

## Mojo-target SFT toolchain

Beyond the live engine, a toolchain-free data + fine-tune stack bootstraps the
C++/Python → Mojo translator adapter (all pure-Python, exercised by pytest):

- **Algorithm dataset generator** — `scripts/sft/build_algorithm_pairs.py`
  emits verified C/C++/Python → Mojo (plus Rust/Zig bonus) SFT pairs into
  `data/sft/algorithms/` from the seed corpus in `examples/`, with a rejects
  channel for unmappable constructs.
- **Qwen2.5-Coder-7B finetune** — `scripts/sft/train_7b.sh` plus the two-phase
  cloud configs in `data/sft/cpp_mojo/sft_7b_phase{1,2}_cloud.yaml`; see
  `docs/finetune_qwen7b.md`.
- **Record/replay shims** — `scripts/sft/record_replay.py` with bool/int
  normalization so captured fixtures replay deterministically.
- **Psychrometric shims** — `scripts/sft/psychro_shims.py` provides a real
  `PsyPsatFnTemp` implementation (replacing the prior `pvstar` stub) for
  EnergyPlus-flavored ports.
- **God-object slicing** — `scripts/sft/god_object_slice.py` emits per-module
  sub-state Mojo structs from a god-object slice.
- **Behavioral divergence classifier** — `src/transpilers/verify/behavioral.py`
  classifies semantic divergences (e.g. floored-vs-truncated integer division)
  in verified output.

## Layout

```
src/transpilers/   frontends · ir (hir/mir/lir) · passes · backends · llm · verify · cli
scripts/           dataset builders · sft/ (fine-tune + eval) · rag/
data/              datasets · the shipped Mojo LoRA adapter · codebase RAG index
examples/          verified corpus (algorithms/ + per-language samples)
tests/             engine test suite (684 tests; 45 Mojo-gated skips auto-skip, ML/torch cases auto-skip without GPU deps)
benchmarks/        transpilation-bench (40-task C++/Python→Mojo)
tools/             migraph (migration dashboard) · cloud (RunPod training bundle)
```

## Tooling

The project leans on a small set of core tools:

- **[uv](https://docs.astral.sh/uv/)** — Python deps · **[just](https://github.com/casey/just)** — task runner · **direnv** — shell activation.
- **[Hugging Face](https://huggingface.co/)** — base models and the fine-tuned
  C++/Python→Mojo LoRA adapter (`data/sft/cpp_mojo/adapter_15b_v2`, base
  `Qwen/Qwen2.5-Coder-1.5B-Instruct`) are pulled/hosted here; the SFT stack runs
  on the HF ecosystem (Transformers / PEFT / LLaMA Factory).
- **[ROCm / HIP SDK](https://rocm.docs.amd.com/)** — local AMD GPU compute for
  fine-tuning on Radeon hardware (RDNA3+); provides the `hipcc` toolchain and the
  ROCm PyTorch backend used by the local SFT runs (see `RUN.md`).
- **[RunPod](https://runpod.io/)** — cloud GPU fine-tuning for runs too big for
  local hardware (see `tools/cloud/` and `docs/runpod_guide.md`).

## Try it

```sh
direnv allow      # one-time: auto-activates .venv on cd
just setup        # uv sync
just example      # transpile examples/algorithms/fibonacci.py
just check        # lint + full test suite
```

Without `direnv`/`just`:

```sh
uv sync
uv run transpile examples/algorithms/fibonacci.py --target rust --verify
uv run pytest
```

End-to-end matrix — compile every output and verify it matches Python's stdout:

```sh
uv run python scripts/run_matrix.py examples/algorithms
uv run python scripts/transpile_matrix.py examples/samples/Python rust   # transpile only
```

Failure taxonomy — classify every verify-gate failure over a corpus into
`parse · unresolved-symbol · unfilled-UnknownT-hole · type-inference-miss ·
target-compile-error · output-mismatch · structural-divergence · timeout`
and roll it up per (source, target) pair (CSV + markdown):

```sh
uv run python scripts/failure_taxonomy.py examples/samples --targets rust,mojo --csv out.csv
```

## Fidelity dial

`transpile --fidelity {structural|idiomatic}` (default `structural`) controls
how faithful the output must be to the source's architecture. Under
`structural`, `--verify` additionally runs the **structural-fidelity
verifier**: the output's module/function/struct skeleton and control-flow
nesting must be isomorphic to the source's (idiom mapping is allowed only at
the statement/expression level — e.g. foreach→indexed loop, Rust's
struct+impl split, Fortran methods as free functions). Added, dropped, merged
or renamed functions and flattened control flow fail the gate. `idiomatic`
skips the skeleton gate, permitting parent-level rewrites.

Fine-tuning the Mojo translator adapter is documented in `RUN.md` (local
ROCm/WSL) and `tools/cloud/` (RunPod).

## Roadmap

Tracked in [GitHub Issues](https://github.com/Tokarzewski/transpilers/issues).
Highlights: dynamic array growth on Zig/Fortran (done on C), `OptionT`/`RefT` in
the type lattice, and statement-expression sequencing in HIR (to lift postfix
`i++`/`i--` out of expression contexts).

set shell := ["bash", "-cu"]
set dotenv-load := false

# List recipes
default:
    @just --list

# Install / refresh dependencies
setup:
    uv sync

# Run the test suite
test *args:
    uv run pytest {{args}}

# Transpile a file end-to-end with rustc verification
transpile file target="rust":
    uv run transpile {{file}} --target {{target}} --verify

# Transpile the bundled example
example:
    @just transpile examples/add.py

# Transpile every example to every supported target
examples-all:
    @for src in examples/*; do \
        case "$src" in *.py|*.c|*.cpp|*.java|*.cs|*.ts|*.js|*.f90|*.vb) ;; *) continue ;; esac; \
        for target in rust c mojo python fortran; do \
            echo "=== $src -> $target ==="; \
            just transpile $src $target || true; \
        done; \
    done

# Lint
lint:
    uv run ruff check src tests

# Format
fmt:
    uv run ruff format src tests

# Lint + tests — what CI should run
check: lint test

# Fine-tune on a cloud RunPod GPU (needs RUNPOD_API_KEY). Deps pulled ephemerally.
cloud-train model="Qwen/Qwen2.5-Coder-1.5B-Instruct" gpu="4090" *args:
    uv run --with runpod --with paramiko --with scp \
        python tools/cloud/runpod_train.py --model {{model}} --gpu {{gpu}} {{args}}

# Remove caches and build artefacts (leaves .venv)
clean:
    rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
    find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Data flywheel: pipe verified repairs back into stdlib_maps/ + SFT (issue #51).
# Run with GITHUB_TOKEN to crawl + repair, or without (--skip-crawl) to only
# promote + refresh metrics from the current repair_outcomes.jsonl.
flywheel mode="skip-crawl":
	uv run python scripts/sft/flywheel_run.py {{ if mode == "skip-crawl" { "--skip-crawl" } else { "" } }}

# Refresh the metrics snapshot + markdown from the current repair log
flywheel-metrics:
	uv run python scripts/sft/flywheel_metrics.py

# Pipe current repair log into stdlib_maps/ + SFT corpus
flywheel-promote:
	uv run python scripts/sft/promote_repair.py --refresh-metrics

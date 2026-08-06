"""LLMLift-style kernel classifier for C++ functions.

Classifies C++ functions into "kernel" (pure numeric/tensor computation,
suitable for Mojo and SIMD lifting) vs "application" (I/O, control flow,
business logic) to route them through the appropriate translation pipeline.

Inspired by:
    "LLMLift: Using Large Language Models for Lifting C to Verified LLVM-IR"
    and related work on verified lifting for high-performance code.

Design:
    - A function is a "kernel" if it is:
        * Dominated by arithmetic/numeric operations
        * Has array/pointer parameters (suggesting data parallelism)
        * No I/O, no exceptions, no global state mutation
        * Body is bounded-loop dominated (for / while with numeric bounds)
    - "Application" functions have I/O, string manipulation, polymorphism,
      dynamic dispatch, exception handling, or complex control flow.
    - "Mixed" functions contain both — split at statement level before lifting.

Usage:
    from transpilers.pipeline.kernel_classifier import classify_function, KernelClassifier

    result = classify_function(cpp_source="void saxpy(float* y, const float* x, float a, int n) { ... }")
    print(result.kind)       # FunctionKind.KERNEL
    print(result.confidence) # 0.92
    print(result.reasons)    # ["array pointer params", "arithmetic-dominated body", ...]

    # Batch classification
    clf = KernelClassifier()
    for func in parsed_functions:
        r = clf.classify(func.source)
        if r.is_kernel:
            pipeline.route_to_mojo_lift(func)
        else:
            pipeline.route_to_python_pivot(func)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class FunctionKind(str, Enum):
    KERNEL = "kernel"  # Pure numeric/tensor — route to Mojo direct lift
    APPLICATION = "application"  # I/O / business logic — route to Python pivot
    MIXED = "mixed"  # Contains both — recommend splitting
    UNKNOWN = "unknown"  # Couldn't determine


@dataclass
class ClassificationResult:
    kind: FunctionKind
    confidence: float  # 0.0 – 1.0
    reasons: list[str] = field(default_factory=list)
    kernel_score: float = 0.0
    app_score: float = 0.0

    @property
    def is_kernel(self) -> bool:
        return self.kind == FunctionKind.KERNEL

    @property
    def is_application(self) -> bool:
        return self.kind == FunctionKind.APPLICATION


# ---------------------------------------------------------------------------
# Heuristic signal patterns
# ---------------------------------------------------------------------------

# Kernel signals (numeric / SIMD-friendly)
_KERNEL_PATTERNS: list[tuple[str, float, str]] = [
    # (regex, weight, label)
    (r"\bfor\s*\(.*;\s*\w+\s*[<>]=?\s*\w+\s*;", 0.3, "bounded for-loop"),
    (r"\bwhile\s*\(", 0.15, "while-loop"),
    (r"\b(?:float|double|int32_t|int64_t|__m256|__m128)\b", 0.2, "numeric types"),
    (r"\[\s*\w+\s*\]", 0.2, "array indexing"),
    (r"\*\w+\s*[+\-\*/]", 0.15, "pointer arithmetic"),
    (r"\b(?:sqrt|pow|exp|log|sin|cos|fabs|fma|cbrt)\b", 0.25, "math intrinsics"),
    (r"#pragma\s+(?:omp|simd|ivdep|unroll)", 0.3, "parallelism pragma"),
    (r"\bstd::transform\b|\bstd::accumulate\b", 0.2, "STL algorithm"),
    (r"__builtin_|_mm256_|_mm512_|vld1q_|vmulq_", 0.4, "SIMD intrinsics"),
    (r"\bconst\s+\w+\s*\*", 0.1, "const pointer param"),
    (r"->data\(\)|\.data\(\)", 0.15, "contiguous buffer access"),
    (r"\bEigen::", 0.3, "Eigen matrix ops"),
    (r"\bblaze::|armadillo::|arma::", 0.3, "numerical library"),
    (r"\bstd::vector<(?:float|double|int)", 0.2, "numeric vector"),
]

# Application signals (I/O, exceptions, business logic)
_APP_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bstd::cout\b|\bprintf\b|\bfprintf\b|\bstd::cerr\b", 0.4, "console I/O"),
    (r"\bstd::cin\b|\bgetline\b|\bscanf\b", 0.3, "console input"),
    (r"\bstd::fstream\b|\bopen\b.*\bfile\b|\bfopen\b", 0.3, "file I/O"),
    (r"\bthrow\b|\bcatch\b|\btry\b", 0.35, "exception handling"),
    (r"\bnew\b|\bdelete\b", 0.2, "manual heap alloc"),
    (r"\bstd::string\b", 0.2, "string type"),
    (r"\bvirtual\b|\boverride\b|\bpolymorphi", 0.35, "OOP/polymorphism"),
    (r"\bdynamic_cast\b|\btypeid\b", 0.3, "dynamic dispatch"),
    (r"\bstd::map\b|\bstd::unordered_map\b", 0.2, "associative container"),
    (r"\bgoto\b", 0.2, "goto"),
    (r"\bsocket\b|\bbind\b|\baccept\b|\brecv\b", 0.5, "network I/O"),
    (r"\bgetenv\b|\bsystem\b|\bpopen\b", 0.4, "OS calls"),
    (r"\bstd::thread\b|\bstd::mutex\b", 0.25, "threading"),
    (r"\bboost::", 0.1, "Boost dependency"),
    (r"\bstd::shared_ptr\b|\bstd::unique_ptr\b", 0.15, "smart pointer"),
    (r"\bstd::variant\b|\bstd::any\b", 0.2, "type-erasure"),
]


class KernelClassifier:
    """Classifies C++ functions as kernels or application code.

    The classifier uses a weighted pattern-matching heuristic. For higher
    accuracy, plug in an LLM classifier via the `llm_classify` method.
    """

    def __init__(self) -> None:
        self._kernel_re = [
            (re.compile(pat, re.IGNORECASE), w, label)
            for pat, w, label in _KERNEL_PATTERNS
        ]
        self._app_re = [
            (re.compile(pat, re.IGNORECASE), w, label)
            for pat, w, label in _APP_PATTERNS
        ]

    def classify(self, cpp_source: str) -> ClassificationResult:
        """Classify a single C++ function or code block.

        Args:
            cpp_source: Raw C++ source text (function body or whole file).

        Returns:
            ClassificationResult with kind, confidence, and reasons.
        """
        kernel_score = 0.0
        app_score = 0.0
        reasons: list[str] = []

        for pattern, weight, label in self._kernel_re:
            if pattern.search(cpp_source):
                kernel_score += weight
                reasons.append(f"kernel: {label}")

        for pattern, weight, label in self._app_re:
            if pattern.search(cpp_source):
                app_score += weight
                reasons.append(f"app: {label}")

        # Normalize scores to [0, 1]
        total = kernel_score + app_score
        if total < 1e-9:
            return ClassificationResult(
                kind=FunctionKind.UNKNOWN,
                confidence=0.0,
                reasons=["no matching patterns"],
            )

        k_norm = kernel_score / total
        a_norm = app_score / total

        # Classify
        THRESHOLD = 0.65
        if k_norm >= THRESHOLD:
            kind = FunctionKind.KERNEL
            confidence = k_norm
        elif a_norm >= THRESHOLD:
            kind = FunctionKind.APPLICATION
            confidence = a_norm
        else:
            kind = FunctionKind.MIXED
            confidence = max(k_norm, a_norm)

        return ClassificationResult(
            kind=kind,
            confidence=round(confidence, 3),
            reasons=reasons,
            kernel_score=round(kernel_score, 3),
            app_score=round(app_score, 3),
        )

    def classify_functions(
        self, functions: list[dict]
    ) -> list[tuple[dict, ClassificationResult]]:
        """Classify a list of parsed function dicts.

        Each dict should have at minimum a 'source' key with the C++ source.

        Returns:
            List of (function_dict, ClassificationResult) tuples.
        """
        return [(fn, self.classify(fn.get("source", ""))) for fn in functions]

    def route(
        self,
        cpp_source: str,
    ) -> tuple[str, ClassificationResult]:
        """Return the recommended translation pipeline for the given source.

        Returns:
            (pipeline_name, result) where pipeline_name is one of:
                "mojo_direct"    — C++ → Mojo (single-step lift)
                "python_pivot"   — C++ → Python → Mojo
                "split_required" — Split function into kernel+app parts first
                "python_only"    — C++ → Python (no Mojo target needed)
        """
        result = self.classify(cpp_source)

        if result.kind == FunctionKind.KERNEL:
            pipeline = "mojo_direct"
        elif result.kind == FunctionKind.APPLICATION:
            pipeline = "python_pivot"
        elif result.kind == FunctionKind.MIXED:
            pipeline = "split_required"
        else:
            pipeline = "python_pivot"  # safe default

        return pipeline, result


# Module-level convenience function
_default_clf: KernelClassifier | None = None


def classify_function(cpp_source: str) -> ClassificationResult:
    """Module-level convenience wrapper around KernelClassifier.classify()."""
    global _default_clf
    if _default_clf is None:
        _default_clf = KernelClassifier()
    return _default_clf.classify(cpp_source)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kernel_classifier.py <cpp_file_or_snippet>")
        print("\nExample:")
        print(
            '  python kernel_classifier.py "void saxpy(float* y, const float* x, float a, int n) {'
        )
        print('      for (int i=0; i<n; ++i) y[i] = a*x[i] + y[i]; }"')
        sys.exit(0)

    source = sys.argv[1]
    # If it looks like a file path, read it
    import pathlib

    p = pathlib.Path(source)
    if p.exists():
        source = p.read_text()

    clf = KernelClassifier()
    pipeline, result = clf.route(source)

    print(f"Kind        : {result.kind.value}")
    print(f"Confidence  : {result.confidence:.0%}")
    print(f"Kernel score: {result.kernel_score}")
    print(f"App score   : {result.app_score}")
    print(f"Pipeline    : {pipeline}")
    print("\nSignals:")
    for r in result.reasons:
        print(f"  {r}")

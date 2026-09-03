# Classic algorithms corpus

Hand-curated Python implementations of canonical algorithms — used as a
cross-target end-to-end stress test. Each file is intentionally simple:
explicit types, basic control flow, no dependencies. The goal is to
verify that the pipeline carries algorithmic structure across all five
targets and produces output that *runs* and *matches Python*, not just
compiles.

```sh
# End-to-end: compile + run + diff stdout against Python reference.
uv run python scripts/run_matrix.py examples/algorithms

# Single target, with verification.
uv run transpile examples/algorithms/fibonacci.py --target mojo --verify
```

## Current pass rates (compile + run + output matches Python)

| Target | Pass | Notes |
|--------|-----:|-------|
| Rust   | 18/18 | |
| Mojo   | 18/18 | |
| Python | 18/18 | reference |
| Fortran| 18/18 | program-wrap + `pyfloat` helper |
| C      | 17/18 | sieve.py needs dynamic-array growth |

## Files

| File | Pattern |
|------|---------|
| `ackermann.py` | recursion stress (deep call stack) |
| `binary_search.py` | list parameter, integer division (`//`) |
| `bubble_sort.py` | in-place subscript-assign + tuple swap |
| `collatz.py` | unbounded while loop with branching |
| `digit_sum.py` | digit extraction via `//` and `%` |
| `fast_pow.py` | `O(log n)` exponentiation via repeated squaring |
| `fibonacci.py` | recursion + iteration (interprocedural inference) |
| `fizzbuzz.py` | branching + boolean composition |
| `gcd.py` | Euclid's algorithm — destructive while loop |
| `is_prime.py` | nested loops with early return |
| `lcm.py` | two functions in one module (`lcm` calls `gcd`) |
| `linear_search.py` | list iteration + early return |
| `mandelbrot.py` | nested float loops, escape-radius check |
| `newton_sqrt.py` | iterative refinement, float convergence |
| `palindrome_int.py` | pure-int digit reversal returning `bool` |
| `sieve.py` | Eratosthenes — dynamic list growth + index mutation |
| `sum_list.py` | accumulator pattern + max over a list |
| `triangle_num.py` | closed-form vs loop summation cross-check |

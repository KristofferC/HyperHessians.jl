# Why is HyperHessians faster than ForwardDiff?

Reproduction material for the "Anatomy of a speedup" report (`report.html` —
open in a browser, works offline). The report decomposes the measured speedup
into three multiplicative factors and backs each with generated-code evidence.

## Headline result (Apple Silicon, Julia 1.12.6, n = 256, chunk 8 for both)

| comparison | ackley | rosenbrock_1 |
| --- | --- | --- |
| vs **stock** ForwardDiff (v1.3.0 / v1.4.1) | 3.19× | 6.54× |
| vs **patched** ForwardDiff (see below) | 2.23× | 3.67× |

Decomposition vs patched ForwardDiff: **symmetry 1.94×** (k(k+1)/2 = 528
upper-triangular block-pair evaluations vs k² = 1024 full-square evaluations)
× **arithmetic per eval 1.0–1.7×** (muladd-fused 8-wide lanes vs unfused
9-wide lanes at identical 81-float payload) × **overhead ~1.1×** (0 B
allocated vs ~657 KB + full-array reseeding per Hessian).

## The ForwardDiff bug

Most of the per-operation gap against stock ForwardDiff is a missing
`@inline` on `dual_definition_retval` (`ForwardDiff/src/dual.jl`): for fat
nested duals Julia's inliner declines it, so every DiffRules-generated dual
op (including `*`) becomes two out-of-line calls plus a 648-byte `memcpy`.
`forwarddiff-inline.patch` fixes it (confirmed against master `090ddbb`,
v1.4.1) and is worth an upstream PR. Effect at n = 256, chunk 8:
nested-dual `*` 65.4 → 22.7 ns/element; ackley Hessian 28.3 → 20.4 ms;
rosenbrock Hessian 41.2 → 23.9 ms.

To benchmark against the patched ForwardDiff:

```sh
git clone https://github.com/JuliaDiff/ForwardDiff.jl
cd ForwardDiff.jl && git apply ../benchmark/why-faster/forwarddiff-inline.patch && cd ..
julia --project=benchmark -e 'import Pkg; Pkg.develop(path="ForwardDiff.jl")'
```

(This rewrites `benchmark/Manifest.toml` with a local path — don't commit
that. `Pkg.free("ForwardDiff")` restores the registry version.)

## Scripts

Run each with `julia --project=benchmark benchmark/why-faster/<script>`.
All pin the chunk size to 8 for both packages so ForwardDiff's nested
`Dual{Dual{Float64,8},8}` and `HyperDual{8,8}` carry the same 81-float,
648-byte payload.

| script | what it measures |
| --- | --- |
| `exp1_counts.jl` | evaluation counts (1024 vs 528), dual sizes, allocations, per-eval cost, totals, and the evals × cost ≈ total sanity check |
| `exp2_codegen.jl` | native instruction / FMA / call counts for single dual ops (`*`, `+`, `/`, `exp`, `sqrt`, `sin`) |
| `exp3_fullfunc.jl` | LLVM IR stats for whole benchmark functions: vectorization, memcpys, out-of-line call sites |
| `exp4_ops.jl` | per-element timing of single ops + whole-function evals at equal payload |
| `exp5_dump.jl` | dumps full `code_native`/`code_llvm` listings into `dumps/` (source of the report's assembly panels) |
| `exp6_fd8.jl` | end-to-end `hessian!` totals, eval counts, and allocations at chunk 8 |

Run exp2/exp3/exp5 with stock ForwardDiff to see the non-inlined
`j_dual_definition_retval` calls, and with the patch to see them gone.

## Findings that did not survive scrutiny

- Nested duals have **no algebraic redundancy** for `*`/`+` — the expanded
  product is term-for-term the HyperDual cross-term formula.
- Chunk-size choice is a non-factor (pinned to 8 everywhere here; FD's
  default heuristic changes its totals by <10%).
- Transcendental call counts are identical in both packages.
- HyperHessians' scalar `(f, f′, f″)` rule tables are mostly a wash once
  ForwardDiff inlines (`sqrt` 14.7 vs 13.8 ns; `exp` even favors FD); the
  dedicated division rule keeps a small edge.

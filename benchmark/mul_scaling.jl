# Per-multiply cost of HyperDual{n,n} vs nested Dual{Dual{n},n} as n = 1..12.
#
# The two representations carry an identical payload ((1+n)^2 Float64s) and
# their product rules perform the same flops, so any timing gap is data
# layout / codegen, not algorithm. Requires ForwardDiff master (>= 1.4.4,
# with the nested-dual inline fix #825). When the loaded HyperHessians has
# the `simd` flag (HyperDual{N1, N2, T, S}, master PR #57), the explicit
# SIMD.Vec-forced variant is measured as a third series.
#
# Run with: julia --project=benchmark benchmark/mul_scaling.jl
# Writes a TSV to benchmark/results/mulscale-<cpu>.tsv and prints it.

using HyperHessians, ForwardDiff, BenchmarkTools, Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "BENCH_SECONDS", "1.0"))

const L = 128  # elements per array; payload at n=12 is ~170 KiB per array

const HD = HyperHessians.HyperDual
const HAS_SIMD = length(Base.unwrap_unionall(HD).parameters) == 4
hdtype(n) = HAS_SIMD ? HD{n, n, Float64, false} : HD{n, n, Float64}
hdtype_simd(n) = HD{n, n, Float64, true}

rand_hd(::Type{T}, ::Val{n}) where {T, n} = T(
    rand(),
    ntuple(_ -> rand(), Val(n)),
    ntuple(_ -> rand(), Val(n)),
    ntuple(_ -> ntuple(_ -> rand(), Val(n)), Val(n)),
)

rand_inner(::Val{n}) where {n} = ForwardDiff.Dual{Nothing}(rand(), ntuple(_ -> rand(), Val(n)))
rand_nested(::Type, ::Val{n}) where {n} =
    ForwardDiff.Dual{Nothing}(rand_inner(Val(n)), ntuple(_ -> rand_inner(Val(n)), Val(n)))

function bench_mul(make, ::Type{T}, v::Val) where {T}
    a = [make(T, v) for _ in 1:L]
    b = [make(T, v) for _ in 1:L]
    out = similar(a)
    t = @belapsed ($out .= $a .* $b)
    return t / L
end

results = NamedTuple{(:n, :t_hd, :t_simd, :t_dd), Tuple{Int, Float64, Float64, Float64}}[]
for n in 1:12
    v = Val(n)
    t_hd = bench_mul(rand_hd, hdtype(n), v)
    t_simd = HAS_SIMD ? bench_mul(rand_hd, hdtype_simd(n), v) : NaN
    t_dd = bench_mul(rand_nested, Nothing, v)
    push!(results, (; n, t_hd, t_simd, t_dd))
    @printf("n=%2d  HyperDual{%d,%d}: %6.2f ns   simd: %6.2f ns   nested Dual: %6.2f ns\n",
        n, n, n, t_hd * 1e9, t_simd * 1e9, t_dd * 1e9)
end

resdir = joinpath(@__DIR__, "results")
isdir(resdir) || mkdir(resdir)
outfile = joinpath(resdir, "mulscale-$(Sys.CPU_NAME).tsv")
open(outfile, "w") do io
    println(io, "# julia $(VERSION), $(Sys.CPU_NAME), ForwardDiff $(pkgversion(ForwardDiff)), HyperHessians $(pkgversion(HyperHessians))")
    println(io, "n\thyperdual_ns\tsimd_ns\tnested_ns")
    for r in results
        @printf(io, "%d\t%.3f\t%.3f\t%.3f\n", r.n, r.t_hd * 1e9, r.t_simd * 1e9, r.t_dd * 1e9)
    end
end
@info "wrote $outfile"

using HyperHessians
using BenchmarkTools
using Printf

rosenbrock(x) = begin
    s = zero(eltype(x))
    @inbounds @simd for i in 1:length(x)-1
        s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return s
end

# Sweep chunk sizes for a single element type and input length.
function benchmark_chunks(::Type{T}, n; chunksizes = (1, 2, 4, 8, 16, 32)) where {T}
    f = rosenbrock
    x = rand(T, n)
    H = zeros(T, n, n)

    default_chunk = HyperHessians.chunksize(HyperHessians.HessianConfig(x))
    results = Dict{Int, Float64}()

    for c in chunksizes
        c > n && return
        print(c, " ")
        cfg = HyperHessians.HessianConfig(x, HyperHessians.Chunk{c}())
        bench = @btime HyperHessians.hessian!($H, $f, $x, $cfg)
    end

    return
end

prettytime(ns) = begin
    ns < 1.0e3 && return @sprintf("%.3f ns", ns)
    ns < 1.0e6 && return @sprintf("%.3f μs", ns / 1.0e3)
    ns < 1.0e9 && return @sprintf("%.3f ms", ns / 1.0e6)
    return @sprintf("%.3f s", ns / 1.0e9)
end

function main()
    n = 8
    for T in (Float32, Float64)
        println("=== $T ===")
        benchmark_chunks(T, n)
    end
end

main()

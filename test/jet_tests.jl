module JetTests

# The symmetric Jet number computes the whole Hessian in a single evaluation,
# storing the gradient once and only the upper triangle. It is an explicit
# opt-in: `HessianConfig(x, Jet[; simd])`. These tests pin that dispatch and
# check the Jet path against the HyperDual path and ForwardDiff.

using Test
using HyperHessians: HyperHessians, hessian, hessian!, hessian_gradient_value,
    hessian_gradient_value!, HessianConfig, Chunk, Jet
using DiffTests
using ForwardDiff
using NaNMath
using SpecialFunctions
using LogExpFunctions

@testset "config dispatch" begin
    # jets are never the default
    for n in (1, 2, 3, 4, 20)
        @test eltype(HessianConfig(rand(n)).duals) <: HyperHessians.HyperDual
    end
    for n in (1, 3, 8), simd in (false, true)
        cfg = HessianConfig(rand(n), Jet; simd)
        @test eltype(cfg.duals) === Jet{n, HyperHessians.nupper(n), Float64, simd}
        @test HyperHessians.chunksize(cfg) == n
    end
    @test_throws ArgumentError HessianConfig(zeros(0), Jet)
    # malformed triangle length fails at construction, not deep in a rule
    @test_throws ArgumentError Jet(1.0, (1.0, 2.0), (1.0,))
end

issymmetric(H) = H == H'

@testset "jet vs hyperdual vs ForwardDiff: $(f)" for f in (
        DiffTests.ackley,
        DiffTests.rosenbrock_1,
        DiffTests.self_weighted_logit,
        x -> sum(abs2, x),
    )
    for n in (1, 2, 3, 5, 8), simd in (false, true)
        x = rand(n) .+ 0.1
        H_jet = hessian(f, x, HessianConfig(x, Jet; simd))
        H_hd = hessian(f, x, HessianConfig(x, Chunk{n}()))
        H_fd = ForwardDiff.hessian(f, x)
        @test H_jet ≈ H_hd rtol = 1.0e-12
        @test H_jet ≈ H_fd rtol = 1.0e-8
        @test issymmetric(H_jet)
    end
end

@testset "binary and special rules through jets (simd = $simd)" for simd in (false, true)
    fns = (
        v -> v[1]^v[2] + atan(v[1], v[2]) + hypot(v[1], v[2]),
        v -> log(v[2], v[1] + 1) + v[1]^3 + v[2]^-2 + sinpi(v[1] / 4),
        v -> abs(v[1]) * exp(-v[2]) + (v[1] < v[2] ? v[1] * v[2] : v[2]),
        v -> muladd(v[1], 2.0, v[2]) + muladd(v[1], v[2], v[1]) + mod(v[1], 10) + floor(v[2]) * v[1],
    )
    for f in fns
        x = rand(2) .+ 0.5
        H_jet = hessian(f, x, HessianConfig(x, Jet; simd))
        H_fd = ForwardDiff.hessian(f, x)
        @test H_jet ≈ H_fd rtol = 1.0e-8
    end
end

@testset "hessian_gradient_value through jets" begin
    f = DiffTests.ackley
    x = rand(3)
    cfg = HessianConfig(x, Jet)
    res = hessian_gradient_value(f, x, cfg)
    @test res.value ≈ f(x)
    @test res.gradient ≈ ForwardDiff.gradient(f, x)
    @test res.hessian ≈ ForwardDiff.hessian(f, x)

    H = zeros(3, 3)
    G = zeros(3)
    v = hessian_gradient_value!(H, G, f, x, cfg)
    @test v ≈ f(x)
    @test G ≈ res.gradient
    @test H ≈ res.hessian
end

@testset "Float32 jets (simd = $simd)" for simd in (false, true)
    f = x -> sum(abs2, x) + sin(x[1])
    x = rand(Float32, 3)
    cfg = HessianConfig(x, Jet; simd)
    @test eltype(cfg.duals) == Jet{3, 6, Float32, simd}
    H = hessian(f, x, cfg)
    @test eltype(H) == Float32
    @test H ≈ ForwardDiff.hessian(f, x) rtol = 1.0e-4
end

@testset "mixed-precision scalars through jets" begin
    x = Float32[1, 2]
    cfg = HessianConfig(x, Jet)
    cfg_hd = HessianConfig(x, Chunk{2}())
    fns = (
        v -> v[1] + 2.1,
        v -> v[1] - 2.1,
        v -> 2.1 - v[1] * v[2],
        v -> @fastmath(v[1] + 2.1 * v[2] - 1.5),
        v -> mod(v[1], 10.0) * v[2],
    )
    for f in fns
        @test hessian(f, x, cfg) ≈ hessian(f, x, cfg_hd)
    end

    # muladd promotes like its scalar counterpart instead of narrowing to the
    # input type
    for f in (
            v -> muladd(v[1], 2.1, v[2]),
            v -> muladd(2.1, v[1], v[2]),
            v -> muladd(v[1], v[2], 2.1),
            v -> muladd(v[1], 2.1, 0.5),
            v -> muladd(2.1, 0.5, v[1]),
        )
        res = hessian_gradient_value(f, x, cfg)
        @test res.value isa Float64
        @test res.value ≈ f(x)
        @test res.value == hessian_gradient_value(f, x, cfg_hd).value
    end
end

@testset "simd jets behave like tuple jets" begin
    # scalar mixing keeps the backend
    j = Jet(1.5, (1.0, 0.0), (0.0, 0.0, 0.0))
    js = convert(Jet{2, 3, Float64, true}, j)
    for op in (x -> x + 1, x -> 2 * x, x -> x / 2, x -> x^3, x -> muladd(x, 2.0, x), one, x -> mod(x, 2.0), exp, -)
        @test typeof(op(js)) === typeof(js)
        rt = op(j)
        rs = op(js)
        @test rs.v == rt.v && rs.g == rt.g && rs.h == rt.h
    end
    # mixed backends promote toward tuple
    @test typeof(js + j) === typeof(j)
    # Int scalars fall back to the generic tuple ops but still work
    @test (js * 2).v == 3.0

    # mixed backends in binary rules promote, independent of argument order
    for op in (atan, hypot, ^, log, muladd)
        rt = op === muladd ? muladd(j, j, j) : op(j, j)
        for mixed in (
                op === muladd ? (muladd(js, j, js), muladd(j, js, j), muladd(js, js, j)) :
                    (op(js, j), op(j, js))
            )
            @test typeof(mixed) === typeof(j)
            @test mixed.v == rt.v && mixed.g == rt.g && mixed.h == rt.h
        end
    end
    # same-shape jets in scalar slots must not nest into Jet-of-Jet
    @test muladd(js, j, 1.0) isa Jet{2, 3, Float64, false}
    @test muladd(js, j, 1.0).v == muladd(j, j, 1.0).v

    # NaNMath scalar slots promote instead of narrowing to the component type
    j32 = Jet(1.2f0, (1.0f0,), (1.0f0,))
    @test NaNMath.pow(j32, 2.5).v === NaNMath.pow(1.2f0, 2.5)
    @test NaNMath.pow(2.5, j32).v === NaNMath.pow(2.5, 1.2f0)
end

# Minimal non-one-based vector/matrix for testing extraction.
struct OffsetVec{T} <: AbstractVector{T}
    data::Vector{T}
    off::Int
end
Base.size(v::OffsetVec) = size(v.data)
Base.axes(v::OffsetVec) = (Base.IdentityUnitRange((1 + v.off):(length(v.data) + v.off)),)
Base.getindex(v::OffsetVec, i::Int) = v.data[i - v.off]
Base.setindex!(v::OffsetVec, x, i::Int) = (v.data[i - v.off] = x; v)

struct OffsetMat{T} <: AbstractMatrix{T}
    data::Matrix{T}
    off::Int
end
Base.size(m::OffsetMat) = size(m.data)
Base.axes(m::OffsetMat) =
    map(n -> Base.IdentityUnitRange((1 + m.off):(n + m.off)), size(m.data))
Base.getindex(m::OffsetMat, i::Int, j::Int) = m.data[i - m.off, j - m.off]
Base.setindex!(m::OffsetMat, x, i::Int, j::Int) = (m.data[i - m.off, j - m.off] = x; m)

@testset "jet extraction into non-one-based outputs" begin
    f = DiffTests.ackley
    x = rand(3)
    cfg = HessianConfig(x, Jet)
    H = OffsetMat(zeros(3, 3), 7)
    G = OffsetVec(zeros(3), 5)
    v = hessian_gradient_value!(H, G, f, x, cfg)
    @test v ≈ f(x)
    @test G.data ≈ ForwardDiff.gradient(f, x)
    @test H.data ≈ ForwardDiff.hessian(f, x)
end

@testset "no allocations, no spurious promotions" begin
    f = DiffTests.ackley
    x = rand(3)
    H = zeros(3, 3)
    for simd in (false, true)
        cfg = HessianConfig(x, Jet; simd)
        hessian!(H, f, x, cfg) # warm up
        @test @allocated(hessian!(H, f, x, cfg)) == 0
    end
    # Float32 jets stay Float32 through the rules, for both backends
    for S in (false, true)
        j = convert(Jet{2, 3, Float32, S}, Jet(1.2f0, (1.0f0, 0.0f0), (0.0f0, 0.0f0, 0.0f0)))
        for op in (exp, sin, sqrt, log, abs2, x -> x^2, x -> x * 2.0f0, x -> x + 1, inv)
            @test op(j) isa Jet{2, 3, Float32, S}
        end
    end
end

@testset "extension rules through jets: $(name)" for (name, f) in (
        (
            "NaNMath", v -> NaNMath.sqrt(v[1]) + NaNMath.log(v[2]) + NaNMath.pow(v[1], v[2]) +
                NaNMath.pow(v[2], 2.5) + NaNMath.pow(v[1], 3) + NaNMath.sin(v[1]) +
                NaNMath.lgamma(v[2] + 2),
        ),
        ("SpecialFunctions", v -> gamma(v[1]) + erf(v[2]) + digamma(v[1] + 2) + besselj0(v[2])),
        ("LogExpFunctions", v -> logistic(v[1]) + xlogx(v[2]) + log1pexp(v[1]) + logit(v[2] / 2)),
    )
    for n in (2, 3), simd in (false, true)
        x = rand(n) .+ 0.2
        H_jet = hessian(f, x, HessianConfig(x, Jet; simd))
        H_hd = hessian(f, x, HessianConfig(x, Chunk{n}()))
        @test H_jet ≈ H_hd rtol = 1.0e-10
    end
end

@testset "fastmath through jets (simd = $simd)" for simd in (false, true)
    f = x -> @fastmath sin(x[1]) + x[1] * x[2] + x[2]^4 + exp(x[1]) / x[2] +
        sqrt(x[2] + 2) - 3.0 * x[1] + x[1]^x[2] + cos(x[2])
    x = rand(3) .+ 0.5
    H_jet = hessian(f, x, HessianConfig(x, Jet; simd))
    H_hd = hessian(f, x, HessianConfig(x, Chunk{3}()))
    @test H_jet ≈ H_hd rtol = 1.0e-8
end

@testset "fastmath methods exist for jets" begin
    # the fast-path methods must actually exist (not fall back to the slow ops)
    for S in (false, true)
        JT = Jet{2, 3, Float64, S}
        for (fn, sig) in (
                (Base.FastMath.mul_fast, Tuple{JT, JT}),
                (Base.FastMath.add_fast, Tuple{JT, JT}),
                (Base.FastMath.div_fast, Tuple{JT, JT}),
                (Base.FastMath.sin_fast, Tuple{JT}),
                (Base.FastMath.exp_fast, Tuple{JT}),
                (Base.FastMath.sqrt_fast, Tuple{JT}),
                (Base.FastMath.pow_fast, Tuple{JT, JT}),
                (Base.FastMath.pow_fast, Tuple{JT, Int}),
            )
            @test which(fn, sig).module === HyperHessians
        end
    end
end

@testset "jet config at any size" begin
    f = DiffTests.ackley
    for n in (2, 10, 16), simd in (false, true)
        x = rand(n)
        cfg = HessianConfig(x, Jet; simd)
        H_jet = hessian(f, x, cfg)
        H_hd = hessian(f, x, HessianConfig(x, Chunk(x)))
        @test H_jet ≈ H_hd rtol = 1.0e-10
    end
end

@testset "jet config errors" begin
    x = rand(3)
    cfg = HessianConfig(x, Jet)
    @test_throws DimensionMismatch hessian!(zeros(2, 2), DiffTests.ackley, x, cfg)
    @test_throws DimensionMismatch hessian!(zeros(3, 3), DiffTests.ackley, rand(4), cfg)
    @test_throws ErrorException hessian!(zeros(3, 3), x -> x, x, cfg)
end

end # module

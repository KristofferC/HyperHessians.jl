module JetTests

# The default HessianConfig uses the symmetric Jet number for small inputs
# (length(x) <= HyperHessians.JET_VECTOR_MAX_N). These tests pin that
# dispatch and check the Jet path against the HyperDual path and ForwardDiff.

using Test
using HyperHessians: HyperHessians, hessian, hessian!, hessian_gradient_value,
    hessian_gradient_value!, HessianConfig, Chunk, Jet
using DiffTests
using ForwardDiff
using NaNMath
using SpecialFunctions
using LogExpFunctions
using StaticArrays

@testset "default config dispatch" begin
    for n in 1:HyperHessians.JET_VECTOR_MAX_N
        cfg = HessianConfig(rand(n))
        @test eltype(cfg.duals) <: Jet
        @test HyperHessians.chunksize(cfg) == n
    end
    # above the threshold, and for explicit chunks, HyperDuals are used
    n_over = HyperHessians.JET_VECTOR_MAX_N + 1
    @test eltype(HessianConfig(rand(n_over)).duals) <: HyperHessians.HyperDual
    @test eltype(HessianConfig(rand(20)).duals) <: HyperHessians.HyperDual
    @test eltype(HessianConfig(rand(4), Chunk{4}()).duals) <: HyperHessians.HyperDual
    @test eltype(HessianConfig(zeros(0)).duals) <: HyperHessians.HyperDual
end

issymmetric(H) = H == H'

@testset "jet vs hyperdual vs ForwardDiff: $(f)" for f in (
        DiffTests.ackley,
        DiffTests.rosenbrock_1,
        DiffTests.self_weighted_logit,
        x -> sum(abs2, x),
    )
    for n in 1:HyperHessians.JET_VECTOR_MAX_N
        x = rand(n) .+ 0.1
        H_jet = hessian(f, x, HessianConfig(x))
        H_hd = hessian(f, x, HessianConfig(x, Chunk{n}()))
        H_fd = ForwardDiff.hessian(f, x)
        @test H_jet ≈ H_hd rtol = 1.0e-12
        @test H_jet ≈ H_fd rtol = 1.0e-8
        @test issymmetric(H_jet)
    end
end

@testset "binary and special rules through jets" begin
    fns = (
        v -> v[1]^v[2] + atan(v[1], v[2]) + hypot(v[1], v[2]),
        v -> log(v[2], v[1] + 1) + v[1]^3 + v[2]^-2 + sinpi(v[1] / 4),
        v -> abs(v[1]) * exp(-v[2]) + (v[1] < v[2] ? v[1] * v[2] : v[2]),
        v -> muladd(v[1], 2.0, v[2]) + muladd(v[1], v[2], v[1]) + mod(v[1], 10) + floor(v[2]) * v[1],
    )
    for f in fns
        x = rand(2) .+ 0.5
        H_jet = hessian(f, x, HessianConfig(x))
        H_fd = ForwardDiff.hessian(f, x)
        @test H_jet ≈ H_fd rtol = 1.0e-8
    end
end

@testset "hessian_gradient_value through jets" begin
    f = DiffTests.ackley
    x = rand(4)
    cfg = HessianConfig(x)
    @test eltype(cfg.duals) <: Jet
    res = hessian_gradient_value(f, x, cfg)
    @test res.value ≈ f(x)
    @test res.gradient ≈ ForwardDiff.gradient(f, x)
    @test res.hessian ≈ ForwardDiff.hessian(f, x)

    H = zeros(4, 4)
    G = zeros(4)
    v = hessian_gradient_value!(H, G, f, x, cfg)
    @test v ≈ f(x)
    @test G ≈ res.gradient
    @test H ≈ res.hessian
end

@testset "Float32 jets" begin
    f = x -> sum(abs2, x) + sin(x[1])
    x = rand(Float32, 3)
    cfg = HessianConfig(x)
    @test eltype(cfg.duals) == Jet{3, 6, Float32}
    H = hessian(f, x, cfg)
    @test eltype(H) == Float32
    @test H ≈ ForwardDiff.hessian(f, x) rtol = 1.0e-4
end

@testset "extension rules through jets: $(name)" for (name, f) in (
        (
            "NaNMath", v -> NaNMath.sqrt(v[1]) + NaNMath.log(v[2]) + NaNMath.pow(v[1], v[2]) +
                NaNMath.pow(v[2], 2.5) + NaNMath.sin(v[1]) + NaNMath.lgamma(v[2] + 2),
        ),
        ("SpecialFunctions", v -> gamma(v[1]) + erf(v[2]) + digamma(v[1] + 2) + besselj0(v[2])),
        ("LogExpFunctions", v -> logistic(v[1]) + xlogx(v[2]) + log1pexp(v[1]) + logit(v[2] / 2)),
    )
    for n in (2, HyperHessians.JET_VECTOR_MAX_N)
        x = rand(n) .+ 0.2
        cfg = HessianConfig(x)
        @test eltype(cfg.duals) <: Jet
        H_jet = hessian(f, x, cfg)
        H_hd = hessian(f, x, HessianConfig(x, Chunk{n}()))
        @test H_jet ≈ H_hd rtol = 1.0e-10
    end
end

@testset "StaticArrays through jets" begin
    f = DiffTests.ackley
    for n in (2, HyperHessians.JET_VECTOR_MAX_N, HyperHessians.JET_VECTOR_MAX_N + 1, 8)
        xv = rand(n)
        xs = SVector{n}(xv)
        H_static = hessian(f, xs)
        @test H_static isa SMatrix
        @test H_static ≈ hessian(f, xv, HessianConfig(xv, Chunk{n}())) rtol = 1.0e-10
        res = hessian_gradient_value(f, xs)
        @test res.value ≈ f(xv)
        @test res.gradient ≈ ForwardDiff.gradient(f, xv)
        @test res.hessian ≈ H_static
    end
end

@testset "jet config errors" begin
    x = rand(3)
    cfg = HessianConfig(x)
    @test eltype(cfg.duals) <: Jet
    @test_throws DimensionMismatch hessian!(zeros(2, 2), DiffTests.ackley, x, cfg)
    @test_throws DimensionMismatch hessian!(zeros(3, 3), DiffTests.ackley, rand(4), cfg)
    @test_throws ErrorException hessian!(zeros(3, 3), x -> x, x, cfg)
end

end # module

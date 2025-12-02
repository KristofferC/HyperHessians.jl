module HVPTests

using Test
using HyperHessians: hvp, hvp!, hvpgrad, hvpgrad!, DirectionalHVPConfig, Chunk
using DiffTests
using ForwardDiff

include(joinpath(@__DIR__, "helpers.jl"))
using .Helpers: ackley_stable

@testset "DirectionalHVPConfig errors" begin
    x = [1.0, 2.0, 3.0]
    @test_throws ArgumentError DirectionalHVPConfig(x, Chunk{0}())
end

@testset "hvp! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    cfg = DirectionalHVPConfig(x)
    @test_throws DimensionMismatch hvp!(zeros(3), f, x, zeros(2), cfg)
    @test_throws DimensionMismatch hvp!(zeros(2), f, x, zeros(3), cfg)
end

@testset "Hessian-vector products" begin
    # Test multiple functions and sizes like the correctness tests
    for f in (DiffTests.rosenbrock_1, DiffTests.ackley, DiffTests.self_weighted_logit)
        for n in (1, 4, 8, 15)
            if n == 1 && f == DiffTests.rosenbrock_1
                continue  # rosenbrock_1 needs n >= 2
            end
            x = rand(n)
            v = rand(n)
            H = ForwardDiff.hessian(f, x)

            for chunk in (1, max(1, n ÷ 2), n)
                cfg = DirectionalHVPConfig(x, Chunk{chunk}())
                @test hvp(f, x, v, cfg) ≈ H * v
                @test hvp(f, x, (v,), cfg)[1] ≈ H * v
            end
        end
    end

    # Test in-place hvp!
    f = DiffTests.ackley
    x = rand(8)
    v = rand(8)
    hv = zeros(8)
    H = ForwardDiff.hessian(f, x)
    hvp!(hv, f, x, v)
    @test hv ≈ H * v

    # Test Float32
    x32 = rand(Float32, 6)
    v32 = rand(Float32, 6)
    hv32 = hvp(ackley_stable, x32, v32)
    @test hv32 isa Vector{Float32}
    @test hv32 ≈ ForwardDiff.hessian(ackley_stable, x32) * v32
end

@testset "Bundled tangents" begin
    f = DiffTests.ackley
    x = rand(6)
    v1 = rand(6)
    v2 = rand(6)
    H = ForwardDiff.hessian(f, x)

    # Tuple input returns a tuple of hvps
    hv_tuple = hvp(f, x, (v1, v2))
    @test hv_tuple[1] ≈ H * v1
    @test hv_tuple[2] ≈ H * v2

    hv_out = (zeros(length(x)), zeros(length(x)))
    cfg_tuple = DirectionalHVPConfig(x, (v1, v2), Chunk{3}())
    hvp!(hv_out, f, x, (v1, v2), cfg_tuple)
    @test hv_out[1] ≈ H * v1
    @test hv_out[2] ≈ H * v2

    @test_throws DimensionMismatch hvp!((zeros(length(x)), zeros(length(x))), f, x, (v1,))

    # Tuple tangent with wrong length element
    v_bad = rand(4)
    @test_throws DimensionMismatch hvp(f, x, (v1, v_bad))

    # hv tuple element with wrong length
    @test_throws DimensionMismatch hvp!((zeros(length(x)), zeros(4)), f, x, (v1, v2))

    # Config tangent count mismatch
    cfg_2 = DirectionalHVPConfig(x, (v1, v2))
    @test_throws DimensionMismatch hvp!(hv_out, f, x, (v1, v2, rand(6)), cfg_2)

    # A single tangent can be provided without wrapping
    hv_single_out = zeros(length(x))
    hvp!(hv_single_out, f, x, v1, DirectionalHVPConfig(x, v1, Chunk{3}()))
    @test hv_single_out ≈ H * v1
end

@testset "hvpgrad" begin
    f = DiffTests.ackley
    x = rand(6)
    v1 = rand(6)
    v2 = rand(6)
    H = ForwardDiff.hessian(f, x)
    g_expected = ForwardDiff.gradient(f, x)

    # Single tangent allocating
    res1 = hvpgrad(f, x, v1)
    @test res1.gradient ≈ g_expected
    @test res1.hvp ≈ H * v1

    # Multiple tangents allocating
    res2 = hvpgrad(f, x, (v1, v2))
    @test res2.gradient ≈ g_expected
    @test res2.hvp[1] ≈ H * v1
    @test res2.hvp[2] ≈ H * v2

    # In-place
    g_out = zeros(length(x))
    hv_out = (zeros(length(x)), zeros(length(x)))
    cfg = DirectionalHVPConfig(x, (v1, v2), Chunk{3}())
    hvpgrad!(g_out, hv_out, f, x, (v1, v2), cfg)
    @test g_out ≈ g_expected
    @test hv_out[1] ≈ H * v1
    @test hv_out[2] ≈ H * v2
end

@testset "hvp! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    v = rand(n)
    hv = zeros(n)

    # Full chunk (vector path)
    cfg_full = DirectionalHVPConfig(x, Chunk{n}())
    hvp!(hv, f, x, v, cfg_full)
    @test @allocated(hvp!(hv, f, x, v, cfg_full)) == 0

    # Chunked path
    cfg_chunk = DirectionalHVPConfig(x, Chunk{4}())
    hvp!(hv, f, x, v, cfg_chunk)
    @test @allocated(hvp!(hv, f, x, v, cfg_chunk)) == 0

    # Bundled tangents (multiple directions)
    v1 = rand(n)
    v2 = rand(n)
    hv_bundle = (zeros(n), zeros(n))
    cfg_bundle_full = DirectionalHVPConfig(x, hv_bundle, Chunk{n}())
    hvp!(hv_bundle, f, x, (v1, v2), cfg_bundle_full)
    @test @allocated(hvp!(hv_bundle, f, x, (v1, v2), cfg_bundle_full)) == 0 broken = VERSION < v"1.11"

    cfg_bundle_chunk = DirectionalHVPConfig(x, hv_bundle, Chunk{4}())
    hvp!(hv_bundle, f, x, (v1, v2), cfg_bundle_chunk)
    @test @allocated(hvp!(hv_bundle, f, x, (v1, v2), cfg_bundle_chunk)) == 0 broken = VERSION < v"1.11"
end

@testset "hvpgrad! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    v = rand(n)
    g = zeros(n)
    hv = zeros(n)

    # Full chunk (vector path)
    cfg_full = DirectionalHVPConfig(x, Chunk{n}())
    hvpgrad!(g, hv, f, x, v, cfg_full)
    @test @allocated(hvpgrad!(g, hv, f, x, v, cfg_full)) == 0 broken = VERSION < v"1.11"

    # Chunked path
    cfg_chunk = DirectionalHVPConfig(x, Chunk{4}())
    hvpgrad!(g, hv, f, x, v, cfg_chunk)
    @test @allocated(hvpgrad!(g, hv, f, x, v, cfg_chunk)) == 0 broken = VERSION < v"1.11"

    # Bundled tangents (multiple directions)
    v1 = rand(n)
    v2 = rand(n)
    hv_bundle = (zeros(n), zeros(n))
    cfg_bundle_full = DirectionalHVPConfig(x, hv_bundle, Chunk{n}())
    hvpgrad!(g, hv_bundle, f, x, (v1, v2), cfg_bundle_full)
    @test @allocated(hvpgrad!(g, hv_bundle, f, x, (v1, v2), cfg_bundle_full)) == 0 broken = VERSION < v"1.11"

    cfg_bundle_chunk = DirectionalHVPConfig(x, hv_bundle, Chunk{4}())
    hvpgrad!(g, hv_bundle, f, x, (v1, v2), cfg_bundle_chunk)
    @test @allocated(hvpgrad!(g, hv_bundle, f, x, (v1, v2), cfg_bundle_chunk)) == 0 broken = VERSION < v"1.11"
end

end # module

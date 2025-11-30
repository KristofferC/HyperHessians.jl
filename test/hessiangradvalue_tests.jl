module HessianGradValueTests

using Test
using HyperHessians: hessiangradvalue, hessiangradvalue!, hessiangradvalue_simd, hessiangradvalue_simd!, HessianConfig, HessianConfigSIMD, Chunk
using DiffTests
using ForwardDiff

const CONFIGS = (HessianConfig, HessianConfigSIMD)

@testset "hessiangradvalue! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    for cfg in (HessianConfig(x), HessianConfigSIMD(x))
        @test_throws DimensionMismatch hessiangradvalue!(zeros(2, 2), zeros(3), f, x, cfg)
        @test_throws DimensionMismatch hessiangradvalue!(zeros(3, 3), zeros(2), f, x, cfg)
    end
    for cfg in (HessianConfig(x, Chunk{2}()), HessianConfigSIMD(x, Chunk{2}()))
        @test_throws DimensionMismatch hessiangradvalue!(zeros(2, 2), zeros(3), f, x, cfg)
        @test_throws DimensionMismatch hessiangradvalue!(zeros(3, 3), zeros(2), f, x, cfg)
    end
end

@testset "hessiangradvalue" begin
    f = DiffTests.ackley
    x = rand(8)
    H = zeros(length(x), length(x))
    G = zeros(length(x))
    for cfg_ctor in CONFIGS
        cfg_chunk = cfg_ctor(x, Chunk{4}())
        fill!(H, 0)
        fill!(G, 0)
        value = hessiangradvalue!(H, G, f, x, cfg_chunk)
        @test value ≈ f(x)
        @test G ≈ ForwardDiff.gradient(f, x)
        @test H ≈ ForwardDiff.hessian(f, x)

        H2 = similar(H)
        G2 = similar(G)
        cfg_vec = cfg_ctor(x, Chunk{length(x)}())
        value_vec = hessiangradvalue!(H2, G2, f, x, cfg_vec)
        @test value_vec ≈ f(x)
        @test G2 ≈ ForwardDiff.gradient(f, x)
        @test H2 ≈ ForwardDiff.hessian(f, x)

        res = hessiangradvalue(f, x, cfg_vec)
        @test res.value ≈ f(x)
        @test res.gradient ≈ ForwardDiff.gradient(f, x)
        @test res.hessian ≈ ForwardDiff.hessian(f, x)
    end

    res_simd = hessiangradvalue_simd(f, x)
    @test res_simd.value ≈ f(x)
    @test res_simd.gradient ≈ ForwardDiff.gradient(f, x)
    @test res_simd.hessian ≈ ForwardDiff.hessian(f, x)
    H_simd = zeros(length(x), length(x))
    G_simd = zeros(length(x))
    hessiangradvalue_simd!(H_simd, G_simd, f, x)
    @test G_simd ≈ ForwardDiff.gradient(f, x)
    @test H_simd ≈ ForwardDiff.hessian(f, x)
end

@testset "hessiangradvalue! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    H = zeros(n, n)
    G = zeros(n)

    for cfg_ctor in CONFIGS
        # Full chunk (vector path)
        cfg_full = cfg_ctor(x, Chunk{n}())
        hessiangradvalue!(H, G, f, x, cfg_full)
        @test @allocated(hessiangradvalue!(H, G, f, x, cfg_full)) == 0 broken = VERSION < v"1.11"

        # Chunked path
        cfg_chunk = cfg_ctor(x, Chunk{4}())
        hessiangradvalue!(H, G, f, x, cfg_chunk)
        @test @allocated(hessiangradvalue!(H, G, f, x, cfg_chunk)) == 0 broken = VERSION < v"1.11"
    end
end

end # module

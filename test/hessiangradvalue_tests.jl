module HessianGradValueTests

using Test
using HyperHessians: hessiangradvalue, hessiangradvalue!, HessianConfig, Chunk
using DiffTests
using ForwardDiff

@testset "hessiangradvalue! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    cfg = HessianConfig(x)
    cfg_chunked = HessianConfig(x, Chunk{2}())
    @test_throws DimensionMismatch hessiangradvalue!(zeros(2, 2), zeros(3), f, x, cfg)
    @test_throws DimensionMismatch hessiangradvalue!(zeros(3, 3), zeros(2), f, x, cfg)
    @test_throws DimensionMismatch hessiangradvalue!(zeros(2, 2), zeros(3), f, x, cfg_chunked)
    @test_throws DimensionMismatch hessiangradvalue!(zeros(3, 3), zeros(2), f, x, cfg_chunked)
end

@testset "hessiangradvalue" begin
    f = DiffTests.ackley
    x = rand(8)
    H = zeros(length(x), length(x))
    G = zeros(length(x))
    cfg_chunk = HessianConfig(x, Chunk{4}())
    value = hessiangradvalue!(H, G, f, x, cfg_chunk)
    @test value ≈ f(x)
    @test G ≈ ForwardDiff.gradient(f, x)
    @test H ≈ ForwardDiff.hessian(f, x)

    H2 = similar(H)
    G2 = similar(G)
    cfg_vec = HessianConfig(x, Chunk{length(x)}())
    value_vec = hessiangradvalue!(H2, G2, f, x, cfg_vec)
    @test value_vec ≈ f(x)
    @test G2 ≈ ForwardDiff.gradient(f, x)
    @test H2 ≈ ForwardDiff.hessian(f, x)

    res = hessiangradvalue(f, x, cfg_vec)
    @test res.value ≈ f(x)
    @test res.gradient ≈ ForwardDiff.gradient(f, x)
    @test res.hessian ≈ ForwardDiff.hessian(f, x)
end

@testset "hessiangradvalue! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    H = zeros(n, n)
    G = zeros(n)

    # Full chunk (vector path)
    cfg_full = HessianConfig(x, Chunk{n}())
    hessiangradvalue!(H, G, f, x, cfg_full)
    @test @allocated(hessiangradvalue!(H, G, f, x, cfg_full)) == 0 broken = VERSION < v"1.11"

    # Chunked path
    cfg_chunk = HessianConfig(x, Chunk{4}())
    hessiangradvalue!(H, G, f, x, cfg_chunk)
    @test @allocated(hessiangradvalue!(H, G, f, x, cfg_chunk)) == 0 broken = VERSION < v"1.11"
end

end # module

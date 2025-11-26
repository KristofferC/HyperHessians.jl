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

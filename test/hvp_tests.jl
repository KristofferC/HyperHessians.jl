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
end

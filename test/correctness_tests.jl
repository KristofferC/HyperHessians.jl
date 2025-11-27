@testset "scalar" begin
    f(x) = exp(x) / sqrt(sin(x)^3 + cos(x)^3)
    x = rand()
    @test hessian(f, x) ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)
end

@testset "HessianConfig errors" begin
    x = [1.0, 2.0, 3.0]
    @test_throws ArgumentError HessianConfig(x, Chunk{0}())
end

@testset "hessian! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    cfg = HessianConfig(x)
    cfg_chunked = HessianConfig(x, Chunk{2}())
    @test_throws DimensionMismatch hessian!(zeros(2, 3), f, x, cfg)
    @test_throws DimensionMismatch hessian!(zeros(3, 2), f, x, cfg)
    @test_throws DimensionMismatch hessian!(zeros(2, 2), f, x, cfg)
    @test_throws DimensionMismatch hessian!(zeros(2, 2), f, x, cfg_chunked)
end

@testset "correctness" begin
    for f in (
            DiffTests.rosenbrock_1,
            DiffTests.rosenbrock_2,
            DiffTests.rosenbrock_3,
            #DiffTests.rosenbrock_4,
            DiffTests.ackley,
            DiffTests.self_weighted_logit,
            DiffTests.nested_array_mul,
        )
        for n in (1, 4, 8, 8 + 7)
            if n == 1 && (f == DiffTests.rosenbrock_4 || f == DiffTests.nested_array_mul)
                continue
            end
            x = rand(n)
            H = ForwardDiff.hessian(f, x)

            for chunk in (1, n ÷ 2, n)
                if chunk <= 0
                    continue
                end
                # @info "f=$f, n=$n, chunk=$chunk"

                cfg = HessianConfig(x, Chunk{chunk}())
                @test H ≈ hessian(f, x, cfg)
                @test H ≈ hessian(f, x, cfg)
            end
        end
    end
end

@testset "hessian! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    H = zeros(n, n)

    # Full chunk (vector path)
    cfg_full = HessianConfig(x, Chunk{n}())
    hessian!(H, f, x, cfg_full)
    @test @allocated(hessian!(H, f, x, cfg_full)) == 0

    # Chunked path
    cfg_chunk = HessianConfig(x, Chunk{4}())
    hessian!(H, f, x, cfg_chunk)
    @test @allocated(hessian!(H, f, x, cfg_chunk)) == 0
end

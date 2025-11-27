@testset "GradientConfig errors" begin
    x = [1.0, 2.0, 3.0]
    @test_throws ArgumentError GradientConfig(x, Chunk{0}())
end

@testset "gradient! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    cfg = GradientConfig(x)
    cfg_chunked = GradientConfig(x, Chunk{2}())
    @test_throws DimensionMismatch gradient!(zeros(2), f, x, cfg)
    @test_throws DimensionMismatch gradient!(zeros(4), f, x, cfg)
    @test_throws DimensionMismatch gradient!(zeros(2), f, x, cfg_chunked)
end

@testset "gradient correctness" begin
    for f in (
            DiffTests.rosenbrock_1,
            DiffTests.rosenbrock_2,
            DiffTests.ackley,
            DiffTests.self_weighted_logit,
        )
        for n in (1, 4, 8, 15)
            if n == 1 && f == DiffTests.rosenbrock_1
                continue  # rosenbrock_1 needs n >= 2
            end
            x = rand(n)
            g = ForwardDiff.gradient(f, x)

            for chunk in (1, max(1, n ÷ 2), n)
                cfg = GradientConfig(x, Chunk{chunk}())
                @test g ≈ gradient(f, x, cfg)
                @test g ≈ gradient(f, x, cfg)
            end
        end
    end
end

@testset "gradientvalue" begin
    f = x -> sum(x .^ 3)
    x = rand(6)
    cfg = GradientConfig(x, Chunk{3}())

    result = gradientvalue(f, x, cfg)
    @test result.value ≈ f(x)
    @test result.gradient ≈ ForwardDiff.gradient(f, x)

    G = similar(x)
    val = gradientvalue!(G, f, x, cfg)
    @test val ≈ f(x)
    @test G ≈ ForwardDiff.gradient(f, x)
end

@testset "gradient! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    G = zeros(n)

    cfg_full = GradientConfig(x, Chunk{n}())
    gradient!(G, f, x, cfg_full)
    @test @allocated(gradient!(G, f, x, cfg_full)) == 0

    cfg_chunk = GradientConfig(x, Chunk{4}())
    gradient!(G, f, x, cfg_chunk)
    @test @allocated(gradient!(G, f, x, cfg_chunk)) == 0
end

module CorrectnessTests

using Test
using HyperHessians: hessian, hessian!, hessian_simd, hessian_simd!, HessianConfig, HessianConfigSIMD, Chunk
using DiffTests
using ForwardDiff

const CONFIGS = (HessianConfig, HessianConfigSIMD)

@testset "scalar" begin
    f(x) = exp(x) / sqrt(sin(x)^3 + cos(x)^3)
    x = rand()
    @test hessian(f, x) ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)
    @test hessian_simd(f, x) ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)
end

@testset "HessianConfig errors" begin
    x = [1.0, 2.0, 3.0]
    for cfg_ctor in CONFIGS
        @test_throws ArgumentError cfg_ctor(x, Chunk{0}())
    end
end

@testset "hessian! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    cfgs = (HessianConfig(x), HessianConfigSIMD(x))
    cfg_chunked = (HessianConfig(x, Chunk{2}()), HessianConfigSIMD(x, Chunk{2}()))
    for cfg in cfgs
        @test_throws DimensionMismatch hessian!(zeros(2, 3), f, x, cfg)
        @test_throws DimensionMismatch hessian!(zeros(3, 2), f, x, cfg)
        @test_throws DimensionMismatch hessian!(zeros(2, 2), f, x, cfg)
    end
    for cfg in cfg_chunked
        @test_throws DimensionMismatch hessian!(zeros(2, 2), f, x, cfg)
    end
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
                for cfg_ctor in CONFIGS
                    cfg = cfg_ctor(x, Chunk{chunk}())
                    @test H ≈ hessian(f, x, cfg)
                end
            end
            @test H ≈ hessian_simd(f, x)
            H_simd_bang = zeros(n, n)
            hessian_simd!(H_simd_bang, f, x)
            @test H_simd_bang ≈ H
        end
    end
end

@testset "hessian! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    H = zeros(n, n)

    for cfg_ctor in CONFIGS
        cfg_full = cfg_ctor(x, Chunk{n}())
        hessian!(H, f, x, cfg_full)
        @test @allocated(hessian!(H, f, x, cfg_full)) == 0

        cfg_chunk = cfg_ctor(x, Chunk{4}())
        hessian!(H, f, x, cfg_chunk)
        @test @allocated(hessian!(H, f, x, cfg_chunk)) == 0
    end
end

end # module

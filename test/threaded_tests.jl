module ThreadedTests

using Test
using HyperHessians: HyperHessians, hessian, hessian!, hessian_gradient_value,
    hessian_gradient_value!, HessianConfig, ThreadedHessianConfig, Chunk,
    linear_to_pair
using ForwardDiff
using DiffTests

# Real parallelism requires the worker to have threads (see runtests.jl);
# with 1 thread these tests still run, but only check task machinery.
@info "threaded_tests running with $(Threads.nthreads()) thread(s)"

@testset "linear_to_pair" begin
    for k in 1:40
        expected = [(i, j) for i in 1:k for j in i:k]
        got = [linear_to_pair(p, k) for p in 1:(k * (k + 1) ÷ 2)]
        @test got == expected
    end
end

@testset "correctness against serial" begin
    for f in (DiffTests.rosenbrock_1, DiffTests.ackley, x -> sum(cumprod(x)))
        for n in (1, 5, 8, 16, 33)
            x = 0.1 .+ rand(n)
            H_ref = ForwardDiff.hessian(f, x)
            for chunk in (1, 3, 4, 8), ntasks in (1, 3, 8)
                chunk > n && continue
                cfg = ThreadedHessianConfig(x, Chunk{chunk}(); ntasks)
                @test hessian(f, x, cfg) ≈ H_ref
                # reuse of the same config
                @test hessian(f, x, cfg) ≈ H_ref

                H = similar(x, n, n)
                G = similar(x)
                val = hessian_gradient_value!(H, G, f, x, cfg)
                @test val ≈ f(x)
                @test G ≈ ForwardDiff.gradient(f, x)
                @test H ≈ H_ref

                res = hessian_gradient_value(f, x, cfg)
                @test res.value ≈ f(x)
                @test res.gradient ≈ G
                @test res.hessian ≈ H_ref
            end
        end
    end
end

@testset "argument errors" begin
    x = rand(4)
    @test_throws ArgumentError ThreadedHessianConfig(x, Chunk{0}())
    @test_throws ArgumentError ThreadedHessianConfig(x; ntasks = 0)
    cfg = ThreadedHessianConfig(x, Chunk{2}())
    f = x -> sum(abs2, x)
    @test_throws DimensionMismatch hessian!(zeros(3, 3), f, x, cfg)
    @test_throws DimensionMismatch hessian(f, rand(5), cfg)
end

@testset "edge cases" begin
    # empty input
    xe = Float64[]
    cfg = ThreadedHessianConfig(xe)
    @test size(hessian(sum, xe, cfg)) == (0, 0)

    # chunk == n and chunk > n: single evaluation, single buffer
    x = rand(6)
    for c in (6, 8)
        cfg = ThreadedHessianConfig(x, Chunk{c}())
        @test length(cfg.duals) == 1
        @test hessian(DiffTests.ackley, x, cfg) ≈ ForwardDiff.hessian(DiffTests.ackley, x)
    end

    # more tasks than block pairs: buffers clamped to npairs
    x = rand(8)
    cfg = ThreadedHessianConfig(x, Chunk{4}(); ntasks = 8)
    @test length(cfg.duals) == 3
    @test hessian(DiffTests.rosenbrock_1, x, cfg) ≈ ForwardDiff.hessian(DiffTests.rosenbrock_1, x)

    # array (matrix) input
    X = rand(3, 4)
    cfg = ThreadedHessianConfig(X, Chunk{4}())
    f = x -> sum(abs2, x)
    @test hessian(f, X, cfg) ≈ HyperHessians.hessian(f, X)
    res = hessian_gradient_value(f, X, cfg)
    @test res.gradient ≈ 2 .* X
end

@testset "constant output value type" begin
    fconst = x -> big(1.0)
    x = rand(8)
    cfg = ThreadedHessianConfig(x, Chunk{3}())
    H = similar(x, 8, 8)
    G = similar(x)
    val = hessian_gradient_value!(H, G, fconst, x, cfg)
    @test val isa BigFloat && val == big(1.0)
    @test all(iszero, H)
    @test all(iszero, G)
end

@testset "inference" begin
    f = x -> sum(abs2, x)
    x = rand(32)
    H = similar(x, 32, 32)
    G = similar(x)
    cfg = ThreadedHessianConfig(x, Chunk{8}())
    @test (@inferred hessian(f, x, cfg)) isa Matrix{Float64}
    @test (@inferred hessian!(H, f, x, cfg)) isa Matrix{Float64}
    @test (@inferred hessian_gradient_value!(H, G, f, x, cfg)) isa Float64
    res = @inferred hessian_gradient_value(f, x, cfg)
    @test res.value isa Float64
end

@testset "throwing f leaves no running tasks" begin
    fthrow = x -> (sum(x) > -Inf && error("boom"); sum(x))
    x = rand(32)
    cfg = ThreadedHessianConfig(x, Chunk{4}(); ntasks = 8)
    H = similar(x, 32, 32)
    @test_throws Exception hessian!(H, fthrow, x, cfg)
    # immediate reuse must be safe and correct
    @test hessian!(H, DiffTests.ackley, x, cfg) ≈ ForwardDiff.hessian(DiffTests.ackley, x)
end

end # module

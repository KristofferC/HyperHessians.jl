using HyperHessians, Test
using HyperHessians: HyperDual, HessianConfig, ThreadedHessianConfig, Chunk
import ForwardDiff, DiffTests

@testset "simd config option" begin
    fastackley = x -> @fastmath DiffTests.ackley(x)
    for T in (Float64, Float32), f in (DiffTests.rosenbrock_1, DiffTests.ackley, fastackley)
        for n in (1, 2, 3, 5, 8, 20)
            x = T[0.1 + 0.9 * (i - 1) / max(n - 1, 1) for i in 1:n]
            Href = ForwardDiff.hessian(f, x)
            for chunk in unique((min(n, 3), min(n, 8))), simd in (false, true)
                cfg = HessianConfig(x, Chunk{chunk}(); simd)
                H = HyperHessians.hessian(f, x, cfg)
                @test isapprox(H, Href; rtol = sqrt(eps(T)))
                @test eltype(cfg.duals) === HyperDual{chunk, chunk, T, simd}
            end
        end
    end

    x = collect(range(0.1, 1.0, length = 12))
    f = DiffTests.rosenbrock_1
    cfg = HessianConfig(x, Chunk{4}(); simd = true)
    res = HyperHessians.hessian_gradient_value(f, x, cfg)
    @test res.value ≈ f(x)
    @test res.gradient ≈ ForwardDiff.gradient(f, x)
    @test res.hessian ≈ ForwardDiff.hessian(f, x)

    tcfg = ThreadedHessianConfig(x, Chunk{4}(); simd = true)
    @test HyperHessians.hessian(f, x, tcfg) ≈ ForwardDiff.hessian(f, x)

    # scalar mixing keeps the backend; simd duals behave like tuple duals
    h = HyperHessians.HyperDual(1.5, (1.0, 0.0), (0.0, 1.0))
    hs = convert(HyperDual{2, 2, Float64, true}, h)
    for op in (x -> x + 1, x -> 2 * x, x -> x / 2, x -> x^3, x -> muladd(x, 2.0, x), one, x -> mod(x, 2.0), exp, -)
        @test typeof(op(hs)) === typeof(hs)
        rt = op(h)
        rs = op(hs)
        @test rs.v == rt.v && rs.ϵ1 == rt.ϵ1 && rs.ϵ2 == rt.ϵ2 && rs.ϵ12 == rt.ϵ12
    end
    # mixed backends promote toward tuple
    @test typeof(hs + h) === typeof(h)
    # Int scalars fall back to the generic tuple ops but still work
    @test (hs * 2).v == 3.0

    # config type inference: literal Bool and Val fold to a concrete type,
    # runtime Bool gives the two-config union
    cfg_lit(x) = HessianConfig(x, Chunk{4}(); simd = true)
    cfg_val(x, s) = HessianConfig(x, Chunk{4}(); simd = s)
    @test isconcretetype(Base.promote_op(cfg_lit, Vector{Float64}))
    @test isconcretetype(Base.promote_op(cfg_val, Vector{Float64}, Val{true}))
    @test Base.promote_op(cfg_val, Vector{Float64}, Bool) isa Union

    # BigFloat eltype with simd flag silently uses the generic path
    xb = big.(collect(range(0.1, 1.0, length = 4)))
    cfgb = HessianConfig(xb, Chunk{4}(); simd = true)
    @test HyperHessians.hessian(DiffTests.rosenbrock_1, xb, cfgb) ≈
        ForwardDiff.hessian(DiffTests.rosenbrock_1, collect(range(0.1, 1.0, length = 4))) rtol = 1.0e-8
end

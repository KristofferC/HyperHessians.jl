module HyperDualInterfaceTests

using Test
using HyperHessians
using HyperHessians: HyperDual

@testset "comparisons compare primal values" begin
    h = HyperDual(2.0, (1.0,), (1.0,), ((0.0,),))
    hsame = HyperDual(2.0, (0.5,), (2.0,), ((1.0,),)) # same value, different seeds
    hbig = HyperDual(3.0, (1.0,), (1.0,), ((0.0,),))

    @test h == hsame
    @test isequal(h, hsame)
    @test hash(h) == hash(hsame)
    @test h != hbig
    @test h == 2.0 && 2.0 == h
    @test h < hbig && hbig > h
    @test h < 3.0 && 1.0 < h
    @test h <= hsame && h >= hsame
    @test h <= 2.0 && 2.0 <= h
    @test isless(h, hbig) && isless(h, 3.0) && isless(1.0, h)
    @test max(h, hbig) === hbig
    @test min(h, hbig) === h
    @test clamp(hbig, 0.0, 2.5).v == 2.5
end

@testset "predicates act on the primal value" begin
    h = HyperDual(2.0, (1.0,), (1.0,), ((0.0,),))
    hnan = HyperDual(NaN, (1.0,), (1.0,), ((0.0,),))
    hinf = HyperDual(Inf, (1.0,), (1.0,), ((0.0,),))

    @test isnan(hnan)
    @test !isnan(h)
    @test isinf(hinf)
    @test !isinf(h)
    @test isfinite(h)
    @test !isfinite(hnan)
    @test signbit(-h)
    @test !signbit(h)
    @test iszero(h - h)
    @test isinteger(h)
    @test iseven(h)
    @test !isodd(h)
end

@testset "branching functions differentiate" begin
    f(x) = sum(xi -> xi > 0.5 ? xi^2 : sin(xi), x)
    H = HyperHessians.hessian(f, [0.3, 0.7])
    @test H[1, 1] ≈ -sin(0.3)
    @test H[2, 2] == 2.0
    @test H[1, 2] == H[2, 1] == 0.0

    Hm = HyperHessians.hessian(x -> maximum(x) + minimum(x)^2, [0.3, 0.7])
    @test Hm == [2.0 0.0; 0.0 0.0]

    Hc = HyperHessians.hessian(x -> clamp(x[1], 0.0, 0.5)^2, [0.3])
    @test Hc == fill(2.0, 1, 1)
end

@testset "rounding" begin
    h = HyperDual(2.7, (1.0,), (1.0,), ((3.0,),))
    for (f, expected) in ((floor, 2.0), (ceil, 3.0), (trunc, 2.0), (round, 3.0))
        r = f(h)
        @test r isa typeof(h)
        @test r.v == expected
        @test r.ϵ1 == (0.0,) && r.ϵ2 == (0.0,) && r.ϵ12 == ((0.0,),)
    end
    @test round(h, RoundDown).v == 2.0
    @test round(h, RoundUp).v == 3.0
    @test floor(Int, h) === 2
    @test ceil(Int, h) === 3
    @test trunc(Int, h) === 2
    @test round(Int, h) === 3
end

@testset "mod/rem pass derivatives through" begin
    h = HyperDual(2.0, (1.0,), (1.0,), ((3.0,),))
    for r in (mod(h, 1.5), rem(h, 1.5), mod2pi(h), rem2pi(h, RoundNearest))
        @test r.ϵ1 == h.ϵ1 && r.ϵ2 == h.ϵ2 && r.ϵ12 == h.ϵ12
    end
    @test mod(h, 1.5).v == 0.5
    @test rem(h, 1.5).v == 0.5
    @test mod2pi(h).v == 2.0
    # two-dual and Real-first methods (via mod(x, y) = x - fld(x, y) * y)
    hd = mod(h, HyperDual(1.5, (0.0,), (0.0,), ((0.0,),)))
    @test hd.v == 0.5 && hd.ϵ1 == (1.0,)
    @test mod(5.3, h).v ≈ mod(5.3, 2.0)
    # chain rule through mod: d²/dx² mod(3x, 2)² = 18 away from wraps
    res = HyperHessians.hessian_gradient_value(x -> mod(3x, 2.0)^2, 1.4)
    @test res.gradient ≈ 2 * (3 * 1.4 - 4) * 3
    @test res.hessian ≈ 18.0
end

@testset "muladd with two plain reals" begin
    z = HyperDual(1.0e16, (1.0,), (2.0,), ((3.0,),))
    r = muladd(3.0, 7.0, z)
    @test r.v == muladd(3.0, 7.0, z.v)
    @test r.ϵ1 == z.ϵ1 && r.ϵ2 == z.ϵ2 && r.ϵ12 == z.ϵ12
    H = HyperHessians.hessian(x -> muladd(2.0, 3.0, x[1] * x[1]), [1.3])
    @test H == fill(2.0, 1, 1)
end

@testset "division" begin
    x, y = 1.2, 0.7
    res = HyperHessians.hessian_gradient_value(v -> v[1] / v[2], [x, y])
    @test res.value ≈ x / y
    @test res.gradient ≈ [1 / y, -x / y^2]
    @test res.hessian ≈ [0.0 -1 / y^2; -1 / y^2 2x / y^3]
    # mixed precision promotes
    h64 = HyperDual(2.0, (1.0,), (1.0,), ((0.0,),))
    h32 = HyperDual(3.0f0, (1.0f0,), (0.0f0,), ((0.0f0,),))
    @test h64 / h32 isa HyperDual{1, 1, Float64}
    @test (h64 / h32).v ≈ 2 / 3
end

@testset "conversions" begin
    h32 = HyperDual(2.0f0, (1.0f0,), (1.0f0,), ((0.5f0,),))
    h64 = HyperDual{1, 1, Float64}(h32)
    @test h64 isa HyperDual{1, 1, Float64}
    @test h64.v == 2.0 && h64.ϵ12 == ((0.5,),)
    @test convert(HyperDual{1, 1, Float64}, h32) == h64
end

end # module

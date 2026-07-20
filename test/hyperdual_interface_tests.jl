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

@testset "muladd with HyperDual multiplicands (issue #53)" begin
    x = HyperDual(2.0, (1.0,), (2.0,), ((3.0,),))
    y = HyperDual(5.0, (0.5,), (1.5,), ((0.7,),))
    z = HyperDual(7.0, (0.1,), (0.2,), ((0.3,),))
    @test muladd(x, y, z) == x * y + z
    @test muladd(x, y, 4.0) == x * y + 4.0
    H = HyperHessians.hessian(v -> muladd(v[1], v[1], v[1]), [1.3])
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

@testset "disambiguation against Base numeric types" begin
    h = HyperDual(1.3, (1.0,), (1.0,), ((0.0,),))
    # comparison with irrationals resolves and compares the primal value
    @test (h == π) == false && (π == h) == false
    @test (h == 1.3) && !(h == 1.4)
    # two-argument power/log against Base's ℯ- and Rational-specialized methods
    @test h^(3 // 2) ≈ 1.3^(3 // 2)
    @test (ℯ^h).v ≈ exp(1.3)
    @test log(ℯ, h).v ≈ log(1.3)
    # construction from Base numeric conversion types
    @test HyperDual{1, 1}(1.0 + 0.0im).v == 1.0
    @test HyperDual{1, 1, Float64}(1.0 + 0.0im).v == 1.0
    @test HyperDual{1, 1}(h) === h
end

# Guard against ambiguities with Base/Core methods only. Internal
# HyperDual-vs-HyperDual pairs are only reachable with mismatched chunk sizes
# (a nonsensical call). Clashes with other <:Real dual types (e.g.
# ForwardDiff.Dual, if co-loaded) are mutually ambiguous but unresolvable
# without depending on those packages, and mixing two AD systems in one op is
# nonsensical anyway. Both are filtered out.
@testset "no ambiguities against Base methods" begin
    ambs = Test.detect_ambiguities(HyperHessians; recursive = true)
    isbasecore(m) = (r = Base.moduleroot(m.module); r === Base || r === Core)
    ishh(m) = Base.moduleroot(m.module) === HyperHessians
    external = filter(ambs) do (m1, m2)
        (ishh(m1) && isbasecore(m2)) || (isbasecore(m1) && ishh(m2))
    end
    @test isempty(external)
end

@testset "conversions" begin
    h32 = HyperDual(2.0f0, (1.0f0,), (1.0f0,), ((0.5f0,),))
    h64 = HyperDual{1, 1, Float64}(h32)
    @test h64 isa HyperDual{1, 1, Float64}
    @test h64.v == 2.0 && h64.ϵ12 == ((0.5,),)
    @test convert(HyperDual{1, 1, Float64}, h32) == h64
end

end # module

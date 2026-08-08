module ForwardDiffTests

using Test
using HyperHessians
using HyperHessians: HyperDual, Jet
using ForwardDiff

@testset "extension loads" begin
    @test Base.get_extension(HyperHessians, :HyperHessiansForwardDiffExt) !== nothing
end

@testset "no HyperDual/Jet vs Dual ambiguities" begin
    ambs = Test.detect_ambiguities(Base, HyperHessians, ForwardDiff; recursive = false)
    bad = filter(ambs) do (m1, m2)
        s = string(m1.sig) * string(m2.sig)
        occursin(r"ForwardDiff\.Dual", s) || return false
        # Pairs mixing Jet with HyperDual are only reachable through
        # nonsensical three-way Jet/HyperDual/Dual calls and are deliberately
        # left ambiguous (see hyperdual_interface_tests.jl); 1.10's
        # detect_ambiguities reports them, 1.11+ does not.
        occursin("HyperDual", s) && occursin("Jet", s) && return false
        return occursin("HyperDual", s) || occursin("Jet", s)
    end
    isempty(bad) || foreach(p -> println(p[1].sig, "\n    <-> ", p[2].sig), bad)
    @test isempty(bad)
end

@testset "mixed HyperDual/Dual arithmetic" begin
    h = HyperDual(2.0, (1.0,), (1.0,))
    d = ForwardDiff.Dual{Nothing}(3.0, 1.0)

    # Dual wraps the HyperDual; the primal parts must match plain arithmetic
    primal(x) = x isa HyperDual ? x.v : ForwardDiff.value(x)
    for op in (+, -, *, /, ^, atan, hypot, log, mod, rem)
        for (a, b) in ((h, d), (d, h))
            r = op(a, b)
            @test r isa ForwardDiff.Dual{Nothing, <:HyperDual}
            @test HyperHessians.value(ForwardDiff.value(r)) ≈ op(primal(a), primal(b)) rtol = 1.0e-14
        end
    end
    @test ForwardDiff.NaNMath.pow(h, d) isa ForwardDiff.Dual{Nothing, <:HyperDual}
    @test ForwardDiff.NaNMath.pow(d, h) isa ForwardDiff.Dual{Nothing, <:HyperDual}

    # comparisons act on primal values
    @test h < d
    @test d > h
    @test h <= d
    @test d >= h
    @test isless(h, d)
    @test !isless(d, h)
    @test !(h == d)
    @test promote_type(typeof(h), typeof(d)) == ForwardDiff.Dual{Nothing, typeof(h), 1}

    # muladd mixes, including Integer/Irrational/Rational crossings
    for args in (
            (h, d, 1.0), (d, h, 1.0), (1.0, h, d), (1.0, d, h), (h, 1.0, d), (d, 1.0, h),
            (h, d, h), (d, h, h), (h, h, d), (d, d, h), (h, d, d), (d, h, d),
            (d, π, h), (h, 2, d), (h, d, 1 // 2), (2, d, h),
        )
        r = muladd(args...)
        @test r isa ForwardDiff.Dual{Nothing, <:HyperDual}
        vals = map(a -> a isa HyperDual ? a.v : a isa ForwardDiff.Dual ? ForwardDiff.value(a) : a, args)
        @test HyperHessians.value(ForwardDiff.value(r)) ≈ muladd(float.(vals)...) rtol = 1.0e-14
    end
end

@testset "mixed Jet/Dual arithmetic" begin
    j = Jet(2.0, (1.0,), (1.0,))
    d = ForwardDiff.Dual{Nothing}(3.0, 1.0)

    primal(x) = x isa Jet ? x.v : ForwardDiff.value(x)
    for op in (+, -, *, /, ^, atan, hypot, log, mod, rem)
        for (a, b) in ((j, d), (d, j))
            r = op(a, b)
            @test r isa ForwardDiff.Dual{Nothing, <:Jet}
            @test HyperHessians.value(ForwardDiff.value(r)) ≈ op(primal(a), primal(b)) rtol = 1.0e-14
        end
    end
    @test ForwardDiff.NaNMath.pow(j, d) isa ForwardDiff.Dual{Nothing, <:Jet}
    @test ForwardDiff.NaNMath.pow(d, j) isa ForwardDiff.Dual{Nothing, <:Jet}
    @test j < d && isless(j, d) && !(j == d)
    @test promote_type(typeof(j), typeof(d)) == ForwardDiff.Dual{Nothing, typeof(j), 1}

    for args in (
            (j, d, 1.0), (d, j, 1.0), (1.0, j, d), (j, 1.0, d), (j, d, j), (d, j, d),
            (d, π, j), (j, 2, d), (j, d, 1 // 2), (2, d, j),
        )
        r = muladd(args...)
        @test r isa ForwardDiff.Dual{Nothing, <:Jet}
        vals = map(a -> a isa Jet ? a.v : a isa ForwardDiff.Dual ? ForwardDiff.value(a) : a, args)
        @test HyperHessians.value(ForwardDiff.value(r)) ≈ muladd(float.(vals)...) rtol = 1.0e-14
    end
end

@testset "ForwardDiff.derivative inside jet hessian" begin
    f(x) = ForwardDiff.derivative(y -> y^2 * sqrt(sum(abs2, x)) + atan(y, x[1]) + hypot(x[2], y), 2.0)
    x = [0.3, 0.4, 0.5]
    cfg = HyperHessians.HessianConfig(x, Jet)
    @test HyperHessians.hessian(f, x, cfg) ≈ ForwardDiff.hessian(f, x)
end

@testset "ForwardDiff.derivative inside hessian" begin
    # g(x) = d/dy sin(x*y) at y = 1.3  =>  g(x) = x*cos(1.3x)
    g(x) = ForwardDiff.derivative(y -> sin(x * y), 1.3)
    g′′(x) = -2.6 * sin(1.3 * x) - 1.69 * x * cos(1.3 * x)
    res = HyperHessians.hessian_gradient_value(g, 0.7)
    @test res.value ≈ g(0.7)
    @test res.hessian ≈ g′′(0.7)

    # vector input, mixed-op inner function
    f(x) = ForwardDiff.derivative(y -> y^2 * sqrt(sum(abs2, x)) + atan(y, x[1]) + hypot(x[2], y), 2.0)
    x = [0.3, 0.4, 0.5]
    H = HyperHessians.hessian(f, x)
    Href = ForwardDiff.hessian(f, x)
    @test H ≈ Href
end

end # module

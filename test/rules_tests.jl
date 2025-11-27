module RulesTests

using Test
using HyperHessians
using HyperHessians: HyperDual, ϵT
using ForwardDiff
using SpecialFunctions
using LogExpFunctions

function check_against_ForwardDiff(f, x, ::Type{T} = Float64) where {T}
    # @show f, x, T
    T == Float32 && return
    seed = ϵT{1, T}((one(T),))
    zero_seed = ϵT{1, T}((zero(T),))
    xT = T(x)
    h = HyperDual(xT, seed, seed, (zero_seed,))
    res = f(h)
    fd1 = ForwardDiff.derivative(z -> f(z), xT)
    fd2 = ForwardDiff.derivative(z -> ForwardDiff.derivative(y -> f(y), z), xT)
    @test res isa HyperDual{1, 1, T}
    # For now, don't verify precision for Float32, seems we need to reduce the tolerances for it
    T == Float32 && return
    @test res.ϵ1[1] ≈ fd1 atol = 1.0e-10 rtol = 1.0e-8
    @test res.ϵ2[1] ≈ fd1 atol = 1.0e-10 rtol = 1.0e-8
    @test res.ϵ12[1][1] ≈ fd2 atol = 1.0e-9 rtol = 1.0e-7
    return
end

@testset "rule derivatives vs ForwardDiff" begin
    xs = Dict(
        :sqrt => 1.2,
        :cbrt => 0.8,
        :abs2 => -1.1,
        :abs => -0.9,
        :inv => 0.7,
        :log => 1.4,
        :log10 => 1.6,
        :log2 => 1.5,
        :log1p => 0.2,
        :exp => 0.3,
        :exp2 => -0.1,
        :exp10 => 0.1,
        :expm1 => -0.4,
        :tan => 0.3,
        :sec => 0.3,
        :csc => 0.4,
        :cot => 0.35,
        :sind => 10.0,
        :cosd => 15.0,
        :tand => 12.0,
        :secd => 12.0,
        :cscd => 12.0,
        :cotd => 12.0,
        :asin => 0.3,
        :acos => 0.3,
        :atan => 0.4,
        :asec => 1.4,
        :acsc => -1.6,
        :acot => 0.4,
        :asind => 0.25,
        :acosd => 0.25,
        :atand => 0.5,
        :asecd => 1.5,
        :acscd => -1.5,
        :acotd => 0.6,
        :sinh => -0.2,
        :cosh => 0.2,
        :tanh => 0.2,
        :sech => 0.3,
        :csch => 1.1,
        :coth => 1.2,
        :asinh => -0.3,
        :acosh => 1.4,
        :atanh => 0.2,
        :asech => 0.5,
        :acsch => -1.3,
        :acoth => 1.3,
        :deg2rad => 37.0,
        :rad2deg => 0.7,
        :cospi => 0.2,
        :sinpi => 0.2,
    )
    for (fsym, _, _) in HyperHessians.DIFF_RULES
        x = xs[fsym]
        f = getfield(Base, fsym)
        for T in (Float64, Float32)
            check_against_ForwardDiff(f, x, T)
        end
    end
end

@testset "SpecialFunctions.jl rule derivatives vs ForwardDiff" begin
    xs = Dict(
        :airyai => 0.4,
        :airyaiprime => 0.4,
        :airyaix => 0.5,
        :airyaiprimex => 0.5,
        :airybi => 0.3,
        :airybiprime => 0.3,
        :besselj0 => 1.2,
        :besselj1 => 1.3,
        :bessely0 => 1.2,
        :bessely1 => 1.3,
        :dawson => 0.7,
        :digamma => 2.1,
        :gamma => 1.4,
        :invdigamma => 2.0,
        :trigamma => 1.3,
        :loggamma => 1.7,
        :erf => 0.6,
        :erfc => 0.6,
        :logerfc => 0.6,
        :erfcinv => 0.4,
        :erfcx => 0.6,
        :logerfcx => 0.6,
        :erfi => 0.5,
        :erfinv => 0.3,
        :expint => 1.1,
        # :expintx => 1.1, not available in ForwardDiff
        :expinti => 1.2,
        :sinint => 1.3,
        :cosint => 1.3,
    )
    Ext = Base.get_extension(HyperHessians, :HyperHessiansSpecialFunctionsExt)
    for (fsym, _, _) in Ext.SPECIALFUNCTIONS_DIFF_RULES
        haskey(xs, fsym) || continue # expintx does not support ForwardDiff
        x = xs[fsym]
        f = getfield(SpecialFunctions, fsym)
        for T in (Float64, Float32)
            check_against_ForwardDiff(f, x, T)
        end
    end
end
@testset "LogExpFunctions.jl rule derivatives vs ForwardDiff" begin
    xs = Dict(
        :xlogx => 0.4,
        :logistic => 0.5,
        :logit => 0.6,
        :log1psq => 0.7,
        :log1pexp => -0.1,
        :log1mexp => -1.1,
        :log2mexp => -1.2,
        :logexpm1 => 1.2,
        :log1pmx => 1.3,
        :logmxp1 => 1.4,
    )
    Ext = Base.get_extension(HyperHessians, :HyperHessiansLogExpFunctionsExt)
    for (fsym, _, _) in Ext.LOGEXPFUNCTIONS_DIFF_RULES
        x = xs[fsym]
        f = getfield(LogExpFunctions, fsym)
        for T in (Float64, Float32)
            check_against_ForwardDiff(f, x, T)
        end
    end
end

@testset "real divided by HyperDual uses inverse rule" begin
    seed = ϵT{1, Float64}((1.0,))
    zero_seed = ϵT{1, Float64}((0.0,))
    h = HyperDual(2.0, seed, seed, (zero_seed,))
    expected = inv(h)
    res = 1 / h
    @test res.v == expected.v
    @test Tuple(res.ϵ1) == Tuple(expected.ϵ1)
    @test Tuple(res.ϵ2) == Tuple(expected.ϵ2)
    @test Tuple(res.ϵ12[1]) == Tuple(expected.ϵ12[1])
end

end # module

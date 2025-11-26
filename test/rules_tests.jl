@testset "rule derivatives vs ForwardDiff" begin
    seed = Vec{1, Float64}((1.0,))
    zero_seed = Vec{1, Float64}((0.0,))
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
        h = HyperDual(x, seed, seed, (zero_seed,))
        res = f(h)
        fd1 = ForwardDiff.derivative(f, x)
        fd2 = ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), x)
        @test res.ϵ1[1] ≈ fd1 atol = 1.0e-10 rtol = 1.0e-8
        @test res.ϵ2[1] ≈ fd1 atol = 1.0e-10 rtol = 1.0e-8
        @test res.ϵ12[1][1] ≈ fd2 atol = 1.0e-9 rtol = 1.0e-7
    end
end

module FastMathTests

using Test
using HyperHessians
using HyperHessians: HyperDual, hessian_gradient_value

function test_dual_approx(actual::HyperDual, expected::HyperDual; rtol = 1.0e-12)
    @test actual.v ≈ expected.v rtol = rtol
    @test all(isapprox(a, b; rtol) for (a, b) in zip(actual.ϵ1, expected.ϵ1))
    @test all(isapprox(a, b; rtol) for (a, b) in zip(actual.ϵ2, expected.ϵ2))
    @test all(
        isapprox(a, b; rtol) for
            (actual_row, expected_row) in zip(actual.ϵ12, expected.ϵ12) for
            (a, b) in zip(actual_row, expected_row)
    )
    return
end

@testset "user @fastmath reaches HyperDual methods" begin
    f(x) = @fastmath sin(x[1]) + exp(x[2]) + x[1] * x[2] + x[1]^4 / x[2]
    f_ieee(x) = sin(x[1]) + exp(x[2]) + x[1] * x[2] + x[1]^4 / x[2]
    x = [0.4, 1.3]

    fast = hessian_gradient_value(f, x)
    ieee = hessian_gradient_value(f_ieee, x)
    @test fast.value ≈ ieee.value rtol = 1.0e-12
    @test fast.gradient ≈ ieee.gradient rtol = 1.0e-12
    @test fast.hessian ≈ ieee.hessian rtol = 1.0e-12
end

@testset "fast arithmetic" begin
    h1 = HyperDual(1.25, (0.7, -0.2), (0.4,), ((0.1,), (-0.3,)))
    h2 = HyperDual(0.8, (-0.1, 0.6), (0.9,), ((0.2,), (0.5,)))

    for (fast, ieee) in (
            (Base.FastMath.add_fast(h1, h2), h1 + h2),
            (Base.FastMath.sub_fast(h1, h2), h1 - h2),
            (Base.FastMath.mul_fast(h1, h2), h1 * h2),
            (Base.FastMath.div_fast(h1, h2), h1 / h2),
            (Base.FastMath.add_fast(h1, 0.3), h1 + 0.3),
            (Base.FastMath.add_fast(0.3, h1), 0.3 + h1),
            (Base.FastMath.sub_fast(h1, 0.3), h1 - 0.3),
            (Base.FastMath.sub_fast(0.3, h1), 0.3 - h1),
            (Base.FastMath.mul_fast(h1, 0.3), h1 * 0.3),
            (Base.FastMath.mul_fast(0.3, h1), 0.3 * h1),
            (Base.FastMath.div_fast(h1, 0.3), h1 / 0.3),
            (Base.FastMath.div_fast(0.3, h1), 0.3 / h1),
            (Base.FastMath.sub_fast(h1), -h1),
        )
        test_dual_approx(fast, ieee)
    end

    h32 = HyperDual(0.8f0, (-0.1f0, 0.6f0), (0.9f0,), ((0.2f0,), (0.5f0,)))
    @test Base.FastMath.add_fast(h1, h32) isa HyperDual{2, 1, Float64}
    @test Base.FastMath.mul_fast(h1, h32) isa HyperDual{2, 1, Float64}
    @test Base.FastMath.div_fast(h1, h32) isa HyperDual{2, 1, Float64}

    mixed_rdiv = Base.FastMath.div_fast(2.0, h32)
    @test mixed_rdiv isa HyperDual{2, 1, Float64}
    h32_promoted = convert(HyperDual{2, 1, Float64}, h32)
    test_dual_approx(mixed_rdiv, 2.0 / h32_promoted)
end

@testset "fast integer powers" begin
    for n in (2, 3, 4, 5, 8, -1, -3), x in (1.3, -1.2)
        h = HyperDual(x, (1.0,), (1.0,), ((0.0,),))
        test_dual_approx(Base.FastMath.pow_fast(h, n), h^n)
        test_dual_approx(Base.FastMath.pow_fast(h, Val(n)), h^n)
    end

    for n in (2, 3, 4, 5)
        result = hessian_gradient_value(x -> @fastmath(x^n), 0.0)
        @test result.value == 0.0
        @test result.gradient == 0.0
        @test result.hessian == (n == 2 ? 2.0 : 0.0)
    end
    literal = hessian_gradient_value(x -> @fastmath(x^4), 0.0)
    @test literal.gradient == 0.0
    @test literal.hessian == 0.0
end

@testset "fast unary rules" begin
    points = Dict(
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
        :asin => 0.3,
        :acos => 0.3,
        :atan => 0.4,
        :sinh => -0.2,
        :cosh => 0.2,
        :tanh => 0.2,
        :asinh => -0.3,
        :acosh => 1.4,
        :atanh => 0.2,
    )

    for (ordinary_sym, fast_sym) in HyperHessians.FAST_UNARY_OPS
        h = HyperDual(points[ordinary_sym], (0.7,), (-0.2,), ((0.3,),))
        ordinary = getfield(Base, ordinary_sym)
        fast = getfield(Base.FastMath, fast_sym)
        @test which(fast, (typeof(h),)).module === HyperHessians
        test_dual_approx(fast(h), ordinary(h); rtol = 1.0e-10)
    end

    h = HyperDual(0.4, (0.7,), (-0.2,), ((0.3,),))
    for (fast, ordinary) in (
            (Base.FastMath.sin_fast, sin),
            (Base.FastMath.cos_fast, cos),
        )
        @test which(fast, (typeof(h),)).module === HyperHessians
        test_dual_approx(fast(h), ordinary(h))
    end
    fast_s, fast_c = Base.FastMath.sincos_fast(h)
    ieee_s, ieee_c = sincos(h)
    test_dual_approx(fast_s, ieee_s)
    test_dual_approx(fast_c, ieee_c)
end

@testset "fast binary rules" begin
    points = Dict(
        :^ => (1.3, 0.7),
        :atan => (0.4, 0.9),
        :hypot => (0.8, 1.1),
        :log => (1.7, 2.3),
    )

    for (ordinary_sym, fast_sym) in HyperHessians.FAST_BINARY_OPS
        x, y = points[ordinary_sym]
        hx = HyperDual(x, (0.7,), (-0.2,), ((0.3,),))
        hy = HyperDual(y, (-0.4,), (0.6,), ((-0.1,),))
        ordinary = getfield(Base, ordinary_sym)
        fast = getfield(Base.FastMath, fast_sym)
        @test which(fast, (typeof(hx), typeof(hy))).module === HyperHessians
        test_dual_approx(fast(hx, hy), ordinary(hx, hy); rtol = 1.0e-10)
        test_dual_approx(fast(hx, y), ordinary(hx, y); rtol = 1.0e-10)
        test_dual_approx(fast(x, hy), ordinary(x, hy); rtol = 1.0e-10)
    end
end

end # module

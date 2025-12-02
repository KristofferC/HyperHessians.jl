module hessian_gradient_valueTests

using Test
using HyperHessians: hessian_gradient_value, hessian_gradient_value!, HessianConfig, Chunk
using DiffTests
using ForwardDiff

@testset "hessian_gradient_value! DimensionMismatch" begin
    f(x) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]
    cfg = HessianConfig(x)
    cfg_chunked = HessianConfig(x, Chunk{2}())
    @test_throws DimensionMismatch hessian_gradient_value!(zeros(2, 2), zeros(3), f, x, cfg)
    @test_throws DimensionMismatch hessian_gradient_value!(zeros(3, 3), zeros(2), f, x, cfg)
    @test_throws DimensionMismatch hessian_gradient_value!(zeros(2, 2), zeros(3), f, x, cfg_chunked)
    @test_throws DimensionMismatch hessian_gradient_value!(zeros(3, 3), zeros(2), f, x, cfg_chunked)
end

@testset "hessian_gradient_value scalar input" begin
    # Scalar input, scalar output
    f_scalar(x) = x^3 - 2x^2 + x
    x = 2.5
    res = hessian_gradient_value(f_scalar, x)
    @test res.value ≈ f_scalar(x)
    @test res.gradient ≈ ForwardDiff.derivative(f_scalar, x)
    @test res.hessian ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(f_scalar, x), x)

    # Scalar input, tuple output
    f_tuple(x) = (x^2, x^3)
    res_tuple = hessian_gradient_value(f_tuple, x)
    @test res_tuple.value == (x^2, x^3)
    @test res_tuple.gradient[1] ≈ ForwardDiff.derivative(x -> x^2, x)
    @test res_tuple.gradient[2] ≈ ForwardDiff.derivative(x -> x^3, x)
    @test res_tuple.hessian[1] ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> y^2, x), x)
    @test res_tuple.hessian[2] ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> y^3, x), x)

    # Scalar input, array output
    f_array(x) = [x^2, x^3, sin(x)]
    res_array = hessian_gradient_value(f_array, x)
    @test res_array.value ≈ f_array(x)
    @test res_array.gradient[1] ≈ ForwardDiff.derivative(x -> x^2, x)
    @test res_array.gradient[2] ≈ ForwardDiff.derivative(x -> x^3, x)
    @test res_array.gradient[3] ≈ ForwardDiff.derivative(sin, x)
    @test res_array.hessian[1] ≈ 2.0  # d²/dx² x² = 2
    @test res_array.hessian[2] ≈ 6x   # d²/dx² x³ = 6x
    @test res_array.hessian[3] ≈ -sin(x)  # d²/dx² sin(x) = -sin(x)
end

@testset "hessian_gradient_value" begin
    f = DiffTests.ackley
    x = rand(8)
    H = zeros(length(x), length(x))
    G = zeros(length(x))
    cfg_chunk = HessianConfig(x, Chunk{4}())
    value = hessian_gradient_value!(H, G, f, x, cfg_chunk)
    @test value ≈ f(x)
    @test G ≈ ForwardDiff.gradient(f, x)
    @test H ≈ ForwardDiff.hessian(f, x)

    H2 = similar(H)
    G2 = similar(G)
    cfg_vec = HessianConfig(x, Chunk{length(x)}())
    value_vec = hessian_gradient_value!(H2, G2, f, x, cfg_vec)
    @test value_vec ≈ f(x)
    @test G2 ≈ ForwardDiff.gradient(f, x)
    @test H2 ≈ ForwardDiff.hessian(f, x)

    res = hessian_gradient_value(f, x, cfg_vec)
    @test res.value ≈ f(x)
    @test res.gradient ≈ ForwardDiff.gradient(f, x)
    @test res.hessian ≈ ForwardDiff.hessian(f, x)
end

@testset "hessian_gradient_value! zero allocations" begin
    f = x -> sum(abs2, x)
    n = 8
    x = rand(n)
    H = zeros(n, n)
    G = zeros(n)

    # Full chunk (vector path)
    cfg_full = HessianConfig(x, Chunk{n}())
    hessian_gradient_value!(H, G, f, x, cfg_full)
    @test @allocated(hessian_gradient_value!(H, G, f, x, cfg_full)) == 0 broken = VERSION < v"1.11"

    # Chunked path
    cfg_chunk = HessianConfig(x, Chunk{4}())
    hessian_gradient_value!(H, G, f, x, cfg_chunk)
    @test @allocated(hessian_gradient_value!(H, G, f, x, cfg_chunk)) == 0 broken = VERSION < v"1.11"
end

end # module

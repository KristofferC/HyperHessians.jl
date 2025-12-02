module arrayTests

using Test
using ForwardDiff
using HyperHessians: hessian, hessian!, hessian_gradient_value, hessian_gradient_value!, HessianConfig, Chunk

@testset "array inputs" begin
    A = reshape(collect(1.0:4.0), 2, 2)
    f_sum(x) = sum(x)
    f_sq(x) = sum(abs2, x)

    @test hessian(f_sum, A) == ForwardDiff.hessian(f_sum, A)
    @test hessian(f_sq, A) == ForwardDiff.hessian(f_sq, A)

    cfg = HessianConfig(A, Chunk(2))
    @test hessian(f_sq, A, cfg) == ForwardDiff.hessian(f_sq, A)

    B = reshape(collect(1.0:8.0), 2, 2, 2)
    f_cube(x) = sum(x .^ 3)
    res = hessian_gradient_value(f_cube, B)

    @test res.hessian ≈ ForwardDiff.hessian(f_cube, B)
    @test res.gradient ≈ ForwardDiff.gradient(f_cube, B)

    H = zeros(length(B), length(B))
    G = similar(B)
    value = hessian_gradient_value!(H, G, f_cube, B)

    @test value ≈ f_cube(B)
    @test G ≈ ForwardDiff.gradient(f_cube, B)
    @test H ≈ ForwardDiff.hessian(f_cube, B)

    cfg_full = HessianConfig(B)
    f_noalloc(x) = sum(abs2, x)
    Halloc = similar(H)
    Galloc = similar(G)
    @test @allocated(hessian!(Halloc, f_noalloc, B, cfg_full)) == 0 broken = true
    @test @allocated(hessian_gradient_value!(Halloc, Galloc, f_noalloc, B, cfg_full)) == 0 broken = true
end

end

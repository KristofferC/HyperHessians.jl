module arrayTests

using Test
using ForwardDiff
using HyperHessians: hessian, hessian!, hessian_gradient_value, hessian_gradient_value!, hvp, hvp!, hvp_gradient_value, hvp_gradient_value!, HessianConfig, HVPConfig, Chunk

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

    C = reshape(collect(0.1:0.1:0.6), 2, 3)
    g_fun(x) = sum(sin, x)
    t1 = fill(1.0, size(C))
    t2 = fill(2.0, size(C))
    cfg_hvp = HVPConfig(C, (t1, t2), Chunk(6))

    hv_expected = reshape(ForwardDiff.hessian(y -> g_fun(reshape(y, size(C))), vec(C)) * vec(t1), size(C))
    @test hvp(g_fun, C, t1) ≈ hv_expected
    hv_tuple = hvp(g_fun, C, (t1, t2), cfg_hvp)
    @test hv_tuple[1] ≈ hv_expected
    @test hv_tuple[2] ≈ reshape(ForwardDiff.hessian(y -> g_fun(reshape(y, size(C))), vec(C)) * vec(t2), size(C))

    hv_out = similar(C)
    hvp!(hv_out, g_fun, C, t1)
    @test hv_out ≈ hv_expected

    gv_res = hvp_gradient_value(g_fun, C, t1)
    @test gv_res.gradient ≈ ForwardDiff.gradient(g_fun, C)
    @test gv_res.hvp ≈ hv_expected

    hv_out2 = (similar(C), similar(C))
    grad_out = similar(C)
    hvp_gradient_value!(hv_out2, grad_out, g_fun, C, (t1, t2), cfg_hvp)
    @test grad_out ≈ ForwardDiff.gradient(g_fun, C)
    @test hv_out2[1] ≈ hv_tuple[1]
    @test hv_out2[2] ≈ hv_tuple[2]
end

end

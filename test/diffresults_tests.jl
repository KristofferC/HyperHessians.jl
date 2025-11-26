@testset "DiffResults extension" begin
    f = DiffTests.rosenbrock_3
    x = rand(4)

    res = DiffResults.HessianResult(x)
    HyperHessians.hessian!(res, f, x)

    @test DiffResults.value(res) ≈ f(x)
    @test DiffResults.gradient(res) ≈ ForwardDiff.gradient(f, x)
    @test DiffResults.hessian(res) ≈ ForwardDiff.hessian(f, x)
end

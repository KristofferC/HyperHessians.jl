@testset "StaticArrays" begin
    x = rand(SVector{4})
    f = DiffTests.rosenbrock_3
    @test ForwardDiff.hessian(f, x) ≈ hessian(f, x)
    @test hessian(f, x) isa MMatrix
    @test_broken hessian(f, x) isa SMatrix
end

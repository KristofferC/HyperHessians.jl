@testset "Float32" begin
    x = rand(Float32, 8)
    @test hessian(ackley_stable, x) isa Matrix{Float32}
    @test hessian(ackley_stable, x) ≈ ForwardDiff.hessian(ackley_stable, x)
    @test hessian(DiffTests.ackley, x) ≈ ForwardDiff.hessian(DiffTests.ackley, x)
end

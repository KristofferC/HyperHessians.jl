@testset "StaticArrays" begin
    x = rand(SVector{4})
    f = DiffTests.rosenbrock_3
    @test ForwardDiff.hessian(f, x) ≈ hessian(f, x)
    @test hessian(f, x) isa SMatrix

    g = x -> sum(abs2, x)
    expected_H = @SMatrix [2.0 0 0 0; 0 2.0 0 0; 0 0 2.0 0; 0 0 0 2.0]
    H = hessian(g, x)
    @test H === expected_H
    @test @allocated(hessian(g, x)) == 0 broken = VERSION < v"1.11"

    dir = SVector{4}(1.0, 2.0, 3.0, 4.0)
    hv = hvp(g, x, dir)
    @test hv isa SVector
    @test hv === 2 .* dir
    @test @allocated(hvp(g, x, dir)) == 0 broken = VERSION < v"1.11"


    res = hessiangradvalue(g, x)
    @test res.gradient === 2 .* x
    @test res.hessian === expected_H
    @test @allocated(hessiangradvalue(g, x)) == 0 broken = VERSION < v"1.11"
end

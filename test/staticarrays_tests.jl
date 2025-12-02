module StaticArraysTests

using Test
using HyperHessians: hessian, hvp, hvp_gradient_value, hessian_gradient_value
using DiffTests
using ForwardDiff
using StaticArrays

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

    res = hvp_gradient_value(g, x, dir)
    @test res.gradient === 2 .* x
    @test res.hvp === 2 .* dir

    # Bundled tangents (multiple directions)
    dir2 = SVector{4}(4.0, 3.0, 2.0, 1.0)
    hv_bundle = hvp(g, x, (dir, dir2))
    @test hv_bundle[1] === 2 .* dir
    @test hv_bundle[2] === 2 .* dir2
    @test @allocated(hvp(g, x, (dir, dir2))) == 0 broken = VERSION < v"1.11"

    res_bundle = hvp_gradient_value(g, x, (dir, dir2))
    @test res_bundle.gradient === 2 .* x
    @test res_bundle.hvp[1] === 2 .* dir
    @test res_bundle.hvp[2] === 2 .* dir2


    res = hessian_gradient_value(g, x)
    @test res.gradient === 2 .* x
    @test res.hessian === expected_H
    @test @allocated(hessian_gradient_value(g, x)) == 0 broken = VERSION < v"1.11"
end

end # module

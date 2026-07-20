module NaNMathStandaloneTests

using Test
using HyperHessians
using NaNMath

@testset "NaNMath extension loads independently" begin
    @test Base.get_extension(HyperHessians, :HyperHessiansNaNMathExt) !== nothing
    @test HyperHessians.hessian(x -> NaNMath.sin(x), 0.3) ≈ -sin(0.3)
end

end

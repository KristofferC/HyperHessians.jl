using HyperHessians: HyperHessians, hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, hvp!, Chunk, HessianConfig, DirectionalHVPConfig, HyperDual, ϵT
using DiffTests
using ForwardDiff
using DiffResults
using Test
if HyperHessians.USE_SIMD
    using SIMD
end
using StaticArrays
using LogExpFunctions
using SpecialFunctions

include("helpers.jl")
include("rules_tests.jl")
include("correctness_tests.jl")
include("hvp_tests.jl")
include("hessiangradvalue_tests.jl")
include("float32_tests.jl")
include("staticarrays_tests.jl")
include("no_spurious_promotions_tests.jl")
include("diffresults_tests.jl")

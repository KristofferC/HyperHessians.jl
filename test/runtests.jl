using HyperHessians: HyperHessians, hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, hvp!, Chunk, HessianConfig, DirectionalHVPConfig, HyperDual
using DiffTests
using ForwardDiff
using Test
using SIMD
using StaticArrays

include("helpers.jl")
include("rules_tests.jl")
include("correctness_tests.jl")
include("hvp_tests.jl")
include("hessiangradvalue_tests.jl")
include("float32_tests.jl")
include("staticarrays_tests.jl")
include("no_spurious_promotions_tests.jl")

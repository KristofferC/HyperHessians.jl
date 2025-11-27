module HyperHessians

if VERSION >= v"1.11.0-"
    eval(Meta.parse("public gradient, gradient!, gradientvalue, gradientvalue!, hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, hvp!, GradientConfig, HessianConfig, DirectionalHVPConfig, Chunk"))
end

using SIMD: SIMD, Vec, vstore
using CommonSubexpressions: cse

include("hyperdual.jl")
include("rules.jl")
include("chunks.jl")
include("hessian.jl")

end # module

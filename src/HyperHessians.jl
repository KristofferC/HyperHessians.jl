module HyperHessians

if VERSION >= v"1.11.0-"
    eval(Meta.parse("public hessian, hessian!, hessian_gradient_value, hessian_gradient_value!, hvp, hvp!, hvp_gradient_value, hvp_gradient_value!, vhvp, vhvp_gradient_value, VHVPConfig, HessianConfig, HVPConfig, Chunk"))
end


using CommonSubexpressions: cse, binarize

include("hyperdual.jl")
include("rules.jl")
include("chunks.jl")
include("hessian.jl")

end # module

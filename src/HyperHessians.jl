module HyperHessians

if VERSION >= v"1.11.0-"
    eval(Meta.parse("public hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, hvp!, hvpgrad, hvpgrad!, HessianConfig, DirectionalHVPConfig, Chunk"))
end


using CommonSubexpressions: cse, binarize

include("hyperdual.jl")
include("rules.jl")
include("chunks.jl")
include("hessian.jl")

end # module

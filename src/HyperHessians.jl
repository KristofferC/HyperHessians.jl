module HyperHessians

using SIMD: SIMD, Vec


if VERSION >= v"1.11.0-"
    eval(Meta.parse("public hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, hvp!, HessianConfig, DirectionalHVPConfig, Chunk"))
end


using CommonSubexpressions: cse, binarize

include("hyperdual.jl")
include("simd_ops.jl")
include("rules.jl")
include("chunks.jl")
include("hessian.jl")

end # module

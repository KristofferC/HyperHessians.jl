module HyperHessians

using SIMD: SIMD, Vec, vstore
using CommonSubexpressions: cse

include("hyperdual.jl")
include("rules.jl")
include("chunks.jl")
include("hessian.jl")

end # module

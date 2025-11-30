module HyperHessians

# Allows using `SIMD.Vec` for partials but that doesn't seem to be faster in practice
# You need to add SIMD to the package for this to work.
const USE_SIMD = false

if USE_SIMD
    using SIMD: SIMD, Vec
end


if VERSION >= v"1.11.0-"
    eval(Meta.parse("public hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, hvp!, HessianConfig, DirectionalHVPConfig, Chunk, pick_chunksize"))
end


using CommonSubexpressions: cse, binarize

include("hyperdual.jl")
if USE_SIMD
    include("simd_ops.jl")
end
include("rules.jl")
include("chunks.jl")
include("hessian.jl")
include("chunk_picker.jl")

end # module

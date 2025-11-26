module HyperHessiansDiffResultsExt

using HyperHessians
using HyperHessians: HessianConfig, hessiangradvalue!, AbstractHessianConfig
using DiffResults

function HyperHessians.hessian!(result::DiffResults.DiffResult, f, x::AbstractVector, cfg::AbstractHessianConfig = HessianConfig(x))
    H = DiffResults.hessian(result)
    G = DiffResults.gradient(result)
    val = hessiangradvalue!(H, G, f, x, cfg)
    DiffResults.value!(result, val)
    DiffResults.gradient!(result, G)
    DiffResults.hessian!(result, H)
    return result
end

end

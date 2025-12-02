module HyperHessiansDiffResultsExt

using HyperHessians
using HyperHessians: HessianConfig, hessian_gradient_value!
using DiffResults

function HyperHessians.hessian!(result::DiffResults.DiffResult, f, x::AbstractVector, cfg::HessianConfig = HessianConfig(x))
    H = DiffResults.hessian(result)
    G = DiffResults.gradient(result)
    val = hessian_gradient_value!(H, G, f, x, cfg)
    DiffResults.value!(result, val)
    DiffResults.gradient!(result, G)
    DiffResults.hessian!(result, H)
    return result
end

end

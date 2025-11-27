module HyperHessiansDiffResultsExt

using HyperHessians
using HyperHessians: GradientConfig, HessianConfig, gradientvalue!, hessiangradvalue!
using DiffResults

function HyperHessians.hessian!(result::DiffResults.DiffResult, f, x::AbstractVector, cfg::HessianConfig = HessianConfig(x))
    H = DiffResults.hessian(result)
    G = DiffResults.gradient(result)
    val = hessiangradvalue!(H, G, f, x, cfg)
    DiffResults.value!(result, val)
    DiffResults.gradient!(result, G)
    DiffResults.hessian!(result, H)
    return result
end

function HyperHessians.gradient!(result::DiffResults.DiffResult, f, x::AbstractVector, cfg::GradientConfig = GradientConfig(x))
    G = DiffResults.gradient(result)
    val = gradientvalue!(G, f, x, cfg)
    DiffResults.value!(result, val)
    DiffResults.gradient!(result, G)
    return result
end

end

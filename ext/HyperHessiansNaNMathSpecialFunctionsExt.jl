module HyperHessiansNaNMathSpecialFunctionsExt

using HyperHessians: HyperDual, chain_rule_dual
using NaNMath
using SpecialFunctions: digamma, trigamma

@inline function NaNMath.lgamma(h::HyperDual{N1, N2}) where {N1, N2}
    x = h.v
    f = NaNMath.lgamma(x)
    return chain_rule_dual(h, f, digamma(x), trigamma(x))
end

end

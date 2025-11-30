module HyperHessiansNaNMathExt

using HyperHessians
using HyperHessians: changeprecision, chain_rule_dual, HyperDual
using CommonSubexpressions: cse
using NaNMath
using SpecialFunctions: digamma, trigamma

# NaNMath-specific rule_expr that qualifies function names with NaNMath
function nanmath_rule_expr(f, f′, f′′)
    ex = quote
        f = NaNMath.$f(x)
        f′ = $f′
        f′′ = $f′′
    end
    return Base.remove_linenums!(ex)
end

# runic: off
const NANMATH_DIFF_RULES = [
    (:sqrt  , :(1 / 2 / f)                                     , :(-2 * f′^3))
    (:sin   , :(NaNMath.cos(x))                                , :(-f))
    (:cos   , :(-NaNMath.sin(x))                               , :(-f))
    (:tan   , :(1 + f^2)                                       , :(2 * f′ * f))
    (:asin  , :(1 / NaNMath.sqrt(1 - x^2))                     , :(x * f′^3))
    (:acos  , :(-1 / NaNMath.sqrt(1 - x^2))                    , :(x * f′^3))
    (:acosh , :(1 / NaNMath.sqrt(x^2 - 1))                     , :(-x * f′^3))
    (:atanh , :(1 / (1 - x^2))                                 , :(2 * x * f′^2))
    (:log   , :(inv(x))                                        , :(-f′^2))
    (:log2  , :(inv(x) / log(2))                               , :(-log(2) * f′^2))
    (:log10 , :(inv(x) / log(10))                              , :(-log(10) * f′^2))
    (:log1p , :(inv(x + 1))                                    , :(-f′^2))
    (:lgamma, :(digamma(x))                                    , :(trigamma(x)))
]
# runic: on

for (i, rule) in enumerate(NANMATH_DIFF_RULES)
    prec_f′ = changeprecision(rule[2])
    prec_f′′ = changeprecision(rule[3])
    NANMATH_DIFF_RULES[i] = (rule[1], prec_f′, prec_f′′)
end

for (f, f′, f′′) in NANMATH_DIFF_RULES
    expr = nanmath_rule_expr(f, f′, f′′)
    cse_expr = cse(expr; warn = false)
    @eval @inline function NaNMath.$f(h::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = h.v
        $cse_expr
        return chain_rule_dual(h, f, f′, f′′)
    end
end

end

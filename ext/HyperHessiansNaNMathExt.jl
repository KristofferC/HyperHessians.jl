module HyperHessiansNaNMathExt

using HyperHessians
using HyperHessians: changeprecision, chain_rule_dual, HyperDual
using CommonSubexpressions: cse, binarize
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

function nanmath_binary_rule_expr(f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
    ex = quote
        f = NaNMath.$f(x, y)
        fₓ = $fₓ
        fᵧ = $fᵧ
        fₓₓ = $fₓₓ
        fₓᵧ = $fₓᵧ
        fᵧᵧ = $fᵧᵧ
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

# runic: off
const NANMATH_BINARY_DIFF_RULES = [
    (
        :pow,
        :(y * f / x),
        :(f * NaNMath.log(x)),
        :(y * (y - one(y)) * f / (x * x)),
        :((f / x) * (one(x) + y * NaNMath.log(x))),
        :(f * (NaNMath.log(x)^2)),
    ),
]
# runic: on

for (i, rule) in enumerate(NANMATH_DIFF_RULES)
    prec_f′ = changeprecision(rule[2])
    prec_f′′ = changeprecision(rule[3])
    NANMATH_DIFF_RULES[i] = (rule[1], prec_f′, prec_f′′)
end

for (i, rule) in enumerate(NANMATH_BINARY_DIFF_RULES)
    prec_fₓ = changeprecision(rule[2])
    prec_fᵧ = changeprecision(rule[3])
    prec_fₓₓ = changeprecision(rule[4])
    prec_fₓᵧ = changeprecision(rule[5])
    prec_fᵧᵧ = changeprecision(rule[6])
    NANMATH_BINARY_DIFF_RULES[i] = (rule[1], prec_fₓ, prec_fᵧ, prec_fₓₓ, prec_fₓᵧ, prec_fᵧᵧ)
end

for (f, f′, f′′) in NANMATH_DIFF_RULES
    expr = nanmath_rule_expr(f, f′, f′′)
    cse_expr = cse(binarize(expr); warn = false)
    @eval @inline function NaNMath.$f(h::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = h.v
        $cse_expr
        return chain_rule_dual(h, f, f′, f′′)
    end
end

for (f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ) in NANMATH_BINARY_DIFF_RULES
    expr = nanmath_binary_rule_expr(f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
    cse_expr = cse(binarize(expr); warn = false)
    @eval @inline function NaNMath.$f(hx::HyperDual{N1, N2, T}, hy::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = hx.v
        y = hy.v
        $cse_expr
        return chain_rule_dual(hx, hy, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
    end
    @eval @inline function NaNMath.$f(hx::HyperDual{N1, N2, T1}, hy::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2}
        return NaNMath.$f(promote(hx, hy)...)
    end
    @eval @inline function NaNMath.$f(hx::HyperDual{N1, N2, T}, y_raw::Real) where {N1, N2, T}
        x = hx.v
        y = T(y_raw)
        $cse_expr
        return chain_rule_dual(hx, f, fₓ, fₓₓ)
    end
    @eval @inline function NaNMath.$f(x_raw::Real, hy::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = T(x_raw)
        y = hy.v
        $cse_expr
        return chain_rule_dual(hy, f, fᵧ, fᵧᵧ)
    end
end

end

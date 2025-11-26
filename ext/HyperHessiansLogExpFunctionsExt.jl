module HyperHessiansLogExpFunctionsExt

using HyperHessians
using HyperHessians: changeprecision, rule_expr
using CommonSubexpressions: cse
using LogExpFunctions

const LOGEXPFUNCTIONS_DIFF_RULES = [
    (:xlogx, :(1 + log(x)), :(1 / x))
    (:logistic, :(v*(1 - v)), :(f′*(1 - 2*v)))
    (:logit, :(inv(x * (1 - x))), :(f′ * f′ * (2*x - 1)))
    (:log1psq, :(2 * x / (1 + x^2)), :(-(2 * (x^2 - 1))/(1 + x^2)^2))
    (:log1pexp, :(logistic(x)), :(f′*(1 - f′)))
    (:log1mexp, :(-exp(x - v)), :(-f′ * f′ * exp(-x)))
    (:log2mexp, :(-exp(x - v)), :(-f′ * f′ * 2 * exp(-x)))
    (:logexpm1, :(exp(x - v)), :(-f′ * f′ * exp(-x))))
    (:log1pmx, :(-x / (1 + x)), :(1 / (1 + x)^2))
    (:logmxp1, :((1 - x) / x), :(-1 / x^2))
]

for (i, rule) in enumerate(LOGEXPFUNCTIONS_DIFF_RULES)
    prec_f′ = changeprecision(rule[2])
    prec_f′′ = changeprecision(rule[3])
    LOGEXPFUNCTIONS_DIFF_RULES[i] = (rule[1], prec_f′, prec_f′′)
end

for (f, f′, f′′) in LOGEXPFUNCTIONS_DIFF_RULES
    expr = rule_expr(f, f′, f′′)
    cse_expr = cse(expr; warn = false)
    @eval @inline function LogExpFunctions.$f(h::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = h.v
        $cse_expr
    end
end

end

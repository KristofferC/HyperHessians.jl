module HyperHessiansSpecialFunctionsExt

using HyperHessians
using HyperHessians: rule_expr, HyperDual, ⊗
using CommonSubexpressions: cse
using SpecialFunctions
using SpecialFunctions: sqrtπ

# runic: off
const SPECIALFUNCTIONS_DIFF_RULES = [
    (:airyai      , :(airyaiprime(x))                                          , :(x * f))
    (:airyaiprime , :(x * airyai(x))                                           , :(airyai(x) + x * f))
    (:airyaix     , :(airyaiprimex(x) + sqrt(x) * f)                           , :(x * f + sqrt(x) * airyaiprimex(x) + f / (2 * sqrt(x)) + sqrt(x) * f′))
    (:airyaiprimex, :(x * airyaix(x) + sqrt(x) * f)                            , :(airyaix(x) + x * (airyaiprimex(x) + sqrt(x) * airyaix(x)) + f / (2 * sqrt(x)) + sqrt(x) * f′))
    (:airybi      , :(airybiprime(x))                                          , :(x * f))
    (:airybiprime , :(x * airybi(x))                                           , :(airybi(x) + x * f))
    (:besselj0    , :(-besselj1(x))                                            , :(-(besselj0(x) - besselj(2, x)) / 2))
    (:besselj1    , :((besselj0(x) - besselj(2, x)) / 2)                       , :((besselj(3, x) - 3 * besselj1(x)) / 4))
    (:bessely0    , :(-bessely1(x))                                            , :(-(bessely0(x) - bessely(2, x)) / 2))
    (:bessely1    , :((bessely0(x) - bessely(2, x)) / 2)                       , :((bessely(3, x) - 3 * bessely1(x)) / 4))
    (:dawson      , :(1 - 2 * x * f)                                           , :(-2 * f - 2 * x * f′))
    (:digamma     , :(trigamma(x))                                             , :(polygamma(2, x)))
    (:gamma       , :(f * digamma(x))                                          , :(f * (digamma(x)^2 + trigamma(x))))
    (:invdigamma  , :(inv(trigamma(f)))                                        , :(-f′^3 * polygamma(2, f)))
    (:trigamma    , :(polygamma(2, x))                                         , :(polygamma(3, x)))
    (:loggamma    , :(digamma(x))                                              , :(trigamma(x)))
    (:erf         , :(2 * exp(-x^2) / sqrtπ)                                   , :(-2 * x * f′))
    (:erfc        , :(-2 * exp(-x^2) / sqrtπ)                                  , :(-2 * x * f′))
    (:logerfc     , :(-(2 * exp(-x^2 - f)) / sqrtπ)                            , :(f′ * (-2 * x - f′)))
    (:erfcinv     , :(-(sqrtπ * (exp(f^2) / 2)))                               , :(2 * f * f′^2))
    (:erfcx       , :(2 * (x * f - inv(oftype(f, sqrtπ))))                     , :(2 * f + 2 * x * f′))
    (:logerfcx    , :(2 * (x - exp(-f) / sqrtπ))                               , :(2 + 2 * exp(-f) * f′ / sqrtπ))
    (:erfi        , :(2 * exp(x^2) / sqrtπ)                                    , :(2 * x * f′))
    (:erfinv      , :((sqrtπ * exp(f^2)) / 2)                                  , :(2 * f * f′^2))
    (:expint      , :(-exp(-x) / x)                                            , :(exp(-x) / x + exp(-x) / x^2))
    (:expintx     , :(f - inv(x))                                              , :(f′ + inv(x^2)))
    (:expinti     , :(exp(x) / x)                                              , :(f′ * (1 - inv(x))))
    (:sinint      , :(sin(x) / x)                                              , :((x * cos(x) - sin(x)) / x^2))
    (:cosint      , :(cos(x) / x)                                              , :(-(x * sin(x) + cos(x)) / x^2))
]
# runic: on

for (f, f′, f′′) in SPECIALFUNCTIONS_DIFF_RULES
    expr = rule_expr(f, f′, f′′)
    cse_expr = cse(expr; warn = false)
    @eval @inline function SpecialFunctions.$f(h::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = h.v
        $cse_expr
    end
end

end

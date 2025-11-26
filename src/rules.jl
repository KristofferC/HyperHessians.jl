# From HyperDualNumbers
# runic: off
const DIFF_RULES = [
    (:sqrt     , :(1 / 2 / f)                                     , :(-f′ / 2 / x))
    (:cbrt     , :(1 / 3 / f^2)                                   , :(-2 * f′ / 3 / x))
    (:abs2     , :(2 * x)                                         , :(2))
    (:abs      , :(signbit(x) ? -one(x) : one(x))                 , :(0))
    (:inv      , :(-f^2)                                          , :(2 * f^3))
    (:log      , :(inv(x))                                        , :(-f′^2))
    (:log10    , :(inv(x) / log(10))                              , :(-log(10) * f′^2))
    (:log2     , :(inv(x) / log(2))                               , :(-log(2) * f′^2))
    (:log1p    , :(inv(x + 1))                                    , :(-f′^2))
    (:exp      , :(f)                                             , :(f))
    (:exp2     , :(f * log(2))                                    , :(f′ * log(2)))
    (:exp10    , :(f * log(10))                                   , :(f′ * log(10)))
    (:expm1    , :(f + 1)                                         , :(f + 1))
    #( :sin    , :(cos(x))                                        , :(-sin(x)))
    #( :cos    , :(-sin(x))                                       , :(-cos(x)))
    (:tan      , :(1 + f^2)                                       , :(2 * f′ * f))
    (:sec      , :(f * tan(x))                                    , :(f * (tan(x)^2 + f^2)))
    (:csc      , :(-cot(x) * f)                                   , :(f * (cot(x)^2 + f^2)))
    (:cot      , :(-(1 + f^2))                                    , :(2 * f * (1 + f^2)))
    (:sind     , :(π * cos(π * x / 180) / 180)                    , :(-π / 180 * π / 180 * f))
    (:cosd     , :(-π * sin(π * x / 180) / 180)                   , :(-π / 180 * π / 180 * f))
    (:tand     , :(π * (1 + f^2) / 180)                           , :((π / 180)^2 * 2 * f * (1 + f^2)))
    (:secd     , :(π * f * tan(π * x / 180) / 180)                , :((π / 180)^2 * f * (tan(π * x / 180)^2 + f^2)))
    (:cscd     , :(-π * cot(π * x / 180) * f / 180)               , :((π / 180)^2 * f * (cot(π * x / 180)^2 + f^2)))
    (:cotd     , :(-π * (1 + f^2) / 180)                          , :((π / 180)^2 * 2 * f * (1 + f^2)))
    (:asin     , :(1 / sqrt(1 - x^2))                             , :(x * f′^3))
    (:acos     , :(-1 / sqrt(1 - x^2))                            , :(x * f′^3))
    (:atan     , :(1 / (1 + x^2))                                 , :(-2 * x * f′^2))
    (:asec     , :(1 / (x * sqrt(x^2 - 1)))                       , :(-(2 * x^2 - 1) / (x^2 * (x^2 - 1)^(3 / 2))))
    (:acsc     , :(-1 / (abs(x) * sqrt(x^2 - 1)))                 , :(sign(x) * (2 * x^2 - 1) / (x^2 * (x^2 - 1)^(3 / 2))))
    (:acot     , :(-1 / (1 + x^2))                                , :(2 * x * f′^2))
    (:asind    , :((180 / π) / sqrt(1 - x^2))                     , :((180 / π) * x / (1 - x^2)^(3 / 2)))
    (:acosd    , :(-(180 / π) / sqrt(1 - x^2))                    , :(-(180 / π) * x / (1 - x^2)^(3 / 2)))
    (:atand    , :((180 / π) / (1 + x^2))                         , :(-2 * x * f′^2 / (180 / π)))
    (:asecd    , :((180 / π) / (x * sqrt(x^2 - 1)))               , :(-(180 / π) * (2 * x^2 - 1) / (x^2 * (x^2 - 1)^(3 / 2))))
    (:acscd    , :(-(180 / π) / (abs(x) * sqrt(x^2 - 1)))         , :((180 / π) * sign(x) * (2 * x^2 - 1) / (x^2 * (x^2 - 1)^(3 / 2))))
    (:acotd    , :(-(180 / π) / (1 + x^2))                        , :(2 * x * f′^2 / (180 / π)))
    (:sinh     , :(cosh(x))                                       , :(f))
    (:cosh     , :(sinh(x))                                       , :(f))
    (:tanh     , :(1 - f^2)                                       , :(-2 * f′ * f))
    (:sech     , :(-f * tanh(x))                                  , :(f * (2 * tanh(x)^2 - 1)))
    (:csch     , :(-coth(x) * f)                                  , :((2 * f^2 + 1) * f))
    (:coth     , :(1 - f^2)                                       , :(-2 * f′ * f))
    (:asinh    , :(1 / sqrt(1 + x^2))                             , :(-x * f′^3))
    (:acosh    , :(1 / sqrt(x^2 - 1))                             , :(-x * f′^3))
    (:atanh    , :(1 / (1 - x^2))                                 , :(2 * x * f′^2))
    (:asech    , :(-1 / (x * sqrt(1 - x^2)))                      , :((1 - 2 * x^2) / (x^2 * (1 - x^2)^(3 / 2))))
    (:acsch    , :(-1 / (abs(x) * sqrt(1 + x^2)))                 , :(sign(x) * (2 * x^2 + 1) / (x^2 * (1 + x^2)^(3 / 2))))
    (:acoth    , :(1 / (1 - x^2))                                 , :(2 * x * f′^2))
    (:deg2rad  , :(π / 180)                                       , :(0))
    (:rad2deg  , :(180 / π)                                       , :(0))
    # ( :erf   , :(2*exp(-x^2)/sqrt(π))                           , :(-4*x*exp(-x^2)/sqrt(π)))
    # ( :erfinv, :(1/2*sqrt(π)*exp(erfinv(x)^2))                  , :(1/2*π*erfinv(x)*exp(2*erfinv(x)^2)))
    # ( :erfc  , :(-2*exp(-x^2)/sqrt(π))                          , :(4*x*exp(-x^2)/sqrt(π)))
    # ( :erfi  , :(2*exp(x^2)/sqrt(π))                            , :(4*x*exp(x^2)/sqrt(π)))
]
# runic: on


changeprecision(x) = x
changeprecision(x::Integer) = :(T($x))
function changeprecision(x::Symbol)
    return x == :π ? :(T($x)) : x
end
function changeprecision(ex::Expr)
    if Meta.isexpr(ex, :call, 3) && ex.args[1] == :^ && ex.args[3] isa Int
        return Expr(:call, :^, changeprecision(ex.args[2]), ex.args[3])
    else
        return Expr(ex.head, changeprecision.(ex.args)...)
    end
end

for (i, rule) in enumerate(DIFF_RULES)
    prec_f′ = changeprecision(rule[2])
    prec_f′′ = changeprecision(rule[3])
    DIFF_RULES[i] = (rule[1], prec_f′, prec_f′′)
end

function rule_expr(f, f′, f′′)
    ex = quote
        # Verify that the cse still works properly when changing this.
        f = $f(x)
        f′ = $f′
        f′′ = $f′′
        x23 = (f′′ ⊙ h.ϵ1) ⊗ h.ϵ2
        return HyperDual(f, h.ϵ1 ⊙ f′, h.ϵ2 ⊙ f′, ntuple(i -> h.ϵ12[i] ⊙ f′ ⊕ x23[i], Val(N1)))
    end
    # Drop line number metadata so debug output is cleaner and CSE vars are shorter.
    return Base.remove_linenums!(ex)
end

@inline function Base.sin(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincos(h.v)
    f′, f′′ = c, -s
    x23 = (f′′ ⊙ h.ϵ1) ⊗ h.ϵ2
    return HyperDual(s, h.ϵ1 ⊙ f′, h.ϵ2 ⊙ f′, ntuple(i -> h.ϵ12[i] ⊙ f′ ⊕ x23[i], Val(N1)))
end

@inline function Base.cos(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincos(h.v)
    f′, f′′ = -s, -c
    x23 = (f′′ ⊙ h.ϵ1) ⊗ h.ϵ2
    return HyperDual(c, h.ϵ1 ⊙ f′, h.ϵ2 ⊙ f′, ntuple(i -> h.ϵ12[i] ⊙ f′ ⊕ x23[i], Val(N1)))
end

@inline function Base.sinpi(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincospi(h.v)
    f′, f′′ = π * c, -π^2 * s
    x23 = (f′′ ⊙ h.ϵ1) ⊗ h.ϵ2
    return HyperDual(s, h.ϵ1 ⊙ f′, h.ϵ2 ⊙ f′, ntuple(i -> h.ϵ12[i] ⊙ f′ ⊕ x23[i], Val(N1)))
end

@inline function Base.cospi(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincospi(h.v)
    f′, f′′ = -π * s, -π^2 * c
    x23 = (f′′ ⊙ h.ϵ1) ⊗ h.ϵ2
    return HyperDual(c, h.ϵ1 ⊙ f′, h.ϵ2 ⊙ f′, ntuple(i -> h.ϵ12[i] ⊙ f′ ⊕ x23[i], Val(N1)))
end


for (f, f′, f′′) in DIFF_RULES
    expr = rule_expr(f, f′, f′′)
    cse_expr = cse(expr; warn = false)
    @eval @inline function Base.$f(h::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = h.v
        $cse_expr
    end
end

"""
    debug_rule_cse(io=stdout; warn=false)

Print the rule template before and after common subexpression elimination for
each entry in `DIFF_RULES`. Useful when manually inspecting how the CSE pass
is transforming the generated code.
"""
function debug_rule_cse(io::IO = stdout; warn::Bool = false)
    for (f, f′, f′′) in DIFF_RULES
        expr = rule_expr(f, f′, f′′)
        println(io, "\n### ", f, " ###")
        println(io, "before CSE:\n", expr)
        println(io, "\nafter CSE:\n", cse(expr; warn))
    end
    return nothing
end

"""
    dump_rule_cse(dir="cse_dump"; warn=false)

Write one file per rule with the before/after CSE expressions for offline
inspection. Each file is named `<function>.jl` inside `dir`.
"""
function dump_rule_cse(dir::AbstractString = "cse_dump"; warn::Bool = false)
    mkpath(dir)
    for (f, f′, f′′) in DIFF_RULES
        expr = rule_expr(f, f′, f′′)
        cse_expr = cse(expr; warn)
        before_str = sprint(show, MIME"text/plain"(), expr)
        after_str = sprint(show, MIME"text/plain"(), cse_expr)
        open(joinpath(dir, string(f) * ".jl"), "w") do io
            println(io, "# before CSE")
            println(io, before_str)
            println(io, "\n# after CSE")
            println(io, after_str)
        end
    end
    return dir
end

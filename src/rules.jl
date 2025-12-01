# From HyperDualNumbers
# runic: off
const DIFF_RULES = [
    (:sqrt     , :(1 / 2 / f)                                     , :(-2 * f′^3))
    (:cbrt     , :(1 / 3 / f^2)                                   , :(-6 * f * f′^3))
    (:abs2     , :(2 * x)                                         , :(2))
    (:abs      , :(signbit(x) ? -one(x) : one(x))                 , :(0))
    (:inv      , :(-f^2)                                          , :(-2 * f′ * f))
    (:log      , :(inv(x))                                        , :(-f′^2))
    (:log10    , :(inv(x) / log(10))                              , :(-log(10) * f′^2))
    (:log2     , :(inv(x) / log(2))                               , :(-log(2) * f′^2))
    (:log1p    , :(inv(x + 1))                                    , :(-f′^2))
    (:exp      , :(f)                                             , :(f))
    (:exp2     , :(f * log(2))                                    , :(f′ * log(2)))
    (:exp10    , :(f * log(10))                                   , :(f′ * log(10)))
    (:expm1    , :(f + 1)                                         , :(f + 1))
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
    (:sinc     , :(cosc(x))                                       , :(iszero(x) ? -π^2 / 3 : -π^2 * f - 2 * f′ / x))
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

# runic: off
const BINARY_DIFF_RULES = [
    (
        :^,
        :(y * f / x),
        :(begin
            cond = x isa Real && x <= T(0)
            logx = cond ? zero(float(x)) : log(x)
            cond
            logx
            f * logx
        end),
        :(y * (y - one(y)) * f / (x * x)),
        :(cond ? Base.oftype(float(x), NaN) : (f / x) * (one(x) + y * logx)),
        :(cond ? Base.oftype(float(x), NaN) : f * logx^2),
    ),
    (
        :atan,
        :(y / (x^2 + y^2)),
        :(-x / (x^2 + y^2)),
        :(-2 * x * y / (x^2 + y^2)^2),
        :( (x^2 - y^2) / (x^2 + y^2)^2 ),
        :(2 * x * y / (x^2 + y^2)^2),
    ),
    (
        :hypot,
        :(x / f),
        :(y / f),
        :(y^2 / f^3),
        :(-x * y / f^3),
        :(x^2 / f^3),
    ),
    (
        :log,
        :( log(y) * inv(-log(x)^2 * x) ),
        :( inv(y) / log(x) ),
        :( log(y) * (log(x) + 2) / (x^2 * log(x)^3) ),
        :( -inv(x * y * log(x)^2) ),
        :( -inv(y^2 * log(x)) ),
    ),
]
# runic: on

for (i, rule) in enumerate(BINARY_DIFF_RULES)
    prec_fₓ = changeprecision(rule[2])
    prec_fᵧ = changeprecision(rule[3])
    prec_fₓₓ = changeprecision(rule[4])
    prec_fₓᵧ = changeprecision(rule[5])
    prec_fᵧᵧ = changeprecision(rule[6])
    BINARY_DIFF_RULES[i] = (rule[1], prec_fₓ, prec_fᵧ, prec_fₓₓ, prec_fₓᵧ, prec_fᵧᵧ)
end

function rule_expr(f, f′, f′′)
    ex = quote
        # Verify that the cse still works properly when changing this.
        f = $f(x)
        f′ = $f′
        f′′ = $f′′
    end
    # Drop line number metadata so debug output is cleaner and CSE vars are shorter.
    return Base.remove_linenums!(ex)
end

function binary_rule_expr(f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
    ex = quote
        f = $f(x, y)
        fₓ = $fₓ
        fᵧ = $fᵧ
        fₓₓ = $fₓₓ
        fₓᵧ = $fₓᵧ
        fᵧᵧ = $fᵧᵧ
    end
    return Base.remove_linenums!(ex)
end

"""
    chain_rule_dual(h::HyperDual, f, f′, f′′)

Apply chain rule to HyperDual `h` given primal `f`, first derivative `f′`, and second derivative `f′′`.
Returns a new HyperDual with properly propagated derivatives.
"""
@inline function chain_rule_dual(h::HyperDual{N1, N2}, f, f′, f′′) where {N1, N2}
    x23 = (f′′ ⊙ h.ϵ1) ⊗ h.ϵ2
    return HyperDual(f, h.ϵ1 ⊙ f′, h.ϵ2 ⊙ f′, ntuple(i -> _muladd(f′, h.ϵ12[i], x23[i]), Val(N1)))
end

"""
    chain_rule_dual(hx::HyperDual, hy::HyperDual, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)

Apply chain rule to a scalar function of two HyperDual inputs given first and
second partial derivatives.
"""
@inline function chain_rule_dual(
        hx::HyperDual{N1, N2},
        hy::HyperDual{N1, N2},
        f,
        fₓ,
        fᵧ,
        fₓₓ,
        fₓᵧ,
        fᵧᵧ,
    ) where {N1, N2}
    ϵ1 = _muladd(fₓ, hx.ϵ1, hy.ϵ1 ⊙ fᵧ)
    ϵ2 = _muladd(fₓ, hx.ϵ2, hy.ϵ2 ⊙ fᵧ)
    @inline g(i) = begin
        acc = _muladd(fₓ, hx.ϵ12[i], hy.ϵ12[i] ⊙ fᵧ)
        acc = _muladd(hx.ϵ1[i] * fₓₓ + hy.ϵ1[i] * fₓᵧ, hx.ϵ2, acc)
        acc = _muladd(hx.ϵ1[i] * fₓᵧ + hy.ϵ1[i] * fᵧᵧ, hy.ϵ2, acc)
        acc
    end
    return HyperDual(f, ϵ1, ϵ2, ntuple(g, Val(N1)))
end

@inline function Base.sin(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincos(h.v)
    return chain_rule_dual(h, s, c, -s)
end

@inline function Base.cos(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincos(h.v)
    return chain_rule_dual(h, c, -s, -c)
end

@inline function Base.sinpi(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincospi(h.v)
    return chain_rule_dual(h, s, π * c, -π^2 * s)
end

@inline function Base.cospi(h::HyperDual{N1, N2}) where {N1, N2}
    s, c = sincospi(h.v)
    return chain_rule_dual(h, c, -π * s, -π^2 * c)
end

for (f, f′, f′′) in DIFF_RULES
    expr = rule_expr(f, f′, f′′)
    cse_expr = cse(binarize(expr); warn = false)
    @eval @inline function Base.$f(h::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = h.v
        $cse_expr
        return chain_rule_dual(h, f, f′, f′′)
    end
end

for (f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ) in BINARY_DIFF_RULES
    expr = binary_rule_expr(f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
    cse_expr = cse(binarize(expr); warn = false)
    @eval @inline function Base.$f(hx::HyperDual{N1, N2, T}, hy::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = hx.v
        y = hy.v
        $cse_expr
        return chain_rule_dual(hx, hy, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
    end
    @eval @inline function Base.$f(hx::HyperDual{N1, N2, T1}, hy::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2}
        return Base.$f(promote(hx, hy)...)
    end
    @eval @inline function Base.$f(hx::HyperDual{N1, N2, T}, y_raw::Real) where {N1, N2, T}
        x = hx.v
        y = T(y_raw)
        $cse_expr
        return chain_rule_dual(hx, f, fₓ, fₓₓ)
    end
    @eval @inline function Base.$f(x_raw::Real, hy::HyperDual{N1, N2, T}) where {N1, N2, T}
        x = T(x_raw)
        y = hy.v
        $cse_expr
        return chain_rule_dual(hy, f, fᵧ, fᵧᵧ)
    end
end


"""
    normalize_cse_vars(expr)

Replace gensym-generated variable names (like `var"##718"`) with deterministic
names (`cse1`, `cse2`, ...) for reproducible output across Julia sessions.
"""
function normalize_cse_vars(expr)
    gensym_to_normalized = Dict{Symbol, Symbol}()
    counter = Ref(0)

    function is_gensym_var(s::Symbol)
        str = string(s)
        return startswith(str, "##") || startswith(str, "var\"##")
    end

    function get_normalized(s::Symbol)
        if haskey(gensym_to_normalized, s)
            return gensym_to_normalized[s]
        else
            counter[] += 1
            normalized = Symbol("cse", counter[])
            gensym_to_normalized[s] = normalized
            return normalized
        end
    end

    function normalize(x)
        return x
    end

    function normalize(s::Symbol)
        return is_gensym_var(s) ? get_normalized(s) : s
    end

    function normalize(ex::Expr)
        return Expr(ex.head, map(normalize, ex.args)...)
    end

    return normalize(expr)
end

"""
    dump_rule_cse(dir="cse_dump"; warn=false)

Write one file per rule with the before/after CSE expressions for offline
inspection. Each file is named `<function>.jl` inside `dir`. Handles both
unary (`DIFF_RULES`) and binary (`BINARY_DIFF_RULES`) rules.
"""
function dump_rule_cse(dir::AbstractString = "cse_dump"; warn::Bool = false)
    mkpath(dir)

    for (f, f′, f′′) in DIFF_RULES
        expr = rule_expr(f, f′, f′′)
        cse_expr = normalize_cse_vars(cse(binarize(expr); warn))
        before_str = sprint(show, MIME"text/plain"(), expr)
        after_str = sprint(show, MIME"text/plain"(), cse_expr)
        open(joinpath(dir, string(f) * ".jl"), "w") do io
            println(io, "# before CSE")
            println(io, before_str)
            println(io, "\n# after CSE")
            println(io, after_str)
        end
    end

    for (f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ) in BINARY_DIFF_RULES
        expr = binary_rule_expr(f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)
        cse_expr = normalize_cse_vars(cse(binarize(expr); warn))
        before_str = sprint(show, MIME"text/plain"(), expr)
        after_str = sprint(show, MIME"text/plain"(), cse_expr)
        open(joinpath(dir, string(f) * "_binary.jl"), "w") do io
            println(io, "# before CSE")
            println(io, before_str)
            println(io, "\n# after CSE")
            println(io, after_str)
        end
    end

    return dir
end

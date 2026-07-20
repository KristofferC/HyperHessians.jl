#=
# Jets: symmetric second-order numbers

A `Jet` is the specialization of `HyperDual{N, N}` to the case where ϵ₁ and ϵ₂
carry the same seeds — which is exactly full-vector mode (one evaluation with
identity seeds, chunk size == input length). In that case ϵ₁ == ϵ₂ at every
intermediate value and ϵ₁₂ is symmetric, so a HyperDual stores (and computes)
the gradient twice and both triangles of the Hessian block. A Jet stores each
once:

    j = v + gᵀε + εᵀHε,   H symmetric, upper triangle stored row-major

which is 1 + N + N(N+1)/2 floats instead of HyperDual{N,N}'s 1 + 2N + N².
The rules follow from the HyperDual rules with a = b = g and A = H:

    (j₁ * j₂).h[i,j] = v₁H₂[i,j] + H₁[i,j]v₂ + g₁[i]g₂[j] + g₂[i]g₁[j]
    f(j).h[i,j]      = f′(v)H[i,j] + f″(v)g[i]g[j]

Measured tradeoff (see benchmark/why-faster/exp9_jet.jl): jets win for small
inputs where register pressure dominates, but the ragged triangle rows
(N, N-1, …, 1) vectorize poorly as they grow — multiply-heavy functions
regress from N = 5, and at N = 8 a HyperDual{8,8}'s 8-wide lanes each fill
one register exactly. `HessianConfig` therefore uses jets for
`length(x) <= JET_VECTOR_MAX_N` and HyperDuals above that.
=#

struct Jet{N, M, T} <: Real
    v::T
    g::NTuple{N, T}
    h::NTuple{M, T}
end

const SecondOrderNumber = Union{HyperDual, Jet}

@inline nupper(N::Int) = N * (N + 1) ÷ 2

Jet{N, M}(v::T) where {N, M, T} =
    Jet{N, M, T}(v, ntuple(_ -> zero(T), Val(N)), ntuple(_ -> zero(T), Val(M)))
Jet{N, M, T}(v::Real) where {N, M, T} = Jet{N, M}(T(v))
Jet{N, M, T}(v::Jet{N, M, T}) where {N, M, T} = v
Jet{N, M, T}(v::Jet{N, M}) where {N, M, T} = convert(Jet{N, M, T}, v)
Jet{N, M}(v::Jet{N, M}) where {N, M} = v

# Disambiguate against Base's numeric conversion constructors, mirroring
# HyperDual (see hyperdual.jl).
for R in (:Complex, :AbstractChar, :(Base.TwicePrecision))
    @eval Jet{N, M, T}(v::$R) where {N, M, T} = Jet{N, M}(T(_scalar(v)))
    @eval Jet{N, M}(v::$R) where {N, M} = Jet{N, M}(_scalar(v))
end

@inline value(x::Jet) = x.v

Base.promote_rule(::Type{Jet{N, M, T1}}, ::Type{Jet{N, M, T2}}) where {N, M, T1, T2} =
    Jet{N, M, promote_type(T1, T2)}
Base.promote_rule(::Type{Jet{N, M, T1}}, ::Type{T2}) where {N, M, T1, T2 <: Real} =
    Jet{N, M, promote_type(T1, T2)}
Base.convert(::Type{Jet{N, M, T1}}, j::Jet{N, M, T2}) where {N, M, T1, T2} =
    Jet{N, M, T1}(T1(j.v), to_ϵ(ϵT{N, T1}, j.g), to_ϵ(ϵT{M, T1}, j.h))
Base.convert(::Type{Jet{N, M, T}}, x::Real) where {N, M, T} = Jet{N, M, T}(T(x))

function Base.show(io::IO, j::Jet)
    print(io, j.v, " + ", Tuple(j.g), "ε + ", Tuple(j.h), "ε² (upper)")
    return
end

Base.one(::Type{Jet{N, M, T}}) where {N, M, T} = Jet{N, M}(one(T))
Base.zero(::Type{Jet{N, M, T}}) where {N, M, T} = Jet{N, M}(zero(T))
Base.one(::Jet{N, M, T}) where {N, M, T} = one(Jet{N, M, T})
Base.zero(::Jet{N, M, T}) where {N, M, T} = zero(Jet{N, M, T})
Base.float(j::Jet{N, M, T}) where {N, M, T} = convert(Jet{N, M, float(T)}, j)

# Upper-triangle helpers, fully unrolled with literal indices so LLVM sees
# straight-line tuple code.
@inline @generated function symouter(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T}
    ex = Expr(:tuple)
    for i in 1:N, j in i:N
        push!(ex.args, :(muladd(a[$i], b[$j], b[$i] * a[$j])))
    end
    return ex
end

@inline @generated function halfouter(a::NTuple{N, T}) where {N, T}
    ex = Expr(:tuple)
    for i in 1:N, j in i:N
        push!(ex.args, :(a[$i] * a[$j]))
    end
    return ex
end

"""
    chain_rule_jet(j::Jet, f, f′, f′′)

Apply the chain rule to `j` given scalar primal `f`, first derivative `f′`,
and second derivative `f′′`.
"""
@inline function chain_rule_jet(j::Jet{N, M, T}, f, f′, f′′) where {N, M, T}
    return Jet(f, f′ .* j.g, muladd.(f′, j.h, f′′ .* halfouter(j.g)))
end

"""
    chain_rule_jet(jx::Jet, jy::Jet, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)

Apply the chain rule to a scalar function of two `Jet` inputs given first and
second partial derivatives.
"""
@inline function chain_rule_jet(
        jx::Jet{N, M, T},
        jy::Jet{N, M, T},
        f,
        fₓ,
        fᵧ,
        fₓₓ,
        fₓᵧ,
        fᵧᵧ,
    ) where {N, M, T}
    g = muladd.(fₓ, jx.g, fᵧ .* jy.g)
    h = muladd.(fₓ, jx.h, fᵧ .* jy.h)
    h = muladd.(fₓₓ, halfouter(jx.g), h)
    h = muladd.(fₓᵧ, symouter(jx.g, jy.g), h)
    h = muladd.(fᵧᵧ, halfouter(jy.g), h)
    return Jet(f, g, h)
end

# Arithmetic
@inline Base.:+(j::Jet) = j
@inline Base.:-(j::Jet{N, M, T}) where {N, M, T} = Jet{N, M, T}(-j.v, .-j.g, .-j.h)

@inline Base.:+(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T} =
    Jet{N, M, T}(a.v + b.v, a.g .+ b.g, a.h .+ b.h)
@inline Base.:+(a::Jet{N, M, T1}, b::Jet{N, M, T2}) where {N, M, T1, T2} = +(promote(a, b)...)
@inline Base.:+(j::Jet{N, M}, r::Real) where {N, M} = Jet(j.v + r, j.g, j.h)
@inline Base.:+(r::Real, j::Jet{N, M}) where {N, M} = Jet(r + j.v, j.g, j.h)

@inline Base.:-(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T} =
    Jet{N, M, T}(a.v - b.v, a.g .- b.g, a.h .- b.h)
@inline Base.:-(a::Jet{N, M, T1}, b::Jet{N, M, T2}) where {N, M, T1, T2} = -(promote(a, b)...)
@inline Base.:-(j::Jet{N, M}, r::Real) where {N, M} = Jet(j.v - r, j.g, j.h)
@inline Base.:-(r::Real, j::Jet{N, M}) where {N, M} = Jet(r - j.v, .-j.g, .-j.h)

@inline function Base.:*(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T}
    return Jet{N, M, T}(
        a.v * b.v,
        muladd.(a.v, b.g, b.v .* a.g),
        muladd.(a.v, b.h, muladd.(b.v, a.h, symouter(a.g, b.g))),
    )
end
@inline Base.:*(a::Jet{N, M, T1}, b::Jet{N, M, T2}) where {N, M, T1, T2} = *(promote(a, b)...)
@inline Base.:*(j::Jet{N, M}, r::Real) where {N, M} = Jet(j.v * r, j.g .* r, j.h .* r)
@inline Base.:*(r::Real, j::Jet{N, M}) where {N, M} = Jet(r * j.v, r .* j.g, r .* j.h)

# Dedicated division rule, mirroring _div_hyperdual:
# f = x/y, fₓ = 1/y, fᵧ = -f/y, fₓₓ = 0, fₓᵧ = -1/y², fᵧᵧ = 2f/y²
@inline function Base.:/(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T}
    x, y = a.v, b.v
    invy = inv(y)
    f = x * invy
    fᵧ = -f * invy
    fₓᵧ = -invy * invy
    fᵧᵧ = -2 * fᵧ * invy
    return chain_rule_jet(a, b, f, invy, fᵧ, zero(invy), fₓᵧ, fᵧᵧ)
end
@inline Base.:/(a::Jet{N, M, T1}, b::Jet{N, M, T2}) where {N, M, T1, T2} = /(promote(a, b)...)
@inline Base.:/(j::Jet{N, M}, r::Real) where {N, M} = Jet(j.v / r, j.g ./ r, j.h ./ r)
@inline Base.:/(r::Real, j::Jet) = r * inv(j)

# Integer powers, mirroring _pow_hyperdual: monomial derivatives directly
# (NaN-free at x = 0, cheaper than repeated multiplication), and resolving the
# ambiguity with Base.^(::Number, ::Integer).
@inline function _pow_jet(j::Jet, n::Integer)
    x = j.v
    if n == 0
        return one(j)
    elseif n == 1
        return j
    elseif n == 2
        return chain_rule_jet(j, x * x, 2 * x, 2 * one(x))
    elseif n == 3
        x2 = x * x
        return chain_rule_jet(j, x2 * x, 3 * x2, 6 * x)
    else
        p = x^(n - 2)
        xp = x * p
        f = x * xp
        f′ = n * xp
        f′′ = n * (n - 1) * p
        return chain_rule_jet(j, f, f′, f′′)
    end
end
@inline Base.:(^)(j::Jet, n::Integer) = _pow_jet(j, n)
@inline Base.:(^)(j::Jet, r::Rational) = j^float(r)
@inline Base.literal_pow(::typeof(^), j::Jet, ::Val{0}) = one(typeof(j))
@inline Base.literal_pow(::typeof(^), j::Jet, ::Val{1}) = j
@inline Base.literal_pow(::typeof(^), j::Jet, ::Val{p}) where {p} = j^p

@inline function Base.muladd(x::Jet{N, M, T}, y::Real, z::Jet{N, M, T}) where {N, M, T}
    return Jet{N, M, T}(
        muladd(x.v, y, z.v),
        muladd.(y, x.g, z.g),
        muladd.(y, x.h, z.h),
    )
end
@inline function Base.muladd(x::Real, y::Jet{N, M, T}, z::Jet{N, M, T}) where {N, M, T}
    return muladd(y, x, z)
end
@inline Base.muladd(x::Jet{N, M, T}, y::Real, z::Real) where {N, M, T} =
    Jet(muladd(x.v, y, z), x.g .* y, x.h .* y)
@inline Base.muladd(x::Real, y::Jet{N, M, T}, z::Real) where {N, M, T} = muladd(y, x, z)
@inline Base.muladd(x::Real, y::Real, z::Jet{N, M, T}) where {N, M, T} =
    Jet(muladd(x, y, z.v), z.g, z.h)
@inline Base.muladd(x::Jet{N, M, T}, y::Jet{N, M, T}, z::Jet{N, M, T}) where {N, M, T} =
    x * y + z
@inline Base.muladd(x::Jet{N, M, T}, y::Jet{N, M, T}, z::Real) where {N, M, T} =
    x * y + z

# Comparisons and predicates act on the primal value (HyperDual semantics).
@inline Base.isless(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = isless(a.v, b.v)
@inline Base.isless(j::Jet, r::Real) = isless(j.v, r)
@inline Base.isless(r::Real, j::Jet) = isless(r, j.v)
@inline Base.isless(j::Jet, r::AbstractFloat) = isless(j.v, r)
@inline Base.isless(r::AbstractFloat, j::Jet) = isless(r, j.v)
@inline Base.:<(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a.v < b.v
@inline Base.:<(j::Jet, r::Real) = j.v < r
@inline Base.:<(r::Real, j::Jet) = r < j.v
@inline Base.:<=(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a.v <= b.v
@inline Base.:<=(j::Jet, r::Real) = j.v <= r
@inline Base.:<=(r::Real, j::Jet) = r <= j.v
@inline Base.:(==)(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a.v == b.v
@inline Base.:(==)(j::Jet, r::Real) = j.v == r
@inline Base.:(==)(r::Real, j::Jet) = r == j.v
@inline Base.:(==)(j::Jet, r::AbstractIrrational) = j.v == r
@inline Base.:(==)(r::AbstractIrrational, j::Jet) = r == j.v
Base.hash(j::Jet, u::UInt) = hash(j.v, u)

for f in (:isnan, :isinf, :isfinite, :signbit, :isinteger, :iseven, :isodd)
    @eval @inline Base.$f(j::Jet) = $f(j.v)
end

# Rounding: derivative is zero almost everywhere.
for f in (:floor, :ceil, :trunc)
    @eval @inline Base.$f(j::Jet{N, M}) where {N, M} = Jet{N, M}($f(j.v))
    @eval @inline Base.$f(::Type{I}, j::Jet) where {I <: Real} = $f(I, j.v)
end
@inline Base.round(j::Jet{N, M}, r::RoundingMode = RoundNearest) where {N, M} =
    Jet{N, M}(round(j.v, r))
@inline Base.round(::Type{I}, j::Jet) where {I <: Real} = round(I, j.v)

# mod/rem: derivative w.r.t. x is 1 almost everywhere.
@inline Base.mod(j::Jet{N, M}, r::Real) where {N, M} = Jet(mod(j.v, r), j.g, j.h)
@inline Base.rem(j::Jet{N, M}, r::Real) where {N, M} = Jet(rem(j.v, r), j.g, j.h)
@inline Base.mod(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a - fld(a.v, b.v) * b
@inline Base.rem(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a - div(a.v, b.v) * b
@inline Base.mod2pi(j::Jet{N, M}) where {N, M} = Jet(mod2pi(j.v), j.g, j.h)
@inline Base.rem2pi(j::Jet{N, M}, r::RoundingMode) where {N, M} =
    Jet(rem2pi(j.v, r), j.g, j.h)

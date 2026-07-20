#=
# HyperDual Numbers

A HyperDual number extends dual numbers to compute second derivatives (Hessians).
A HyperDual is defined as:

    h = v + ϵ₁ᵀa + ϵ₂ᵀb + ϵ₁ᵀAϵ₂

where:
- v is the primal value
- ϵ₁, ϵ₂ are independent infinitesimals with ϵ₁² = ϵ₂² = 0
- a, b are vectors (first derivative components)
- A is a matrix (second derivative / Hessian component)
- The cross term ϵ₁ᵢϵ₂ⱼ ≠ 0 captures mixed partials

## Multiplication rule

For h₁ = v₁ + ϵ₁ᵀa₁ + ϵ₂ᵀb₁ + ϵ₁ᵀA₁ϵ₂ and h₂ = v₂ + ϵ₁ᵀa₂ + ϵ₂ᵀb₂ + ϵ₁ᵀA₂ϵ₂:

    h₁ * h₂ = v₁v₂
            + ϵ₁ᵀ(v₁a₂ + a₁v₂)
            + ϵ₂ᵀ(v₁b₂ + b₁v₂)
            + ϵ₁ᵀ(v₁A₂ + A₁v₂ + a₁b₂ᵀ + a₂b₁ᵀ)ϵ₂

This follows from the product rule and ϵ² = 0.

## Chain rule for f(h)

For a scalar function f applied to h = v + ϵ₁ᵀa + ϵ₂ᵀb + ϵ₁ᵀAϵ₂:

    f(h) = f(v) + ϵ₁ᵀ(f'(v)a) + ϵ₂ᵀ(f'(v)b) + ϵ₁ᵀ(f'(v)A + f''(v)abᵀ)ϵ₂

This gives us the first and second derivatives via f' and f''.
=#

# Allow non-square partial lengths: ϵ₁ ∈ ℝᴺ¹, ϵ₂ ∈ ℝᴺ².
const ϵT{N, T} = NTuple{N, T}

# Tuple implementations (default)
@inline zero_ϵ(::Type{NTuple{N, T}}) where {N, T} = ntuple(_ -> zero(T), Val(N))
@inline zero_ϵ(x::NTuple{N, T}) where {N, T} = zero_ϵ(NTuple{N, T})
@inline to_ϵ(::Type{NTuple{N, T}}, x) where {N, T} = convert(NTuple{N, T}, x)
@inline convert_cross(::Type{NTuple{N, T}}, xs::NTuple{M, Any}) where {N, M, T} =
    ntuple(i -> to_ϵ(NTuple{N, T}, xs[i]), Val(M))

@inline ⊕(a::Real, b::Real) = a + b
@inline ⊕(a::NTuple{N, A}, b::NTuple{N, B}) where {N, A, B} = ntuple(i -> ⊕(a[i], b[i]), Val(N))
@inline ⊟(a::Real) = -a
@inline ⊟(a::NTuple{N, A}) where {N, A} = ntuple(i -> ⊟(a[i]), Val(N))
@inline ⊖(a, b) = ⊕(a, ⊟(b))
@inline ⊙(a::Real, r::Real) = a * r
@inline ⊙(a::NTuple{N, A}, r::Real) where {N, A} = ntuple(i -> ⊙(a[i], r), Val(N))
@inline ⊙(r::Real, a::NTuple{N, A}) where {N, A} = ntuple(i -> ⊙(r, a[i]), Val(N))
@inline ⊘(a::Real, r::Real) = a / r
@inline ⊘(a::NTuple{N, A}, r::Real) where {N, A} = ntuple(i -> ⊘(a[i], r), Val(N))
@inline _muladd(a::Real, b::NTuple{N, A}, c::NTuple{N, C}) where {N, A, C} =
    ntuple(i -> muladd(a, b[i], c[i]), Val(N))
@inline _muladd(a::NTuple{N, A}, b::Real, c::NTuple{N, C}) where {N, A, C} =
    ntuple(i -> muladd(a[i], b, c[i]), Val(N))
@inline ⊗(t1::NTuple{N1, T1}, t2::NTuple{N2, T2}) where {N1, N2, T1, T2} = ntuple(i -> ⊙(t2, t1[i]), Val(N1))

struct HyperDual{N1, N2, T} <: Real
    v::T
    ϵ1::ϵT{N1, T}
    ϵ2::ϵT{N2, T}
    ϵ12::NTuple{N1, ϵT{N2, T}}
end
HyperDual(v::T, ϵ1::ϵT{N1, T}, ϵ2::ϵT{N2, T}) where {N1, N2, T} =
    HyperDual(v, ϵ1, ϵ2, ntuple(_ -> ntuple(_ -> zero(T), Val(N2)), Val(N1)))
HyperDual{N1, N2}(v::T) where {N1, N2, T} =
    HyperDual(v, ntuple(_ -> zero(T), Val(N1)), ntuple(_ -> zero(T), Val(N2)))
HyperDual{N1, N2, T}(v) where {N1, N2, T} = HyperDual{N1, N2}(T(v))
HyperDual{N1, N2, T}(v::HyperDual{N1, N2, T}) where {N1, N2, T} = v
HyperDual{N1, N2, T}(v::HyperDual{N1, N2}) where {N1, N2, T} = convert(HyperDual{N1, N2, T}, v)

function HyperDual(v::T1, ϵ1::ϵT{N1, T2}, ϵ2::ϵT{N2, T2}, ϵ12::NTuple{N1, ϵT{N2, T2}}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual(T(v), to_ϵ(ϵT{N1, T}, ϵ1), to_ϵ(ϵT{N2, T}, ϵ2), convert_cross(ϵT{N2, T}, ϵ12))
end

# Accessor Functions
@inline value(x) = x
@inline value(x::HyperDual) = x.v

@inline mapϵ12(f, h::HyperDual{N1, N2}) where {N1, N2} = ntuple(i -> f(h.ϵ12[i]), Val(N1))
@inline mapϵ12(f, h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} =
    ntuple(i -> f(h1.ϵ12[i], h2.ϵ12[i]), Val(N1))

Base.promote_rule(::Type{HyperDual{N1, N2, T1}}, ::Type{HyperDual{N1, N2, T2}}) where {N1, N2, T1, T2} =
    HyperDual{N1, N2, promote_type(T1, T2)}
Base.promote_rule(::Type{HyperDual{N1, N2, T1}}, ::Type{T2}) where {N1, N2, T1, T2 <: Real} =
    HyperDual{N1, N2, promote_type(T1, T2)}
Base.convert(::Type{HyperDual{N1, N2, T1}}, h::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} =
    HyperDual{N1, N2, T1}(T1(h.v), to_ϵ(ϵT{N1, T1}, h.ϵ1), to_ϵ(ϵT{N2, T1}, h.ϵ2), convert_cross(ϵT{N2, T1}, h.ϵ12))
Base.convert(::Type{HyperDual{N1, N2, T}}, x::Real) where {N1, N2, T} = HyperDual{N1, N2, T}(T(x))

function Base.show(io::IO, h::HyperDual)
    print(io, h.v, " + ", Tuple(h.ϵ1), "ϵ1", " + ", Tuple(h.ϵ2), "ϵ2", " + ", map(Tuple, h.ϵ12), "ϵ12")
    return
end

Base.one(::Type{HyperDual{N1, N2, T}}) where {N1, N2, T} = HyperDual{N1, N2}(one(T))
Base.zero(::Type{HyperDual{N1, N2, T}}) where {N1, N2, T} = HyperDual{N1, N2}(zero(T))
Base.one(::HyperDual{N1, N2, T}) where {N1, N2, T} = one(HyperDual{N1, N2, T})
Base.zero(::HyperDual{N1, N2, T}) where {N1, N2, T} = zero(HyperDual{N1, N2, T})
Base.float(h::HyperDual{N1, N2, T}) where {N1, N2, T} = convert(HyperDual{N1, N2, float(T)}, h)

@inline Base.:(-)(h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(-h.v, ⊟(h.ϵ1), ⊟(h.ϵ2), mapϵ12(⊟, h))
@inline Base.:(+)(h::HyperDual) = h

@inline Base.:+(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    HyperDual(h1.v + h2.v, h1.ϵ1 ⊕ h2.ϵ1, h1.ϵ2 ⊕ h2.ϵ2, mapϵ12(⊕, h1, h2))
@inline Base.:+(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = +(promote(h1, h2)...)
@inline Base.:+(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v + r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:+(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r + h.v, h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:-(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    HyperDual(h1.v - h2.v, h1.ϵ1 ⊖ h2.ϵ1, h1.ϵ2 ⊖ h2.ϵ2, mapϵ12(⊖, h1, h2))
@inline Base.:-(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = -(promote(h1, h2)...)
@inline Base.:-(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v - r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:-(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r - h.v, ⊟(h.ϵ1), ⊟(h.ϵ2), mapϵ12(⊟, h))

@inline Base.:*(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v * r, h.ϵ1 ⊙ r, h.ϵ2 ⊙ r, mapϵ12(ϵ -> ϵ ⊙ r, h))
@inline Base.:/(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v / r, h.ϵ1 ⊘ r, h.ϵ2 ⊘ r, mapϵ12(ϵ -> ϵ ⊘ r, h))
@inline Base.:(*)(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r * h.v, r ⊙ h.ϵ1, r ⊙ h.ϵ2, mapϵ12(ϵ -> r ⊙ ϵ, h))

@inline Base.:(/)(r::Real, h::HyperDual{N1, N2}) where {N1, N2} = r * inv(h)
# Dedicated division rule: cheaper than h1 * inv(h2) (fewer chain-rule products).
# f = x/y, fₓ = 1/y, fᵧ = -f/y, fₓₓ = 0, fₓᵧ = -1/y², fᵧᵧ = 2f/y²
@inline function Base.:(/)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T}
    x, y = h1.v, h2.v
    invy = inv(y)
    f = x * invy
    fᵧ = -f * invy
    return chain_rule_dual(h1, h2, f, invy, fᵧ, zero(invy), -invy * invy, -2 * fᵧ * invy)
end
@inline Base.:(/)(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = /(promote(h1, h2)...)

@inline function Base.muladd(x::HyperDual{N1, N2, T}, y::Real, z::HyperDual{N1, N2, T}) where {N1, N2, T}
    return HyperDual(
        muladd(x.v, y, z.v),
        _muladd(y, x.ϵ1, z.ϵ1),
        _muladd(y, x.ϵ2, z.ϵ2),
        ntuple(i -> _muladd(y, x.ϵ12[i], z.ϵ12[i]), Val(N1))
    )
end
@inline function Base.muladd(x::Real, y::HyperDual{N1, N2, T}, z::HyperDual{N1, N2, T}) where {N1, N2, T}
    return HyperDual(
        muladd(x, y.v, z.v),
        _muladd(x, y.ϵ1, z.ϵ1),
        _muladd(x, y.ϵ2, z.ϵ2),
        ntuple(i -> _muladd(x, y.ϵ12[i], z.ϵ12[i]), Val(N1)),
    )
end
@inline function Base.muladd(x::HyperDual{N1, N2, T}, y::Real, z::Real) where {N1, N2, T}
    return HyperDual(
        muladd(x.v, y, z),
        x.ϵ1 ⊙ y,
        x.ϵ2 ⊙ y,
        ntuple(i -> x.ϵ12[i] ⊙ y, Val(N1)),
    )
end
@inline function Base.muladd(x::Real, y::HyperDual{N1, N2, T}, z::Real) where {N1, N2, T}
    return HyperDual(
        muladd(x, y.v, z),
        y.ϵ1 ⊙ x,
        y.ϵ2 ⊙ x,
        ntuple(i -> y.ϵ12[i] ⊙ x, Val(N1)),
    )
end
@inline Base.muladd(x::Real, y::Real, z::HyperDual{N1, N2, T}) where {N1, N2, T} =
    HyperDual(muladd(x, y, z.v), z.ϵ1, z.ϵ2, z.ϵ12)

# Comparisons and predicates act on the primal value (ForwardDiff semantics):
# derivative components are ignored so branching code sees plain numbers.
@inline Base.isless(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = isless(h1.v, h2.v)
@inline Base.isless(h::HyperDual, r::Real) = isless(h.v, r)
@inline Base.isless(r::Real, h::HyperDual) = isless(r, h.v)
# Disambiguate against Base.isless(::Real, ::AbstractFloat) and its mirror.
@inline Base.isless(h::HyperDual, r::AbstractFloat) = isless(h.v, r)
@inline Base.isless(r::AbstractFloat, h::HyperDual) = isless(r, h.v)
@inline Base.:<(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1.v < h2.v
@inline Base.:<(h::HyperDual, r::Real) = h.v < r
@inline Base.:<(r::Real, h::HyperDual) = r < h.v
@inline Base.:<=(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1.v <= h2.v
@inline Base.:<=(h::HyperDual, r::Real) = h.v <= r
@inline Base.:<=(r::Real, h::HyperDual) = r <= h.v
@inline Base.:(==)(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1.v == h2.v
@inline Base.:(==)(h::HyperDual, r::Real) = h.v == r
@inline Base.:(==)(r::Real, h::HyperDual) = r == h.v
Base.hash(h::HyperDual, u::UInt) = hash(h.v, u)

for f in (:isnan, :isinf, :isfinite, :signbit, :isinteger, :iseven, :isodd)
    @eval @inline Base.$f(h::HyperDual) = $f(h.v)
end

# Rounding: derivative is zero almost everywhere.
for f in (:floor, :ceil, :trunc)
    @eval @inline Base.$f(h::HyperDual{N1, N2}) where {N1, N2} = HyperDual{N1, N2}($f(h.v))
    @eval @inline Base.$f(::Type{I}, h::HyperDual) where {I <: Real} = $f(I, h.v)
end
@inline Base.round(h::HyperDual{N1, N2}, r::RoundingMode = RoundNearest) where {N1, N2} =
    HyperDual{N1, N2}(round(h.v, r))
@inline Base.round(::Type{I}, h::HyperDual) where {I <: Real} = round(I, h.v)

# mod/rem: derivative w.r.t. x is 1 almost everywhere, so ϵ parts pass through.
# The two-dual methods use mod(x, y) = x - fld(x, y) * y with fld constant a.e.
@inline Base.mod(h::HyperDual{N1, N2}, r::Real) where {N1, N2} = HyperDual(mod(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.rem(h::HyperDual{N1, N2}, r::Real) where {N1, N2} = HyperDual(rem(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.mod(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1 - fld(h1.v, h2.v) * h2
@inline Base.rem(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1 - div(h1.v, h2.v) * h2
@inline Base.mod2pi(h::HyperDual{N1, N2}) where {N1, N2} = HyperDual(mod2pi(h.v), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.rem2pi(h::HyperDual{N1, N2}, r::RoundingMode) where {N1, N2} =
    HyperDual(rem2pi(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:(*)(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = *(promote(h1, h2)...)
@inline function Base.:(*)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T}
    r = h1.v * h2.v
    ϵ1 = _muladd(h1.v, h2.ϵ1, h1.ϵ1 ⊙ h2.v)
    ϵ2 = _muladd(h1.v, h2.ϵ2, h1.ϵ2 ⊙ h2.v)
    # Inline outer products with FMA: h1.ϵ1[i]*h2.ϵ2 + h2.ϵ1[i]*h1.ϵ2
    @inline g(i) = _muladd(h1.v, h2.ϵ12[i], _muladd(h1.ϵ12[i], h2.v, _muladd(h1.ϵ1[i], h2.ϵ2, h2.ϵ1[i] ⊙ h1.ϵ2)))
    ϵ12 = ntuple(g, Val(N1))
    return HyperDual(r, ϵ1, ϵ2, ϵ12)
end
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{p}) where {p} = x^p

# Integer powers use the monomial derivatives n*x^(n-1) and n*(n-1)*x^(n-2)
# directly (the generic `^` rule divides by x and returns NaN at x = 0).
# This is also cheaper than repeated HyperDual multiplication.
# Also resolves the ambiguity with Base.^(::Number, ::Integer).
@inline function Base.:(^)(h::HyperDual{N1, N2, T}, n::Integer) where {N1, N2, T}
    x = h.v
    if n == 0
        return one(h)
    elseif n == 1
        return h
    elseif n == 2
        # Small exponents get explicit branches so that literal_pow constant-folds
        # them down to plain multiplications (no runtime pow call).
        return chain_rule_dual(h, x * x, 2 * x, 2 * one(x))
    elseif n == 3
        x2 = x * x
        return chain_rule_dual(h, x2 * x, 3 * x2, 6 * x)
    else
        p = x^(n - 2)
        xp = x * p
        f = x * xp
        f′ = n * xp
        f′′ = (n * (n - 1)) * p
        return chain_rule_dual(h, f, f′, f′′)
    end
end

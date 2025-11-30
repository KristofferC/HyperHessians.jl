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
abstract type AbstractHyperDualNumber{N1, N2, T} <: Real end

# Tuple-based HyperDual (default)
struct HyperDual{N1, N2, T} <: AbstractHyperDualNumber{N1, N2, T}
    v::T
    ϵ1::NTuple{N1, T}
    ϵ2::NTuple{N2, T}
    ϵ12::NTuple{N1, NTuple{N2, T}}
end

# SIMD-based HyperDual
struct HyperDualSIMD{N1, N2, T} <: AbstractHyperDualNumber{N1, N2, T}
    v::T
    ϵ1::Vec{N1, T}
    ϵ2::Vec{N2, T}
    ϵ12::NTuple{N1, Vec{N2, T}}
end

# Tuple implementations (default)
@inline zero_ϵ(::Type{NTuple{N, T}}) where {N, T} = ntuple(_ -> zero(T), Val(N))
@inline zero_ϵ(x::NTuple{N, T}) where {N, T} = zero_ϵ(NTuple{N, T})
@inline to_ϵ(::Type{NTuple{N, T}}, x) where {N, T} = convert(NTuple{N, T}, x)
@inline convert_cross(::Type{NTuple{N, T}}, xs::NTuple{M, Any}) where {N, M, T} =
    ntuple(i -> to_ϵ(NTuple{N, T}, xs[i]), Val(M))

# Tuple operators
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

# Helper constructors for creating partial types
struct ϵT{N, T} end
(::Type{ϵT{N, T}})(x::NTuple{N}) where {N, T} = NTuple{N, T}(x)
(::Type{ϵT{N, T}})(x::Vec{N}) where {N, T} = Vec(NTuple{N, T}(Tuple(x)))

struct ϵT_SIMD{N, T} end
(::Type{ϵT_SIMD{N, T}})(x::NTuple{N}) where {N, T} = Vec(NTuple{N, T}(x))
(::Type{ϵT_SIMD{N, T}})(x::Vec{N}) where {N, T} = Vec(NTuple{N, T}(Tuple(x)))

# HyperDual constructors (tuple-based)
function HyperDual(v::T1, ϵ1::NTuple{N1, T2}, ϵ2::NTuple{N2, T2}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual{N1, N2, T}(T(v), to_ϵ(NTuple{N1, T}, ϵ1), to_ϵ(NTuple{N2, T}, ϵ2), ntuple(_ -> zero_ϵ(NTuple{N2, T}), Val(N1)))
end
HyperDual{N1, N2}(v::T) where {N1, N2, T} = HyperDual{N1, N2, T}(T(v), zero_ϵ(NTuple{N1, T}), zero_ϵ(NTuple{N2, T}), ntuple(_ -> zero_ϵ(NTuple{N2, T}), Val(N1)))
HyperDual{N1, N2, T}(v) where {N1, N2, T} = HyperDual{N1, N2}(T(v))

function HyperDual(v::T1, ϵ1::NTuple{N1, T2}, ϵ2::NTuple{N2, T2}, ϵ12::NTuple{N1, NTuple{N2, T2}}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual{N1, N2, T}(T(v), to_ϵ(NTuple{N1, T}, ϵ1), to_ϵ(NTuple{N2, T}, ϵ2), convert_cross(NTuple{N2, T}, ϵ12))
end

# HyperDualSIMD constructors
function HyperDualSIMD(v::T1, ϵ1::Vec{N1, T2}, ϵ2::Vec{N2, T2}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDualSIMD{N1, N2, T}(T(v), to_ϵ(Vec{N1, T}, ϵ1), to_ϵ(Vec{N2, T}, ϵ2), ntuple(_ -> zero_ϵ(Vec{N2, T}), Val(N1)))
end
HyperDualSIMD{N1, N2}(v::T) where {N1, N2, T} = HyperDualSIMD{N1, N2, T}(T(v), zero_ϵ(Vec{N1, T}), zero_ϵ(Vec{N2, T}), ntuple(_ -> zero_ϵ(Vec{N2, T}), Val(N1)))
HyperDualSIMD{N1, N2, T}(v) where {N1, N2, T} = HyperDualSIMD{N1, N2}(T(v))

function HyperDualSIMD(v::T1, ϵ1::Vec{N1, T2}, ϵ2::Vec{N2, T2}, ϵ12::NTuple{N1, Vec{N2, T2}}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDualSIMD{N1, N2, T}(T(v), to_ϵ(Vec{N1, T}, ϵ1), to_ϵ(Vec{N2, T}, ϵ2), convert_cross(Vec{N2, T}, ϵ12))
end

# Common methods via abstract type
@inline mapϵ12(f, h::AbstractHyperDualNumber{N1, N2, T}) where {N1, N2, T} = ntuple(i -> f(h.ϵ12[i]), Val(N1))
@inline mapϵ12(f, h1::H, h2::H) where {N1, N2, T, H <: AbstractHyperDualNumber{N1, N2, T}} =
    ntuple(i -> f(h1.ϵ12[i], h2.ϵ12[i]), Val(N1))

# HyperDual promote/convert
Base.promote_rule(::Type{HyperDual{N1, N2, T1}}, ::Type{HyperDual{N1, N2, T2}}) where {N1, N2, T1, T2} =
    HyperDual{N1, N2, promote_type(T1, T2)}
Base.promote_rule(::Type{HyperDual{N1, N2, T1}}, ::Type{T2}) where {N1, N2, T1, T2 <: Real} =
    HyperDual{N1, N2, promote_type(T1, T2)}
Base.convert(::Type{HyperDual{N1, N2, T1}}, h::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} =
    HyperDual{N1, N2, T1}(T1(h.v), to_ϵ(NTuple{N1, T1}, h.ϵ1), to_ϵ(NTuple{N2, T1}, h.ϵ2), convert_cross(NTuple{N2, T1}, h.ϵ12))
Base.convert(::Type{HyperDual{N1, N2, T}}, x::Real) where {N1, N2, T} = HyperDual{N1, N2, T}(T(x))

# HyperDualSIMD promote/convert
Base.promote_rule(::Type{HyperDualSIMD{N1, N2, T1}}, ::Type{HyperDualSIMD{N1, N2, T2}}) where {N1, N2, T1, T2} =
    HyperDualSIMD{N1, N2, promote_type(T1, T2)}
Base.promote_rule(::Type{HyperDualSIMD{N1, N2, T1}}, ::Type{T2}) where {N1, N2, T1, T2 <: Real} =
    HyperDualSIMD{N1, N2, promote_type(T1, T2)}
Base.convert(::Type{HyperDualSIMD{N1, N2, T1}}, h::HyperDualSIMD{N1, N2, T2}) where {N1, N2, T1, T2} =
    HyperDualSIMD{N1, N2, T1}(T1(h.v), to_ϵ(Vec{N1, T1}, h.ϵ1), to_ϵ(Vec{N2, T1}, h.ϵ2), convert_cross(Vec{N2, T1}, h.ϵ12))
Base.convert(::Type{HyperDualSIMD{N1, N2, T}}, x::Real) where {N1, N2, T} = HyperDualSIMD{N1, N2, T}(T(x))

function Base.show(io::IO, h::HyperDual)
    print(io, h.v, " + ", Tuple(h.ϵ1), "ϵ1", " + ", Tuple(h.ϵ2), "ϵ2", " + ", map(Tuple, h.ϵ12), "ϵ12")
    return
end

# HyperDual one/zero/float
Base.one(::Type{HyperDual{N1, N2, T}}) where {N1, N2, T} = HyperDual{N1, N2, T}(one(T))
Base.zero(::Type{HyperDual{N1, N2, T}}) where {N1, N2, T} = HyperDual{N1, N2, T}(zero(T))
Base.one(::HyperDual{N1, N2, T}) where {N1, N2, T} = one(HyperDual{N1, N2, T})
Base.zero(::HyperDual{N1, N2, T}) where {N1, N2, T} = zero(HyperDual{N1, N2, T})
Base.float(h::HyperDual{N1, N2, T}) where {N1, N2, T} = convert(HyperDual{N1, N2, float(T)}, h)

# HyperDualSIMD one/zero/float
Base.one(::Type{HyperDualSIMD{N1, N2, T}}) where {N1, N2, T} = HyperDualSIMD{N1, N2, T}(one(T))
Base.zero(::Type{HyperDualSIMD{N1, N2, T}}) where {N1, N2, T} = HyperDualSIMD{N1, N2, T}(zero(T))
Base.one(::HyperDualSIMD{N1, N2, T}) where {N1, N2, T} = one(HyperDualSIMD{N1, N2, T})
Base.zero(::HyperDualSIMD{N1, N2, T}) where {N1, N2, T} = zero(HyperDualSIMD{N1, N2, T})
Base.float(h::HyperDualSIMD{N1, N2, T}) where {N1, N2, T} = convert(HyperDualSIMD{N1, N2, float(T)}, h)

# Unified arithmetic for AbstractHyperDualNumber
@inline Base.:(-)(h::AbstractHyperDualNumber) = typeof(h)(-h.v, ⊟(h.ϵ1), ⊟(h.ϵ2), mapϵ12(⊟, h))
@inline Base.:(+)(h::AbstractHyperDualNumber) = h

@inline Base.:+(h1::H, h2::H) where {H <: AbstractHyperDualNumber} =
    H(h1.v + h2.v, h1.ϵ1 ⊕ h2.ϵ1, h1.ϵ2 ⊕ h2.ϵ2, mapϵ12(⊕, h1, h2))
@inline Base.:+(h1::H1, h2::H2) where {H1 <: AbstractHyperDualNumber, H2 <: AbstractHyperDualNumber} = +(promote(h1, h2)...)
@inline Base.:+(h::AbstractHyperDualNumber, r::Real) = typeof(h)(h.v + r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:+(r::Real, h::AbstractHyperDualNumber) = typeof(h)(r + h.v, h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:-(h1::H, h2::H) where {H <: AbstractHyperDualNumber} =
    H(h1.v - h2.v, h1.ϵ1 ⊖ h2.ϵ1, h1.ϵ2 ⊖ h2.ϵ2, mapϵ12(⊖, h1, h2))
@inline Base.:-(h1::H1, h2::H2) where {H1 <: AbstractHyperDualNumber, H2 <: AbstractHyperDualNumber} = -(promote(h1, h2)...)
@inline Base.:-(h::AbstractHyperDualNumber, r::Real) = typeof(h)(h.v - r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:-(r::Real, h::AbstractHyperDualNumber) = typeof(h)(r - h.v, ⊟(h.ϵ1), ⊟(h.ϵ2), mapϵ12(⊟, h))

@inline Base.:*(h::AbstractHyperDualNumber, r::Real) = typeof(h)(h.v * r, h.ϵ1 ⊙ r, h.ϵ2 ⊙ r, mapϵ12(ϵ -> ϵ ⊙ r, h))
@inline Base.:/(h::AbstractHyperDualNumber, r::Real) = typeof(h)(h.v / r, h.ϵ1 ⊘ r, h.ϵ2 ⊘ r, mapϵ12(ϵ -> ϵ ⊘ r, h))
@inline Base.:*(r::Real, h::AbstractHyperDualNumber) = typeof(h)(r * h.v, r ⊙ h.ϵ1, r ⊙ h.ϵ2, mapϵ12(ϵ -> r ⊙ ϵ, h))

@inline Base.:(/)(r::Real, h::AbstractHyperDualNumber) = r * inv(h)
@inline Base.:(/)(h1::H, h2::H) where {H <: AbstractHyperDualNumber} = h1 * inv(h2)

@inline function Base.muladd(x::H, y::Real, z::H) where {N1, H <: AbstractHyperDualNumber{N1}}
    return H(
        muladd(x.v, y, z.v),
        _muladd(y, x.ϵ1, z.ϵ1),
        _muladd(y, x.ϵ2, z.ϵ2),
        ntuple(i -> _muladd(y, x.ϵ12[i], z.ϵ12[i]), Val(N1))
    )
end
@inline function Base.muladd(x::Real, y::H, z::H) where {N1, H <: AbstractHyperDualNumber{N1}}
    return H(
        muladd(x, y.v, z.v),
        _muladd(x, y.ϵ1, z.ϵ1),
        _muladd(x, y.ϵ2, z.ϵ2),
        ntuple(i -> _muladd(x, y.ϵ12[i], z.ϵ12[i]), Val(N1)),
    )
end
@inline function Base.muladd(x::H, y::Real, z::Real) where {N1, H <: AbstractHyperDualNumber{N1}}
    return H(
        muladd(x.v, y, z),
        x.ϵ1 ⊙ y,
        x.ϵ2 ⊙ y,
        ntuple(i -> x.ϵ12[i] ⊙ y, Val(N1)),
    )
end
@inline function Base.muladd(x::Real, y::H, z::Real) where {N1, H <: AbstractHyperDualNumber{N1}}
    return H(
        muladd(x, y.v, z),
        y.ϵ1 ⊙ x,
        y.ϵ2 ⊙ x,
        ntuple(i -> y.ϵ12[i] ⊙ x, Val(N1)),
    )
end
@inline Base.muladd(x::Real, y::Real, z::AbstractHyperDualNumber) = muladd(x, y, z.v) + z - z.v

@inline Base.:(*)(h1::H1, h2::H2) where {H1 <: AbstractHyperDualNumber, H2 <: AbstractHyperDualNumber} = *(promote(h1, h2)...)
@inline function Base.:(*)(h1::H, h2::H) where {N1, H <: AbstractHyperDualNumber{N1}}
    r = h1.v * h2.v
    ϵ1 = _muladd(h1.v, h2.ϵ1, h1.ϵ1 ⊙ h2.v)
    ϵ2 = _muladd(h1.v, h2.ϵ2, h1.ϵ2 ⊙ h2.v)
    # Inline outer products with FMA: h1.ϵ1[i]*h2.ϵ2 + h2.ϵ1[i]*h1.ϵ2
    @inline g(i) = _muladd(h1.v, h2.ϵ12[i], _muladd(h1.ϵ12[i], h2.v, _muladd(h1.ϵ1[i], h2.ϵ2, h2.ϵ1[i] ⊙ h1.ϵ2)))
    ϵ12 = ntuple(g, Val(N1))
    return H(r, ϵ1, ϵ2, ϵ12)
end

# Common literal_pow for both types
@inline Base.literal_pow(::typeof(^), x::AbstractHyperDualNumber, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::AbstractHyperDualNumber, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::AbstractHyperDualNumber, ::Val{2}) = x * x
@inline Base.literal_pow(::typeof(^), x::AbstractHyperDualNumber, ::Val{3}) = x * x * x

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
if USE_SIMD
    const ϵT{N, T} = Vec{N, T}
else
    const ϵT{N, T} = NTuple{N, T}
end

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
    HyperDual(v, ϵ1, ϵ2, ntuple(_ -> zero_ϵ(ϵT{N2, T}), Val(N1)))
HyperDual{N1, N2}(v::T) where {N1, N2, T} = HyperDual(v, zero_ϵ(ϵT{N1, T}), zero_ϵ(ϵT{N2, T}))
HyperDual{N1, N2, T}(v) where {N1, N2, T} = HyperDual{N1, N2}(T(v))

function HyperDual(v::T1, ϵ1::ϵT{N1, T2}, ϵ2::ϵT{N2, T2}, ϵ12::NTuple{N1, ϵT{N2, T2}}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual(T(v), to_ϵ(ϵT{N1, T}, ϵ1), to_ϵ(ϵT{N2, T}, ϵ2), convert_cross(ϵT{N2, T}, ϵ12))
end

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
    HyperDual(-h.v, ⊟(h.ϵ1), ⊟(h.ϵ2), ntuple(i -> ⊟(h.ϵ12[i]), Val(N1)))
@inline Base.:(+)(h::HyperDual) = h

@inline Base.:+(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    HyperDual(h1.v + h2.v, h1.ϵ1 ⊕ h2.ϵ1, h1.ϵ2 ⊕ h2.ϵ2, ntuple(i -> h1.ϵ12[i] ⊕ h2.ϵ12[i], Val(N1)))
@inline Base.:+(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = +(promote(h1, h2)...)
@inline Base.:+(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v + r, h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:-(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    HyperDual(h1.v - h2.v, h1.ϵ1 ⊖ h2.ϵ1, h1.ϵ2 ⊖ h2.ϵ2, ntuple(i -> h1.ϵ12[i] ⊖ h2.ϵ12[i], Val(N1)))
@inline Base.:-(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = -(promote(h1, h2)...)
@inline Base.:-(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v - r, h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:+(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r + h.v, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:-(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r - h.v, ⊟(h.ϵ1), ⊟(h.ϵ2), ntuple(i -> ⊟(h.ϵ12[i]), Val(N1)))

@inline Base.:*(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v * r, h.ϵ1 ⊙ r, h.ϵ2 ⊙ r, ntuple(i -> h.ϵ12[i] ⊙ r, Val(N1)))
@inline Base.:/(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v / r, h.ϵ1 ⊘ r, h.ϵ2 ⊘ r, ntuple(i -> h.ϵ12[i] ⊘ r, Val(N1)))
@inline Base.:(*)(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r * h.v, r ⊙ h.ϵ1, r ⊙ h.ϵ2, ntuple(i -> r ⊙ h.ϵ12[i], Val(N1)))

@inline Base.:(/)(r::Real, h::HyperDual{N1, N2}) where {N1, N2} = r * inv(h)
@inline Base.:(/)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} = h1 * inv(h2)

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
@inline Base.muladd(x::Real, y::Real, z::HyperDual{N1, N2, T}) where {N1, N2, T} = muladd(x, y, z.v) + z - z.v

@inline Base.:(*)(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = *(promote(h1, h2)...)
@inline function Base.:(*)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T}
    r = h1.v * h2.v
    ϵ1 = _muladd(h1.v, h2.ϵ1, h1.ϵ1 ⊙ h2.v)
    ϵ2 = _muladd(h1.v, h2.ϵ2, h1.ϵ2 ⊙ h2.v)
    ϵ12_1 = h1.ϵ1 ⊗ h2.ϵ2
    ϵ12_2 = h2.ϵ1 ⊗ h1.ϵ2
    ϵ12 = ntuple(i -> _muladd(h1.v, h2.ϵ12[i], _muladd(h1.ϵ12[i], h2.v, ϵ12_1[i] ⊕ ϵ12_2[i])), Val(N1))
    return HyperDual(r, ϵ1, ϵ2, ϵ12)
end
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{2}) = x * x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{3}) = x * x * x

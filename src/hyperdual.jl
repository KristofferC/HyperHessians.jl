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
struct HyperDual{N1, N2, T} <: Real
    v::T
    ϵ1::Vec{N1, T}
    ϵ2::Vec{N2, T}
    ϵ12::NTuple{N1, Vec{N2, T}}
end
HyperDual(v::T, ϵ1::Vec{N1, T}, ϵ2::Vec{N2, T}) where {N1, N2, T} =
    HyperDual(v, ϵ1, ϵ2, ntuple(_ -> zero(Vec{N2, T}), Val(N1)))
HyperDual{N1, N2}(v::T) where {N1, N2, T} = HyperDual(v, zero(Vec{N1, T}), zero(Vec{N2, T}))
HyperDual{N1, N2, T}(v) where {N1, N2, T} = HyperDual{N1, N2}(T(v))

function HyperDual(v::T1, ϵ1::Vec{N1, T2}, ϵ2::Vec{N2, T2}, ϵ12::NTuple{N1, Vec{N2, T2}}) where {N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual(T(v), convert(Vec{N1, T}, ϵ1), convert(Vec{N2, T}, ϵ2), convert.(Vec{N2, T}, ϵ12))
end

Base.promote_rule(::Type{HyperDual{N1, N2, T1}}, ::Type{HyperDual{N1, N2, T2}}) where {N1, N2, T1, T2} =
    HyperDual{N1, N2, promote_type(T1, T2)}
Base.promote_rule(::Type{HyperDual{N1, N2, T1}}, ::Type{T2}) where {N1, N2, T1, T2 <: Real} =
    HyperDual{N1, N2, promote_type(T1, T2)}
Base.convert(::Type{HyperDual{N1, N2, T1}}, h::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} =
    HyperDual{N1, N2, T1}(T1(h.v), convert(Vec{N1, T1}, h.ϵ1), convert(Vec{N2, T1}, h.ϵ2), convert.(Vec{N2, T1}, h.ϵ12))
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
    HyperDual(-h.v, -h.ϵ1, -h.ϵ2, ntuple(i -> -h.ϵ12[i], Val(N1)))
@inline Base.:(+)(h::HyperDual) = h

for f in (:+, :-)
    @eval begin
        @inline Base.$f(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
            HyperDual($f(h1.v, h2.v), $f(h1.ϵ1, h2.ϵ1), $f(h1.ϵ2, h2.ϵ2), ntuple(i -> $f(h1.ϵ12[i], h2.ϵ12[i]), Val(N1)))
        @inline Base.$f(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = $f(promote(h1, h2)...)
        @inline Base.$f(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
            HyperDual($f(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
        @inline Base.$f(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
            HyperDual($f(r, h.v), h.ϵ1, h.ϵ2, h.ϵ12)
    end
end

for f in (:*, :/)
    @eval @inline Base.$f(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
        HyperDual($f(h.v, r), $f(h.ϵ1, r), $f(h.ϵ2, r), ntuple(i -> $f(h.ϵ12[i], r), Val(N1)))
    @eval @inline Base.$f(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
        HyperDual($f(r, h.v), $f(r, h.ϵ1), $f(r, h.ϵ2), ntuple(i -> $f(r, h.ϵ12[i]), Val(N1)))
end

@inline Base.:(/)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} = h1 * inv(h2)

@inline Base.muladd(x::HyperDual{N1, N2}, y::Real, z::HyperDual{N1, N2}) where {N1, N2} = x * y + z
@inline Base.muladd(x::Real, y::HyperDual{N1, N2}, z::HyperDual{N1, N2}) where {N1, N2} = x * y + z
@inline Base.muladd(x::HyperDual{N1, N2}, y::HyperDual{N1, N2}, z::Real) where {N1, N2} = x * y + z
@inline Base.muladd(x::HyperDual{N1, N2}, y::Real, z::Real) where {N1, N2} = x * y + z
@inline Base.muladd(x::Real, y::HyperDual{N1, N2}, z::Real) where {N1, N2} = x * y + z
@inline Base.muladd(x::Real, y::Real, z::HyperDual{N1, N2}) where {N1, N2} = muladd(x, y, z.v) + z - z.v
@inline Base.muladd(x::HyperDual{N1, N2}, y::HyperDual{N1, N2}, z::HyperDual{N1, N2}) where {N1, N2} = x * y + z

unsafe_getindex(v::SIMD.Vec, i::SIMD.IntegerTypes) = SIMD.Intrinsics.extractelement(v.data, i - 1)
@inline ⊗(v1::Vec{N1, T}, v2::Vec{N2, T}) where {N1, N2, T} = ntuple(i -> unsafe_getindex(v1, i) * v2, Val(N1))

@inline Base.:(*)(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = *(promote(h1, h2)...)
@inline function Base.:(*)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T}
    r = h1.v * h2.v
    ϵ1 = muladd(h1.v, h2.ϵ1, h1.ϵ1 * h2.v)
    ϵ2 = muladd(h1.v, h2.ϵ2, h1.ϵ2 * h2.v)
    ϵ12_1 = h1.ϵ1 ⊗ h2.ϵ2
    ϵ12_2 = h2.ϵ1 ⊗ h1.ϵ2
    ϵ12 = ntuple(i -> muladd(h1.v, h2.ϵ12[i], muladd(h1.ϵ12[i], h2.v, ϵ12_1[i] + ϵ12_2[i])), Val(N1))
    return HyperDual(r, ϵ1, ϵ2, ϵ12)
end
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{2}) = x * x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{3}) = x * x * x

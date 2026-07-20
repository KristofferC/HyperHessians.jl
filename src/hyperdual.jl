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

abstract type ArithmeticMode end
struct IEEEArithmetic <: ArithmeticMode end
struct FastArithmetic <: ArithmeticMode end

const IEEE_MODE = IEEEArithmetic()
const FAST_MODE = FastArithmetic()

@inline _add(::IEEEArithmetic, a::Real, b::Real) = a + b
@inline _add(::FastArithmetic, a::Real, b::Real) = Base.FastMath.add_fast(a, b)
@inline _add(mode::ArithmeticMode, a::NTuple{N, A}, b::NTuple{N, B}) where {N, A, B} =
    ntuple(i -> _add(mode, a[i], b[i]), Val(N))

@inline _neg(::IEEEArithmetic, a::Real) = -a
@inline _neg(::FastArithmetic, a::Real) = Base.FastMath.sub_fast(a)
@inline _neg(mode::ArithmeticMode, a::NTuple{N, A}) where {N, A} =
    ntuple(i -> _neg(mode, a[i]), Val(N))
@inline _sub(mode::ArithmeticMode, a, b) = _add(mode, a, _neg(mode, b))

@inline _mul(::IEEEArithmetic, a::Real, b::Real) = a * b
@inline _mul(::FastArithmetic, a::Real, b::Real) = Base.FastMath.mul_fast(a, b)
@inline _mul(mode::ArithmeticMode, a::NTuple{N, A}, r::Real) where {N, A} =
    ntuple(i -> _mul(mode, a[i], r), Val(N))
@inline _mul(mode::ArithmeticMode, r::Real, a::NTuple{N, A}) where {N, A} =
    ntuple(i -> _mul(mode, r, a[i]), Val(N))

@inline _div(::IEEEArithmetic, a::Real, b::Real) = a / b
@inline _div(::FastArithmetic, a::Real, b::Real) = Base.FastMath.div_fast(a, b)
@inline _div(mode::ArithmeticMode, a::NTuple{N, A}, r::Real) where {N, A} =
    ntuple(i -> _div(mode, a[i], r), Val(N))

@inline _inv(::IEEEArithmetic, x::Real) = inv(x)
@inline _inv(::FastArithmetic, x::Real) = Base.FastMath.inv_fast(x)
@inline _pow(::IEEEArithmetic, x::Real, n::Integer) = x^n
@inline _pow(::FastArithmetic, x::Real, n::Integer) = Base.FastMath.pow_fast(x, n)

@inline _muladd(::IEEEArithmetic, a::Real, b::Real, c::Real) = muladd(a, b, c)
@inline _muladd(::FastArithmetic, a::Real, b::Real, c::Real) =
    Base.FastMath.add_fast(Base.FastMath.mul_fast(a, b), c)
@inline _muladd(mode::ArithmeticMode, a::Real, b::NTuple{N, A}, c::NTuple{N, C}) where {N, A, C} =
    ntuple(i -> _muladd(mode, a, b[i], c[i]), Val(N))
@inline _muladd(mode::ArithmeticMode, a::NTuple{N, A}, b::Real, c::NTuple{N, C}) where {N, A, C} =
    ntuple(i -> _muladd(mode, a[i], b, c[i]), Val(N))
@inline _outer(mode::ArithmeticMode, t1::NTuple{N1, T1}, t2::NTuple{N2, T2}) where {N1, N2, T1, T2} =
    ntuple(i -> _mul(mode, t2, t1[i]), Val(N1))

# Shorthand for the ordinary IEEE arithmetic used throughout the public methods.
@inline ⊕(a, b) = _add(IEEE_MODE, a, b)
@inline ⊟(a) = _neg(IEEE_MODE, a)
@inline ⊖(a, b) = _sub(IEEE_MODE, a, b)
@inline ⊙(a, b) = _mul(IEEE_MODE, a, b)
@inline ⊘(a, b) = _div(IEEE_MODE, a, b)
@inline _muladd(a, b, c) = _muladd(IEEE_MODE, a, b, c)
@inline ⊗(t1, t2) = _outer(IEEE_MODE, t1, t2)

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
HyperDual{N1, N2}(v::HyperDual{N1, N2}) where {N1, N2} = v

# Disambiguate against Base's numeric conversion constructors (Complex, Char,
# TwicePrecision), which also target Real/Number. Route through the scalar value.
_scalar(z::Complex) = real(typeof(z))(z)
_scalar(c::AbstractChar) = Int(c)
_scalar(v::Base.TwicePrecision{T}) where {T} = T(v)
for R in (:Complex, :AbstractChar, :(Base.TwicePrecision))
    @eval HyperDual{N1, N2, T}(v::$R) where {N1, N2, T} = HyperDual{N1, N2}(T(v))
    @eval HyperDual{N1, N2}(v::$R) where {N1, N2} = HyperDual{N1, N2}(_scalar(v))
end

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

@inline function _neg_hyperdual(mode::ArithmeticMode, h::HyperDual{N1, N2}) where {N1, N2}
    return HyperDual(
        _neg(mode, h.v),
        _neg(mode, h.ϵ1),
        _neg(mode, h.ϵ2),
        mapϵ12(x -> _neg(mode, x), h),
    )
end

@inline function _add_hyperdual(
        mode::ArithmeticMode,
        h1::HyperDual{N1, N2, T},
        h2::HyperDual{N1, N2, T},
    ) where {N1, N2, T}
    return HyperDual(
        _add(mode, h1.v, h2.v),
        _add(mode, h1.ϵ1, h2.ϵ1),
        _add(mode, h1.ϵ2, h2.ϵ2),
        mapϵ12((x, y) -> _add(mode, x, y), h1, h2),
    )
end

@inline function _sub_hyperdual(
        mode::ArithmeticMode,
        h1::HyperDual{N1, N2, T},
        h2::HyperDual{N1, N2, T},
    ) where {N1, N2, T}
    return HyperDual(
        _sub(mode, h1.v, h2.v),
        _sub(mode, h1.ϵ1, h2.ϵ1),
        _sub(mode, h1.ϵ2, h2.ϵ2),
        mapϵ12((x, y) -> _sub(mode, x, y), h1, h2),
    )
end

@inline Base.:(-)(h::HyperDual) = _neg_hyperdual(IEEE_MODE, h)
@inline Base.:(+)(h::HyperDual) = h

@inline Base.:+(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    _add_hyperdual(IEEE_MODE, h1, h2)
@inline Base.:+(h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}) where {N1, N2, T1, T2} = +(promote(h1, h2)...)
@inline Base.:+(h::HyperDual{N1, N2}, r::Real) where {N1, N2} =
    HyperDual(h.v + r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:+(r::Real, h::HyperDual{N1, N2}) where {N1, N2} =
    HyperDual(r + h.v, h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:-(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    _sub_hyperdual(IEEE_MODE, h1, h2)
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

# Dedicated division rule: cheaper than h1 * inv(h2) (fewer chain-rule products).
# f = x/y, fₓ = 1/y, fᵧ = -f/y, fₓₓ = 0, fₓᵧ = -1/y², fᵧᵧ = 2f/y²
@inline function _div_hyperdual(
        mode::ArithmeticMode,
        h1::HyperDual{N1, N2, T},
        h2::HyperDual{N1, N2, T},
    ) where {N1, N2, T}
    x, y = h1.v, h2.v
    invy = _inv(mode, y)
    f = _mul(mode, x, invy)
    fᵧ = _neg(mode, _mul(mode, f, invy))
    fₓᵧ = _mul(mode, _neg(mode, invy), invy)
    fᵧᵧ = _mul(mode, _mul(mode, _neg(mode, 2), fᵧ), invy)
    return chain_rule_dual(mode, h1, h2, f, invy, fᵧ, zero(invy), fₓᵧ, fᵧᵧ)
end

# Specialized rule for f(y) = r/y. Unlike promotion to HyperDual/HyperDual,
# this propagates only the nonzero denominator derivatives. Keep the cross
# derivative calculation fused here: materializing its intermediate outer
# product causes tuple spills for larger chunk sizes.
@inline function _rdiv_hyperdual(
        mode::ArithmeticMode,
        r::Real,
        h::HyperDual{N1, N2},
    ) where {N1, N2}
    r, y = promote(r, h.v)
    invy = _inv(mode, y)
    f = _mul(mode, r, invy)
    f′ = _neg(mode, _mul(mode, f, invy))
    f′′ = _mul(mode, _mul(mode, _neg(mode, 2), f′), invy)
    ϵ1 = _mul(mode, h.ϵ1, f′)
    ϵ2 = _mul(mode, h.ϵ2, f′)
    @inline g(i) = _muladd(
        mode,
        f′,
        h.ϵ12[i],
        _mul(mode, _mul(mode, f′′, h.ϵ1[i]), h.ϵ2),
    )
    return HyperDual(f, ϵ1, ϵ2, ntuple(g, Val(N1)))
end

@inline Base.:(/)(r::Real, h::HyperDual) = r * inv(h)
@inline Base.:(/)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    _div_hyperdual(IEEE_MODE, h1, h2)
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
# All-HyperDual multiplicands are ambiguous between the mixed methods above, so
# resolve them explicitly via the general product-then-sum.
@inline Base.muladd(x::HyperDual{N1, N2, T}, y::HyperDual{N1, N2, T}, z::HyperDual{N1, N2, T}) where {N1, N2, T} =
    x * y + z
@inline Base.muladd(x::HyperDual{N1, N2, T}, y::HyperDual{N1, N2, T}, z::Real) where {N1, N2, T} =
    x * y + z

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
# Disambiguate against Base.==(::Real, ::AbstractIrrational) and its mirror.
@inline Base.:(==)(h::HyperDual, r::AbstractIrrational) = h.v == r
@inline Base.:(==)(r::AbstractIrrational, h::HyperDual) = r == h.v
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
@inline function _mul_hyperdual(
        mode::ArithmeticMode,
        h1::HyperDual{N1, N2, T},
        h2::HyperDual{N1, N2, T},
    ) where {N1, N2, T}
    r = _mul(mode, h1.v, h2.v)
    ϵ1 = _muladd(mode, h1.v, h2.ϵ1, _mul(mode, h1.ϵ1, h2.v))
    ϵ2 = _muladd(mode, h1.v, h2.ϵ2, _mul(mode, h1.ϵ2, h2.v))
    # Inline outer products with FMA: h1.ϵ1[i]*h2.ϵ2 + h2.ϵ1[i]*h1.ϵ2
    @inline g(i) = _muladd(
        mode,
        h1.v,
        h2.ϵ12[i],
        _muladd(
            mode,
            h1.ϵ12[i],
            h2.v,
            _muladd(mode, h1.ϵ1[i], h2.ϵ2, _mul(mode, h2.ϵ1[i], h1.ϵ2)),
        ),
    )
    ϵ12 = ntuple(g, Val(N1))
    return HyperDual(r, ϵ1, ϵ2, ϵ12)
end
@inline Base.:(*)(h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}) where {N1, N2, T} =
    _mul_hyperdual(IEEE_MODE, h1, h2)
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{p}) where {p} = x^p

# Integer powers use the monomial derivatives n*x^(n-1) and n*(n-1)*x^(n-2)
# directly (the generic `^` rule divides by x and returns NaN at x = 0).
# This is also cheaper than repeated HyperDual multiplication.
# Also resolves the ambiguity with Base.^(::Number, ::Integer).
@inline function _pow_hyperdual(mode::ArithmeticMode, h::HyperDual, n::Integer)
    x = h.v
    if n == 0
        return one(h)
    elseif n == 1
        return h
    elseif n == 2
        return chain_rule_dual(
            mode,
            h,
            _mul(mode, x, x),
            _mul(mode, 2, x),
            _mul(mode, 2, one(x)),
        )
    elseif n == 3
        x2 = _mul(mode, x, x)
        return chain_rule_dual(
            mode,
            h,
            _mul(mode, x2, x),
            _mul(mode, 3, x2),
            _mul(mode, 6, x),
        )
    else
        p = _pow(mode, x, n - 2)
        xp = _mul(mode, x, p)
        f = _mul(mode, x, xp)
        f′ = _mul(mode, n, xp)
        f′′ = _mul(mode, n * (n - 1), p)
        return chain_rule_dual(mode, h, f, f′, f′′)
    end
end
@inline Base.:(^)(h::HyperDual, n::Integer) = _pow_hyperdual(IEEE_MODE, h, n)
# Disambiguate against Base.^(::Number, ::Rational): use the general real-power rule.
@inline Base.:(^)(h::HyperDual, r::Rational) = h^float(r)

# `@fastmath` rewrites user arithmetic to these functions. The methods below
# select fast arithmetic for both the primal and all derivative components.
@inline Base.FastMath.sub_fast(h::HyperDual) = _neg_hyperdual(FAST_MODE, h)

@inline Base.FastMath.add_fast(
    h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}
) where {N1, N2, T} = _add_hyperdual(FAST_MODE, h1, h2)
@inline Base.FastMath.add_fast(
    h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}
) where {N1, N2, T1, T2} = Base.FastMath.add_fast(promote(h1, h2)...)
@inline Base.FastMath.add_fast(h::HyperDual, r::Real) =
    HyperDual(Base.FastMath.add_fast(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.FastMath.add_fast(r::Real, h::HyperDual) =
    HyperDual(Base.FastMath.add_fast(r, h.v), h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.FastMath.sub_fast(
    h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}
) where {N1, N2, T} = _sub_hyperdual(FAST_MODE, h1, h2)
@inline Base.FastMath.sub_fast(
    h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}
) where {N1, N2, T1, T2} = Base.FastMath.sub_fast(promote(h1, h2)...)
@inline Base.FastMath.sub_fast(h::HyperDual, r::Real) =
    HyperDual(Base.FastMath.sub_fast(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.FastMath.sub_fast(r::Real, h::HyperDual) = HyperDual(
    Base.FastMath.sub_fast(r, h.v),
    _neg(FAST_MODE, h.ϵ1),
    _neg(FAST_MODE, h.ϵ2),
    mapϵ12(x -> _neg(FAST_MODE, x), h),
)

@inline Base.FastMath.mul_fast(
    h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}
) where {N1, N2, T} = _mul_hyperdual(FAST_MODE, h1, h2)
@inline Base.FastMath.mul_fast(
    h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}
) where {N1, N2, T1, T2} = Base.FastMath.mul_fast(promote(h1, h2)...)
@inline Base.FastMath.mul_fast(h::HyperDual, r::Real) = HyperDual(
    Base.FastMath.mul_fast(h.v, r),
    _mul(FAST_MODE, h.ϵ1, r),
    _mul(FAST_MODE, h.ϵ2, r),
    mapϵ12(x -> _mul(FAST_MODE, x, r), h),
)
@inline Base.FastMath.mul_fast(r::Real, h::HyperDual) = HyperDual(
    Base.FastMath.mul_fast(r, h.v),
    _mul(FAST_MODE, r, h.ϵ1),
    _mul(FAST_MODE, r, h.ϵ2),
    mapϵ12(x -> _mul(FAST_MODE, r, x), h),
)

@inline Base.FastMath.div_fast(
    h1::HyperDual{N1, N2, T}, h2::HyperDual{N1, N2, T}
) where {N1, N2, T} = _div_hyperdual(FAST_MODE, h1, h2)
@inline Base.FastMath.div_fast(
    h1::HyperDual{N1, N2, T1}, h2::HyperDual{N1, N2, T2}
) where {N1, N2, T1, T2} = Base.FastMath.div_fast(promote(h1, h2)...)
@inline Base.FastMath.div_fast(h::HyperDual, r::Real) = HyperDual(
    Base.FastMath.div_fast(h.v, r),
    _div(FAST_MODE, h.ϵ1, r),
    _div(FAST_MODE, h.ϵ2, r),
    mapϵ12(x -> _div(FAST_MODE, x, r), h),
)
@inline Base.FastMath.div_fast(r::Real, h::HyperDual) =
    _rdiv_hyperdual(FAST_MODE, r, h)

@inline Base.FastMath.pow_fast(h::HyperDual, n::Integer) =
    _pow_hyperdual(FAST_MODE, h, n)
@inline Base.FastMath.pow_fast(h::HyperDual, ::Val{p}) where {p} =
    _pow_hyperdual(FAST_MODE, h, p)

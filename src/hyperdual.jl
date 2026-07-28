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
@inline zero_ϵ(::Type{Tuple{}}) = () # NTuple{0, T} leaves T unbound
@inline zero_ϵ(x::NTuple{N, T}) where {N, T} = zero_ϵ(NTuple{N, T})
@inline to_ϵ(::Type{NTuple{N, T}}, x) where {N, T} = convert(NTuple{N, T}, x)
@inline convert_cross(::Type{NTuple{N, T}}, xs::NTuple{M, Any}) where {N, M, T} =
    ntuple(i -> to_ϵ(NTuple{N, T}, xs[i]), Val(M))

# The `simd` flag selects the SIMD.Vec-backed tuple operations below; scalar
# operations are identical for both flag values.
abstract type ArithmeticMode end
struct IEEEArithmetic{simd} <: ArithmeticMode end
struct FastArithmetic{simd} <: ArithmeticMode end

const IEEE_MODE = IEEEArithmetic{false}()
const FAST_MODE = FastArithmetic{false}()

@inline _simd(::IEEEArithmetic{S}) where {S} = S
@inline _simd(::FastArithmetic{S}) where {S} = S

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

# SIMD.Vec-forced implementations, selected by HyperDuals carrying S = true
# (`HessianConfig(x; simd = true)`). Only same-eltype Float32/Float64 tuples
# are forced; anything else (nested duals, BigFloat, mixed precision, empty
# tuples, non-matching scalar types) falls back to the generic ntuple code
# above, which auto-vectorizes.
const SIMDMode = Union{IEEEArithmetic{true}, FastArithmetic{true}}
const SIMDFloat = Union{Float32, Float64}

@inline _add(::IEEEArithmetic{true}, a::NTuple{N, T}, b::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(Vec{N, T}(a) + Vec{N, T}(b))
@inline _mul(::IEEEArithmetic{true}, a::NTuple{N, T}, r::T) where {N, T <: SIMDFloat} =
    Tuple(Vec{N, T}(a) * r)
@inline _mul(::IEEEArithmetic{true}, r::T, a::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(r * Vec{N, T}(a))
@inline _div(::IEEEArithmetic{true}, a::NTuple{N, T}, r::T) where {N, T <: SIMDFloat} =
    Tuple(Vec{N, T}(a) / r)
@inline _muladd(::IEEEArithmetic{true}, a::T, b::NTuple{N, T}, c::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(muladd(Vec{N, T}(a), Vec{N, T}(b), Vec{N, T}(c)))
@inline _muladd(::IEEEArithmetic{true}, a::NTuple{N, T}, b::T, c::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(muladd(Vec{N, T}(a), Vec{N, T}(b), Vec{N, T}(c)))

# Fast-mode Vec ops carry LLVM fast flags like their scalar counterparts.
@inline _add(::FastArithmetic{true}, a::NTuple{N, T}, b::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(Base.FastMath.add_fast(Vec{N, T}(a), Vec{N, T}(b)))
@inline _mul(::FastArithmetic{true}, a::NTuple{N, T}, r::T) where {N, T <: SIMDFloat} =
    Tuple(Base.FastMath.mul_fast(Vec{N, T}(a), r))
@inline _mul(::FastArithmetic{true}, r::T, a::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(Base.FastMath.mul_fast(r, Vec{N, T}(a)))
@inline _div(::FastArithmetic{true}, a::NTuple{N, T}, r::T) where {N, T <: SIMDFloat} =
    Tuple(Base.FastMath.div_fast(Vec{N, T}(a), r))
@inline _muladd(::FastArithmetic{true}, a::T, b::NTuple{N, T}, c::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(Base.FastMath.add_fast(Base.FastMath.mul_fast(Vec{N, T}(a), Vec{N, T}(b)), Vec{N, T}(c)))
@inline _muladd(::FastArithmetic{true}, a::NTuple{N, T}, b::T, c::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(Base.FastMath.add_fast(Base.FastMath.mul_fast(Vec{N, T}(a), Vec{N, T}(b)), Vec{N, T}(c)))

# Negation is exact, so one method covers both modes.
@inline _neg(::SIMDMode, a::NTuple{N, T}) where {N, T <: SIMDFloat} =
    Tuple(-Vec{N, T}(a))

# Length-1 lanes (e.g. the ϵ₂/ϵ₁₂ rows of directional HVP duals): Vec{1} is
# pure overhead, keep those scalar. Length-0 lanes would build the illegal
# Vec{0} whenever a scalar argument binds T. Both are defined per concrete
# mode so they stay strictly more specific than the per-mode Vec methods.
for M in (:(IEEEArithmetic{true}), :(FastArithmetic{true}))
    @eval begin
        @inline _add(mode::$M, a::Tuple{T}, b::Tuple{T}) where {T <: SIMDFloat} = (_add(mode, a[1], b[1]),)
        @inline _neg(mode::$M, a::Tuple{T}) where {T <: SIMDFloat} = (_neg(mode, a[1]),)
        @inline _mul(mode::$M, a::Tuple{T}, r::T) where {T <: SIMDFloat} = (_mul(mode, a[1], r),)
        @inline _mul(mode::$M, r::T, a::Tuple{T}) where {T <: SIMDFloat} = (_mul(mode, r, a[1]),)
        @inline _div(mode::$M, a::Tuple{T}, r::T) where {T <: SIMDFloat} = (_div(mode, a[1], r),)
        @inline _muladd(mode::$M, a::T, b::Tuple{T}, c::Tuple{T}) where {T <: SIMDFloat} = (_muladd(mode, a, b[1], c[1]),)
        @inline _muladd(mode::$M, a::Tuple{T}, b::T, c::Tuple{T}) where {T <: SIMDFloat} = (_muladd(mode, a[1], b, c[1]),)

        @inline _add(mode::$M, a::Tuple{}, b::Tuple{}) = ()
        @inline _neg(mode::$M, a::Tuple{}) = ()
        @inline _mul(mode::$M, a::Tuple{}, r::T) where {T <: SIMDFloat} = ()
        @inline _mul(mode::$M, r::T, a::Tuple{}) where {T <: SIMDFloat} = ()
        @inline _div(mode::$M, a::Tuple{}, r::T) where {T <: SIMDFloat} = ()
        @inline _muladd(mode::$M, a::T, b::Tuple{}, c::Tuple{}) where {T <: SIMDFloat} = ()
        @inline _muladd(mode::$M, a::Tuple{}, b::T, c::Tuple{}) where {T <: SIMDFloat} = ()
    end
end

# S selects the arithmetic backend: false = plain tuple ops (works for any T),
# true = SIMD.Vec-forced ops for Float32/Float64 components.
struct HyperDual{N1, N2, T, S} <: Real
    v::T
    ϵ1::ϵT{N1, T}
    ϵ2::ϵT{N2, T}
    ϵ12::NTuple{N1, ϵT{N2, T}}
end

# Internal constructors preserving the backend flag of the operands.
@inline hyperdual(mode::ArithmeticMode, v, ϵ1, ϵ2, ϵ12) = hyperdual(Val(_simd(mode)), v, ϵ1, ϵ2, ϵ12)
@inline hyperdual(::Val{S}, v::T, ϵ1::ϵT{N1, T}, ϵ2::ϵT{N2, T}, ϵ12::NTuple{N1, ϵT{N2, T}}) where {S, N1, N2, T} =
    HyperDual{N1, N2, T, S}(v, ϵ1, ϵ2, ϵ12)
@inline function hyperdual(::Val{S}, v::T1, ϵ1::ϵT{N1, T2}, ϵ2::ϵT{N2, T2}, ϵ12::NTuple{N1, ϵT{N2, T2}}) where {S, N1, N2, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual{N1, N2, T, S}(T(v), to_ϵ(ϵT{N1, T}, ϵ1), to_ϵ(ϵT{N2, T}, ϵ2), convert_cross(ϵT{N2, T}, ϵ12))
end

@inline _ieee_mode(::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} = IEEEArithmetic{S}()
@inline _fast_mode(::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} = FastArithmetic{S}()

# Public constructors default to the tuple backend.
HyperDual(v::T1, ϵ1::ϵT{N1, T2}, ϵ2::ϵT{N2, T2}, ϵ12::NTuple{N1, ϵT{N2, T2}}) where {N1, N2, T1, T2} =
    hyperdual(Val(false), v, ϵ1, ϵ2, ϵ12)
HyperDual(v::T, ϵ1::ϵT{N1, T}, ϵ2::ϵT{N2, T}) where {N1, N2, T} =
    HyperDual(v, ϵ1, ϵ2, ntuple(_ -> ntuple(_ -> zero(T), Val(N2)), Val(N1)))
HyperDual{N1, N2}(v::T) where {N1, N2, T} =
    HyperDual(v, ntuple(_ -> zero(T), Val(N1)), ntuple(_ -> zero(T), Val(N2)))
HyperDual{N1, N2, T}(v) where {N1, N2, T} = HyperDual{N1, N2}(T(v))
HyperDual{N1, N2, T}(v::HyperDual{N1, N2, T}) where {N1, N2, T} = v
HyperDual{N1, N2, T}(v::HyperDual{N1, N2, <:Any, S}) where {N1, N2, T, S} = convert(HyperDual{N1, N2, T, S}, v)
HyperDual{N1, N2}(v::HyperDual{N1, N2}) where {N1, N2} = v

HyperDual{N1, N2, T, S}(v::Real) where {N1, N2, T, S} = hyperdual(
    Val(S), T(v),
    ntuple(_ -> zero(T), Val(N1)), ntuple(_ -> zero(T), Val(N2)),
    ntuple(_ -> ntuple(_ -> zero(T), Val(N2)), Val(N1)),
)
HyperDual{N1, N2, T, S}(v::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} = v
HyperDual{N1, N2, T, S}(v::HyperDual{N1, N2}) where {N1, N2, T, S} = convert(HyperDual{N1, N2, T, S}, v)

# Disambiguate against Base's numeric conversion constructors (Complex, Char,
# TwicePrecision), which also target Real/Number. Route through the scalar value.
_scalar(z::Complex) = real(typeof(z))(z)
_scalar(c::AbstractChar) = Int(c)
_scalar(v::Base.TwicePrecision{T}) where {T} = T(v)
for R in (:Complex, :AbstractChar, :(Base.TwicePrecision))
    @eval HyperDual{N1, N2, T}(v::$R) where {N1, N2, T} = HyperDual{N1, N2}(T(v))
    @eval HyperDual{N1, N2}(v::$R) where {N1, N2} = HyperDual{N1, N2}(_scalar(v))
    @eval HyperDual{N1, N2, T, S}(v::$R) where {N1, N2, T, S} = HyperDual{N1, N2, T, S}(T(v))
end

# Accessor Functions
@inline value(x) = x
@inline value(x::HyperDual) = x.v

@inline mapϵ12(f, h::HyperDual{N1, N2}) where {N1, N2} = ntuple(i -> f(h.ϵ12[i]), Val(N1))
@inline mapϵ12(f, h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} =
    ntuple(i -> f(h1.ϵ12[i], h2.ϵ12[i]), Val(N1))

# Mixed backends promote to the tuple backend.
Base.promote_rule(::Type{HyperDual{N1, N2, T1, S1}}, ::Type{HyperDual{N1, N2, T2, S2}}) where {N1, N2, T1, T2, S1, S2} =
    HyperDual{N1, N2, promote_type(T1, T2), S1 && S2}
Base.promote_rule(::Type{HyperDual{N1, N2, T1, S}}, ::Type{T2}) where {N1, N2, T1, S, T2 <: Real} =
    HyperDual{N1, N2, promote_type(T1, T2), S}
Base.convert(::Type{HyperDual{N1, N2, T1, S}}, h::HyperDual{N1, N2, T2, S2}) where {N1, N2, T1, T2, S, S2} =
    HyperDual{N1, N2, T1, S}(T1(h.v), to_ϵ(ϵT{N1, T1}, h.ϵ1), to_ϵ(ϵT{N2, T1}, h.ϵ2), convert_cross(ϵT{N2, T1}, h.ϵ12))
Base.convert(::Type{HyperDual{N1, N2, T, S}}, x::Real) where {N1, N2, T, S} = HyperDual{N1, N2, T, S}(T(x))

function Base.show(io::IO, h::HyperDual)
    print(io, h.v, " + ", Tuple(h.ϵ1), "ϵ1", " + ", Tuple(h.ϵ2), "ϵ2", " + ", map(Tuple, h.ϵ12), "ϵ12")
    return
end

Base.one(::Type{HyperDual{N1, N2, T, S}}) where {N1, N2, T, S} = HyperDual{N1, N2, T, S}(one(T))
Base.zero(::Type{HyperDual{N1, N2, T, S}}) where {N1, N2, T, S} = HyperDual{N1, N2, T, S}(zero(T))
Base.one(::Type{HyperDual{N1, N2, T}}) where {N1, N2, T} = one(HyperDual{N1, N2, T, false})
Base.zero(::Type{HyperDual{N1, N2, T}}) where {N1, N2, T} = zero(HyperDual{N1, N2, T, false})
Base.one(::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} = one(HyperDual{N1, N2, T, S})
Base.zero(::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} = zero(HyperDual{N1, N2, T, S})
Base.float(h::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} = convert(HyperDual{N1, N2, float(T), S}, h)

@inline function _neg_hyperdual(mode::ArithmeticMode, h::HyperDual{N1, N2}) where {N1, N2}
    return hyperdual(
        mode,
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
    return hyperdual(
        mode,
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
    return hyperdual(
        mode,
        _sub(mode, h1.v, h2.v),
        _sub(mode, h1.ϵ1, h2.ϵ1),
        _sub(mode, h1.ϵ2, h2.ϵ2),
        mapϵ12((x, y) -> _sub(mode, x, y), h1, h2),
    )
end

@inline Base.:(-)(h::HyperDual) = _neg_hyperdual(_ieee_mode(h), h)
@inline Base.:(+)(h::HyperDual) = h

@inline Base.:+(h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    _add_hyperdual(IEEEArithmetic{S}(), h1, h2)
@inline Base.:+(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = +(promote(h1, h2)...)
@inline Base.:+(h::HyperDual{N1, N2, T, S}, r::Real) where {N1, N2, T, S} =
    hyperdual(Val(S), h.v + r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.:+(r::Real, h::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    hyperdual(Val(S), r + h.v, h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:-(h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    _sub_hyperdual(IEEEArithmetic{S}(), h1, h2)
@inline Base.:-(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = -(promote(h1, h2)...)
@inline Base.:-(h::HyperDual{N1, N2, T, S}, r::Real) where {N1, N2, T, S} =
    hyperdual(Val(S), h.v - r, h.ϵ1, h.ϵ2, h.ϵ12)
@inline function Base.:-(r::Real, h::HyperDual{N1, N2}) where {N1, N2}
    mode = _ieee_mode(h)
    return hyperdual(mode, r - h.v, _neg(mode, h.ϵ1), _neg(mode, h.ϵ2), mapϵ12(x -> _neg(mode, x), h))
end

@inline function Base.:*(h::HyperDual{N1, N2}, r::Real) where {N1, N2}
    mode = _ieee_mode(h)
    return hyperdual(mode, h.v * r, _mul(mode, h.ϵ1, r), _mul(mode, h.ϵ2, r), mapϵ12(ϵ -> _mul(mode, ϵ, r), h))
end
@inline function Base.:/(h::HyperDual{N1, N2}, r::Real) where {N1, N2}
    mode = _ieee_mode(h)
    return hyperdual(mode, h.v / r, _div(mode, h.ϵ1, r), _div(mode, h.ϵ2, r), mapϵ12(ϵ -> _div(mode, ϵ, r), h))
end
@inline function Base.:(*)(r::Real, h::HyperDual{N1, N2}) where {N1, N2}
    mode = _ieee_mode(h)
    return hyperdual(mode, r * h.v, _mul(mode, r, h.ϵ1), _mul(mode, r, h.ϵ2), mapϵ12(ϵ -> _mul(mode, r, ϵ), h))
end

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
    return hyperdual(mode, f, ϵ1, ϵ2, ntuple(g, Val(N1)))
end

@inline Base.:(/)(r::Real, h::HyperDual) = r * inv(h)
@inline Base.:(/)(h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    _div_hyperdual(IEEEArithmetic{S}(), h1, h2)
@inline Base.:(/)(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = /(promote(h1, h2)...)

@inline function Base.muladd(x::HyperDual{N1, N2, T, S}, y::Real, z::HyperDual{N1, N2, T, S}) where {N1, N2, T, S}
    mode = IEEEArithmetic{S}()
    return hyperdual(
        mode,
        muladd(x.v, y, z.v),
        _muladd(mode, y, x.ϵ1, z.ϵ1),
        _muladd(mode, y, x.ϵ2, z.ϵ2),
        ntuple(i -> _muladd(mode, y, x.ϵ12[i], z.ϵ12[i]), Val(N1))
    )
end
@inline function Base.muladd(x::Real, y::HyperDual{N1, N2, T, S}, z::HyperDual{N1, N2, T, S}) where {N1, N2, T, S}
    mode = IEEEArithmetic{S}()
    return hyperdual(
        mode,
        muladd(x, y.v, z.v),
        _muladd(mode, x, y.ϵ1, z.ϵ1),
        _muladd(mode, x, y.ϵ2, z.ϵ2),
        ntuple(i -> _muladd(mode, x, y.ϵ12[i], z.ϵ12[i]), Val(N1)),
    )
end
@inline function Base.muladd(x::HyperDual{N1, N2, T, S}, y::Real, z::Real) where {N1, N2, T, S}
    mode = IEEEArithmetic{S}()
    return hyperdual(
        mode,
        muladd(x.v, y, z),
        _mul(mode, x.ϵ1, y),
        _mul(mode, x.ϵ2, y),
        ntuple(i -> _mul(mode, x.ϵ12[i], y), Val(N1)),
    )
end
@inline function Base.muladd(x::Real, y::HyperDual{N1, N2, T, S}, z::Real) where {N1, N2, T, S}
    mode = IEEEArithmetic{S}()
    return hyperdual(
        mode,
        muladd(x, y.v, z),
        _mul(mode, y.ϵ1, x),
        _mul(mode, y.ϵ2, x),
        ntuple(i -> _mul(mode, y.ϵ12[i], x), Val(N1)),
    )
end
@inline Base.muladd(x::Real, y::Real, z::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    hyperdual(Val(S), muladd(x, y, z.v), z.ϵ1, z.ϵ2, z.ϵ12)
# All-HyperDual multiplicands are ambiguous between the mixed methods above, so
# resolve them explicitly via the general product-then-sum.
@inline Base.muladd(x::HyperDual{N1, N2, T, S}, y::HyperDual{N1, N2, T, S}, z::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    x * y + z
@inline Base.muladd(x::HyperDual{N1, N2, T, S}, y::HyperDual{N1, N2, T, S}, z::Real) where {N1, N2, T, S} =
    x * y + z
# Same-shape duals with mismatched T or S in any combination of slots are
# otherwise dispatch-ambiguous among the scalar-slot methods above (a Real
# slot matches a dual of a different parameterization). Enumerate every
# pattern explicitly and promote to a common type; different-shape duals in
# Real slots keep their nesting semantics through the scalar-slot methods.
@inline Base.muladd(x::HyperDual{N1, N2, T1, S1}, y::HyperDual{N1, N2, T2, S2}, z::HyperDual{N1, N2, T1, S1}) where {N1, N2, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::HyperDual{N1, N2, T2, S2}, y::HyperDual{N1, N2, T1, S1}, z::HyperDual{N1, N2, T1, S1}) where {N1, N2, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::HyperDual{N1, N2, T1, S1}, y::HyperDual{N1, N2, T1, S1}, z::HyperDual{N1, N2, T2, S2}) where {N1, N2, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::HyperDual{N1, N2, T1, S1}, y::HyperDual{N1, N2, T2, S2}, z::HyperDual{N1, N2, T3, S3}) where {N1, N2, T1, T2, T3, S1, S2, S3} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::HyperDual{N1, N2, T1, S1}, y::HyperDual{N1, N2, T2, S2}, z::Real) where {N1, N2, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::HyperDual{N1, N2, T1, S1}, y::Real, z::HyperDual{N1, N2, T2, S2}) where {N1, N2, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Real, y::HyperDual{N1, N2, T1, S1}, z::HyperDual{N1, N2, T2, S2}) where {N1, N2, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)

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
    @eval @inline function Base.$f(h::HyperDual{N1, N2, T, S}) where {N1, N2, T, S}
        fv = $f(h.v)
        return HyperDual{N1, N2, typeof(fv), S}(fv)
    end
    @eval @inline Base.$f(::Type{I}, h::HyperDual) where {I <: Real} = $f(I, h.v)
end
@inline function Base.round(h::HyperDual{N1, N2, T, S}, r::RoundingMode = RoundNearest) where {N1, N2, T, S}
    fv = round(h.v, r)
    return HyperDual{N1, N2, typeof(fv), S}(fv)
end
@inline Base.round(::Type{I}, h::HyperDual) where {I <: Real} = round(I, h.v)

# mod/rem: derivative w.r.t. x is 1 almost everywhere, so ϵ parts pass through.
# The two-dual methods use mod(x, y) = x - fld(x, y) * y with fld constant a.e.
@inline Base.mod(h::HyperDual{N1, N2, T, S}, r::Real) where {N1, N2, T, S} =
    hyperdual(Val(S), mod(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.rem(h::HyperDual{N1, N2, T, S}, r::Real) where {N1, N2, T, S} =
    hyperdual(Val(S), rem(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.mod(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1 - fld(h1.v, h2.v) * h2
@inline Base.rem(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = h1 - div(h1.v, h2.v) * h2
@inline Base.mod2pi(h::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    hyperdual(Val(S), mod2pi(h.v), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.rem2pi(h::HyperDual{N1, N2, T, S}, r::RoundingMode) where {N1, N2, T, S} =
    hyperdual(Val(S), rem2pi(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.:(*)(h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}) where {N1, N2} = *(promote(h1, h2)...)
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
    return hyperdual(mode, r, ϵ1, ϵ2, ϵ12)
end
@inline Base.:(*)(h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    _mul_hyperdual(IEEEArithmetic{S}(), h1, h2)
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
@inline Base.:(^)(h::HyperDual, n::Integer) = _pow_hyperdual(_ieee_mode(h), h, n)
# Disambiguate against Base.^(::Number, ::Rational): use the general real-power rule.
@inline Base.:(^)(h::HyperDual, r::Rational) = h^float(r)

# `@fastmath` rewrites user arithmetic to these functions. The methods below
# select fast arithmetic for both the primal and all derivative components.
@inline Base.FastMath.sub_fast(h::HyperDual) = _neg_hyperdual(_fast_mode(h), h)

@inline Base.FastMath.add_fast(
    h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}
) where {N1, N2, T, S} = _add_hyperdual(FastArithmetic{S}(), h1, h2)
@inline Base.FastMath.add_fast(
    h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}
) where {N1, N2} = Base.FastMath.add_fast(promote(h1, h2)...)
@inline Base.FastMath.add_fast(h::HyperDual{N1, N2, T, S}, r::Real) where {N1, N2, T, S} =
    hyperdual(Val(S), Base.FastMath.add_fast(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline Base.FastMath.add_fast(r::Real, h::HyperDual{N1, N2, T, S}) where {N1, N2, T, S} =
    hyperdual(Val(S), Base.FastMath.add_fast(r, h.v), h.ϵ1, h.ϵ2, h.ϵ12)

@inline Base.FastMath.sub_fast(
    h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}
) where {N1, N2, T, S} = _sub_hyperdual(FastArithmetic{S}(), h1, h2)
@inline Base.FastMath.sub_fast(
    h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}
) where {N1, N2} = Base.FastMath.sub_fast(promote(h1, h2)...)
@inline Base.FastMath.sub_fast(h::HyperDual{N1, N2, T, S}, r::Real) where {N1, N2, T, S} =
    hyperdual(Val(S), Base.FastMath.sub_fast(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
@inline function Base.FastMath.sub_fast(r::Real, h::HyperDual{N1, N2}) where {N1, N2}
    mode = _fast_mode(h)
    return hyperdual(
        mode,
        Base.FastMath.sub_fast(r, h.v),
        _neg(mode, h.ϵ1),
        _neg(mode, h.ϵ2),
        mapϵ12(x -> _neg(mode, x), h),
    )
end

@inline Base.FastMath.mul_fast(
    h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}
) where {N1, N2, T, S} = _mul_hyperdual(FastArithmetic{S}(), h1, h2)
@inline Base.FastMath.mul_fast(
    h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}
) where {N1, N2} = Base.FastMath.mul_fast(promote(h1, h2)...)
@inline function Base.FastMath.mul_fast(h::HyperDual{N1, N2}, r::Real) where {N1, N2}
    mode = _fast_mode(h)
    return hyperdual(
        mode,
        Base.FastMath.mul_fast(h.v, r),
        _mul(mode, h.ϵ1, r),
        _mul(mode, h.ϵ2, r),
        mapϵ12(x -> _mul(mode, x, r), h),
    )
end
@inline function Base.FastMath.mul_fast(r::Real, h::HyperDual{N1, N2}) where {N1, N2}
    mode = _fast_mode(h)
    return hyperdual(
        mode,
        Base.FastMath.mul_fast(r, h.v),
        _mul(mode, r, h.ϵ1),
        _mul(mode, r, h.ϵ2),
        mapϵ12(x -> _mul(mode, r, x), h),
    )
end

@inline Base.FastMath.div_fast(
    h1::HyperDual{N1, N2, T, S}, h2::HyperDual{N1, N2, T, S}
) where {N1, N2, T, S} = _div_hyperdual(FastArithmetic{S}(), h1, h2)
@inline Base.FastMath.div_fast(
    h1::HyperDual{N1, N2}, h2::HyperDual{N1, N2}
) where {N1, N2} = Base.FastMath.div_fast(promote(h1, h2)...)
@inline function Base.FastMath.div_fast(h::HyperDual{N1, N2}, r::Real) where {N1, N2}
    mode = _fast_mode(h)
    return hyperdual(
        mode,
        Base.FastMath.div_fast(h.v, r),
        _div(mode, h.ϵ1, r),
        _div(mode, h.ϵ2, r),
        mapϵ12(x -> _div(mode, x, r), h),
    )
end
@inline Base.FastMath.div_fast(r::Real, h::HyperDual) =
    _rdiv_hyperdual(_fast_mode(h), r, h)

@inline Base.FastMath.pow_fast(h::HyperDual, n::Integer) =
    _pow_hyperdual(_fast_mode(h), h, n)
@inline Base.FastMath.pow_fast(h::HyperDual, ::Val{p}) where {p} =
    _pow_hyperdual(_fast_mode(h), h, p)

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

Measured tradeoff: jets tend to win for small inputs where register pressure
dominates, but the ragged triangle rows (N, N-1, …, 1) vectorize poorly as
they grow, and a full-vector HyperDual{N,N} becomes register-exact once N
reaches the SIMD width (N = 4 on AVX2, N = 8 on AVX-512), winning again from
there. The crossover is function- and CPU-dependent, so jets are never
selected by default: opt in with `HessianConfig(x, Jet)`, or let
ChunkPicker.jl benchmark the candidates.

Like HyperDual, all arithmetic is parameterized over an `ArithmeticMode` and
the `S` backend flag: `@fastmath` in the differentiated function propagates
through the derivative components (see the `Base.FastMath` methods at the
bottom and the fast rule variants generated in rules.jl), and `S = true`
(`HessianConfig(x, Jet; simd = true)`) forces SIMD.Vec arithmetic on the
gradient and triangle tuples via the shared `_add`/`_mul`/… backend ops in
hyperdual.jl.
=#

"""
    Jet{N, M, T, S} <: Real

Symmetric second-order number: a value, a gradient of length `N`, and the
upper triangle of the Hessian (`M = N(N+1)/2` entries, row-major). Computes a
whole Hessian in a single function evaluation; select it with
`HessianConfig(x, Jet; simd)`. `S` selects the arithmetic backend as for
`HyperDual`: `false` = plain tuple ops (works for any `T`), `true` =
SIMD.Vec-forced ops for `Float32`/`Float64` components.
"""
struct Jet{N, M, T, S} <: Real
    v::T
    g::NTuple{N, T}
    h::NTuple{M, T}
end

const SecondOrderNumber = Union{HyperDual, Jet}

@inline nupper(N::Int) = N * (N + 1) ÷ 2

# Internal constructors preserving the backend flag of the operands.
@inline jet(mode::ArithmeticMode, v, g, h) = jet(Val(_simd(mode)), v, g, h)
@inline jet(::Val{S}, v::T, g::ϵT{N, T}, h::ϵT{M, T}) where {S, N, M, T} =
    Jet{N, M, T, S}(v, g, h)
@inline function jet(::Val{S}, v::T1, g::ϵT{N, T2}, h::ϵT{M, T3}) where {S, N, M, T1, T2, T3}
    T = promote_type(T1, T2, T3)
    return Jet{N, M, T, S}(T(v), to_ϵ(ϵT{N, T}, g), to_ϵ(ϵT{M, T}, h))
end

@inline _ieee_mode(::Jet{N, M, T, S}) where {N, M, T, S} = IEEEArithmetic{S}()
@inline _fast_mode(::Jet{N, M, T, S}) where {N, M, T, S} = FastArithmetic{S}()

# Public constructors default to the tuple backend.
function Jet(v::T1, g::ϵT{N, T2}, h::ϵT{M, T3}) where {N, M, T1, T2, T3}
    M == nupper(N) || throw(ArgumentError(lazy"h must hold the N(N+1)/2 = $(nupper(N)) upper-triangle entries, got $M"))
    return jet(Val(false), v, g, h)
end
Jet{N, M}(v::T) where {N, M, T} =
    Jet{N, M, T, false}(v, ntuple(_ -> zero(T), Val(N)), ntuple(_ -> zero(T), Val(M)))
Jet{N, M, T}(v) where {N, M, T} = Jet{N, M}(T(v))
Jet{N, M, T}(v::Jet{N, M, T}) where {N, M, T} = v
Jet{N, M, T}(v::Jet{N, M, <:Any, S}) where {N, M, T, S} = convert(Jet{N, M, T, S}, v)
Jet{N, M}(v::Jet{N, M}) where {N, M} = v

Jet{N, M, T, S}(v::Real) where {N, M, T, S} = jet(
    Val(S), T(v),
    ntuple(_ -> zero(T), Val(N)), ntuple(_ -> zero(T), Val(M)),
)
Jet{N, M, T, S}(v::Jet{N, M, T, S}) where {N, M, T, S} = v
Jet{N, M, T, S}(v::Jet{N, M}) where {N, M, T, S} = convert(Jet{N, M, T, S}, v)

# Disambiguate against Base's numeric conversion constructors, mirroring
# HyperDual (see hyperdual.jl).
for R in (:Complex, :AbstractChar, :(Base.TwicePrecision))
    @eval Jet{N, M, T}(v::$R) where {N, M, T} = Jet{N, M}(T(_scalar(v)))
    @eval Jet{N, M}(v::$R) where {N, M} = Jet{N, M}(_scalar(v))
    @eval Jet{N, M, T, S}(v::$R) where {N, M, T, S} = Jet{N, M, T, S}(T(_scalar(v)))
end

@inline value(x::Jet) = x.v

# Mixed backends promote to the tuple backend.
Base.promote_rule(::Type{Jet{N, M, T1, S1}}, ::Type{Jet{N, M, T2, S2}}) where {N, M, T1, T2, S1, S2} =
    Jet{N, M, promote_type(T1, T2), S1 && S2}
Base.promote_rule(::Type{Jet{N, M, T1, S}}, ::Type{T2}) where {N, M, T1, S, T2 <: Real} =
    Jet{N, M, promote_type(T1, T2), S}
Base.convert(::Type{Jet{N, M, T1, S}}, j::Jet{N, M, T2, S2}) where {N, M, T1, T2, S, S2} =
    Jet{N, M, T1, S}(T1(j.v), to_ϵ(ϵT{N, T1}, j.g), to_ϵ(ϵT{M, T1}, j.h))
Base.convert(::Type{Jet{N, M, T, S}}, x::Real) where {N, M, T, S} = Jet{N, M, T, S}(T(x))

function Base.show(io::IO, j::Jet)
    print(io, j.v, " + ", Tuple(j.g), "ε + ", Tuple(j.h), "ε² (upper)")
    return
end

Base.one(::Type{Jet{N, M, T, S}}) where {N, M, T, S} = Jet{N, M, T, S}(one(T))
Base.zero(::Type{Jet{N, M, T, S}}) where {N, M, T, S} = Jet{N, M, T, S}(zero(T))
Base.one(::Type{Jet{N, M, T}}) where {N, M, T} = one(Jet{N, M, T, false})
Base.zero(::Type{Jet{N, M, T}}) where {N, M, T} = zero(Jet{N, M, T, false})
Base.one(::Jet{N, M, T, S}) where {N, M, T, S} = one(Jet{N, M, T, S})
Base.zero(::Jet{N, M, T, S}) where {N, M, T, S} = zero(Jet{N, M, T, S})
Base.float(j::Jet{N, M, T, S}) where {N, M, T, S} = convert(Jet{N, M, float(T), S}, j)

# Upper-triangle helpers, fully unrolled with literal indices so LLVM sees
# straight-line tuple code.
@inline @generated function symouter(mode::ArithmeticMode, a::NTuple{N, T}, b::NTuple{N, T}) where {N, T}
    ex = Expr(:tuple)
    for i in 1:N, j in i:N
        push!(ex.args, :(_muladd(mode, a[$i], b[$j], _mul(mode, b[$i], a[$j]))))
    end
    return ex
end

@inline @generated function halfouter(mode::ArithmeticMode, a::NTuple{N, T}) where {N, T}
    ex = Expr(:tuple)
    for i in 1:N, j in i:N
        push!(ex.args, :(_mul(mode, a[$i], a[$j])))
    end
    return ex
end

"""
    chain_rule_jet(j::Jet, f, f′, f′′)

Apply the chain rule to `j` given scalar primal `f`, first derivative `f′`,
and second derivative `f′′`.
"""
@inline chain_rule_jet(j::Jet, f, f′, f′′) = chain_rule_jet(_ieee_mode(j), j, f, f′, f′′)

@inline function chain_rule_jet(mode::ArithmeticMode, j::Jet{N, M, T}, f, f′, f′′) where {N, M, T}
    return jet(
        mode,
        f,
        _mul(mode, j.g, f′),
        _muladd(mode, f′, j.h, _mul(mode, halfouter(mode, j.g), f′′)),
    )
end

"""
    chain_rule_jet(jx::Jet, jy::Jet, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)

Apply the chain rule to a scalar function of two `Jet` inputs given first and
second partial derivatives.
"""
@inline chain_rule_jet(jx::Jet, jy::Jet, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ) =
    chain_rule_jet(_ieee_mode(jx), jx, jy, f, fₓ, fᵧ, fₓₓ, fₓᵧ, fᵧᵧ)

@inline function chain_rule_jet(
        mode::ArithmeticMode,
        jx::Jet{N, M, T},
        jy::Jet{N, M, T},
        f,
        fₓ,
        fᵧ,
        fₓₓ,
        fₓᵧ,
        fᵧᵧ,
    ) where {N, M, T}
    g = _muladd(mode, fₓ, jx.g, _mul(mode, jy.g, fᵧ))
    h = _muladd(mode, fₓ, jx.h, _mul(mode, jy.h, fᵧ))
    h = _muladd(mode, fₓₓ, halfouter(mode, jx.g), h)
    h = _muladd(mode, fₓᵧ, symouter(mode, jx.g, jy.g), h)
    h = _muladd(mode, fᵧᵧ, halfouter(mode, jy.g), h)
    return jet(mode, f, g, h)
end

# Arithmetic
@inline function _neg_jet(mode::ArithmeticMode, j::Jet{N, M, T}) where {N, M, T}
    return jet(mode, _neg(mode, j.v), _neg(mode, j.g), _neg(mode, j.h))
end

@inline function _add_jet(mode::ArithmeticMode, a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T}
    return jet(mode, _add(mode, a.v, b.v), _add(mode, a.g, b.g), _add(mode, a.h, b.h))
end

@inline function _sub_jet(mode::ArithmeticMode, a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T}
    return jet(mode, _sub(mode, a.v, b.v), _sub(mode, a.g, b.g), _sub(mode, a.h, b.h))
end

@inline function _mul_jet(mode::ArithmeticMode, a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T}
    return jet(
        mode,
        _mul(mode, a.v, b.v),
        _muladd(mode, a.v, b.g, _mul(mode, a.g, b.v)),
        _muladd(mode, a.v, b.h, _muladd(mode, b.v, a.h, symouter(mode, a.g, b.g))),
    )
end

# Dedicated division rule, mirroring _div_hyperdual:
# f = x/y, fₓ = 1/y, fᵧ = -f/y, fₓₓ = 0, fₓᵧ = -1/y², fᵧᵧ = 2f/y²
@inline function _div_jet(mode::ArithmeticMode, a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T}
    x, y = a.v, b.v
    invy = _inv(mode, y)
    f = _mul(mode, x, invy)
    fᵧ = _neg(mode, _mul(mode, f, invy))
    fₓᵧ = _mul(mode, _neg(mode, invy), invy)
    fᵧᵧ = _mul(mode, _mul(mode, _neg(mode, 2), fᵧ), invy)
    return chain_rule_jet(mode, a, b, f, invy, fᵧ, zero(invy), fₓᵧ, fᵧᵧ)
end

# f(y) = r/y: f′ = -f/y, f′′ = 2f/y²
@inline function _rdiv_jet(mode::ArithmeticMode, r::Real, j::Jet{N, M}) where {N, M}
    r, y = promote(r, j.v)
    invy = _inv(mode, y)
    f = _mul(mode, r, invy)
    f′ = _neg(mode, _mul(mode, f, invy))
    f′′ = _mul(mode, _mul(mode, _neg(mode, 2), f′), invy)
    return chain_rule_jet(mode, j, f, f′, f′′)
end

@inline Base.:+(j::Jet) = j
@inline Base.:-(j::Jet) = _neg_jet(_ieee_mode(j), j)

@inline Base.:+(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _add_jet(IEEEArithmetic{S}(), a, b)
@inline Base.:+(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = +(promote(a, b)...)
@inline Base.:+(j::Jet{N, M, T, S}, r::Real) where {N, M, T, S} =
    jet(Val(S), j.v + r, j.g, j.h)
@inline Base.:+(r::Real, j::Jet{N, M, T, S}) where {N, M, T, S} =
    jet(Val(S), r + j.v, j.g, j.h)

@inline Base.:-(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _sub_jet(IEEEArithmetic{S}(), a, b)
@inline Base.:-(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = -(promote(a, b)...)
@inline Base.:-(j::Jet{N, M, T, S}, r::Real) where {N, M, T, S} =
    jet(Val(S), j.v - r, j.g, j.h)
@inline function Base.:-(r::Real, j::Jet{N, M}) where {N, M}
    mode = _ieee_mode(j)
    return jet(mode, r - j.v, _neg(mode, j.g), _neg(mode, j.h))
end

@inline Base.:*(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _mul_jet(IEEEArithmetic{S}(), a, b)
@inline Base.:*(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = *(promote(a, b)...)
@inline function Base.:*(j::Jet{N, M}, r::Real) where {N, M}
    mode = _ieee_mode(j)
    return jet(mode, j.v * r, _mul(mode, j.g, r), _mul(mode, j.h, r))
end
@inline function Base.:*(r::Real, j::Jet{N, M}) where {N, M}
    mode = _ieee_mode(j)
    return jet(mode, r * j.v, _mul(mode, r, j.g), _mul(mode, r, j.h))
end

@inline Base.:/(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _div_jet(IEEEArithmetic{S}(), a, b)
@inline Base.:/(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = /(promote(a, b)...)
@inline function Base.:/(j::Jet{N, M}, r::Real) where {N, M}
    mode = _ieee_mode(j)
    return jet(mode, j.v / r, _div(mode, j.g, r), _div(mode, j.h, r))
end
@inline Base.:/(r::Real, j::Jet) = _rdiv_jet(_ieee_mode(j), r, j)

# Integer powers, mirroring _pow_hyperdual: monomial derivatives directly
# (NaN-free at x = 0, cheaper than repeated multiplication), and resolving the
# ambiguity with Base.^(::Number, ::Integer).
@inline function _pow_jet(mode::ArithmeticMode, j::Jet, n::Integer)
    x = j.v
    if n == 0
        return one(j)
    elseif n == 1
        return j
    elseif n == 2
        return chain_rule_jet(mode, j, _mul(mode, x, x), _mul(mode, 2, x), _mul(mode, 2, one(x)))
    elseif n == 3
        x2 = _mul(mode, x, x)
        return chain_rule_jet(mode, j, _mul(mode, x2, x), _mul(mode, 3, x2), _mul(mode, 6, x))
    else
        p = _pow(mode, x, n - 2)
        xp = _mul(mode, x, p)
        f = _mul(mode, x, xp)
        f′ = _mul(mode, n, xp)
        f′′ = _mul(mode, n * (n - 1), p)
        return chain_rule_jet(mode, j, f, f′, f′′)
    end
end
@inline Base.:(^)(j::Jet, n::Integer) = _pow_jet(_ieee_mode(j), j, n)
@inline Base.:(^)(j::Jet, r::Rational) = j^float(r)
@inline Base.literal_pow(::typeof(^), j::Jet, ::Val{0}) = one(typeof(j))
@inline Base.literal_pow(::typeof(^), j::Jet, ::Val{1}) = j
@inline Base.literal_pow(::typeof(^), j::Jet, ::Val{p}) where {p} = j^p

@inline function Base.muladd(x::Jet{N, M, T, S}, y::Real, z::Jet{N, M, T, S}) where {N, M, T, S}
    mode = IEEEArithmetic{S}()
    return jet(
        mode,
        muladd(x.v, y, z.v),
        _muladd(mode, y, x.g, z.g),
        _muladd(mode, y, x.h, z.h),
    )
end
@inline Base.muladd(x::Real, y::Jet{N, M, T, S}, z::Jet{N, M, T, S}) where {N, M, T, S} =
    muladd(y, x, z)
@inline function Base.muladd(x::Jet{N, M, T, S}, y::Real, z::Real) where {N, M, T, S}
    mode = IEEEArithmetic{S}()
    return jet(mode, muladd(x.v, y, z), _mul(mode, x.g, y), _mul(mode, x.h, y))
end
@inline Base.muladd(x::Real, y::Jet{N, M, T, S}, z::Real) where {N, M, T, S} = muladd(y, x, z)
@inline Base.muladd(x::Real, y::Real, z::Jet{N, M, T, S}) where {N, M, T, S} =
    jet(Val(S), muladd(x, y, z.v), z.g, z.h)
# All-Jet multiplicands are ambiguous between the mixed methods above, so
# resolve them explicitly via the general product-then-sum.
@inline Base.muladd(x::Jet{N, M, T, S}, y::Jet{N, M, T, S}, z::Jet{N, M, T, S}) where {N, M, T, S} =
    x * y + z
@inline Base.muladd(x::Jet{N, M, T, S}, y::Jet{N, M, T, S}, z::Real) where {N, M, T, S} =
    x * y + z
# Same-shape jets with mismatched T or S in any combination of slots are
# otherwise dispatch-ambiguous among the scalar-slot methods above (a Real
# slot matches a jet of a different parameterization). Enumerate every
# pattern explicitly and promote to a common type, mirroring HyperDual.
@inline Base.muladd(x::Jet{N, M, T1, S1}, y::Jet{N, M, T2, S2}, z::Jet{N, M, T1, S1}) where {N, M, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Jet{N, M, T2, S2}, y::Jet{N, M, T1, S1}, z::Jet{N, M, T1, S1}) where {N, M, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Jet{N, M, T1, S1}, y::Jet{N, M, T1, S1}, z::Jet{N, M, T2, S2}) where {N, M, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Jet{N, M, T1, S1}, y::Jet{N, M, T2, S2}, z::Jet{N, M, T3, S3}) where {N, M, T1, T2, T3, S1, S2, S3} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Jet{N, M, T1, S1}, y::Jet{N, M, T2, S2}, z::Real) where {N, M, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Jet{N, M, T1, S1}, y::Real, z::Jet{N, M, T2, S2}) where {N, M, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)
@inline Base.muladd(x::Real, y::Jet{N, M, T1, S1}, z::Jet{N, M, T2, S2}) where {N, M, T1, T2, S1, S2} =
    muladd(promote(x, y, z)...)

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
    @eval @inline function Base.$f(j::Jet{N, M, T, S}) where {N, M, T, S}
        fv = $f(j.v)
        return Jet{N, M, typeof(fv), S}(fv)
    end
    @eval @inline Base.$f(::Type{I}, j::Jet) where {I <: Real} = $f(I, j.v)
end
@inline function Base.round(j::Jet{N, M, T, S}, r::RoundingMode = RoundNearest) where {N, M, T, S}
    fv = round(j.v, r)
    return Jet{N, M, typeof(fv), S}(fv)
end
@inline Base.round(::Type{I}, j::Jet) where {I <: Real} = round(I, j.v)

# mod/rem: derivative w.r.t. x is 1 almost everywhere.
@inline Base.mod(j::Jet{N, M, T, S}, r::Real) where {N, M, T, S} =
    jet(Val(S), mod(j.v, r), j.g, j.h)
@inline Base.rem(j::Jet{N, M, T, S}, r::Real) where {N, M, T, S} =
    jet(Val(S), rem(j.v, r), j.g, j.h)
@inline Base.mod(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a - fld(a.v, b.v) * b
@inline Base.rem(a::Jet{N, M}, b::Jet{N, M}) where {N, M} = a - div(a.v, b.v) * b
@inline Base.mod2pi(j::Jet{N, M, T, S}) where {N, M, T, S} =
    jet(Val(S), mod2pi(j.v), j.g, j.h)
@inline Base.rem2pi(j::Jet{N, M, T, S}, r::RoundingMode) where {N, M, T, S} =
    jet(Val(S), rem2pi(j.v, r), j.g, j.h)

# `@fastmath` rewrites user arithmetic to these functions. As for HyperDual,
# the methods below select fast arithmetic for the primal and all derivative
# components; the fast variants of the derivative rules are generated in
# rules.jl.
@inline Base.FastMath.sub_fast(j::Jet) = _neg_jet(_fast_mode(j), j)

@inline Base.FastMath.add_fast(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _add_jet(FastArithmetic{S}(), a, b)
@inline Base.FastMath.add_fast(a::Jet{N, M}, b::Jet{N, M}) where {N, M} =
    Base.FastMath.add_fast(promote(a, b)...)
@inline Base.FastMath.add_fast(j::Jet{N, M, T, S}, r::Real) where {N, M, T, S} =
    jet(Val(S), Base.FastMath.add_fast(j.v, r), j.g, j.h)
@inline Base.FastMath.add_fast(r::Real, j::Jet{N, M, T, S}) where {N, M, T, S} =
    jet(Val(S), Base.FastMath.add_fast(r, j.v), j.g, j.h)

@inline Base.FastMath.sub_fast(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _sub_jet(FastArithmetic{S}(), a, b)
@inline Base.FastMath.sub_fast(a::Jet{N, M}, b::Jet{N, M}) where {N, M} =
    Base.FastMath.sub_fast(promote(a, b)...)
@inline Base.FastMath.sub_fast(j::Jet{N, M, T, S}, r::Real) where {N, M, T, S} =
    jet(Val(S), Base.FastMath.sub_fast(j.v, r), j.g, j.h)
@inline function Base.FastMath.sub_fast(r::Real, j::Jet{N, M}) where {N, M}
    mode = _fast_mode(j)
    return jet(
        mode,
        Base.FastMath.sub_fast(r, j.v),
        _neg(mode, j.g),
        _neg(mode, j.h),
    )
end

@inline Base.FastMath.mul_fast(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _mul_jet(FastArithmetic{S}(), a, b)
@inline Base.FastMath.mul_fast(a::Jet{N, M}, b::Jet{N, M}) where {N, M} =
    Base.FastMath.mul_fast(promote(a, b)...)
@inline function Base.FastMath.mul_fast(j::Jet{N, M}, r::Real) where {N, M}
    mode = _fast_mode(j)
    return jet(
        mode,
        Base.FastMath.mul_fast(j.v, r),
        _mul(mode, j.g, r),
        _mul(mode, j.h, r),
    )
end
@inline function Base.FastMath.mul_fast(r::Real, j::Jet{N, M}) where {N, M}
    mode = _fast_mode(j)
    return jet(
        mode,
        Base.FastMath.mul_fast(r, j.v),
        _mul(mode, r, j.g),
        _mul(mode, r, j.h),
    )
end

@inline Base.FastMath.div_fast(a::Jet{N, M, T, S}, b::Jet{N, M, T, S}) where {N, M, T, S} =
    _div_jet(FastArithmetic{S}(), a, b)
@inline Base.FastMath.div_fast(a::Jet{N, M}, b::Jet{N, M}) where {N, M} =
    Base.FastMath.div_fast(promote(a, b)...)
@inline function Base.FastMath.div_fast(j::Jet{N, M}, r::Real) where {N, M}
    mode = _fast_mode(j)
    return jet(
        mode,
        Base.FastMath.div_fast(j.v, r),
        _div(mode, j.g, r),
        _div(mode, j.h, r),
    )
end
@inline Base.FastMath.div_fast(r::Real, j::Jet) = _rdiv_jet(_fast_mode(j), r, j)

@inline Base.FastMath.pow_fast(j::Jet, n::Integer) = _pow_jet(_fast_mode(j), j, n)
@inline Base.FastMath.pow_fast(j::Jet, ::Val{p}) where {p} = _pow_jet(_fast_mode(j), j, p)

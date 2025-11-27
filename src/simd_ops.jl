# SIMD.Vec operations for HyperDual numbers

@inline zero_ϵ(::Type{Vec{N, T}}) where {N, T} = zero(Vec{N, T})
@inline zero_ϵ(::Vec{N, T}) where {N, T} = zero(Vec{N, T})
@inline to_ϵ(::Type{Vec{N, T}}, x) where {N, T} = convert(Vec{N, T}, x)
@inline convert_cross(::Type{Vec{N, T}}, xs::NTuple{M, Any}) where {N, M, T} =
    ntuple(i -> to_ϵ(Vec{N, T}, xs[i]), Val(M))

@inline ⊕(a::Vec, b::Vec) = a + b
@inline ⊟(a::Vec) = -a
@inline ⊙(a::Vec, r::Real) = a * r
@inline ⊙(r::Real, a::Vec) = r * a
@inline ⊘(a::Vec, r::Real) = a / r

@inline _muladd(a::Real, b::Vec{N, T}, c::Vec{N, T}) where {N, T} = muladd(a, b, c)
@inline _muladd(a::Vec{N, T}, b::Real, c::Vec{N, T}) where {N, T} = muladd(a, b, c)

unsafe_getindex(v::SIMD.Vec, i::SIMD.IntegerTypes) = SIMD.Intrinsics.extractelement(v.data, i - 1)
@inline ⊗(v1::Vec{N1, T}, v2::Vec{N2, T}) where {N1, N2, T} = ntuple(i -> unsafe_getindex(v1, i) * v2, Val(N1))

module HyperHessians

using SIMD: SIMD, Vec, vstore
using CommonSubexpressions: cse

include("rules.jl")

################
# Tagging
################

"""
    SmallTag{H}

Compact tag carrying only a small hash, used to prevent perturbation confusion
without embedding large function types in `HyperDual` signatures.
"""
struct SmallTag{H} end

smalltag(::Type{F}, ::Type{V}, ::Val{N}) where {F, V, N} = SmallTag{UInt(hash(F) ⊻ hash(V) ⊻ hash(N))}()
smalltag(f, ::Type{V}, ::Val{N}) where {V, N} = smalltag(typeof(f), V, Val(N))
smalltag(::Nothing, ::Type{V}, ::Val{N}) where {V, N} = SmallTag{UInt(0)}()

struct PerturbationConfusion{Expected, Observed} <: Exception end
Base.showerror(io::IO, ::PerturbationConfusion{E, O}) where {E, O} =
    print(io, "perturbation confusion: expected tag ", E, ", observed ", O)

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

# This is currently "artifically" restricted to "square" HyperDuals where
# ϵ1 and ϵ2 have the same length.
struct HyperDual{Tag, N, T} <: Real
    v::T
    ϵ1::Vec{N, T} # Vec{M,T}
    ϵ2::Vec{N, T}
    ϵ12::NTuple{N, Vec{N, T}} # NTuple{M, Vec{N,T}}
end
HyperDual{Tag}(v::T, ϵ1::Vec{N, T}, ϵ2::Vec{N, T}) where {Tag, N, T} =
    HyperDual{Tag, N, T}(v, ϵ1, ϵ2, ntuple(i -> zero(Vec{N, T}), Val(N)))
HyperDual{Tag}(v::T, ϵ1::Vec{N, T}, ϵ2::Vec{N, T}, ϵ12::NTuple{N, Vec{N, T}}) where {Tag, N, T} =
    HyperDual{Tag, N, T}(v, ϵ1, ϵ2, ϵ12)
HyperDual{Tag, N}(v::T) where {Tag, N, T} = HyperDual{Tag}(v, zero(Vec{N, T}), zero(Vec{N, T}))
HyperDual{Tag, N, T}(v) where {Tag, N, T} = HyperDual{Tag, N}(T(v))

function HyperDual{Tag}(v::T1, ϵ1::Vec{N, T2}, ϵ2::Vec{N, T2}, ϵ12::NTuple{N, Vec{N, T2}}) where {Tag, N, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual{Tag}(T(v), convert(Vec{N, T}, ϵ1), convert(Vec{N, T}, ϵ2), convert.(Vec{N, T}, ϵ12))
end

HyperDual(tag::Tag, v::T, ϵ1::Vec{N, T}, ϵ2::Vec{N, T}) where {Tag, N, T} =
    HyperDual{Tag}(v, ϵ1, ϵ2)
HyperDual(tag::Tag, v::T, ϵ1::Vec{N, T}, ϵ2::Vec{N, T}, ϵ12::NTuple{N, Vec{N, T}}) where {Tag, N, T} =
    HyperDual{Tag}(v, ϵ1, ϵ2, ϵ12)

@inline _check_tag(::Type{Tag}, ::Type{Tag}) where {Tag} = nothing
@inline function _check_tag(::Type{Expected}, ::Type{Observed}) where {Expected, Observed}
    throw(PerturbationConfusion{Expected, Observed}())
end
@inline _check_tag(expected, observed) = _check_tag(typeof(expected), typeof(observed))

@inline _check_same_tag(::HyperDual{Tag1}, ::HyperDual{Tag2}) where {Tag1, Tag2} =
    _check_tag(Tag1, Tag2)

Base.promote_rule(::Type{HyperDual{Tag1, N, T1}}, ::Type{HyperDual{Tag2, N, T2}}) where {Tag1, Tag2, N, T1, T2} =
    (_check_tag(Tag1, Tag2); HyperDual{Tag1, N, promote_type(T1, T2)})
Base.promote_rule(::Type{HyperDual{Tag, N, T1}}, ::Type{T2}) where {Tag, N, T1, T2 <: Real} = HyperDual{Tag, N, promote_type(T1, T2)}
Base.convert(::Type{HyperDual{Tag, N, T1}}, h::HyperDual{Tag, N, T2}) where {Tag, N, T1, T2} =
    HyperDual{Tag, N, T1}(T1(h.v), convert(Vec{N, T1}, h.ϵ1), convert(Vec{N, T1}, h.ϵ2), convert.(Vec{N, T1}, h.ϵ12))
Base.convert(::Type{HyperDual{Tag, N, T}}, x::Real) where {Tag, N, T} = HyperDual{Tag, N, T}(T(x))

Base.one(::Type{HyperDual{Tag, N, T}}) where {Tag, N, T} = HyperDual{Tag, N}(one(T))
Base.zero(::Type{HyperDual{Tag, N, T}}) where {Tag, N, T} = HyperDual{Tag, N}(zero(T))
Base.one(::HyperDual{Tag, N, T}) where {Tag, N, T} = one(HyperDual{Tag, N, T})
Base.zero(::HyperDual{Tag, N, T}) where {Tag, N, T} = zero(HyperDual{Tag, N, T})
Base.float(h::HyperDual{Tag, N, T}) where {Tag, N, T} = convert(HyperDual{Tag, N, float(T)}, h)

# Unary
@inline Base.:(-)(h::HyperDual{Tag, N}) where {Tag, N} = HyperDual{Tag}(-h.v, -h.ϵ1, -h.ϵ2, ntuple(i -> -h.ϵ12[i], Val(N)))
@inline Base.:(+)(h::HyperDual) = h

# Binary
for f in (:+, :-)
    @eval begin
        @inline Base.$f(h1::HyperDual{Tag, N, T}, h2::HyperDual{Tag, N, T}) where {Tag, N, T} =
            HyperDual{Tag}($f(h1.v, h2.v), $f(h1.ϵ1, h2.ϵ1), $f(h1.ϵ2, h2.ϵ2), ntuple(i -> $f(h1.ϵ12[i], h2.ϵ12[i]), Val(N)))
        @inline Base.$f(h1::HyperDual{Tag1, N, T1}, h2::HyperDual{Tag2, N, T2}) where {Tag1, Tag2, N, T1, T2} =
            (_check_same_tag(h1, h2); $f(promote(h1, h2)...))
        @inline Base.$f(h::HyperDual{Tag, N}, r::Real) where {Tag, N} =
            HyperDual{Tag}($f(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
        @inline Base.$f(r::Real, h::HyperDual{Tag, N}) where {Tag, N} =
            HyperDual{Tag}($f(r, h.v), h.ϵ1, h.ϵ2, h.ϵ12)
    end
end

for f in (:*, :/)
    @eval @inline Base.$f(h::HyperDual{Tag, N}, r::Real) where {Tag, N} =
        HyperDual{Tag}($f(h.v, r), $f(h.ϵ1, r), $f(h.ϵ2, r), ntuple(i -> $f(h.ϵ12[i], r), Val(N)))
    @eval @inline Base.$f(r::Real, h::HyperDual{Tag, N}) where {Tag, N} =
        HyperDual{Tag}($f(r, h.v), $f(r, h.ϵ1), $f(r, h.ϵ2), ntuple(i -> $f(r, h.ϵ12[i]), Val(N)))
end

@inline Base.:(/)(h1::HyperDual{Tag1, N, T1}, h2::HyperDual{Tag2, N, T2}) where {Tag1, Tag2, N, T1, T2} =
    (_check_same_tag(h1, h2); h1 * inv(h2))

# muladd: x*y + z
@inline Base.muladd(x::HyperDual{Tag, N}, y::Real, z::HyperDual{Tag, N}) where {Tag, N} = x * y + z
@inline Base.muladd(x::Real, y::HyperDual{Tag, N}, z::HyperDual{Tag, N}) where {Tag, N} = x * y + z
@inline Base.muladd(x::HyperDual{Tag, N}, y::HyperDual{Tag, N}, z::Real) where {Tag, N} = x * y + z
@inline Base.muladd(x::HyperDual{Tag, N}, y::Real, z::Real) where {Tag, N} = x * y + z
@inline Base.muladd(x::Real, y::HyperDual{Tag, N}, z::Real) where {Tag, N} = x * y + z
@inline Base.muladd(x::Real, y::Real, z::HyperDual{Tag, N}) where {Tag, N} = muladd(x, y, z.v) + z - z.v
@inline Base.muladd(x::HyperDual{Tag, N}, y::HyperDual{Tag, N}, z::HyperDual{Tag, N}) where {Tag, N} = x * y + z

# Get's rid of a bunch of boundserror throwing in the LLVM IR, not sure it matters for runtime performance...
unsafe_getindex(v::SIMD.Vec, i::SIMD.IntegerTypes) = SIMD.Intrinsics.extractelement(v.data, i - 1)
@inline ⊗(v1::Vec{N}, v2::Vec{N}) where {N} = ntuple(i -> unsafe_getindex(v1, i) * v2, Val(N))

@inline Base.:(*)(h1::HyperDual{Tag, N, T1}, h2::HyperDual{Tag, N, T2}) where {Tag, N, T1, T2} =
    *(promote(h1, h2)...)
@inline function Base.:(*)(h1::HyperDual{Tag1, N, T1}, h2::HyperDual{Tag2, N, T2}) where {Tag1, Tag2, N, T1, T2}
    _check_same_tag(h1, h2)
    return *(promote(h1, h2)...)
end
@inline function Base.:(*)(h1::HyperDual{Tag, N, T}, h2::HyperDual{Tag, N, T}) where {Tag, N, T}
    r = h1.v * h2.v
    ϵ1 = muladd(h1.v, h2.ϵ1, h1.ϵ1 * h2.v)
    ϵ2 = muladd(h1.v, h2.ϵ2, h1.ϵ2 * h2.v)
    ϵ12_1 = h1.ϵ1 ⊗ h2.ϵ2
    ϵ12_2 = h2.ϵ1 ⊗ h1.ϵ2
    ϵ12 = ntuple(i -> muladd(h1.v, h2.ϵ12[i], muladd(h1.ϵ12[i], h2.v, ϵ12_1[i] + ϵ12_2[i])), Val(N))
    return HyperDual{Tag}(r, ϵ1, ϵ2, ϵ12)
end
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{2}) = x * x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{3}) = x * x * x

for (f, f′, f′′) in DIFF_RULES
    expr = quote
        # Verify that the cse still works properly when changing this.
        v = $f(x)
        f′ = $f′
        f′′ = $f′′
        x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
        return HyperDual{Tag}(v, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i] * f′ + x23[i], Val(N)))
    end
    cse_expr = cse(expr; warn = false)
    @eval @inline function Base.$f(h::HyperDual{Tag, N, T}) where {Tag, N, T}
        x = h.v
        $cse_expr
    end
end

@inline function Base.sin(h::HyperDual{Tag, N}) where {Tag, N}
    s, c = sincos(h.v)
    f′, f′′ = c, -s
    x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
    return HyperDual{Tag}(s, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i] * f′ + x23[i], Val(N)))
end

@inline function Base.cos(h::HyperDual{Tag, N}) where {Tag, N}
    s, c = sincos(h.v)
    f′, f′′ = -s, -c
    x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
    return HyperDual{Tag}(c, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i] * f′ + x23[i], Val(N)))
end


##########
# Chunks #
##########

struct Chunk{N} end

function Chunk(input_length::Integer)
    return Chunk{pickchunksize(input_length)}()
end

Chunk(x::AbstractArray) = Chunk(length(x))

# TODO: More testing needed here.
function pickchunksize(input_length)
    input_length == 1 && return 1
    input_length == 2 && return 2
    3 <= input_length <= 4 && return 4 # 3 seems to be very bad for some reason
    return 8
end

chunksize(::Chunk{N}) where {N} = N::Int


##################
# Hessian Config #
##################

abstract type AbstractHessianConfig end

struct HessianConfig{Tag, D <: AbstractVector{<:HyperDual{Tag}}, S} <: AbstractHessianConfig
    duals::D
    seeds::S
end
(chunksize(cfg::HessianConfig)::Int) = length(cfg.seeds)
tagtype(::HessianConfig{Tag}) where {Tag} = Tag

function HessianConfig(f, x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    @assert 0 < N
    tag = smalltag(f, T, Val(N))
    Tag = typeof(tag)
    duals = similar(x, HyperDual{Tag, N, T}) # not Vector
    seeds = collect(construct_seeds(NTuple{N, T}))
    return HessianConfig{Tag, typeof(duals), typeof(seeds)}(duals, seeds)
end

# Tag checking is intentionally a no-op in hot paths; tags are baked into the
# dual types themselves, so mixing contexts will already error when combining
# duals with different tags. This avoids per-call hashing overhead.
@inline checktag(cfg::HessianConfig, f, x) = nothing

###########
# Seeding #
###########

@generated function single_seed(::Type{NTuple{N, T}}, ::Val{i}) where {N, T, i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(T)), :(zero(T))) for j in 1:N]...)
    return :(Vec($(ex)))
end

@generated construct_seeds(::Type{NTuple{N, T}}) where {N, T} =
    Expr(:tuple, [:(single_seed(NTuple{N, T}, Val{$i}())) for i in 1:N]...)

seed_epsilon_1(d::HyperDual{Tag, N, T}, ϵ1) where {Tag, N, T} = HyperDual{Tag, N, T}(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::HyperDual{Tag, N, T}, ϵ2) where {Tag, N, T} = HyperDual{Tag, N, T}(d.v, d.ϵ1, ϵ2, d.ϵ12)

function seed!(d::AbstractVector{<:HyperDual{Tag, N}}, x, seeds, block_i, block_j) where {Tag, N}
    d .= HyperDual{Tag, N}.(x)
    index_i = (block_i - 1) * N + 1
    index_j = (block_j - 1) * N + 1
    range_i = index_i:min(length(x), (index_i + N - 1))
    range_j = index_j:min(length(x), (index_j + N - 1))
    chunks_i = length(range_i)
    chunks_j = length(range_j)

    d[range_i] .= seed_epsilon_1.(view(d, range_i), view(seeds, 1:chunks_i))
    d[range_j] .= seed_epsilon_2.(view(d, range_j), view(seeds, 1:chunks_j))
    return d
end

# Hessian
@noinline check_scalar(x) =
    x isa Number || throw(error("expected a scalar to be returned from function passed to `hessian`"))

function extract_hessian!(H::AbstractMatrix, v::HyperDual)
    Base.require_one_based_indexing(H)
    @inbounds for i in 1:size(H, 2)
        H[:, i] .= Tuple(v.ϵ12[i])
    end
    return H
end

function symmetrize!(H::AbstractMatrix)
    Base.require_one_based_indexing(H)
    for i in 1:size(H, 1)
        for j in i:size(H, 2)
            H[j, i] = H[i, j]
        end
    end
    return H
end

function extract_hessian!(H::AbstractMatrix, v::HyperDual{Tag, N}, block_i::Int, block_j::Int) where {Tag, N}
    index_i = (block_i - 1) * N + 1
    index_j = (block_j - 1) * N + 1
    range_i = index_i:(index_i + N - 1)
    range_j = index_j:(index_j + N - 1)

    for (I, i) in enumerate(range_i)
        for (J, j) in enumerate(range_j)
            if checkbounds(Bool, H, i, j)
                H[i, j] = v.ϵ12[I][J]
            end
        end
    end
    return H
end


function extract_gradient!(G::AbstractVector, v::HyperDual{Tag, N}, block_i::Int) where {Tag, N}
    Base.require_one_based_indexing(G)
    index_i = (block_i - 1) * N + 1
    range_i = index_i:min(length(G), (index_i + N - 1))
    for (I, i) in enumerate(range_i)
        G[i] = v.ϵ1[I]
    end
    return G
end


###############
# Hessian API #
###############

# Scalar
function hessian(f, x::Real)
    tag = smalltag(f, typeof(x), Val(1))
    Tag = typeof(tag)
    dual = HyperDual{Tag}(x, Vec(one(x)), Vec(one(x)))
    v = f(dual)
    check_scalar(v)
    return @inbounds v.ϵ12[1][1]
end

hessian(f, x::AbstractVector) = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfig(f, x))
hessian(f, x::AbstractVector, cfg::AbstractHessianConfig) = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, cfg)

function hessian!(H::AbstractMatrix, f, x::AbstractVector{T}, cfg::AbstractHessianConfig) where {T}
    if chunksize(cfg) == length(x)
        return hessian_vector!(H, f, x, cfg)
    else
        return hessian_chunk!(H, f, x, cfg)
    end
end

function hessian_vector!(H::AbstractMatrix, f, x::AbstractVector, cfg::HessianConfig)
    @assert size(H, 1) == size(H, 2) == length(x)
    Tag = tagtype(cfg)
    cfg.duals .= HyperDual{Tag}.(x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    return extract_hessian!(H, v)
end

function hessian_chunk!(H::AbstractMatrix, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    @assert size(H, 1) == size(H, 2) == length(x)
    n_chunks = ceil(Int, length(x) / chunksize(cfg))
    for i in 1:n_chunks
        for j in i:n_chunks
            seed!(cfg.duals, x, cfg.seeds, i, j)
            v = f(cfg.duals)
            check_scalar(v)
            extract_hessian!(H, v, i, j)
        end
    end
    symmetrize!(H)
    return H
end

function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector{T}, cfg::AbstractHessianConfig) where {T}
    @assert size(H, 1) == size(H, 2) == length(x)
    @assert length(G) == length(x)
    if chunksize(cfg) == length(x)
        return hessiangradvalue_vector!(H, G, f, x, cfg)
    else
        return hessiangradvalue_chunk!(H, G, f, x, cfg)
    end
end

function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector)
    cfg = HessianConfig(f, x)
    return hessiangradvalue!(H, G, f, x, cfg)
end

function hessiangradvalue(f, x::AbstractVector, cfg::AbstractHessianConfig)
    G = similar(x, axes(x, 1))
    H = similar(x, axes(x, 1), axes(x, 1))
    value = hessiangradvalue!(H, G, f, x, cfg)
    return (; value = value, gradient = G, hessian = H)
end

function hessiangradvalue(f, x::AbstractVector)
    cfg = HessianConfig(f, x)
    return hessiangradvalue(f, x, cfg)
end

function hessiangradvalue_vector!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector, cfg::HessianConfig)
    @assert size(H, 1) == size(H, 2) == length(x)
    Tag = tagtype(cfg)
    cfg.duals .= HyperDual{Tag}.(x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    G .= Tuple(v.ϵ1)
    extract_hessian!(H, v)
    return v.v
end

function hessiangradvalue_chunk!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    @assert size(H, 1) == size(H, 2) == length(x)
    @assert length(G) == length(x)
    n_chunks = ceil(Int, length(x) / chunksize(cfg))
    value = zero(T)
    for i in 1:n_chunks
        for j in i:n_chunks
            seed!(cfg.duals, x, cfg.seeds, i, j)
            v = f(cfg.duals)
            check_scalar(v)
            value = v.v
            extract_hessian!(H, v, i, j)
            if j == i
                extract_gradient!(G, v, i)
            end
        end
    end
    symmetrize!(H)
    return value
end

end # module

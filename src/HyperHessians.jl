module HyperHessians

using SIMD: SIMD, Vec, vstore
using CommonSubexpressions: cse

include("rules.jl")

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
struct HyperDual{N, T} <: Real
    v::T
    ϵ1::Vec{N, T} # Vec{M,T}
    ϵ2::Vec{N, T}
    ϵ12::NTuple{N, Vec{N, T}} # NTuple{M, Vec{N,T}}
end
HyperDual(v::T, ϵ1::Vec{N, T}, ϵ2::Vec{N, T}) where {N, T} = HyperDual(v, ϵ1, ϵ2, ntuple(i -> zero(Vec{N, T}), Val(N)))
HyperDual{N}(v::T) where {N, T} = HyperDual(v, zero(Vec{N, T}), zero(Vec{N, T}))
HyperDual{N, T}(v) where {N, T} = HyperDual{N}(T(v))

function HyperDual(v::T1, ϵ1::Vec{N, T2}, ϵ2::Vec{N, T2}, ϵ12::NTuple{N, Vec{N, T2}}) where {N, T1, T2}
    T = promote_type(T1, T2)
    return HyperDual(T(v), convert(Vec{N, T}, ϵ1), convert(Vec{N, T}, ϵ2), convert.(Vec{N, T}, ϵ12))
end

Base.promote_rule(::Type{HyperDual{N, T1}}, ::Type{HyperDual{N, T2}}) where {N, T1, T2} = HyperDual{N, promote_type(T1, T2)}
Base.promote_rule(::Type{HyperDual{N, T1}}, ::Type{T2}) where {N, T1, T2 <: Real} = HyperDual{N, promote_type(T1, T2)}
Base.convert(::Type{HyperDual{N, T1}}, h::HyperDual{N, T2}) where {N, T1, T2} =
    HyperDual{N, T1}(T1(h.v), convert(Vec{N, T1}, h.ϵ1), convert(Vec{N, T1}, h.ϵ2), convert.(Vec{N, T1}, h.ϵ12))
Base.convert(::Type{HyperDual{N, T}}, x::Real) where {N, T} = HyperDual{N, T}(T(x))

# Make this look nicer
function Base.show(io::IO, h::HyperDual)
    print(io, h.v, " + ", Tuple(h.ϵ1), "ϵ1", " + ", Tuple(h.ϵ2), "ϵ2", " + ", map(Tuple, h.ϵ12), "ϵ12")
    return
end

Base.one(::Type{HyperDual{N, T}}) where {N, T} = HyperDual{N}(one(T))
Base.zero(::Type{HyperDual{N, T}}) where {N, T} = HyperDual{N}(zero(T))
Base.one(::HyperDual{N, T}) where {N, T} = one(HyperDual{N, T})
Base.zero(::HyperDual{N, T}) where {N, T} = zero(HyperDual{N, T})
Base.float(h::HyperDual{N, T}) where {N, T} = convert(HyperDual{N, float(T)}, h)

# Unary
@inline Base.:(-)(h::HyperDual{N}) where {N} = HyperDual(-h.v, -h.ϵ1, -h.ϵ2, ntuple(i -> -h.ϵ12[i], Val(N)))
@inline Base.:(+)(h::HyperDual) = h

# Binary
for f in (:+, :-)
    @eval begin
        @inline Base.$f(h1::HyperDual{N, T}, h2::HyperDual{N, T}) where {N, T} =
            HyperDual($f(h1.v, h2.v), $f(h1.ϵ1, h2.ϵ1), $f(h1.ϵ2, h2.ϵ2), ntuple(i -> $f(h1.ϵ12[i], h2.ϵ12[i]), Val(N)))
        @inline Base.$f(h1::HyperDual{N, T1}, h2::HyperDual{N, T2}) where {N, T1, T2} = $f(promote(h1, h2)...)
        @inline Base.$f(h::HyperDual{N}, r::Real) where {N} =
            HyperDual($f(h.v, r), h.ϵ1, h.ϵ2, h.ϵ12)
        @inline Base.$f(r::Real, h::HyperDual{N}) where {N} =
            HyperDual($f(r, h.v), h.ϵ1, h.ϵ2, h.ϵ12)
    end
end

for f in (:*, :/)
    @eval @inline Base.$f(h::HyperDual{N}, r::Real) where {N} =
        HyperDual($f(h.v, r), $f(h.ϵ1, r), $f(h.ϵ2, r), ntuple(i -> $f(h.ϵ12[i], r), Val(N)))
    @eval @inline Base.$f(r::Real, h::HyperDual{N}) where {N} =
        HyperDual($f(r, h.v), $f(r, h.ϵ1), $f(r, h.ϵ2), ntuple(i -> $f(r, h.ϵ12[i]), Val(N)))
end

@inline Base.:(/)(h1::HyperDual{N, T}, h2::HyperDual{N, T}) where {N, T} = h1 * inv(h2)

# muladd: x*y + z
@inline Base.muladd(x::HyperDual{N}, y::Real, z::HyperDual{N}) where {N} = x * y + z
@inline Base.muladd(x::Real, y::HyperDual{N}, z::HyperDual{N}) where {N} = x * y + z
@inline Base.muladd(x::HyperDual{N}, y::HyperDual{N}, z::Real) where {N} = x * y + z
@inline Base.muladd(x::HyperDual{N}, y::Real, z::Real) where {N} = x * y + z
@inline Base.muladd(x::Real, y::HyperDual{N}, z::Real) where {N} = x * y + z
@inline Base.muladd(x::Real, y::Real, z::HyperDual{N}) where {N} = muladd(x, y, z.v) + z - z.v
@inline Base.muladd(x::HyperDual{N}, y::HyperDual{N}, z::HyperDual{N}) where {N} = x * y + z

# Get's rid of a bunch of boundserror throwing in the LLVM IR, not sure it matters for runtime performance...
unsafe_getindex(v::SIMD.Vec, i::SIMD.IntegerTypes) = SIMD.Intrinsics.extractelement(v.data, i - 1)
@inline ⊗(v1::Vec{N}, v2::Vec{N}) where {N} = ntuple(i -> unsafe_getindex(v1, i) * v2, Val(N))

@inline Base.:(*)(h1::HyperDual{N, T1}, h2::HyperDual{N, T2}) where {N, T1, T2} = *(promote(h1, h2)...)
@inline function Base.:(*)(h1::HyperDual{N, T}, h2::HyperDual{N, T}) where {N, T}
    r = h1.v * h2.v
    ϵ1 = muladd(h1.v, h2.ϵ1, h1.ϵ1 * h2.v)
    ϵ2 = muladd(h1.v, h2.ϵ2, h1.ϵ2 * h2.v)
    ϵ12_1 = h1.ϵ1 ⊗ h2.ϵ2
    ϵ12_2 = h2.ϵ1 ⊗ h1.ϵ2
    ϵ12 = ntuple(i -> muladd(h1.v, h2.ϵ12[i], muladd(h1.ϵ12[i], h2.v, ϵ12_1[i] + ϵ12_2[i])), Val(N))
    return HyperDual(r, ϵ1, ϵ2, ϵ12)
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
        return HyperDual(v, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i] * f′ + x23[i], Val(N)))
    end
    cse_expr = cse(expr; warn = false)
    @eval @inline function Base.$f(h::HyperDual{N, T}) where {N, T}
        x = h.v
        $cse_expr
    end
end

@inline function Base.sin(h::HyperDual{N}) where {N}
    s, c = sincos(h.v)
    f′, f′′ = c, -s
    x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
    return HyperDual(s, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i] * f′ + x23[i], Val(N)))
end

@inline function Base.cos(h::HyperDual{N}) where {N}
    s, c = sincos(h.v)
    f′, f′′ = -s, -c
    x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
    return HyperDual(c, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i] * f′ + x23[i], Val(N)))
end


##########
# Chunks #
##########

struct Chunk{N} end

function Chunk(input_length::Integer, ::Type{T} = Float64) where {T}
    return Chunk{pickchunksize(input_length, T)}()
end

Chunk(x::AbstractArray) = Chunk(length(x), eltype(x))

maxchunksize(::Type{Float64}) = 4
maxchunksize(::Type{Float32}) = 8
maxchunksize(::Type{T}) where {T} = 4

function pickchunksize(input_length, ::Type{T}) where {T}
    max_size = maxchunksize(T)
    input_length <= max_size && return input_length
    return max_size
end

chunksize(::Chunk{N}) where {N} = N::Int


##################
# Hessian Config #
##################

abstract type AbstractHessianConfig end

struct HessianConfig{D <: AbstractVector{<:HyperDual}, S} <: AbstractHessianConfig
    duals::D
    seeds::S
end
(chunksize(cfg::HessianConfig)::Int) = length(cfg.seeds)

function HessianConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    @assert 0 < N
    duals = similar(x, HyperDual{N, T}) # not Vector
    seeds = collect(construct_seeds(NTuple{N, T}))
    return HessianConfig(duals, seeds)
end

###########
# Seeding #
###########

@generated function single_seed(::Type{NTuple{N, T}}, ::Val{i}) where {N, T, i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(T)), :(zero(T))) for j in 1:N]...)
    return :(Vec($(ex)))
end

@generated construct_seeds(::Type{NTuple{N, T}}) where {N, T} =
    Expr(:tuple, [:(single_seed(NTuple{N, T}, Val{$i}())) for i in 1:N]...)

seed_epsilon_1(d::HyperDual{N, T}, ϵ1) where {N, T} = HyperDual{N, T}(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::HyperDual{N, T}, ϵ2) where {N, T} = HyperDual{N, T}(d.v, d.ϵ1, ϵ2, d.ϵ12)

function seed!(d::AbstractVector{<:HyperDual{N}}, x, seeds, block_i, block_j) where {N}
    d .= HyperDual{N}.(x)
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

function extract_hessian!(H::AbstractMatrix, v::HyperDual{N}, block_i::Int, block_j::Int) where {N}
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


function extract_gradient!(G::AbstractVector, v::HyperDual{N}, block_i::Int) where {N}
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
    dual = HyperDual(x, Vec(one(x)), Vec(one(x)))
    v = f(dual)
    check_scalar(v)
    return @inbounds v.ϵ12[1][1]
end

hessian(f, x::AbstractVector) = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfig(x))
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
    cfg.duals .= HyperDual.(x, cfg.seeds, cfg.seeds)
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
    cfg = HessianConfig(x)
    return hessiangradvalue!(H, G, f, x, cfg)
end

function hessiangradvalue(f, x::AbstractVector, cfg::AbstractHessianConfig)
    G = similar(x, axes(x, 1))
    H = similar(x, axes(x, 1), axes(x, 1))
    value = hessiangradvalue!(H, G, f, x, cfg)
    return (; value = value, gradient = G, hessian = H)
end

function hessiangradvalue(f, x::AbstractVector)
    cfg = HessianConfig(x)
    return hessiangradvalue(f, x, cfg)
end

function hessiangradvalue_vector!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector, cfg::HessianConfig)
    @assert size(H, 1) == size(H, 2) == length(x)
    cfg.duals .= HyperDual.(x, cfg.seeds, cfg.seeds)
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

##############################
# Hessian-vector interactions #
##############################

"""
    hvp(f, x, v[, cfg])

Compute the Hessian–vector product `H(x) * v`.
"""
function hvp(f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::HessianConfig) where {T}
    hv = similar(x, T)
    return hvp!(hv, f, x, v, cfg)
end

hvp(f, x::AbstractVector, v::AbstractVector) = hvp(f, x, v, HessianConfig(x))

function hvp!(hv::AbstractVector{T}, f::F, x::AbstractVector{T}, v::AbstractVector{T}, cfg::HessianConfig) where {T, F}
    @assert length(x) == length(v)
    @assert length(hv) == length(x)
    if chunksize(cfg) == length(x)
        return hvp_vector!(hv, f, x, v, cfg)
    else
        return hvp_chunk!(hv, f, x, v, cfg)
    end
end

hvp!(hv::AbstractVector, f, x::AbstractVector, v::AbstractVector) = hvp!(hv, f, x, v, HessianConfig(x))

@inline function hvp_vector!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::HessianConfig) where {T}
    # ε₁ is seeded with the identity directions and ε₂ with diag(v); the mixed
    # ε₁ᵀAε₂ term therefore accumulates ∑ⱼ Hᵢⱼ vⱼ in each slot, giving H * v.
    @inbounds for i in eachindex(x)
        cfg.duals[i] = HyperDual(x[i], cfg.seeds[i], cfg.seeds[i] * v[i])
    end
    out = f(cfg.duals)
    check_scalar(out)

    @inbounds for i in 1:length(x)
        hv[i] = sum(Tuple(out.ϵ12[i]))
    end
    return hv
end

@inline function seed_hvp!(d::AbstractVector{<:HyperDual{N}}, x, seeds, v, block_i::Int, block_j::Int) where {N}
    d .= HyperDual{N}.(x)
    index_i = (block_i - 1) * N + 1
    index_j = (block_j - 1) * N + 1
    range_i = index_i:min(length(x), (index_i + N - 1))
    range_j = index_j:min(length(x), (index_j + N - 1))
    chunks_i = length(range_i)
    chunks_j = length(range_j)

    @inbounds for (offset, idx) in enumerate(range_i)
        d[idx] = seed_epsilon_1(d[idx], seeds[offset])
    end
    @inbounds for (offset, idx) in enumerate(range_j)
        d[idx] = seed_epsilon_2(d[idx], seeds[offset] * v[idx])
    end
    return range_i, range_j
end

function hvp_chunk!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::HessianConfig) where {T}
    N = chunksize(cfg)
    fill!(hv, zero(T))
    n_chunks = ceil(Int, length(x) / N)

    for i in 1:n_chunks
        for j in 1:n_chunks
            range_i, range_j = seed_hvp!(cfg.duals, x, cfg.seeds, v, i, j)
            out = f(cfg.duals)
            check_scalar(out)

            @inbounds for (I, idx_i) in enumerate(range_i)
                block = Tuple(out.ϵ12[I])
                hv[idx_i] += sum(@inbounds block[J] for J in 1:length(range_j))
            end
        end
    end
    return hv
end

@inline function hvp_vector(f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::HessianConfig) where {T}
    hv = similar(x, T)
    return hvp_vector!(hv, f, x, v, cfg)
end

function hvp_chunk(f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::HessianConfig) where {T}
    hv = similar(x, T)
    return hvp_chunk!(hv, f, x, v, cfg)
end

end # module

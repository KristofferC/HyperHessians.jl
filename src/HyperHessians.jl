module HyperHessians

using SIMD: SIMD, Vec, vstore
using CommonSubexpressions: cse

include("rules.jl")

# This is currently "artifically" restricted to "square" HyperDuals where
# ϵ1 and ϵ2 have the same length.
struct HyperDual{N, T} <: Real
    v::T
    ϵ1::Vec{N,T} # Vec{M,T}
    ϵ2::Vec{N,T}
    ϵ12::NTuple{N, Vec{N,T}} # NTuple{M, Vec{N,T}} 
end
HyperDual(v::T, ϵ1::Vec{N,T}, ϵ2::Vec{N,T}) where {N, T} = HyperDual(v, ϵ1, ϵ2, ntuple(i -> zero(Vec{N,T}), Val(N)))
HyperDual{N}(v::T) where {N, T} = HyperDual(v, zero(Vec{N,T}),zero(Vec{N,T}))
HyperDual{N,T}(v) where {N, T} = HyperDual{N}(T(v))

function HyperDual(v::T1, ϵ1::Vec{N,T2}, ϵ2::Vec{N,T2}, ϵ12::NTuple{N, Vec{N,T2}}) where {N, T1, T2}
    T = promote_type(T1, T2)
    HyperDual(T(v), convert(Vec{N,T}, ϵ1), convert(Vec{N,T}, ϵ2), convert.(Vec{N,T}, ϵ12))
end

Base.promote_rule(::Type{HyperDual{N,T1}}, ::Type{HyperDual{N,T2}}) where {N,T1,T2} = HyperDual{N, promote_type(T1, T2)}
Base.convert(::Type{HyperDual{N,T1}}, h::HyperDual{N,T2}) where {N,T1,T2} = 
    HyperDual{N,T1}(T1(h.v), convert(Vec{N,T1}, h.ϵ1), convert(Vec{N,T1}, h.ϵ2), convert.(Vec{N,T1}, h.ϵ12))

# Make this look nicer
function Base.show(io::IO, h::HyperDual)
    print(io, h.v, " + ", Tuple(h.ϵ1), "ϵ1", " + ", Tuple(h.ϵ2), "ϵ2", " + ",  map(Tuple, h.ϵ12), "ϵ12")
end

Base.one(::Type{HyperDual{N, T}}) where {N, T} = HyperDual{N}(one(T))
Base.zero(::Type{HyperDual{N, T}}) where {N, T} = HyperDual{N}(zero(T))
Base.one(::HyperDual{N, T}) where {N, T} = one(HyperDual{N,T})
Base.zero(::HyperDual{N, T}) where {N, T} = zero(HyperDual{N,T})
Base.float(h::HyperDual{N,T}) where {N,T} = convert(HyperDual{N,float(T)}, h)

# Unary
@inline Base.:(-)(h::HyperDual{N}) where {N} = HyperDual(-h.v, -h.ϵ1, -h.ϵ2, ntuple(i -> -h.ϵ12[i], Val(N)))
@inline Base.:(+)(h::HyperDual) = h

# Binary
for f in (:+, :-)
    @eval begin
        @inline Base.$f(h1::HyperDual{N,T}, h2::HyperDual{N,T}) where {N,T} =
            HyperDual($f(h1.v, h2.v), $f(h1.ϵ1, h2.ϵ1), $f(h1.ϵ2, h2.ϵ2), ntuple(i -> $f(h1.ϵ12[i], h2.ϵ12[i]), Val(N)))
        @inline Base.$f(h1::HyperDual{N,T1}, h2::HyperDual{N,T2}) where {N,T1,T2} = $f(promote(h1, h2)...)
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

@inline Base.:(/)(h1::HyperDual{N,T}, h2::HyperDual{N,T}) where {N,T} = h1 * inv(h2)

# Get's rid of a bunch of boundserror throwing in the LLVM IR, not sure it matters for runtime performance...
unsafe_getindex(v::SIMD.Vec, i::SIMD.IntegerTypes) = SIMD.Intrinsics.extractelement(v.data, i-1)
@inline ⊗(v1::Vec{N}, v2::Vec{N}) where {N} = ntuple(i -> unsafe_getindex(v1, i) * v2, Val(N))

@inline Base.:(*)(h1::HyperDual{N,T1}, h2::HyperDual{N,T2}) where {N,T1,T2} = *(promote(h1, h2)...)
@inline function Base.:(*)(h1::HyperDual{N,T}, h2::HyperDual{N,T}) where {N,T}
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
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{2}) = x*x
@inline Base.literal_pow(::typeof(^), x::HyperDual, ::Val{3}) = x*x*x

for (f, f′, f′′) in DIFF_RULES
    expr = quote
        # Verify that the cse still works properly when changing this.
        v = $f(x)
        f′ = $f′
        f′′ = $f′′
        x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
        return HyperDual(v, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i]*f′ + x23[i], Val(N)))
    end
    cse_expr = cse(expr; warn=false)
    @eval @inline function Base.$f(h::HyperDual{N,T}) where {N,T}
        x = h.v
        $cse_expr
    end
end

@inline function Base.sin(h::HyperDual{N}) where {N}
    s, c = sincos(h.v)
    f′, f′′ = c, -s
    x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
    return HyperDual(s, h.ϵ1 * f′, h.ϵ2 * f′, ntuple(i -> h.ϵ12[i]*f′ + x23[i], Val(N)))
end

@inline function Base.cos(h::HyperDual{N}) where {N}
    s, c = sincos(h.v)
    f′,  f′′ = -s, -c
    x23 = (f′′ * h.ϵ1) ⊗ h.ϵ2
    return HyperDual(c, h.ϵ1 * f′,  h.ϵ2 * f′, ntuple(i -> h.ϵ12[i]*f′ + x23[i], Val(N)))
end



##########
# Chunks #
##########

const DEFAULT_CHUNK_THRESHOLD = 8

struct Chunk{N} end

const CHUNKS = [Chunk{i}() for i in 1:DEFAULT_CHUNK_THRESHOLD]

function Chunk(input_length::Integer)
    N = pickchunksize(input_length)
    0 < N <= DEFAULT_CHUNK_THRESHOLD && return CHUNKS[N] # TODO: Check if this matters
    return Chunk{N}()
end

Chunk(x::AbstractArray) = Chunk(length(x))

# TODO: MOre testing needed here.
function pickchunksize(input_length)
    input_length == 1 && return 1
    input_length == 2 && return 2
    3 <= input_length <= 4 && return 4 # 3 seems to be very bad for some reason
    return 8
end

(chunksize(::Chunk{N})::Int) where {N} = N


##################
# Hessian Config #
##################

abstract type AbstractHessianConfig end

struct HessianConfig{D <: AbstractVector{<:HyperDual}, S} <: AbstractHessianConfig
    duals::D
    seeds::S
end
(chunksize(cfg::HessianConfig)::Int) = length(cfg.seeds)

function HessianConfig(x::AbstractVector{T}, chunk=Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    @assert 0 < N
    duals = similar(x, HyperDual{N,T}) # not Vector
    seeds = collect(construct_seeds(NTuple{N,T}))
    return HessianConfig(duals, seeds)
end

struct HessianConfigThreaded{D <: AbstractVector{<:HyperDual}, S} <: AbstractHessianConfig
    duals::Vector{D}
    seeds::S
end

function HessianConfigThreaded(x::AbstractVector{T}, chunk=Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    @assert 0 < N
    duals = [similar(x, HyperDual{N,T}) for i in 1:Threads.nthreads()] # not Vector
    seeds = collect(construct_seeds(NTuple{N,T}))
    return HessianConfigThreaded(duals, seeds)
end

(chunksize(cfg::HessianConfigThreaded)::Int) = length(cfg.seeds)
HessianConfig(cfg::HessianConfigThreaded) = HessianConfig(first(cfg.duals), cfg.seeds)


###########
# Seeding #
###########

@generated function single_seed(::Type{NTuple{N,T}}, ::Val{i}) where {N,T,i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(T)), :(zero(T))) for j in 1:N]...)
    return :(Vec($(ex)))
end

@generated construct_seeds(::Type{NTuple{N,T}}) where {N,T} =
    Expr(:tuple, [:(single_seed(NTuple{N,T}, Val{$i}())) for i in 1:N]...)

seed_epsilon_1(d::HyperDual{N,T}, ϵ1) where {N,T} = HyperDual{N,T}(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::HyperDual{N,T}, ϵ2) where {N,T} = HyperDual{N,T}(d.v, d.ϵ1, ϵ2, d.ϵ12)

function seed!(d::AbstractVector{<:HyperDual{N}}, x, seeds, block_i, block_j) where {N}
    d .= HyperDual{N}.(x)
    index_i = (block_i-1) * N + 1
    index_j = (block_j-1) * N + 1
    range_i = index_i : min(length(x), (index_i + N -1))
    range_j = index_j : min(length(x), (index_j + N -1))
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
        for j in i:size(H,2) 
            H[j,i] = H[i,j]
        end
    end
    return H
end

function extract_hessian!(H::AbstractMatrix, v::HyperDual{N}, block_i::Int, block_j::Int) where {N}
    index_i = (block_i-1) * N + 1
    index_j = (block_j-1) * N + 1
    range_i = index_i : (index_i + N -1)
    range_j = index_j : (index_j + N -1)

    for (I, i) in enumerate(range_i)
        for (J, j) in enumerate(range_j)
            if checkbounds(Bool, H, i, j)
                H[i, j] = v.ϵ12[I][J]
            end
        end
    end
    return H
end


function extract_gradient!(G::AbstractVector, v::HyperDual, block_i::Int)
    index_i = (block_i-1) * N + 1
    range_i = index_i : (index_i + N -1)
    for (I, i) in enumerate(range_i)
        G[i] .= v.ϵ1[I]
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

hessian(f, x::AbstractVector)                             = hessian!(similar(x, axes(x,1), axes(x,1)), f, x, HessianConfig(x))
hessian(f, x::AbstractVector, cfg::AbstractHessianConfig) = hessian!(similar(x, axes(x,1), axes(x,1)), f, x, cfg)

function hessian!(H::AbstractMatrix, f, x::AbstractVector{T}, cfg::AbstractHessianConfig) where {T,N}
    if chunksize(cfg) == length(x)
        if cfg isa HessianConfigThreaded
            cfg = HessianConfig(cfg)
        end
        return hessian_vector!(H, f, x, cfg)
    else
        return hessian_chunk!(H, f, x, cfg)
    end
end

function hessian_vector!(H::AbstractMatrix, f, x::AbstractVector, cfg::HessianConfig)
    @assert size(H,1) == size(H,2) == length(x)
    cfg.duals .= HyperDual.(x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    return extract_hessian!(H, v)
end

function hessian_chunk!(H::AbstractMatrix, f, x::AbstractVector{T}, cfg::HessianConfig) where {T,N}
    @assert size(H,1) == size(H,2) == length(x)
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

# Threaded chunk mode
function linear_to_cartesian(n::Int, k::Int)
    k -= 1
    n += 1
    i = n - 2 - floor(sqrt(-8*k + 4*n*(n-1)-7)/2 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return round(Int, i+1), round(Int, j)
end

function hessian_chunk!(H::AbstractMatrix, f, x::AbstractVector{T}, cfg::HessianConfigThreaded) where {T,N}
    @assert size(H,1) == size(H,2) == length(x)
    n_chunks = ceil(Int, length(x) / chunksize(cfg))
    Threads.@threads for k in 1:((n_chunks*(n_chunks+1)) ÷ 2)
        i, j = linear_to_cartesian(n_chunks, k)
        duals = cfg.duals[Threads.threadid()]
        seed!(duals, x, cfg.seeds, i, j)
        v = f(duals)
        check_scalar(v)
        extract_hessian!(H, v, i, j)
    end
    symmetrize!(H)
    return H
end

end # module
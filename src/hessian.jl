mutable struct HessianConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::D
    const seeds::S
end
@inline _chunksize(::Type{<:NTuple{N}}) where {N} = N
if USE_SIMD
    @inline _chunksize(::Type{<:Vec{N}}) where {N} = N
end
@inline _chunksize(seeds) = something(_chunksize(eltype(seeds)), length(seeds))
(chunksize(cfg::HessianConfig)::Int) = _chunksize(cfg.seeds)::Int

function HessianConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, N, T}) # not Vector
    seeds = collect(construct_seeds(NTuple{N, T}))
    return HessianConfig(duals, seeds)
end

"""
    DirectionalHVPConfig(x; chunk=Chunk(x))

Configuration for Hessian–vector products.
"""
mutable struct DirectionalHVPConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::D
    const seeds::S
end
(chunksize(cfg::DirectionalHVPConfig)::Int) = _chunksize(cfg.seeds)

function DirectionalHVPConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, 1, T}) # directional: one ε₂ lane
    seeds = collect(construct_seeds(NTuple{N, T}))
    return DirectionalHVPConfig(duals, seeds)
end

@generated function single_seed(::Type{NTuple{N, T}}, ::Val{i}) where {N, T, i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(T)), :(zero(T))) for j in 1:N]...)
    if USE_SIMD
        return :(Vec($(ex)))
    else
        return ex
    end
end

@generated construct_seeds(::Type{NTuple{N, T}}) where {N, T} =
    Expr(:tuple, [:(single_seed(NTuple{N, T}, Val{$i}())) for i in 1:N]...)

@inline block_range(block::Int, chunk::Int, ax) = begin
    start = first(ax) + (block - 1) * chunk
    start:min(last(ax), start + chunk - 1)
end

seed_epsilon_1(d::HyperDual{N1, N2, T}, ϵ1) where {N1, N2, T} = HyperDual{N1, N2, T}(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::HyperDual{N1, N2, T}, ϵ2) where {N1, N2, T} = HyperDual{N1, N2, T}(d.v, d.ϵ1, ϵ2, d.ϵ12)

@inline function seed!(d::AbstractVector{<:HyperDual{N1, N2}}, x, seeds, block_i, block_j) where {N1, N2}
    # Full vector path: single block, no need to zero then overwrite.
    if block_i == 1 && block_j == 1 && N1 == length(x) && N2 == length(x)
        @inbounds for idx in eachindex(x)
            seed = seeds[idx]
            d[idx] = HyperDual(x[idx], seed, seed)
        end
        return d
    end

    d .= HyperDual{N1, N2}.(x)

    range_i = block_range(block_i, N1, axes(x, 1))
    if block_i == block_j
        @inbounds for k in 1:length(range_i)
            idx = range_i[k]
            seed = seeds[k]
            d[idx] = HyperDual(x[idx], seed, seed)
        end
    else
        @inbounds for k in 1:length(range_i)
            idx = range_i[k]
            d[idx] = seed_epsilon_1(d[idx], seeds[k])
        end
        range_j = block_range(block_j, N2, axes(x, 1))
        @inbounds for k in 1:length(range_j)
            idx = range_j[k]
            d[idx] = seed_epsilon_2(d[idx], seeds[k])
        end
    end
    return d
end

@noinline check_scalar(x) =
    x isa Number || throw(error("expected a scalar to be returned from function passed to `hessian`"))

function extract_hessian!(H::AbstractMatrix, v::HyperDual)
    @inbounds for (icol, col) in enumerate(axes(H, 2))
        H[:, col] .= Tuple(v.ϵ12[icol])
    end
    return H
end

function symmetrize!(H::AbstractMatrix)
    ax1, ax2 = axes(H, 1), axes(H, 2)
    @inbounds for i_pos in 1:length(ax1)
        i = ax1[i_pos]
        for j_pos in i_pos:length(ax2)
            j = ax2[j_pos]
            H[j, i] = H[i, j]
        end
    end
    return H
end

function extract_hessian!(H::AbstractMatrix, v::HyperDual{N1, N2}, block_i::Int, block_j::Int) where {N1, N2}
    range_i = block_range(block_i, N1, axes(H, 1))
    range_j = block_range(block_j, N2, axes(H, 2))

    for (I, i) in enumerate(range_i)
        for (J, j) in enumerate(range_j)
            H[i, j] = v.ϵ12[I][J]
        end
    end
    return H
end

function extract_gradient!(G::AbstractVector, v::HyperDual{N1, N2}, block_i::Int) where {N1, N2}
    range_i = block_range(block_i, N1, axes(G, 1))
    for (I, i) in enumerate(range_i)
        G[i] = v.ϵ1[I]
    end
    return G
end

function hessian(f, x::Real)
    one_seed = single_seed(NTuple{1, typeof(x)}, Val(1))
    dual = HyperDual(x, one_seed, one_seed)
    v = f(dual)
    check_scalar(v)
    return v.ϵ12[1][1]
end

hessian(f::F, x::AbstractVector) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfig(x))
hessian(f::F, x::AbstractVector, cfg::HessianConfig) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, cfg)

function hessian!(H::AbstractMatrix, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    return hessian_chunk!(H, f, x, cfg)
end

function hessian_chunk!(H::AbstractMatrix, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    n_chunks = ceil(Int, length(x) / chunksize(cfg))
    for i in 1:n_chunks
        for j in i:n_chunks
            seed!(cfg.duals, x, cfg.seeds, i, j)
            v = f(cfg.duals)
            check_scalar(v)
            extract_hessian!(H, v, i, j)
        end
    end
    if n_chunks > 1
        symmetrize!(H)
    end
    return H
end

function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    return hessiangradvalue_chunk!(H, G, f, x, cfg)
end

function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector) where {F}
    cfg = HessianConfig(x)
    return hessiangradvalue!(H, G, f, x, cfg)
end

function hessiangradvalue(f::F, x::AbstractVector, cfg::HessianConfig) where {F}
    G = similar(x, axes(x, 1))
    H = similar(x, axes(x, 1), axes(x, 1))
    value = hessiangradvalue!(H, G, f, x, cfg)
    return (; value = value, gradient = G, hessian = H)
end

function hessiangradvalue(f::F, x::AbstractVector) where {F}
    cfg = HessianConfig(x)
    return hessiangradvalue(f, x, cfg)
end

function hessiangradvalue_chunk!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
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
    if n_chunks > 1
        symmetrize!(H)
    end
    return value
end

"""
    hvp(f, x, v[, cfg])

Compute the Hessian–vector product `H(x) * v`. Uses the directional path by default;
pass a `DirectionalHVPConfig` explicitly to control chunking.
"""
hvp(f::F, x::AbstractVector, v::AbstractVector) where {F} = hvp(f, x, v, DirectionalHVPConfig(x))
hvp(f::F, x::AbstractVector, v::AbstractVector, cfg::DirectionalHVPConfig) where {F} = hvp!(similar(x, eltype(x)), f, x, v, cfg)

hvp!(hv::AbstractVector, f::F, x::AbstractVector, v::AbstractVector) where {F} = hvp!(hv, f, x, v, DirectionalHVPConfig(x))
hvp!(hv::AbstractVector, f::F, x::AbstractVector, v::AbstractVector, cfg::DirectionalHVPConfig) where {F} = hvp_dir!(hv, f, x, v, cfg)

@inline function hvp_dir!(hv::AbstractVector{T}, f::F, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T, F}
    length(x) == length(v) || throw(DimensionMismatch(lazy"x and v must have same length, got $(length(x)) and $(length(v))"))
    length(hv) == length(x) || throw(DimensionMismatch(lazy"hv must have length $(length(x)), got $(length(hv))"))
    return hvp_chunk_dir!(hv, f, x, v, cfg)
end

function hvp_chunk_dir!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T}
    N = chunksize(cfg)
    n_chunks = ceil(Int, length(x) / N)
    fill!(hv, zero(T))
    for i in 1:n_chunks
        # Reset ε₁ to zero and ε₂ to the direction for this chunk pass.
        zeroϵ1 = zero_ϵ(cfg.seeds[1])
        ax = axes(x, 1)
        start_i = first(ax) + (i - 1) * N
        end_i = min(last(ax), start_i + N - 1)
        @inbounds for idx in eachindex(x)
            # Seed ε₁ only for the current block; others stay zeroed.
            if start_i <= idx <= end_i
                offset = idx - start_i + 1
                cfg.duals[idx] = HyperDual(x[idx], cfg.seeds[offset], (@static USE_SIMD ? Vec((v[idx],)) : (v[idx],)))
            else
                cfg.duals[idx] = HyperDual(x[idx], zeroϵ1, (@static USE_SIMD ? Vec((v[idx],)) : (v[idx],)))
            end
        end

        out = f(cfg.duals)
        check_scalar(out)
        @inbounds for (I, idx_i) in enumerate(start_i:end_i)
            hv[idx_i] = out.ϵ12[I][1]
        end
    end
    return hv
end

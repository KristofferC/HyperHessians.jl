# Config types - unified for both HyperDual and HyperDualSIMD
mutable struct HessianConfig{D <: AbstractVector{<:AbstractHyperDualNumber}, S}
    const duals::D
    const seeds::S
end

mutable struct DirectionalHVPConfig{D <: AbstractVector{<:AbstractHyperDualNumber}, S}
    const duals::D
    const seeds::S
end

@inline _chunksize(::Type{<:NTuple{N}}) where {N} = N
@inline _chunksize(::Type{Vec{N, T}}) where {N, T} = N
@inline _chunksize(seeds::AbstractVector) = _chunksize(eltype(seeds))
chunksize(cfg::HessianConfig) = _chunksize(cfg.seeds)
chunksize(cfg::DirectionalHVPConfig) = _chunksize(cfg.seeds)

function HessianConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, N, T})
    seeds = collect(construct_seeds(NTuple{N, T}))
    return HessianConfig(duals, seeds)
end

function HessianConfigSIMD(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDualSIMD{N, N, T})
    seeds = collect(construct_seeds(Vec{N, T}))
    return HessianConfig(duals, seeds)
end

function DirectionalHVPConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, 1, T})
    seeds = collect(construct_seeds(NTuple{N, T}))
    return DirectionalHVPConfig(duals, seeds)
end

function DirectionalHVPConfigSIMD(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDualSIMD{N, 1, T})
    seeds = collect(construct_seeds(Vec{N, T}))
    return DirectionalHVPConfig(duals, seeds)
end

construct_seeds(::Type{NTuple{N, T}}) where {N, T} = ntuple(i -> ntuple(j -> ifelse(i == j, one(T), zero(T)), Val(N)), Val(N))
construct_seeds(::Type{Vec{N, T}}) where {N, T} = ntuple(i -> Vec(ntuple(j -> ifelse(i == j, one(T), zero(T)), Val(N))), Val(N))

# Helpers for initializing duals - dispatch on element type
@inline _init_dual(::Type{HyperDual{N1, N2, T}}, x, ϵ1, ϵ2) where {N1, N2, T} = HyperDual(x, ϵ1, ϵ2)
@inline _init_dual(::Type{HyperDualSIMD{N1, N2, T}}, x, ϵ1, ϵ2) where {N1, N2, T} = HyperDualSIMD(x, ϵ1, ϵ2)
@inline _init_dual_zero(::Type{H}, x) where {H <: AbstractHyperDualNumber} = H(x)

# Helper for HVP direction seeding
@inline _init_hvp_dual(::Type{HyperDual{N1, 1, T}}, x, ϵ1, v) where {N1, T} = HyperDual(x, ϵ1, (v,))
@inline _init_hvp_dual(::Type{HyperDualSIMD{N1, 1, T}}, x, ϵ1, v) where {N1, T} = HyperDualSIMD(x, ϵ1, Vec((v,)))

@inline block_range(block::Int, chunk::Int, ax) = begin
    start = first(ax) + (block - 1) * chunk
    start:min(last(ax), start + chunk - 1)
end

# Seed functions - unified via typeof
seed_epsilon_1(d::AbstractHyperDualNumber, ϵ1) = typeof(d)(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::AbstractHyperDualNumber, ϵ2) = typeof(d)(d.v, d.ϵ1, ϵ2, d.ϵ12)

@inline function seed_block_ϵ1!(d::AbstractVector{<:AbstractHyperDualNumber{N1}}, seeds, block_i, ax) where {N1}
    range_i = block_range(block_i, N1, ax)
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        d[idx] = seed_epsilon_1(d[idx], seeds[k])
    end
    return range_i
end

@inline function seed_block_ϵ2!(d::AbstractVector{<:AbstractHyperDualNumber}, seeds, range_j)
    @inbounds for k in 1:length(range_j)
        idx = range_j[k]
        d[idx] = seed_epsilon_2(d[idx], seeds[k])
    end
    return nothing
end

@inline function zero_block_ϵ2!(d::AbstractVector{<:AbstractHyperDualNumber}, seeds, range_j)
    zeroϵ2 = zero_ϵ(seeds[1])
    @inbounds for k in 1:length(range_j)
        idx = range_j[k]
        d[idx] = seed_epsilon_2(d[idx], zeroϵ2)
    end
    return nothing
end

@inline function zero_block_ϵ1!(d::AbstractVector{<:AbstractHyperDualNumber}, seeds, range_i)
    zeroϵ1 = zero_ϵ(seeds[1])
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        d[idx] = seed_epsilon_1(d[idx], zeroϵ1)
    end
    return nothing
end

@noinline check_scalar(x) =
    x isa Number || throw(error("expected a scalar to be returned from function passed to `hessian`"))

function extract_hessian!(H::AbstractMatrix, v::AbstractHyperDualNumber)
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

function extract_hessian!(H::AbstractMatrix, v::AbstractHyperDualNumber{N1, N2, T}, block_i::Int, block_j::Int) where {N1, N2, T}
    range_i = block_range(block_i, N1, axes(H, 1))
    range_j = block_range(block_j, N2, axes(H, 2))

    for (I, i) in enumerate(range_i)
        for (J, j) in enumerate(range_j)
            H[i, j] = v.ϵ12[I][J]
        end
    end
    return H
end

function extract_gradient!(G::AbstractVector, v::AbstractHyperDualNumber{N1, N2, T}, block_i::Int) where {N1, N2, T}
    range_i = block_range(block_i, N1, axes(G, 1))
    for (I, i) in enumerate(range_i)
        G[i] = v.ϵ1[I]
    end
    return G
end

# Scalar hessian
function hessian(f, x::Real)
    one_seed = ntuple(i -> ifelse(i == 1, one(typeof(x)), zero(typeof(x))), Val(1))
    dual = HyperDual(x, one_seed, one_seed)
    v = f(dual)
    check_scalar(v)
    return v.ϵ12[1][1]
end

function hessian_simd(f, x::Real)
    one_seed = Vec((one(typeof(x)),))
    dual = HyperDualSIMD(x, one_seed, one_seed)
    v = f(dual)
    check_scalar(v)
    return v.ϵ12[1][1]
end

# Vector hessian
hessian(f::F, x::AbstractVector) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfig(x))
hessian_simd(f::F, x::AbstractVector) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfigSIMD(x))
hessian(f::F, x::AbstractVector, cfg::HessianConfig) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, cfg)

function hessian!(H::AbstractMatrix, f::F, x::AbstractVector, cfg::HessianConfig) where {F}
    if chunksize(cfg) == length(x)
        return hessian_vector!(H, f, x, cfg)
    else
        return hessian_chunk!(H, f, x, cfg)
    end
end

hessian_simd!(H::AbstractMatrix, f::F, x::AbstractVector) where {F} = hessian!(H, f, x, HessianConfigSIMD(x))

function hessian_vector!(H::AbstractMatrix, f, x::AbstractVector, cfg::HessianConfig)
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    D = eltype(cfg.duals)
    cfg.duals .= _init_dual.(D, x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    return extract_hessian!(H, v)
end

function hessian_chunk!(H::AbstractMatrix, f, x::AbstractVector, cfg::HessianConfig)
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    ax = axes(x, 1)
    N = chunksize(cfg)
    n_chunks = ceil(Int, length(x) / N)
    D = eltype(cfg.duals)
    cfg.duals .= _init_dual_zero.(D, x)
    prev_range = 0:-1  # empty range sentinel
    for i in 1:n_chunks
        zero_block_ϵ1!(cfg.duals, cfg.seeds, prev_range)
        range_i = seed_block_ϵ1!(cfg.duals, cfg.seeds, i, ax)
        for j in i:n_chunks
            range_j = j == i ? range_i : block_range(j, N, ax)
            seed_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
            v = f(cfg.duals)
            check_scalar(v)
            extract_hessian!(H, v, i, j)
            zero_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
        end
        prev_range = range_i
    end
    symmetrize!(H)
    return H
end

# hessiangradvalue functions
function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector, cfg::HessianConfig) where {F}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    if chunksize(cfg) == length(x)
        return hessiangradvalue_vector!(H, G, f, x, cfg)
    else
        return hessiangradvalue_chunk!(H, G, f, x, cfg)
    end
end

function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector) where {F}
    cfg = HessianConfig(x)
    return hessiangradvalue!(H, G, f, x, cfg)
end

function hessiangradvalue_simd!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector) where {F}
    cfg = HessianConfigSIMD(x)
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

function hessiangradvalue_simd(f::F, x::AbstractVector) where {F}
    cfg = HessianConfigSIMD(x)
    return hessiangradvalue(f, x, cfg)
end

function hessiangradvalue_vector!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector, cfg::HessianConfig)
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    D = eltype(cfg.duals)
    cfg.duals .= _init_dual.(D, x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    G .= Tuple(v.ϵ1)
    extract_hessian!(H, v)
    return v.v
end

function hessiangradvalue_chunk!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    ax = axes(x, 1)
    N = chunksize(cfg)
    n_chunks = ceil(Int, length(x) / N)
    D = eltype(cfg.duals)
    cfg.duals .= _init_dual_zero.(D, x)
    prev_range = 0:-1  # empty range sentinel
    value = zero(T)
    for i in 1:n_chunks
        zero_block_ϵ1!(cfg.duals, cfg.seeds, prev_range)
        range_i = seed_block_ϵ1!(cfg.duals, cfg.seeds, i, ax)
        for j in i:n_chunks
            range_j = j == i ? range_i : block_range(j, N, ax)
            seed_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
            v = f(cfg.duals)
            check_scalar(v)
            value = v.v
            extract_hessian!(H, v, i, j)
            if j == i
                extract_gradient!(G, v, i)
            end
            zero_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
        end
        prev_range = range_i
    end
    symmetrize!(H)
    return value
end

# HVP functions
"""
    hvp(f, x, v[, cfg])

Compute the Hessian–vector product `H(x) * v`. Uses the directional path by default;
pass a `DirectionalHVPConfig` explicitly to control chunking.
"""
hvp(f::F, x::AbstractVector, v::AbstractVector) where {F} = hvp(f, x, v, DirectionalHVPConfig(x))
hvp_simd(f::F, x::AbstractVector, v::AbstractVector) where {F} = hvp(f, x, v, DirectionalHVPConfigSIMD(x))
hvp(f::F, x::AbstractVector, v::AbstractVector, cfg::DirectionalHVPConfig) where {F} = hvp!(similar(x, eltype(x)), f, x, v, cfg)

hvp!(hv::AbstractVector, f::F, x::AbstractVector, v::AbstractVector) where {F} = hvp!(hv, f, x, v, DirectionalHVPConfig(x))
hvp_simd!(hv::AbstractVector, f::F, x::AbstractVector, v::AbstractVector) where {F} = hvp!(hv, f, x, v, DirectionalHVPConfigSIMD(x))
hvp!(hv::AbstractVector, f::F, x::AbstractVector, v::AbstractVector, cfg::DirectionalHVPConfig) where {F} = hvp_dir!(hv, f, x, v, cfg)

@inline function hvp_dir!(hv::AbstractVector{T}, f::F, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T, F}
    length(x) == length(v) || throw(DimensionMismatch(lazy"x and v must have same length, got $(length(x)) and $(length(v))"))
    length(hv) == length(x) || throw(DimensionMismatch(lazy"hv must have length $(length(x)), got $(length(hv))"))
    if chunksize(cfg) == length(x)
        return hvp_vector_dir!(hv, f, x, v, cfg)
    else
        return hvp_chunk_dir!(hv, f, x, v, cfg)
    end
end

@inline function hvp_vector_dir!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T}
    D = eltype(cfg.duals)
    @inbounds for i in eachindex(x)
        cfg.duals[i] = _init_hvp_dual(D, x[i], cfg.seeds[i], v[i])
    end
    out = f(cfg.duals)
    check_scalar(out)
    @inbounds for i in 1:length(x)
        hv[i] = out.ϵ12[i][1]
    end
    return hv
end

@inline function seed_hvp_dir!(d::AbstractVector{<:AbstractHyperDualNumber{N1}}, seeds, block_i::Int, ax) where {N1}
    range_i = block_range(block_i, N1, ax)
    @inbounds for (offset, idx) in enumerate(range_i)
        d[idx] = seed_epsilon_1(d[idx], seeds[offset])
    end
    return range_i
end

function hvp_chunk_dir!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T}
    N = chunksize(cfg)
    fill!(hv, zero(T))
    n_chunks = ceil(Int, length(x) / N)
    ax = axes(x, 1)
    zeroϵ1 = zero_ϵ(cfg.seeds[1])
    D = eltype(cfg.duals)

    @inbounds for j in eachindex(x)
        cfg.duals[j] = _init_hvp_dual(D, x[j], zeroϵ1, v[j])
    end

    for i in 1:n_chunks
        range_i = seed_hvp_dir!(cfg.duals, cfg.seeds, i, ax)
        out = f(cfg.duals)
        check_scalar(out)
        @inbounds for (I, idx_i) in enumerate(range_i)
            hv[idx_i] = out.ϵ12[I][1]
        end
        zero_block_ϵ1!(cfg.duals, cfg.seeds, range_i)
    end
    return hv
end

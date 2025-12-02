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

DirectionalHVPConfig(x::AbstractVector{T}, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    _DirectionalHVPConfig(x, chunk, Val(1))
DirectionalHVPConfig(x::AbstractVector{T}, ::AbstractVector, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    _DirectionalHVPConfig(x, chunk, Val(1))
DirectionalHVPConfig(x::AbstractVector{T}, ::NTuple{N, <:AbstractVector}, chunk::Chunk = Chunk(x)::Chunk) where {T, N} =
    _DirectionalHVPConfig(x, chunk, Val(N))

function _DirectionalHVPConfig(x::AbstractVector{T}, chunk::Chunk, ::Val{ntangents}) where {T, ntangents}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    ntangents > 0 || throw(ArgumentError(lazy"number of tangents must be positive, got $ntangents"))
    duals = similar(x, HyperDual{N, ntangents, T}) # directional: ε₂ has one lane per tangent
    seeds = collect(construct_seeds(NTuple{N, T}))
    return DirectionalHVPConfig{typeof(duals), typeof(seeds)}(duals, seeds)
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

@inline function seed_block_ϵ1!(d::AbstractVector{<:HyperDual{N1, N2}}, seeds, block_i, ax) where {N1, N2}
    range_i = block_range(block_i, N1, ax)
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        d[idx] = seed_epsilon_1(d[idx], seeds[k])
    end
    return range_i
end

@inline function seed_block_ϵ2!(d::AbstractVector{<:HyperDual{N1, N2}}, seeds, range_j) where {N1, N2}
    @inbounds for k in 1:length(range_j)
        idx = range_j[k]
        d[idx] = seed_epsilon_2(d[idx], seeds[k])
    end
    return nothing
end

@inline function zero_block_ϵ2!(d::AbstractVector{<:HyperDual{N1, N2}}, seeds, range_j) where {N1, N2}
    zeroϵ2 = zero_ϵ(seeds[1])
    @inbounds for k in 1:length(range_j)
        idx = range_j[k]
        d[idx] = seed_epsilon_2(d[idx], zeroϵ2)
    end
    return nothing
end

const TangentBundle = Union{AbstractVector, NTuple{N, V} where {N, V <: AbstractVector}}
const HVBundle = Union{AbstractVector, NTuple{N, V} where {N, V <: AbstractVector}}

@inline tangents_count(::AbstractVector) = 1
@inline tangents_count(::NTuple{N, <:AbstractVector}) where {N} = N

@inline ntangents(::Type{<:HyperDual{<:Any, N2, <:Any}}) where {N2} = N2
@inline ntangents(cfg::DirectionalHVPConfig) = ntangents(eltype(cfg.duals))

@inline function zero_block_ϵ1!(d::AbstractVector{<:HyperDual{N1, N2}}, seeds, range_i) where {N1, N2}
    zeroϵ1 = zero_ϵ(seeds[1])
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        d[idx] = seed_epsilon_1(d[idx], zeroϵ1)
    end
    return nothing
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
    if chunksize(cfg) == length(x)
        return hessian_vector!(H, f, x, cfg)
    else
        return hessian_chunk!(H, f, x, cfg)
    end
end

function hessian_vector_core!(H::AbstractMatrix, G::Union{Nothing, AbstractVector}, f, x::AbstractVector, cfg::HessianConfig)
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    G === nothing || length(G) == length(x) || throw(DimensionMismatch(lazy"gradient must have length $(length(x)), got $(length(G))"))
    cfg.duals .= HyperDual.(x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    if G !== nothing
        G .= Tuple(v.ϵ1)
    end
    extract_hessian!(H, v)
    return v.v
end

hessian_vector!(H::AbstractMatrix, f::F, x::AbstractVector, cfg::HessianConfig) where {F} =
    (hessian_vector_core!(H, nothing, f, x, cfg); H)

function hessian_chunk_core!(H::AbstractMatrix, G::Union{Nothing, AbstractVector}, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    G === nothing || length(G) == length(x) || throw(DimensionMismatch(lazy"gradient must have length $(length(x)), got $(length(G))"))
    ax = axes(x, 1)
    n_chunks = ceil(Int, length(x) / chunksize(cfg))
    cfg.duals .= HyperDual{chunksize(cfg), chunksize(cfg)}.(x)
    prev_range = 0:-1  # empty range sentinel
    value = zero(T)
    for i in 1:n_chunks
        zero_block_ϵ1!(cfg.duals, cfg.seeds, prev_range)
        range_i = seed_block_ϵ1!(cfg.duals, cfg.seeds, i, ax)
        for j in i:n_chunks
            range_j = j == i ? range_i : block_range(j, chunksize(cfg), ax)
            seed_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
            v = f(cfg.duals)
            check_scalar(v)
            value = v.v
            extract_hessian!(H, v, i, j)
            if G !== nothing && j == i
                extract_gradient!(G, v, i)
            end
            zero_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
        end
        prev_range = range_i
    end
    symmetrize!(H)
    return value
end

hessian_chunk!(H::AbstractMatrix, f::F, x::AbstractVector, cfg::HessianConfig) where {F} =
    (hessian_chunk_core!(H, nothing, f, x, cfg); H)

function hessiangradvalue!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
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

hessiangradvalue_vector!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector, cfg::HessianConfig) where {F} =
    hessian_vector_core!(H, G, f, x, cfg)

hessiangradvalue_chunk!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T} =
    hessian_chunk_core!(H, G, f, x, cfg)

"""
    hvp(f, x, tangents[, cfg])

Compute Hessian–vector product(s) `H(x) * v`. `tangents` may be a single vector
or a tuple of vectors (use `(v,)` for multiple directions); bundled tangents are
evaluated in one pass. Pass a `DirectionalHVPConfig` explicitly to control
chunking or reuse allocations.
"""
@inline similar_output(x::AbstractVector{T}, _::AbstractVector) where {T} = similar(x, T)
@inline function similar_output(x::AbstractVector{T}, _::NTuple{N, <:AbstractVector}) where {T, N}
    return ntuple(_ -> similar(x, T), Val(N))
end

function check_grad_dims(g::AbstractVector, n::Int)
    length(g) == n || throw(DimensionMismatch(lazy"gradient must have length $n, got $(length(g))"))
    return nothing
end

function check_tangent_dims(x, v::AbstractVector)
    length(v) == length(x) || throw(DimensionMismatch(lazy"tangent must have length $(length(x)), got $(length(v))"))
    return nothing
end
function check_tangent_dims(x, v::NTuple{N, <:AbstractVector}) where {N}
    expected = length(x)
    for (i, tangent) in enumerate(v)
        length(tangent) == expected || throw(DimensionMismatch(lazy"tangent $i must have length $expected, got $(length(tangent))"))
    end
    return nothing
end

function check_output_dims(hv::AbstractVector, n::Int, ntan::Int)
    ntan == 1 || throw(DimensionMismatch(lazy"hv is a vector but $ntan tangents were provided"))
    length(hv) == n || throw(DimensionMismatch(lazy"hv must have length $n, got $(length(hv))"))
    return nothing
end
function check_output_dims(hv::NTuple{NT, <:AbstractVector}, n::Int, ntan::Int) where {NT}
    NT == ntan || throw(DimensionMismatch(lazy"hv tuple length $NT does not match number of tangents $ntan"))
    for (i, h) in enumerate(hv)
        length(h) == n || throw(DimensionMismatch(lazy"hv tangent $i must have length $n, got $(length(h))"))
    end
    return nothing
end

@inline directional_ϵ2(v::AbstractVector, idx, ::Val{1}) =
    (@static USE_SIMD ? Vec((v[idx],)) : (v[idx],))
@inline function directional_ϵ2(v::NTuple{N, <:AbstractVector}, idx, ::Val{N}) where {N}
    vals = ntuple(j -> v[j][idx], Val(N))
    return @static USE_SIMD ? Vec(vals) : vals
end

@inline function store_hvp!(hv::AbstractVector, idx, vals, ::Val{1})
    hv[idx] = vals[1]
    return nothing
end
@inline function store_hvp!(hv::NTuple{N, <:AbstractVector}, idx, vals, ::Val{N}) where {N}
    @inbounds for j in 1:N
        hv[j][idx] = vals[j]
    end
    return nothing
end

@inline fill_output!(hv::AbstractVector, z) = fill!(hv, z)
@inline function fill_output!(hv::NTuple{N, <:AbstractVector}, z) where {N}
    for h in hv
        fill!(h, z)
    end
    return hv
end

hvp(f::F, x::AbstractVector, v::TangentBundle) where {F} =
    hvp(f, x, v, DirectionalHVPConfig(x, v))
hvp(f::F, x::AbstractVector, v::TangentBundle, cfg::DirectionalHVPConfig) where {F} =
    hvp!(similar_output(x, v), f, x, v, cfg)

hvp!(hv::HVBundle, f::F, x::AbstractVector, v::TangentBundle) where {F} =
    hvp!(hv, f, x, v, DirectionalHVPConfig(x, v))
hvp!(hv::HVBundle, f::F, x::AbstractVector, v::TangentBundle, cfg::DirectionalHVPConfig) where {F} =
    hvp_dir!(hv, f, x, v, cfg)

@inline function hvp_dir!(hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig) where {T, F}
    n_tangents = tangents_count(v)
    n_tangents == ntangents(cfg) ||
        throw(DimensionMismatch(lazy"config expects $(ntangents(cfg)) tangents, but $(n_tangents) were provided"))
    check_tangent_dims(x, v)
    check_output_dims(hv, length(x), n_tangents)
    valN = Val(n_tangents)
    if chunksize(cfg) == length(x)
        return hvp_vector_dir!(hv, f, x, v, cfg, valN)
    else
        return hvp_chunk_dir!(hv, f, x, v, cfg, valN)
    end
end

@inline hvp_vector_dir!(hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig, valN::Val{N}) where {T, N} =
    hvpgrad_vector_dir_core!(nothing, hv, f, x, v, cfg, valN)

# Gradient + HVP (directional)
function hvpgrad(f::F, x::AbstractVector, v::TangentBundle) where {F}
    g = similar(x, eltype(x))
    hv = similar_output(x, v)
    hvpgrad!(g, hv, f, x, v, DirectionalHVPConfig(x, v))
    return (; gradient = g, hvp = hv)
end
function hvpgrad(f::F, x::AbstractVector, v::TangentBundle, cfg::DirectionalHVPConfig) where {F}
    g = similar(x, eltype(x))
    hv = similar_output(x, v)
    hvpgrad!(g, hv, f, x, v, cfg)
    return (; gradient = g, hvp = hv)
end

function hvpgrad!(g::AbstractVector, hv::HVBundle, f::F, x::AbstractVector, v::TangentBundle) where {F}
    return hvpgrad!(g, hv, f, x, v, DirectionalHVPConfig(x, v))
end
function hvpgrad!(g::AbstractVector, hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig) where {F, T}
    n_tangents = tangents_count(v)
    n_tangents == ntangents(cfg) ||
        throw(DimensionMismatch(lazy"config expects $(ntangents(cfg)) tangents, but $(n_tangents) were provided"))
    check_grad_dims(g, length(x))
    check_tangent_dims(x, v)
    check_output_dims(hv, length(x), n_tangents)
    valN = Val(n_tangents)
    if chunksize(cfg) == length(x)
        hvpgrad_vector_dir!(g, hv, f, x, v, cfg, valN)
    else
        hvpgrad_chunk_dir!(g, hv, f, x, v, cfg, valN)
    end
    return (; gradient = g, hvp = hv)
end

@inline hvpgrad_vector_dir!(g::AbstractVector{T}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig, valN::Val{N}) where {T, N} =
    hvpgrad_vector_dir_core!(g, hv, f, x, v, cfg, valN)

@inline function hvpgrad_chunk_dir_core!(g::Union{Nothing, AbstractVector{T}}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig, ::Val{N}) where {T, N}
    Nchunk = chunksize(cfg)
    fill_output!(hv, zero(T))
    n_chunks = ceil(Int, length(x) / Nchunk)
    ax = axes(x, 1)
    zeroϵ1 = zero_ϵ(cfg.seeds[1])

    # Initialize ε₂ once and keep ε₁ zeroed globally.
    @inbounds for j in eachindex(x)
        cfg.duals[j] = HyperDual(x[j], zeroϵ1, directional_ϵ2(v, j, Val(N)))
    end

    for i in 1:n_chunks
        range_i = seed_hvp_dir!(cfg.duals, cfg.seeds, i, ax)
        out = f(cfg.duals)
        check_scalar(out)
        @inbounds for (I, idx_i) in enumerate(range_i)
            if g !== nothing
                g[idx_i] = out.ϵ1[I]
            end
            store_hvp!(hv, idx_i, out.ϵ12[I], Val(N))
        end
        zero_block_ϵ1!(cfg.duals, cfg.seeds, range_i)
    end
    return g === nothing ? hv : (g, hv)
end

@inline function seed_hvp_dir!(d::AbstractVector{<:HyperDual{N1, N2, <:Any}}, seeds, block_i::Int, ax) where {N1, N2}
    range_i = block_range(block_i, N1, ax)

    # Seed ε₁ block rows without creating views
    @inbounds for (offset, idx) in enumerate(range_i)
        d[idx] = seed_epsilon_1(d[idx], seeds[offset])
    end
    return range_i
end

function hvp_chunk_dir!(hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig, ::Val{N}) where {F, T, N}
    return hvpgrad_chunk_dir_core!(nothing, hv, f, x, v, cfg, Val(N))
end

@inline function hvpgrad_vector_dir_core!(g::Union{Nothing, AbstractVector{T}}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig, ::Val{N}) where {T, N}
    # Seed ε₁ with identity directions and ε₂ with the bundled tangents;
    # the mixed term ε₁ᵀ A ε₂ yields each H * v column in ϵ₁₂.
    @inbounds for i in eachindex(x)
        cfg.duals[i] = HyperDual(x[i], cfg.seeds[i], directional_ϵ2(v, i, Val(N)))
    end
    out = f(cfg.duals)
    check_scalar(out)
    @inbounds for i in 1:length(x)
        if g !== nothing
            g[i] = out.ϵ1[i]
        end
        store_hvp!(hv, i, out.ϵ12[i], Val(N))
    end
    return g === nothing ? hv : (g, hv)
end

hvpgrad_chunk_dir!(g::AbstractVector{T}, hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::DirectionalHVPConfig, valN::Val{N}) where {F, T, N} =
    hvpgrad_chunk_dir_core!(g, hv, f, x, v, cfg, valN)

struct HessianConfig{D <: AbstractVector{<:HyperDual}, S}
    duals::D
    seeds::S
end
(chunksize(cfg::HessianConfig)::Int) = length(cfg.seeds)

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
struct DirectionalHVPConfig{D <: AbstractVector{<:HyperDual}, S}
    duals::D
    seeds::S
end
(chunksize(cfg::DirectionalHVPConfig)::Int) = length(cfg.seeds)

function DirectionalHVPConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, 1, T}) # directional: one ε₂ lane
    seeds = collect(construct_seeds(NTuple{N, T}))
    return DirectionalHVPConfig(duals, seeds)
end

struct GradientConfig{D <: AbstractVector{<:HyperDual}, S}
    duals::D
    seeds::S
end
(chunksize(cfg::GradientConfig)::Int) = length(cfg.seeds)

function GradientConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, 1, T}) # keep ε₂ length 1 and zero-seeded
    seeds = collect(construct_seeds(NTuple{N, T}))
    return GradientConfig(duals, seeds)
end

@generated function single_seed(::Type{NTuple{N, T}}, ::Val{i}) where {N, T, i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(T)), :(zero(T))) for j in 1:N]...)
    return :(Vec($(ex)))
end

@generated construct_seeds(::Type{NTuple{N, T}}) where {N, T} =
    Expr(:tuple, [:(single_seed(NTuple{N, T}, Val{$i}())) for i in 1:N]...)

seed_epsilon_1(d::HyperDual{N1, N2, T}, ϵ1) where {N1, N2, T} = HyperDual{N1, N2, T}(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::HyperDual{N1, N2, T}, ϵ2) where {N1, N2, T} = HyperDual{N1, N2, T}(d.v, d.ϵ1, ϵ2, d.ϵ12)

function seed!(d::AbstractVector{<:HyperDual{N1, N2}}, x, seeds, block_i, block_j) where {N1, N2}
    d .= HyperDual{N1, N2}.(x)
    index_i = (block_i - 1) * N1 + 1
    index_j = (block_j - 1) * N2 + 1
    range_i = index_i:min(length(x), (index_i + N1 - 1))
    range_j = index_j:min(length(x), (index_j + N2 - 1))
    chunks_i = length(range_i)
    chunks_j = length(range_j)

    d[range_i] .= seed_epsilon_1.(view(d, range_i), view(seeds, 1:chunks_i))
    d[range_j] .= seed_epsilon_2.(view(d, range_j), view(seeds, 1:chunks_j))
    return d
end

@inline function seed_gradient!(d::AbstractVector{<:HyperDual{N1, N2, T}}, x, seeds, block_i, zeroϵ1, zeroϵ2) where {N1, N2, T}
    index_i = (block_i - 1) * N1 + 1
    range_i = index_i:min(length(x), (index_i + N1 - 1))

    @inbounds for j in eachindex(x)
        d[j] = HyperDual(x[j], zeroϵ1, zeroϵ2)
    end

    @inbounds for (offset, idx) in enumerate(range_i)
        d[idx] = seed_epsilon_1(d[idx], seeds[offset])
    end
    return range_i
end

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

function extract_hessian!(H::AbstractMatrix, v::HyperDual{N1, N2}, block_i::Int, block_j::Int) where {N1, N2}
    index_i = (block_i - 1) * N1 + 1
    index_j = (block_j - 1) * N2 + 1
    range_i = index_i:(index_i + N1 - 1)
    range_j = index_j:(index_j + N2 - 1)

    for (I, i) in enumerate(range_i)
        for (J, j) in enumerate(range_j)
            if checkbounds(Bool, H, i, j)
                H[i, j] = v.ϵ12[I][J]
            end
        end
    end
    return H
end

function extract_gradient!(G::AbstractVector, v::HyperDual{N1, N2}, block_i::Int) where {N1, N2}
    Base.require_one_based_indexing(G)
    index_i = (block_i - 1) * N1 + 1
    range_i = index_i:min(length(G), (index_i + N1 - 1))
    for (I, i) in enumerate(range_i)
        G[i] = v.ϵ1[I]
    end
    return G
end

function gradient(f, x::Real)
    dual = HyperDual(x, Vec(one(x)), Vec(zero(x)))
    v = f(dual)
    check_scalar(v)
    return @inbounds v.ϵ1[1]
end

function hessian(f, x::Real)
    dual = HyperDual(x, Vec(one(x)), Vec(one(x)))
    v = f(dual)
    check_scalar(v)
    return @inbounds v.ϵ12[1][1]
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

function hessian_vector!(H::AbstractMatrix, f, x::AbstractVector, cfg::HessianConfig)
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    cfg.duals .= HyperDual.(x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    return extract_hessian!(H, v)
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
    symmetrize!(H)
    return H
end

gradient(f::F, x::AbstractVector) where {F} = gradient!(similar(x, axes(x, 1)), f, x, GradientConfig(x))
gradient(f::F, x::AbstractVector, cfg::GradientConfig) where {F} = gradient!(similar(x, axes(x, 1)), f, x, cfg)

gradient!(G::AbstractVector, f::F, x::AbstractVector) where {F} = gradient!(G, f, x, GradientConfig(x))
function gradient!(G::AbstractVector, f::F, x::AbstractVector{T}, cfg::GradientConfig) where {F, T}
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    if chunksize(cfg) == length(x)
        gradientvalue_vector!(G, f, x, cfg)
    else
        gradientvalue_chunk!(G, f, x, cfg)
    end
    return G
end

function gradientvalue(f::F, x::AbstractVector) where {F}
    cfg = GradientConfig(x)
    return gradientvalue(f, x, cfg)
end
function gradientvalue(f::F, x::AbstractVector, cfg::GradientConfig) where {F}
    G = similar(x, axes(x, 1))
    value = gradientvalue!(G, f, x, cfg)
    return (; value = value, gradient = G)
end

gradientvalue!(G::AbstractVector, f::F, x::AbstractVector) where {F} = gradientvalue!(G, f, x, GradientConfig(x))
function gradientvalue!(G::AbstractVector, f::F, x::AbstractVector{T}, cfg::GradientConfig) where {F, T}
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    if chunksize(cfg) == length(x)
        return gradientvalue_vector!(G, f, x, cfg)
    else
        return gradientvalue_chunk!(G, f, x, cfg)
    end
end

function gradientvalue_vector!(G::AbstractVector, f, x::AbstractVector{T}, cfg::GradientConfig) where {T}
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    zeroϵ2 = zero(Vec{1, T})
    @inbounds for i in eachindex(x)
        cfg.duals[i] = HyperDual(x[i], cfg.seeds[i], zeroϵ2)
    end
    v = f(cfg.duals)
    check_scalar(v)
    G .= Tuple(v.ϵ1)
    return v.v
end

function gradientvalue_chunk!(G::AbstractVector, f, x::AbstractVector{T}, cfg::GradientConfig) where {T}
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    N = chunksize(cfg)
    n_chunks = ceil(Int, length(x) / N)
    zeroϵ1 = zero(cfg.seeds[1])
    zeroϵ2 = zero(Vec{1, T})
    value = zero(T)
    for i in 1:n_chunks
        seed_gradient!(cfg.duals, x, cfg.seeds, i, zeroϵ1, zeroϵ2)
        v = f(cfg.duals)
        check_scalar(v)
        value = v.v
        extract_gradient!(G, v, i)
    end
    return value
end

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

function hessiangradvalue_vector!(H::AbstractMatrix, G::AbstractVector, f, x::AbstractVector, cfg::HessianConfig)
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    cfg.duals .= HyperDual.(x, cfg.seeds, cfg.seeds)
    v = f(cfg.duals)
    check_scalar(v)
    G .= Tuple(v.ϵ1)
    extract_hessian!(H, v)
    return v.v
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
    symmetrize!(H)
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
    if chunksize(cfg) == length(x)
        return hvp_vector_dir!(hv, f, x, v, cfg)
    else
        return hvp_chunk_dir!(hv, f, x, v, cfg)
    end
end

@inline function hvp_vector_dir!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T}
    # Seed ε₁ with identity directions and ε₂ with the directional vector v;
    # the mixed term ε₁ᵀ A ε₂ yields (H * v)[i] in ϵ₁₂[i][1].
    @inbounds for i in eachindex(x)
        cfg.duals[i] = HyperDual(x[i], cfg.seeds[i], Vec((v[i],)))
    end
    out = f(cfg.duals)
    check_scalar(out)
    @inbounds for i in 1:length(x)
        hv[i] = out.ϵ12[i][1]
    end
    return hv
end

@inline function seed_hvp_dir!(d::AbstractVector{<:HyperDual{N1, 1}}, x, seeds, v, block_i::Int) where {N1}
    index_i = (block_i - 1) * N1 + 1
    range_i = index_i:min(length(x), (index_i + N1 - 1))
    chunks_i = length(range_i)

    # Seed ε₁ block rows without creating views
    @inbounds for (offset, idx) in enumerate(range_i)
        d[idx] = seed_epsilon_1(d[idx], seeds[offset])
    end
    return range_i
end

function hvp_chunk_dir!(hv::AbstractVector{T}, f, x::AbstractVector{T}, v::AbstractVector{T}, cfg::DirectionalHVPConfig) where {T}
    N = chunksize(cfg)
    fill!(hv, zero(T))
    n_chunks = ceil(Int, length(x) / N)

    for i in 1:n_chunks
        # Reset ε₁ to zero and ε₂ to the direction for this chunk pass.
        zeroϵ1 = zero(cfg.seeds[1])
        @inbounds for j in eachindex(x)
            cfg.duals[j] = HyperDual(x[j], zeroϵ1, Vec((v[j],)))
        end

        range_i = seed_hvp_dir!(cfg.duals, x, cfg.seeds, v, i)
        out = f(cfg.duals)
        check_scalar(out)
        @inbounds for (I, idx_i) in enumerate(range_i)
            hv[idx_i] = out.ϵ12[I][1]
        end
    end
    return hv
end

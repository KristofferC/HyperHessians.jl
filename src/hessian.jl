mutable struct HessianConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::D
    const seeds::S
end
@inline _chunksize(::Type{<:NTuple{N}}) where {N} = N
@inline _chunksize(seeds) = something(_chunksize(eltype(seeds)), length(seeds))
(chunksize(cfg::HessianConfig)::Int) = _chunksize(cfg.seeds)::Int

function HessianConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk) where {T}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    duals = similar(x, HyperDual{N, N, T}) # not Vector
    seeds = collect(construct_seeds(NTuple{N, T}))
    return HessianConfig(duals, seeds)
end
HessianConfig(x::AbstractArray{T}, chunk = Chunk(x)::Chunk) where {T} =
    HessianConfig(vec(x), chunk)

"""
    HVPConfig(x; chunk=Chunk(x))

Configuration for Hessian–vector products.
"""
mutable struct HVPConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::D
    const seeds::S
end
(chunksize(cfg::HVPConfig)::Int) = _chunksize(cfg.seeds)

HVPConfig(x::AbstractVector{T}, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    _HVPConfig(x, chunk, Val(1))
HVPConfig(x::AbstractVector{T}, ::AbstractVector, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    _HVPConfig(x, chunk, Val(1))
HVPConfig(x::AbstractVector{T}, ::NTuple{N, <:AbstractVector}, chunk::Chunk = Chunk(x)::Chunk) where {T, N} =
    _HVPConfig(x, chunk, Val(N))
HVPConfig(x::AbstractArray{T}, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    HVPConfig(vec(x), chunk)
HVPConfig(x::AbstractArray{T}, v::AbstractVector, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    HVPConfig(vec(x), vec(v), chunk)
HVPConfig(x::AbstractArray{T}, v::NTuple{N, <:AbstractArray}, chunk::Chunk = Chunk(x)::Chunk) where {T, N} =
    HVPConfig(vec(x), ntuple(i -> vec(v[i]), Val(N)), chunk)

function _HVPConfig(x::AbstractVector{T}, chunk::Chunk, ::Val{ntangents}) where {T, ntangents}
    N = chunksize(chunk)
    N > 0 || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    ntangents > 0 || throw(ArgumentError(lazy"number of tangents must be positive, got $ntangents"))
    duals = similar(x, HyperDual{N, ntangents, T}) # directional: ε₂ has one lane per tangent
    seeds = collect(construct_seeds(NTuple{N, T}))
    return HVPConfig{typeof(duals), typeof(seeds)}(duals, seeds)
end

@generated function single_seed(::Type{NTuple{N, T}}, ::Val{i}) where {N, T, i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(T)), :(zero(T))) for j in 1:N]...)
    return ex
end

@generated construct_seeds(::Type{NTuple{N, T}}) where {N, T} =
    Expr(:tuple, [:(single_seed(NTuple{N, T}, Val{$i}())) for i in 1:N]...)

"""
    VHVPConfig(x, v)

Configuration for directional quadratic forms `vhvp`/`vhvp_gradient_value`.
Caches a buffer used for constructing `x + t*v` without allocations.
"""
mutable struct VHVPConfig{B <: AbstractVector}
    buffer::B
end

function VHVPConfig(x::AbstractArray, v::AbstractArray)
    length(vec(x)) == length(vec(v)) || throw(DimensionMismatch(lazy"tangent must have length $(length(x)), got $(length(v))"))
    baseT = promote_type(eltype(x), eltype(v))
    buffer = similar(vec(x), HyperDual{1, 1, baseT})
    return VHVPConfig(buffer)
end

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

const TangentBundle = Union{
    AbstractVector,
    AbstractArray,
    NTuple{N, V} where {N, V <: AbstractVector},
    NTuple{N, V} where {N, V <: AbstractArray},
}
const HVBundle = Union{AbstractVector, NTuple{N, V} where {N, V <: AbstractVector}}

@inline tangents_count(::AbstractVector) = 1
@inline tangents_count(::AbstractArray) = 1
@inline tangents_count(::NTuple{N, <:AbstractVector}) where {N} = N
@inline tangents_count(::NTuple{N, <:AbstractArray}) where {N} = N

@inline ntangents(::Type{<:HyperDual{<:Any, N2, <:Any}}) where {N2} = N2
@inline ntangents(cfg::HVPConfig) = ntangents(eltype(cfg.duals))

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

@inline _vectorize_input(f, x::AbstractVector) = (f, x, nothing)
@inline function _vectorize_input(f, x::AbstractArray)
    shape = size(x)
    f_vec = let f = f, shape = shape
        y -> f(reshape(y, shape))
    end
    return f_vec, vec(x), shape
end

@inline _vectorize_tangents(v::AbstractVector, _shape) = (v, nothing)
@inline function _vectorize_tangents(v::AbstractArray, shape)
    shape === nothing && return (v, nothing)
    return vec(v), shape
end
@inline function _vectorize_tangents(v::NTuple{N, <:AbstractArray}, shape) where {N}
    shape === nothing && return v, nothing
    return ntuple(i -> vec(v[i]), Val(N)), shape
end

function _check_config_length(cfg::HessianConfig, n::Int)
    length(cfg.duals) == n || throw(DimensionMismatch(lazy"config size $(length(cfg.duals)) does not match input length $n"))
    return nothing
end

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
    return hessian_gradient_value(f, x).hessian
end

"""
    hessian_gradient_value(f, x::Real)

Compute value, first derivative, and second derivative for a scalar input.
Returns a named tuple `(value, gradient, hessian)` where `gradient` and
`hessian` are numbers.
"""
function hessian_gradient_value(f, x::Real)
    seed = single_seed(NTuple{1, typeof(x)}, Val(1))
    dual = HyperDual(x, seed, seed)
    v = f(dual)
    return (;
        value = map(v -> v.v, v),
        gradient = map(v -> v.ϵ1[1], v),
        hessian = map(v -> v.ϵ12[1][1], v),
    )
end

hessian(f::F, x::AbstractVector) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfig(x))
hessian(f::F, x::AbstractVector, cfg::HessianConfig) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, cfg)
hessian(f::F, x::AbstractArray) where {F} = begin
    f_vec, x_vec, _ = _vectorize_input(f, x)
    hessian(f_vec, x_vec)
end
function hessian(f::F, x::AbstractArray, cfg::HessianConfig) where {F}
    f_vec, x_vec, _ = _vectorize_input(f, x)
    _check_config_length(cfg, length(x_vec))
    return hessian(f_vec, x_vec, cfg)
end

function hessian!(H::AbstractMatrix, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
    if chunksize(cfg) == length(x)
        return hessian_vector!(H, f, x, cfg)
    else
        return hessian_chunk!(H, f, x, cfg)
    end
end
hessian!(H::AbstractMatrix, f::F, x::AbstractArray) where {F} = begin
    f_vec, x_vec, _ = _vectorize_input(f, x)
    hessian!(H, f_vec, x_vec, HessianConfig(x_vec))
end
function hessian!(H::AbstractMatrix, f::F, x::AbstractArray, cfg::HessianConfig) where {F}
    f_vec, x_vec, _ = _vectorize_input(f, x)
    _check_config_length(cfg, length(x_vec))
    return hessian!(H, f_vec, x_vec, cfg)
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

function hessian_gradient_value!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    if chunksize(cfg) == length(x)
        return hessian_gradient_value_vector!(H, G, f, x, cfg)
    else
        return hessian_gradient_value_chunk!(H, G, f, x, cfg)
    end
end

function hessian_gradient_value!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector) where {F}
    cfg = HessianConfig(x)
    return hessian_gradient_value!(H, G, f, x, cfg)
end
hessian_gradient_value!(H::AbstractMatrix, G::AbstractArray, f::F, x::AbstractArray) where {F} = begin
    f_vec, x_vec, _ = _vectorize_input(f, x)
    g_vec = vec(G)
    length(g_vec) == length(x_vec) || throw(DimensionMismatch(lazy"G must have length $(length(x_vec)), got $(length(g_vec))"))
    hessian_gradient_value!(H, g_vec, f_vec, x_vec, HessianConfig(x_vec))
end
function hessian_gradient_value!(H::AbstractMatrix, G::AbstractArray, f::F, x::AbstractArray, cfg::HessianConfig) where {F}
    f_vec, x_vec, _ = _vectorize_input(f, x)
    g_vec = vec(G)
    length(g_vec) == length(x_vec) || throw(DimensionMismatch(lazy"G must have length $(length(x_vec)), got $(length(g_vec))"))
    _check_config_length(cfg, length(x_vec))
    return hessian_gradient_value!(H, g_vec, f_vec, x_vec, cfg)
end

function hessian_gradient_value(f::F, x::AbstractVector, cfg::HessianConfig) where {F}
    G = similar(x, axes(x, 1))
    H = similar(x, axes(x, 1), axes(x, 1))
    value = hessian_gradient_value!(H, G, f, x, cfg)
    return (; value = value, gradient = G, hessian = H)
end

function hessian_gradient_value(f::F, x::AbstractVector) where {F}
    cfg = HessianConfig(x)
    return hessian_gradient_value(f, x, cfg)
end
hessian_gradient_value(f::F, x::AbstractArray, cfg::HessianConfig) where {F} =
    _hessian_gradient_value_array(f, x, cfg)
hessian_gradient_value(f::F, x::AbstractArray) where {F} =
    _hessian_gradient_value_array(f, x, nothing)

@inline function _hessian_gradient_value_array(f::F, x::AbstractArray, cfg::Union{HessianConfig, Nothing}) where {F}
    f_vec, x_vec, shape = _vectorize_input(f, x)
    res = if cfg === nothing
        hessian_gradient_value(f_vec, x_vec)
    else
        _check_config_length(cfg, length(x_vec))
        hessian_gradient_value(f_vec, x_vec, cfg)
    end
    grad = shape === nothing ? res.gradient : reshape(res.gradient, shape)
    return (; value = res.value, gradient = grad, hessian = res.hessian)
end

hessian_gradient_value_vector!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector, cfg::HessianConfig) where {F} =
    hessian_vector_core!(H, G, f, x, cfg)

hessian_gradient_value_chunk!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T} =
    hessian_chunk_core!(H, G, f, x, cfg)

"""
    hvp(f, x, tangents[, cfg])

Compute Hessian–vector product(s) `H(x) * v`. `tangents` may be a single vector
or a tuple of vectors (use `(v,)` for multiple directions); bundled tangents are
evaluated in one pass. Pass a `HVPConfig` explicitly to control
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

function check_tangent_dims(x::AbstractArray, v::AbstractArray)
    length(v) == length(x) || throw(DimensionMismatch(lazy"tangent must have length $(length(x)), got $(length(v))"))
    return nothing
end
function check_tangent_dims(x::AbstractArray, v::NTuple{N, <:AbstractArray}) where {N}
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

@inline function _vhvp_fill!(buf, x, v, t)
    @inbounds for i in eachindex(x)
        buf[i] = x[i] + t * v[i]
    end
    return buf
end

@inline _vhvp_zero(::VHVPConfig{<:AbstractVector{<:HyperDual{1, 1, T}}}) where {T} = zero(T)

"""
    vhvp(f, x, v[, cfg])

Directional second derivative `v' * H(x) * v`, computed by differentiating
`t -> f(x + t * v)` at `t = 0`.
"""
function vhvp(f::F, x::AbstractArray, v::AbstractArray, cfg::VHVPConfig) where {F}
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, _ = _vectorize_tangents(v, shape)
    check_tangent_dims(x_vec, v_vec)
    length(cfg.buffer) == length(x_vec) || throw(DimensionMismatch(lazy"config size $(length(cfg.buffer)) does not match input length $(length(x_vec))"))
    g = t -> begin
        _vhvp_fill!(cfg.buffer, x_vec, v_vec, t)
        f_vec(cfg.buffer)
    end
    return hessian(g, _vhvp_zero(cfg))
end
vhvp(f::F, x::AbstractArray, v::AbstractArray) where {F} =
    vhvp(f, x, v, VHVPConfig(x, v))
function vhvp(f::F, x::Number, v::Number) where {F}
    t0 = zero(promote_type(typeof(x), typeof(v)))
    return hessian(t -> f(x + t * v), t0)
end

"""
    vhvp_gradient_value(f, x, v[, cfg])

Directional value/gradient/second-derivative along `v`. The returned named
tuple has fields `value`, `gradient` (`dot(∇f(x), v)`), and
`hessian` (`v' * H(x) * v`).
"""
function vhvp_gradient_value(f::F, x::AbstractArray, v::AbstractArray, cfg::VHVPConfig) where {F}
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, _ = _vectorize_tangents(v, shape)
    check_tangent_dims(x_vec, v_vec)
    length(cfg.buffer) == length(x_vec) || throw(DimensionMismatch(lazy"config size $(length(cfg.buffer)) does not match input length $(length(x_vec))"))
    g = t -> begin
        _vhvp_fill!(cfg.buffer, x_vec, v_vec, t)
        f_vec(cfg.buffer)
    end
    return hessian_gradient_value(g, _vhvp_zero(cfg))
end
vhvp_gradient_value(f::F, x::AbstractArray, v::AbstractArray) where {F} =
    vhvp_gradient_value(f, x, v, VHVPConfig(x, v))
function vhvp_gradient_value(f::F, x::Number, v::Number) where {F}
    t0 = zero(promote_type(typeof(x), typeof(v)))
    return hessian_gradient_value(t -> f(x + t * v), t0)
end

@inline directional_ϵ2(v::AbstractArray, idx, ::Val{1}) = (v[idx],)
@inline function directional_ϵ2(v::NTuple{N, <:AbstractArray}, idx, ::Val{N}) where {N}
    return ntuple(j -> v[j][idx], Val(N))
end

@inline function store_hvp!(hv::AbstractArray, idx, vals, ::Val{1})
    hv[idx] = vals[1]
    return nothing
end
@inline function store_hvp!(hv::NTuple{N, <:AbstractArray}, idx, vals, ::Val{N}) where {N}
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

@inline function _reshape_output(hv::AbstractVector, shape, orig_shape)
    return orig_shape === nothing ? hv : reshape(hv, shape)
end
@inline function _reshape_output(hv::NTuple{N, <:AbstractVector}, shape, orig_shape) where {N}
    orig_shape === nothing && return hv
    return ntuple(i -> reshape(hv[i], shape), Val(N))
end

function _hvp_common_array!(hv, g, f, x, v, cfg::Union{HVPConfig, Nothing})
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, orig_shape = _vectorize_tangents(v, shape)
    cfg = cfg === nothing ? HVPConfig(x_vec, v_vec) : cfg
    g_vec = g === nothing ? nothing : vec(g)
    hv_vec = hv isa NTuple ? ntuple(i -> vec(hv[i]), Val(length(hv))) : vec(hv)
    value = g_vec === nothing ?
        hvp!(hv_vec, f_vec, x_vec, v_vec, cfg) :
        hvp_gradient_value!(hv_vec, g_vec, f_vec, x_vec, v_vec, cfg)
    if g_vec !== nothing
        g .= _reshape_output(g_vec, shape, shape)
    end
    hv_out = _reshape_output(hv_vec, shape, orig_shape)
    if hv isa NTuple
        for i in eachindex(hv)
            hv[i] .= hv_out[i]
        end
    else
        hv .= hv_out
    end
    return g_vec === nothing ? hv : value
end

function _hvp_gradient_value_array(f, x, v, cfg::Union{HVPConfig, Nothing})
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, orig_shape = _vectorize_tangents(v, shape)
    cfg = cfg === nothing ? HVPConfig(x_vec, v_vec) : cfg
    res = hvp_gradient_value(f_vec, x_vec, v_vec, cfg)
    grad = _reshape_output(res.gradient, shape, shape)
    hv = _reshape_output(res.hvp, shape, orig_shape)
    return (; value = res.value, gradient = grad, hvp = hv)
end

hvp(f::F, x::AbstractVector, v::TangentBundle) where {F} =
    hvp(f, x, v, HVPConfig(x, v))
hvp(f::F, x::AbstractVector, v::TangentBundle, cfg::HVPConfig) where {F} =
    hvp!(similar_output(x, v), f, x, v, cfg)
hvp(f::F, x::AbstractArray{T, N}, v::TangentBundle) where {F, T, N} = begin
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, orig_shape = _vectorize_tangents(v, shape)
    hv = hvp(f_vec, x_vec, v_vec, HVPConfig(x_vec, v_vec))
    return _reshape_output(hv, shape, orig_shape)
end
function hvp(f::F, x::AbstractArray{T, N}, v::TangentBundle, cfg::HVPConfig) where {F, T, N}
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, orig_shape = _vectorize_tangents(v, shape)
    return _reshape_output(hvp(f_vec, x_vec, v_vec, cfg), shape, orig_shape)
end

hvp!(hv::HVBundle, f::F, x::AbstractVector, v::TangentBundle) where {F} =
    hvp!(hv, f, x, v, HVPConfig(x, v))
hvp!(hv::HVBundle, f::F, x::AbstractVector, v::TangentBundle, cfg::HVPConfig) where {F} =
    hvp_dir!(hv, f, x, v, cfg)
hvp!(hv::AbstractArray, f::F, x::AbstractArray, v::TangentBundle) where {F} = begin
    _hvp_common_array!(hv, nothing, f, x, v, nothing)
end
function hvp!(hv::AbstractArray, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {F}
    return _hvp_common_array!(hv, nothing, f, x, v, cfg)
end
function hvp!(hv::NTuple{N, <:AbstractArray}, f::F, x::AbstractArray, v::TangentBundle) where {N, F}
    return _hvp_common_array!(hv, nothing, f, x, v, nothing)
end
function hvp!(hv::NTuple{N, <:AbstractArray}, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {N, F}
    return _hvp_common_array!(hv, nothing, f, x, v, cfg)
end

@inline function hvp_dir!(hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig) where {T, F}
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

@inline hvp_vector_dir!(hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, valN::Val{N}) where {T, N} =
    hvp_gradient_value_vector_dir_core!(nothing, hv, f, x, v, cfg, valN)

# Gradient + HVP + value (directional)
hvp_gradient_value(f::F, x::AbstractVector, v::TangentBundle) where {F} =
    hvp_gradient_value(f, x, v, HVPConfig(x, v))
function hvp_gradient_value(f::F, x::AbstractVector, v::TangentBundle, cfg::HVPConfig) where {F}
    g = similar(x, eltype(x))
    hv = similar_output(x, v)
    value = hvp_gradient_value!(hv, g, f, x, v, cfg)
    return (; value = value, gradient = g, hvp = hv)
end
hvp_gradient_value(f::F, x::AbstractArray{T, N}, v::TangentBundle) where {F, T, N} =
    _hvp_gradient_value_array(f, x, v, nothing)
function hvp_gradient_value(f::F, x::AbstractArray{T, N}, v::TangentBundle, cfg::HVPConfig) where {F, T, N}
    return _hvp_gradient_value_array(f, x, v, cfg)
end

function hvp_gradient_value!(hv::HVBundle, g::AbstractVector, f::F, x::AbstractVector, v::TangentBundle) where {F}
    return hvp_gradient_value!(hv, g, f, x, v, HVPConfig(x, v))
end
function hvp_gradient_value!(hv::HVBundle, g::AbstractVector, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig) where {F, T}
    n_tangents = tangents_count(v)
    n_tangents == ntangents(cfg) ||
        throw(DimensionMismatch(lazy"config expects $(ntangents(cfg)) tangents, but $(n_tangents) were provided"))
    check_grad_dims(g, length(x))
    check_tangent_dims(x, v)
    check_output_dims(hv, length(x), n_tangents)
    valN = Val(n_tangents)
    if chunksize(cfg) == length(x)
        return hvp_gradient_value_vector_dir!(g, hv, f, x, v, cfg, valN)
    else
        return hvp_gradient_value_chunk_dir!(g, hv, f, x, v, cfg, valN)
    end
end
hvp_gradient_value!(hv::AbstractArray, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle) where {F} =
    _hvp_common_array!(hv, g, f, x, v, nothing)
function hvp_gradient_value!(hv::AbstractArray, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {F}
    return _hvp_common_array!(hv, g, f, x, v, cfg)
end
hvp_gradient_value!(hv::NTuple{N, <:AbstractArray}, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle) where {N, F} =
    _hvp_common_array!(hv, g, f, x, v, nothing)
function hvp_gradient_value!(hv::NTuple{N, <:AbstractArray}, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {N, F}
    return _hvp_common_array!(hv, g, f, x, v, cfg)
end

@inline hvp_gradient_value_vector_dir!(g::AbstractVector{T}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, valN::Val{N}) where {T, N} =
    hvp_gradient_value_vector_dir_core!(g, hv, f, x, v, cfg, valN)

@inline function hvp_gradient_value_chunk_dir_core!(g::Union{Nothing, AbstractVector{T}}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, ::Val{N}) where {T, N}
    Nchunk = chunksize(cfg)
    fill_output!(hv, zero(T))
    n_chunks = ceil(Int, length(x) / Nchunk)
    ax = axes(x, 1)
    zeroϵ1 = zero_ϵ(cfg.seeds[1])
    value = zero(T)

    # Initialize ε₂ once and keep ε₁ zeroed globally.
    @inbounds for j in eachindex(x)
        cfg.duals[j] = HyperDual(x[j], zeroϵ1, directional_ϵ2(v, j, Val(N)))
    end

    for i in 1:n_chunks
        range_i = seed_hvp_dir!(cfg.duals, cfg.seeds, i, ax)
        out = f(cfg.duals)
        check_scalar(out)
        value = out.v
        @inbounds for (I, idx_i) in enumerate(range_i)
            if g !== nothing
                g[idx_i] = out.ϵ1[I]
            end
            store_hvp!(hv, idx_i, out.ϵ12[I], Val(N))
        end
        zero_block_ϵ1!(cfg.duals, cfg.seeds, range_i)
    end
    return g === nothing ? hv : value
end

@inline function seed_hvp_dir!(d::AbstractVector{<:HyperDual{N1, N2, <:Any}}, seeds, block_i::Int, ax) where {N1, N2}
    range_i = block_range(block_i, N1, ax)

    # Seed ε₁ block rows without creating views
    @inbounds for (offset, idx) in enumerate(range_i)
        d[idx] = seed_epsilon_1(d[idx], seeds[offset])
    end
    return range_i
end

function hvp_chunk_dir!(hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, ::Val{N}) where {F, T, N}
    return hvp_gradient_value_chunk_dir_core!(nothing, hv, f, x, v, cfg, Val(N))
end

@inline function hvp_gradient_value_vector_dir_core!(g::Union{Nothing, AbstractVector{T}}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, ::Val{N}) where {T, N}
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
    return g === nothing ? hv : out.v
end

hvp_gradient_value_chunk_dir!(g::AbstractVector{T}, hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, valN::Val{N}) where {F, T, N} =
    hvp_gradient_value_chunk_dir_core!(g, hv, f, x, v, cfg, valN)

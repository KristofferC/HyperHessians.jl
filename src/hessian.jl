mutable struct HessianConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::D
    const seeds::S
end
@inline _chunksize(::Type{<:NTuple{N}}) where {N} = N
@inline _chunksize(seeds) = something(_chunksize(eltype(seeds)), length(seeds))
(chunksize(cfg::HessianConfig)::Int) = _chunksize(cfg.seeds)::Int

"""
    HessianConfig(x[, chunk::Chunk]; simd = false)

Configuration for `hessian`/`hessian!`. `simd = true` uses SIMD.Vec-forced
arithmetic for the derivative components (only effective for `Float32`/
`Float64` elements); whether that is faster depends on the function, chunk
size, and CPU — benchmark, or let ChunkPicker.jl decide.

The flag becomes a type parameter of the config: a constant `simd` infers
to a concrete config type, a runtime one to a two-type `Union`.
"""
Base.@constprop :aggressive function HessianConfig(x::AbstractVector{T}, chunk = Chunk(x)::Chunk; simd::Bool = false) where {T}
    N = chunksize(chunk)
    N > 0 || (N == 0 && isempty(x)) || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    # Branch so inference sees two concrete config types (a constant `simd`
    # folds to one); `HyperDual{N, N, T, simd}` alone stays existential.
    D = simd ? HyperDual{N, N, T, true} : HyperDual{N, N, T, false}
    duals = similar(x, D) # not Vector
    seeds = NTuple{N, T}[construct_seeds(NTuple{N, T})...]
    return HessianConfig(duals, seeds)
end
Base.@constprop :aggressive HessianConfig(x::AbstractArray{T}, chunk = Chunk(x)::Chunk; simd::Bool = false) where {T} =
    HessianConfig(vec(x), chunk; simd)

"""
    ThreadedHessianConfig(x, chunk::Chunk = Chunk(x); ntasks = Threads.nthreads(), simd = false)

Like [`HessianConfig`](@ref), but `hessian`/`hessian!` calls evaluate the
independent Hessian blocks in parallel on `ntasks` tasks.

`f` must be safe to call concurrently and return the same result regardless of
evaluation order: it must not close over shared mutable state (e.g. a
preallocated work buffer), and any side effects run in unspecified order.
A config instance must not be shared between concurrent `hessian` calls, and
the output `H` (and `G`) must be dense arrays without aliased entries.
"""
mutable struct ThreadedHessianConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::Vector{D} # one buffer per task
    const seeds::S
end
(chunksize(cfg::ThreadedHessianConfig)::Int) = _chunksize(cfg.seeds)::Int

n_block_pairs(n::Int, N::Int) = (k = N == 0 ? 0 : cld(n, N); k * (k + 1) ÷ 2)

Base.@constprop :aggressive function ThreadedHessianConfig(x::AbstractVector{T}, chunk::Chunk = Chunk(x); ntasks::Integer = Threads.nthreads(), simd::Bool = false) where {T}
    N = chunksize(chunk)
    N > 0 || (N == 0 && isempty(x)) || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    ntasks > 0 || throw(ArgumentError(lazy"ntasks must be positive, got $ntasks"))
    nbuffers = max(1, min(Int(ntasks), n_block_pairs(length(x), N)))
    D = simd ? HyperDual{N, N, T, true} : HyperDual{N, N, T, false}
    duals = [similar(x, D) for _ in 1:nbuffers]
    seeds = NTuple{N, T}[construct_seeds(NTuple{N, T})...]
    return ThreadedHessianConfig(duals, seeds)
end
Base.@constprop :aggressive ThreadedHessianConfig(x::AbstractArray{T}, chunk::Chunk = Chunk(x); ntasks::Integer = Threads.nthreads(), simd::Bool = false) where {T} =
    ThreadedHessianConfig(vec(x), chunk; ntasks, simd)

const AnyHessianConfig = Union{HessianConfig, ThreadedHessianConfig}

# Serial view of a threaded config, for paths with nothing to parallelize.
_serial_config(cfg::ThreadedHessianConfig) =
    HessianConfig{eltype(cfg.duals), typeof(cfg.seeds)}(cfg.duals[1], cfg.seeds)

"""
    HVPConfig(x[, tangents], chunk::Chunk = Chunk(x))

Configuration for Hessian–vector products. Pass the tangents (a vector, or a
tuple of vectors for multiple directions) so the config allocates one ϵ₂ lane
per tangent.
"""
mutable struct HVPConfig{D <: AbstractVector{<:HyperDual}, S}
    const duals::D
    const seeds::S
end
(chunksize(cfg::HVPConfig)::Int) = _chunksize(cfg.seeds)

HVPConfig(x::AbstractVector{T}, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    _HVPConfig(x, chunk, Val(1), T)
HVPConfig(x::AbstractVector{T}, v::AbstractVector, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    _HVPConfig(x, chunk, Val(1), promote_type(T, eltype(v)))
HVPConfig(x::AbstractVector{T}, v::Tuple{Vararg{AbstractVector, N}}, chunk::Chunk = Chunk(x)::Chunk) where {T, N} =
    _HVPConfig(x, chunk, Val(N), promote_type(T, ntuple(i -> eltype(v[i]), Val(N))...))
HVPConfig(x::AbstractArray{T}, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    HVPConfig(vec(x), chunk)
HVPConfig(x::AbstractArray{T}, v::AbstractVector, chunk::Chunk = Chunk(x)::Chunk) where {T} =
    HVPConfig(vec(x), vec(v), chunk)
HVPConfig(x::AbstractArray{T}, v::Tuple{Vararg{AbstractArray, N}}, chunk::Chunk = Chunk(x)::Chunk) where {T, N} =
    HVPConfig(vec(x), ntuple(i -> vec(v[i]), Val(N)), chunk)

function _HVPConfig(x::AbstractVector, chunk::Chunk, ::Val{ntangents}, ::Type{T}) where {T, ntangents}
    N = chunksize(chunk)
    N > 0 || (N == 0 && isempty(x)) || throw(ArgumentError(lazy"chunk size must be positive, got $N"))
    ntangents > 0 || throw(ArgumentError(lazy"number of tangents must be positive, got $ntangents"))
    duals = similar(x, HyperDual{N, ntangents, T, false}) # directional: ε₂ has one lane per tangent
    seeds = NTuple{N, T}[construct_seeds(NTuple{N, T})...]
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
    buffer = similar(vec(x), HyperDual{1, 1, baseT, false})
    return VHVPConfig(buffer)
end

@inline block_range(block::Int, chunk::Int, ax) = begin
    start = first(ax) + (block - 1) * chunk
    start:min(last(ax), start + chunk - 1)
end

seed_epsilon_1(d::HyperDual{N1, N2, T, S}, ϵ1) where {N1, N2, T, S} = HyperDual{N1, N2, T, S}(d.v, ϵ1, d.ϵ2, d.ϵ12)
seed_epsilon_2(d::HyperDual{N1, N2, T, S}, ϵ2) where {N1, N2, T, S} = HyperDual{N1, N2, T, S}(d.v, d.ϵ1, ϵ2, d.ϵ12)

@inline function seed_block_ϵ1!(d::AbstractVector{<:HyperDual{N1, N2}}, seeds, block_i, ax) where {N1, N2}
    range_i = block_range(block_i, N1, ax)
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        d[idx] = seed_epsilon_1(d[idx], seeds[k])
    end
    return range_i
end

# Seed a diagonal block: write value, ϵ1 and ϵ2 in a single pass (ϵ12 is zeroed).
@inline function seed_block_ϵ1ϵ2!(d::AbstractVector{<:HyperDual{N1, N2}}, x, seeds, block_i, ax) where {N1, N2}
    range_i = block_range(block_i, N1, ax)
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        seed = seeds[k]
        d[idx] = HyperDual(x[idx], seed, seed)
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
    zeroϵ2 = zero_ϵ(eltype(seeds))
    @inbounds for k in 1:length(range_j)
        idx = range_j[k]
        d[idx] = seed_epsilon_2(d[idx], zeroϵ2)
    end
    return nothing
end

const TangentBundle = Union{
    AbstractVector,
    AbstractArray,
    Tuple{Vararg{AbstractVector}},
    Tuple{Vararg{AbstractArray}},
}
const HVBundle = Union{AbstractVector, Tuple{Vararg{AbstractVector}}}

@inline tangents_count(::AbstractVector) = 1
@inline tangents_count(::AbstractArray) = 1
@inline tangents_count(::Tuple{Vararg{AbstractVector, N}}) where {N} = N
@inline tangents_count(::Tuple{Vararg{AbstractArray, N}}) where {N} = N

@inline tangent_eltype(v::AbstractArray) = eltype(v)
@inline tangent_eltype(v::Tuple{Vararg{AbstractArray, N}}) where {N} =
    promote_type(ntuple(i -> eltype(v[i]), Val(N))...)

@inline ntangents(::Type{<:HyperDual{<:Any, N2, <:Any}}) where {N2} = N2
@inline ntangents(cfg::HVPConfig) = ntangents(eltype(cfg.duals))

@inline function zero_block_ϵ1!(d::AbstractVector{<:HyperDual{N1, N2}}, seeds, range_i) where {N1, N2}
    zeroϵ1 = zero_ϵ(eltype(seeds))
    @inbounds for k in 1:length(range_i)
        idx = range_i[k]
        d[idx] = seed_epsilon_1(d[idx], zeroϵ1)
    end
    return nothing
end

@noinline check_scalar(x) =
    x isa Real || throw(ErrorException("expected a real scalar to be returned from function passed to `hessian`"))

@inline _simd_of(::Type{HyperDual{N1, N2, T, S}}) where {N1, N2, T, S} = S
@inline _simd_of(::Type{<:HyperDual}) = false

@inline _ensure_dual(v::HyperDual{N1, N2}, ::AbstractVector{<:HyperDual{N1, N2}}) where {N1, N2} = v
@inline _ensure_dual(v::HyperDual{N1, N2}, ::HyperDual{N1, N2}) where {N1, N2} = v
# Keep typeof(v) as the value type (a constant f may return e.g. BigFloat);
# only the backend flag follows the buffer.
@inline _ensure_dual(v::Real, d::AbstractVector{<:HyperDual{N1, N2}}) where {N1, N2} =
    HyperDual{N1, N2, typeof(v), _simd_of(eltype(d))}(v)
@inline _ensure_dual(v::Real, ::HyperDual{N1, N2}) where {N1, N2} = HyperDual{N1, N2}(v)

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
@inline function _vectorize_tangents(v::Tuple{Vararg{AbstractArray, N}}, shape) where {N}
    shape === nothing && return v, nothing
    return ntuple(i -> vec(v[i]), Val(N)), shape
end

function _check_config_length(cfg::HessianConfig, n::Int)
    length(cfg.duals) == n || throw(DimensionMismatch(lazy"config size $(length(cfg.duals)) does not match input length $n"))
    return nothing
end

function _check_config_length(cfg::HVPConfig, n::Int)
    length(cfg.duals) == n || throw(DimensionMismatch(lazy"config size $(length(cfg.duals)) does not match input length $n"))
    return nothing
end

function _check_config_length(cfg::ThreadedHessianConfig, n::Int)
    for duals in cfg.duals
        length(duals) == n || throw(DimensionMismatch(lazy"config size $(length(duals)) does not match input length $n"))
    end
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

    if block_i == block_j
        # Diagonal blocks are symmetric: store the transposed entry so both the
        # tuple reads and the column writes are contiguous.
        @inbounds for (J, j) in enumerate(range_j)
            for (I, i) in enumerate(range_i)
                H[i, j] = v.ϵ12[J][I]
            end
        end
    else
        @inbounds for (J, j) in enumerate(range_j)
            for (I, i) in enumerate(range_i)
                H[i, j] = v.ϵ12[I][J]
            end
        end
    end
    return H
end

function extract_gradient!(G::AbstractVector, v::HyperDual{N1, N2}, block_i::Int) where {N1, N2}
    range_i = block_range(block_i, N1, axes(G, 1))
    @inbounds for (I, i) in enumerate(range_i)
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
    v = map(v -> _ensure_dual(v, dual), v)
    return (;
        value = map(v -> v.v, v),
        gradient = map(v -> v.ϵ1[1], v),
        hessian = map(v -> v.ϵ12[1][1], v),
    )
end

hessian(f::F, x::AbstractVector) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, HessianConfig(x))
hessian(f::F, x::AbstractVector, cfg::AnyHessianConfig) where {F} = hessian!(similar(x, axes(x, 1), axes(x, 1)), f, x, cfg)
hessian(f::F, x::AbstractArray) where {F} = begin
    f_vec, x_vec, _ = _vectorize_input(f, x)
    hessian(f_vec, x_vec)
end
function hessian(f::F, x::AbstractArray, cfg::AnyHessianConfig) where {F}
    f_vec, x_vec, _ = _vectorize_input(f, x)
    _check_config_length(cfg, length(x_vec))
    return hessian(f_vec, x_vec, cfg)
end

function hessian!(H::AbstractMatrix, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
    hessian_core!(H, nothing, f, x, cfg)
    return H
end
hessian!(H::AbstractMatrix, f::F, x::AbstractArray) where {F} = begin
    f_vec, x_vec, _ = _vectorize_input(f, x)
    hessian!(H, f_vec, x_vec, HessianConfig(x_vec))
end
function hessian!(H::AbstractMatrix, f::F, x::AbstractArray, cfg::AnyHessianConfig) where {F}
    f_vec, x_vec, _ = _vectorize_input(f, x)
    _check_config_length(cfg, length(x_vec))
    return hessian!(H, f_vec, x_vec, cfg)
end

function hessian_core!(H::AbstractMatrix, G::Union{Nothing, AbstractVector}, f, x::AbstractVector{T}, cfg::HessianConfig) where {T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    G === nothing || length(G) == length(x) || throw(DimensionMismatch(lazy"gradient must have length $(length(x)), got $(length(G))"))
    ax = axes(x, 1)
    if length(x) <= chunksize(cfg)
        # Single chunk (including empty x): seed everything in one pass and
        # evaluate f once; the diagonal block is all of H, so no symmetrization.
        seed_block_ϵ1ϵ2!(cfg.duals, x, cfg.seeds, 1, ax)
        v = f(cfg.duals)
        check_scalar(v)
        v = _ensure_dual(v, cfg.duals)
        if length(x) == chunksize(cfg)
            # Chunk covers H exactly: unrolled whole-column stores.
            extract_hessian!(H, v)
        else
            extract_hessian!(H, v, 1, 1)
        end
        if G !== nothing
            extract_gradient!(G, v, 1)
        end
        return v.v
    end
    n_chunks = ceil(Int, length(x) / chunksize(cfg))
    cfg.duals .= eltype(cfg.duals).(x)
    prev_range = 0:-1  # empty range sentinel
    value = zero(T)
    for i in 1:n_chunks
        zero_block_ϵ1!(cfg.duals, cfg.seeds, prev_range)
        range_i = seed_block_ϵ1ϵ2!(cfg.duals, x, cfg.seeds, i, ax)
        for j in i:n_chunks
            range_j = j == i ? range_i : block_range(j, chunksize(cfg), ax)
            if j > i
                seed_block_ϵ2!(cfg.duals, cfg.seeds, range_j)
            end
            v = f(cfg.duals)
            check_scalar(v)
            v = _ensure_dual(v, cfg.duals)
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

# Map linear pair index p ∈ 1:k(k+1)/2 to upper-triangle (i, j), i ≤ j, in
# row-major order: (1,1) (1,2) ... (1,k) (2,2) ... (k,k). Called once per
# task; the pair loop advances (i, j) incrementally from there.
function linear_to_pair(p::Int, k::Int)
    i = ceil(Int, (2k + 1 - sqrt((2k + 1)^2 - 8p)) / 2)
    while (i - 1) * k - ((i - 1) * (i - 2)) ÷ 2 >= p # fp guard
        i -= 1
    end
    while i * k - (i * (i - 1)) ÷ 2 < p # fp guard
        i += 1
    end
    offset = (i - 1) * k - ((i - 1) * (i - 2)) ÷ 2
    j = (p - offset) + i - 1
    return i, j
end

function hessian_pair_range!(H, G, f::F, x, duals::AbstractVector{HyperDual{N, N, T, S}}, seeds, prange, n_chunks) where {F, N, T, S}
    ax = axes(x, 1)
    duals .= HyperDual{N, N, T, S}.(x)
    value = zero(eltype(x))
    isempty(prange) && return value
    i, j = linear_to_pair(first(prange), n_chunks)
    for _ in prange
        range_i = seed_block_ϵ1!(duals, seeds, i, ax)
        range_j = j == i ? range_i : block_range(j, N, ax)
        seed_block_ϵ2!(duals, seeds, range_j)
        v = f(duals)
        check_scalar(v)
        v = _ensure_dual(v, duals)
        value = v.v
        extract_hessian!(H, v, i, j)
        if G !== nothing && j == i
            extract_gradient!(G, v, i)
        end
        zero_block_ϵ2!(duals, seeds, range_j)
        zero_block_ϵ1!(duals, seeds, range_i)
        j += 1
        if j > n_chunks
            i += 1
            j = i
        end
    end
    return value
end

function hessian_chunk_threaded_core!(H::AbstractMatrix, G::Union{Nothing, AbstractVector}, f::F, x::AbstractVector{T}, cfg::ThreadedHessianConfig) where {F, T}
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    G === nothing || length(G) == length(x) || throw(DimensionMismatch(lazy"gradient must have length $(length(x)), got $(length(G))"))
    N = chunksize(cfg)
    n_chunks = cld(length(x), N)
    npairs = n_block_pairs(length(x), N)
    ntasks = min(length(cfg.duals), npairs)
    local value
    if ntasks <= 1
        value = hessian_pair_range!(H, G, f, x, cfg.duals[1], cfg.seeds, 1:npairs, n_chunks)
    else
        tasks = Vector{Task}(undef, ntasks)
        # @sync guarantees every task has finished (or failed) before we
        # return or rethrow, so none can still be writing into H/G/cfg.
        @sync for t in 1:ntasks
            lo = div((t - 1) * npairs, ntasks) + 1
            hi = div(t * npairs, ntasks)
            duals = cfg.duals[t]
            tasks[t] = Threads.@spawn hessian_pair_range!(H, G, f, x, duals, cfg.seeds, lo:hi, n_chunks)
        end
        # The last task owns the final (k, k) pair, matching serial traversal.
        # `fetch(::Task)` is untyped; recover inferrability with a type assert.
        RT = Base.promote_op(
            hessian_pair_range!, typeof(H), typeof(G), F, typeof(x),
            eltype(cfg.duals), typeof(cfg.seeds), UnitRange{Int}, Int
        )
        value = fetch(tasks[ntasks])::RT
    end
    symmetrize!(H)
    return value
end

function hessian!(H::AbstractMatrix, f::F, x::AbstractVector{T}, cfg::ThreadedHessianConfig) where {F, T}
    _check_config_length(cfg, length(x))
    if chunksize(cfg) == length(x) # single evaluation, includes empty input
        hessian_core!(H, nothing, f, x, _serial_config(cfg))
        return H
    else
        return (hessian_chunk_threaded_core!(H, nothing, f, x, cfg); H)
    end
end

function hessian_gradient_value!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::ThreadedHessianConfig) where {F, T}
    _check_config_length(cfg, length(x))
    size(H, 1) == size(H, 2) == length(x) || throw(DimensionMismatch(lazy"H must be square with size matching length(x)=$(length(x)), got size(H)=$(size(H))"))
    length(G) == length(x) || throw(DimensionMismatch(lazy"G must have length $(length(x)), got $(length(G))"))
    if chunksize(cfg) == length(x)
        return hessian_core!(H, G, f, x, _serial_config(cfg))
    else
        return hessian_chunk_threaded_core!(H, G, f, x, cfg)
    end
end

function hessian_gradient_value!(H::AbstractMatrix, G::AbstractVector, f::F, x::AbstractVector{T}, cfg::HessianConfig) where {F, T}
    return hessian_core!(H, G, f, x, cfg)
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
function hessian_gradient_value!(H::AbstractMatrix, G::AbstractArray, f::F, x::AbstractArray, cfg::AnyHessianConfig) where {F}
    f_vec, x_vec, _ = _vectorize_input(f, x)
    g_vec = vec(G)
    length(g_vec) == length(x_vec) || throw(DimensionMismatch(lazy"G must have length $(length(x_vec)), got $(length(g_vec))"))
    _check_config_length(cfg, length(x_vec))
    return hessian_gradient_value!(H, g_vec, f_vec, x_vec, cfg)
end

function hessian_gradient_value(f::F, x::AbstractVector, cfg::AnyHessianConfig) where {F}
    G = similar(x, axes(x, 1))
    H = similar(x, axes(x, 1), axes(x, 1))
    value = hessian_gradient_value!(H, G, f, x, cfg)
    return (; value = value, gradient = G, hessian = H)
end

function hessian_gradient_value(f::F, x::AbstractVector) where {F}
    cfg = HessianConfig(x)
    return hessian_gradient_value(f, x, cfg)
end
hessian_gradient_value(f::F, x::AbstractArray, cfg::AnyHessianConfig) where {F} =
    _hessian_gradient_value_array(f, x, cfg)
hessian_gradient_value(f::F, x::AbstractArray) where {F} =
    _hessian_gradient_value_array(f, x, nothing)

@inline function _hessian_gradient_value_array(f::F, x::AbstractArray, cfg::Union{AnyHessianConfig, Nothing}) where {F}
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

"""
    hvp(f, x, tangents[, cfg])

Compute Hessian–vector product(s) `H(x) * v`. `tangents` may be a single vector
or a tuple of vectors for multiple directions (e.g. `(v1, v2)`); bundled
tangents are evaluated in one pass. Pass a `HVPConfig` explicitly to control
chunking or reuse allocations.
"""
@inline similar_output(x::AbstractVector{T}, v::AbstractVector) where {T} = similar(x, promote_type(T, eltype(v)))
@inline function similar_output(x::AbstractVector{T}, v::Tuple{Vararg{AbstractVector, N}}) where {T, N}
    output_type = promote_type(T, ntuple(i -> eltype(v[i]), Val(N))...)
    return ntuple(_ -> similar(x, output_type), Val(N))
end

function check_grad_dims(g::AbstractVector, n::Int)
    length(g) == n || throw(DimensionMismatch(lazy"gradient must have length $n, got $(length(g))"))
    return nothing
end

function check_tangent_dims(x::AbstractArray, v::AbstractArray)
    length(v) == length(x) || throw(DimensionMismatch(lazy"tangent must have length $(length(x)), got $(length(v))"))
    return nothing
end
function check_tangent_dims(x::AbstractArray, v::Tuple{Vararg{AbstractArray, N}}) where {N}
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
function check_output_dims(hv::Tuple{Vararg{AbstractVector, NT}}, n::Int, ntan::Int) where {NT}
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

@inline directional_ϵ2(v::AbstractArray, idx, ::Val{1}, ::Type{T}) where {T} = (T(v[idx]),)
@inline function directional_ϵ2(v::Tuple{Vararg{AbstractArray, N}}, idx, ::Val{N}, ::Type{T}) where {N, T}
    return ntuple(j -> T(v[j][idx]), Val(N))
end


@inline dual_value_type(::Type{<:HyperDual{N1, N2, T}}) where {N1, N2, T} = T

function check_hvp_config_eltype(cfg::HVPConfig, x, v)
    cfgT = dual_value_type(eltype(cfg.duals))
    requiredT = promote_type(eltype(x), tangent_eltype(v))
    promote_type(cfgT, requiredT) === cfgT ||
        throw(ArgumentError(lazy"config element type $cfgT cannot represent input and tangent element type $requiredT; construct HVPConfig with the current input and tangents"))
    return nothing
end

@inline function store_hvp!(hv::AbstractArray, idx, vals, ::Val{1})
    hv[idx] = vals[1]
    return nothing
end
@inline function store_hvp!(hv::Tuple{Vararg{AbstractArray, N}}, idx, vals, ::Val{N}) where {N}
    @inbounds for j in 1:N
        hv[j][idx] = vals[j]
    end
    return nothing
end

@inline function _reshape_output(hv::AbstractVector, shape, orig_shape)
    return orig_shape === nothing ? hv : reshape(hv, shape)
end
@inline function _reshape_output(hv::Tuple{Vararg{AbstractVector, N}}, shape, orig_shape) where {N}
    orig_shape === nothing && return hv
    return ntuple(i -> reshape(hv[i], shape), Val(N))
end

function _hvp_common_array!(hv, g, f, x, v, cfg::Union{HVPConfig, Nothing})
    f_vec, x_vec, shape = _vectorize_input(f, x)
    v_vec, _ = _vectorize_tangents(v, shape)
    cfg = cfg === nothing ? HVPConfig(x_vec, v_vec) : cfg
    g_vec = g === nothing ? nothing : vec(g)
    # vec/reshape share memory with hv and g, so writes land in place.
    hv_vec = hv isa NTuple ? ntuple(i -> vec(hv[i]), Val(length(hv))) : vec(hv)
    value = g_vec === nothing ?
        hvp!(hv_vec, f_vec, x_vec, v_vec, cfg) :
        hvp_gradient_value!(hv_vec, g_vec, f_vec, x_vec, v_vec, cfg)
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
function hvp!(hv::Tuple{Vararg{AbstractArray, N}}, f::F, x::AbstractArray, v::TangentBundle) where {N, F}
    return _hvp_common_array!(hv, nothing, f, x, v, nothing)
end
function hvp!(hv::Tuple{Vararg{AbstractArray, N}}, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {N, F}
    return _hvp_common_array!(hv, nothing, f, x, v, cfg)
end

@inline function hvp_dir!(hv::HVBundle, f::F, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig) where {T, F}
    _check_config_length(cfg, length(x))
    n_tangents = tangents_count(v)
    n_tangents == ntangents(cfg) ||
        throw(DimensionMismatch(lazy"config expects $(ntangents(cfg)) tangents, but $(n_tangents) were provided"))
    check_hvp_config_eltype(cfg, x, v)
    check_tangent_dims(x, v)
    check_output_dims(hv, length(x), n_tangents)
    return hvp_gradient_value_dir_core!(nothing, hv, f, x, v, cfg, Val(n_tangents))
end

# Gradient + HVP + value (directional)
hvp_gradient_value(f::F, x::AbstractVector, v::TangentBundle) where {F} =
    hvp_gradient_value(f, x, v, HVPConfig(x, v))
function hvp_gradient_value(f::F, x::AbstractVector, v::TangentBundle, cfg::HVPConfig) where {F}
    g = similar(x, dual_value_type(eltype(cfg.duals)))
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
    _check_config_length(cfg, length(x))
    n_tangents = tangents_count(v)
    n_tangents == ntangents(cfg) ||
        throw(DimensionMismatch(lazy"config expects $(ntangents(cfg)) tangents, but $(n_tangents) were provided"))
    check_hvp_config_eltype(cfg, x, v)
    check_grad_dims(g, length(x))
    check_tangent_dims(x, v)
    check_output_dims(hv, length(x), n_tangents)
    return hvp_gradient_value_dir_core!(g, hv, f, x, v, cfg, Val(n_tangents))
end
hvp_gradient_value!(hv::AbstractArray, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle) where {F} =
    _hvp_common_array!(hv, g, f, x, v, nothing)
function hvp_gradient_value!(hv::AbstractArray, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {F}
    return _hvp_common_array!(hv, g, f, x, v, cfg)
end
hvp_gradient_value!(hv::Tuple{Vararg{AbstractArray, N}}, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle) where {N, F} =
    _hvp_common_array!(hv, g, f, x, v, nothing)
function hvp_gradient_value!(hv::Tuple{Vararg{AbstractArray, N}}, g::AbstractArray, f::F, x::AbstractArray, v::TangentBundle, cfg::HVPConfig) where {N, F}
    return _hvp_common_array!(hv, g, f, x, v, cfg)
end

@inline function hvp_gradient_value_dir_core!(g::Union{Nothing, AbstractVector}, hv::HVBundle, f, x::AbstractVector{T}, v::TangentBundle, cfg::HVPConfig, ::Val{N}) where {T, N}
    Nchunk = chunksize(cfg)
    ax = axes(x, 1)
    dualT = dual_value_type(eltype(cfg.duals))
    if length(x) <= Nchunk
        # Single chunk (including empty x): seed ε₁ with the identity and ε₂
        # with the tangents in one pass and evaluate f once.
        range_i = block_range(1, Nchunk, ax)
        @inbounds for k in 1:length(range_i)
            idx = range_i[k]
            cfg.duals[idx] = HyperDual(dualT(x[idx]), cfg.seeds[k], directional_ϵ2(v, idx, Val(N), dualT))
        end
        out = f(cfg.duals)
        check_scalar(out)
        out = _ensure_dual(out, cfg.duals)
        @inbounds for (I, idx_i) in enumerate(range_i)
            if g !== nothing
                g[idx_i] = out.ϵ1[I]
            end
            store_hvp!(hv, idx_i, out.ϵ12[I], Val(N))
        end
        return g === nothing ? hv : out.v
    end

    n_chunks = ceil(Int, length(x) / Nchunk)
    zeroϵ1 = zero_ϵ(eltype(cfg.seeds))
    value = zero(T)

    # Initialize ε₂ with the tangents and seed ε₁ for the first block in one pass.
    range_i = block_range(1, Nchunk, ax)
    @inbounds for j in eachindex(x)
        ϵ1 = j in range_i ? cfg.seeds[j - first(range_i) + 1] : zeroϵ1
        cfg.duals[j] = HyperDual(dualT(x[j]), ϵ1, directional_ϵ2(v, j, Val(N), dualT))
    end

    for i in 1:n_chunks
        if i > 1
            # Zero the previous block's ε₁ and seed the current one.
            zero_block_ϵ1!(cfg.duals, cfg.seeds, range_i)
            range_i = seed_block_ϵ1!(cfg.duals, cfg.seeds, i, ax)
        end
        out = f(cfg.duals)
        check_scalar(out)
        out = _ensure_dual(out, cfg.duals)
        value = out.v
        @inbounds for (I, idx_i) in enumerate(range_i)
            if g !== nothing
                g[idx_i] = out.ϵ1[I]
            end
            store_hvp!(hv, idx_i, out.ϵ12[I], Val(N))
        end
    end
    return g === nothing ? hv : value
end

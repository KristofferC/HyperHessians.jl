module HyperHessiansStaticArraysExt

using HyperHessians
using HyperHessians: HyperDual, check_scalar, construct_seeds, _ensure_dual, ϵT,
    Jet, nupper, JET_VECTOR_MAX_N
using StaticArrays

@generated function hyperdualize(x::S) where {S <: StaticVector}
    N = length(x)
    T = eltype(S)
    dual_exprs = [:(HyperDual(x[$i], seeds[$i], seeds[$i])) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        seeds = construct_seeds(NTuple{$N, $T})
        V = StaticArrays.similar_type(x, HyperDual{$N, $N, $T})
        return V($(Expr(:tuple, dual_exprs...)))
    end
end

@generated function hyperdualize_dir(x::StaticVector{N, T}, tangents::NTuple{M, <:StaticVector{N, T}}) where {N, M, T}
    eps_exprs = Vector{Any}(undef, N)
    for i in 1:N
        vals = [:(tangents[$j][$i]) for j in 1:M]
        eps_exprs[i] = Expr(:tuple, vals...)
    end
    dual_exprs = [:(HyperDual(x[$i], seeds[$i], $(eps_exprs[i]))) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        seeds = construct_seeds(NTuple{$N, $T})
        V = StaticArrays.similar_type(x, HyperDual{$N, $M, $T})
        return V($(Expr(:tuple, dual_exprs...)))
    end
end

@generated function jetize(x::S) where {S <: StaticVector}
    N = length(x)
    T = eltype(S)
    M = nupper(N)
    dual_exprs = [
        :(Jet{$N, $M, $T}(x[$i], seeds[$i], zeroh)) for i in 1:N
    ]
    zeroh_expr = Expr(:tuple, (:(zero($T)) for _ in 1:M)...)
    return quote
        $(Expr(:meta, :inline))
        seeds = construct_seeds(NTuple{$N, $T})
        zeroh = $zeroh_expr
        V = StaticArrays.similar_type(x, Jet{$N, $M, $T})
        return V($(Expr(:tuple, dual_exprs...)))
    end
end

@generated function extract_static_hessian(v::Jet{N, M, TD}, x::StaticVector{N, TX}) where {N, M, TD, TX}
    # upper triangle is stored row-major; mirror into the full matrix
    idx = Dict{Tuple{Int, Int}, Int}()
    k = 0
    for i in 1:N, j in i:N
        k += 1
        idx[(i, j)] = k
        idx[(j, i)] = k
    end
    entries = [:(v.h[$(idx[(i, j)])]) for j in 1:N for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $TX, Size($N, $N))
        return V($(Expr(:tuple, entries...)))
    end
end

@generated function extract_static_gradient(v::Jet{N, M, TD}, x::StaticVector{N, TX}) where {N, M, TD, TX}
    entries = [:(v.g[$i]) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $TX, Size($N))
        return V($(Expr(:tuple, entries...)))
    end
end

@generated function extract_static_hessian(v::HyperDual{N, N, TD}, x::StaticVector{N, TX}) where {N, TD, TX}
    entries = [:(v.ϵ12[$i][$j]) for i in 1:N for j in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $TX, Size($N, $N))
        return V($(Expr(:tuple, entries...)))
    end
end

@generated function extract_static_gradient(v::HyperDual{N, M, TD}, x::StaticVector{N, TX}) where {N, M, TD, TX}
    entries = [:(v.ϵ1[$i]) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $TX, Size($N))
        return V($(Expr(:tuple, entries...)))
    end
end

@generated function extract_static_hvp(v::HyperDual{N, M, TD}, x::StaticVector{N, TX}) where {N, M, TD, TX}
    vectors = Vector{Any}(undef, M)
    for j in 1:M
        entries = [:(v.ϵ12[$i][$j]) for i in 1:N]
        vectors[j] = quote
            V = StaticArrays.similar_type(x, $TX, Size($N))
            V($(Expr(:tuple, entries...)))
        end
    end
    return quote
        $(Expr(:meta, :inline))
        return ($(vectors...),)
    end
end

# N is a type parameter, so this choice folds away at compile time. Small
# static vectors use the symmetric Jet (same threshold and rationale as
# HessianConfig, see src/hessian.jl).
@inline _dualize(x::StaticVector{N}) where {N} =
    N <= JET_VECTOR_MAX_N && N >= 1 ? jetize(x) : hyperdualize(x)

@inline function _static_hessian_core(f, x)
    duals = _dualize(x)
    out = f(duals)
    check_scalar(out)
    return _ensure_dual(out, duals)
end

function HyperHessians.hessian(f::F, x::StaticVector{N, T}) where {F, N, T}
    out = _static_hessian_core(f, x)
    return extract_static_hessian(out, x)
end

function HyperHessians.hessian_gradient_value(f::F, x::StaticVector{N, T}) where {F, N, T}
    out = _static_hessian_core(f, x)
    return (;
        value = out.v,
        gradient = extract_static_gradient(out, x),
        hessian = extract_static_hessian(out, x),
    )
end

function HyperHessians.hvp(f::F, x::StaticVector{N, T}, tangents::NTuple{M, <:StaticVector{N, T}}) where {F, N, T, M}
    duals = hyperdualize_dir(x, tangents)
    out = f(duals)
    check_scalar(out)
    out = _ensure_dual(out, duals)
    return extract_static_hvp(out, x)
end

function HyperHessians.hvp(f::F, x::StaticVector{N, T}, v::StaticVector{N, T}) where {F, N, T}
    return HyperHessians.hvp(f, x, (v,))[1]
end

function HyperHessians.hvp_gradient_value(f::F, x::StaticVector{N, T}, tangents::NTuple{M, <:StaticVector{N, T}}) where {F, N, T, M}
    duals = hyperdualize_dir(x, tangents)
    out = f(duals)
    check_scalar(out)
    out = _ensure_dual(out, duals)
    g = extract_static_gradient(out, x)
    hv = extract_static_hvp(out, x)
    return (; value = out.v, gradient = g, hvp = hv)
end

function HyperHessians.hvp_gradient_value(f::F, x::StaticVector{N, T}, v::StaticVector{N, T}) where {F, N, T}
    res = HyperHessians.hvp_gradient_value(f, x, (v,))
    return (; value = res.value, gradient = res.gradient, hvp = res.hvp[1])
end

end

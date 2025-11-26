module HyperHessiansStaticArraysExt

using HyperHessians
using HyperHessians: HyperDual, check_scalar, construct_seeds, ϵT, USE_TUPLES
using StaticArrays
using SIMD: Vec

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

@generated function hyperdualize_dir(x::S, v::S) where {S <: StaticVector}
    N = length(x)
    T = eltype(S)
    if USE_TUPLES
        dual_exprs = [:(HyperDual(x[$i], seeds[$i], (v[$i],))) for i in 1:N]
    else
        dual_exprs = [:(HyperDual(x[$i], seeds[$i], Vec((v[$i],)))) for i in 1:N]
    end
    return quote
        $(Expr(:meta, :inline))
        seeds = construct_seeds(NTuple{$N, $T})
        V = StaticArrays.similar_type(x, HyperDual{$N, 1, $T})
        return V($(Expr(:tuple, dual_exprs...)))
    end
end

@generated function extract_static_hessian(v::HyperDual{N, N, T}, x::StaticVector{N, T}) where {N, T}
    entries = [:(v.ϵ12[$i][$j]) for i in 1:N for j in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $T, Size($N, $N))
        return V($(Expr(:tuple, entries...)))
    end
end

@generated function extract_static_gradient(v::HyperDual{N, N, T}, x::StaticVector{N, T}) where {N, T}
    entries = [:(v.ϵ1[$i]) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $T, Size($N))
        return V($(Expr(:tuple, entries...)))
    end
end

@generated function extract_static_hvp(v::HyperDual{N, 1, T}, x::StaticVector{N, T}) where {N, T}
    entries = [:(v.ϵ12[$i][1]) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(x, $T, Size($N))
        return V($(Expr(:tuple, entries...)))
    end
end

function HyperHessians.hessian(f::F, x::StaticVector{N, T}) where {F, N, T}
    duals = hyperdualize(x)
    out = f(duals)
    check_scalar(out)
    return extract_static_hessian(out, x)
end

function HyperHessians.hessiangradvalue(f::F, x::StaticVector{N, T}) where {F, N, T}
    duals = hyperdualize(x)
    out = f(duals)
    check_scalar(out)
    return (;
        value = out.v,
        gradient = extract_static_gradient(out, x),
        hessian = extract_static_hessian(out, x),
    )
end

function HyperHessians.hvp(f::F, x::StaticVector{N, T}, v::StaticVector{N, T}) where {F, N, T}
    duals = hyperdualize_dir(x, v)
    out = f(duals)
    check_scalar(out)
    return extract_static_hvp(out, x)
end

end

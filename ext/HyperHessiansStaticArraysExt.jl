module HyperHessiansStaticArraysExt

using HyperHessians
using HyperHessians: HyperDual, check_scalar, construct_seeds, ϵT, USE_SIMD
using StaticArrays
if HyperHessians.USE_SIMD
    using SIMD: Vec
end

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
        tuple_expr = Expr(:tuple, vals...)
        eps_exprs[i] = USE_SIMD ? :(Vec($tuple_expr)) : tuple_expr
    end
    dual_exprs = [:(HyperDual(x[$i], seeds[$i], $(eps_exprs[i]))) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        seeds = construct_seeds(NTuple{$N, $T})
        V = StaticArrays.similar_type(x, HyperDual{$N, $M, $T})
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

@generated function extract_static_hvp(v::HyperDual{N, M, T}, x::StaticVector{N, T}) where {N, M, T}
    vectors = Vector{Any}(undef, M)
    for j in 1:M
        entries = [:(v.ϵ12[$i][$j]) for i in 1:N]
        vectors[j] = quote
            V = StaticArrays.similar_type(x, $T, Size($N))
            V($(Expr(:tuple, entries...)))
        end
    end
    return quote
        $(Expr(:meta, :inline))
        return ($(vectors...),)
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

function HyperHessians.hvp(f::F, x::StaticVector{N, T}, tangents::NTuple{M, <:StaticVector{N, T}}) where {F, N, T, M}
    duals = hyperdualize_dir(x, tangents)
    out = f(duals)
    check_scalar(out)
    return extract_static_hvp(out, x)
end

function HyperHessians.hvp(f::F, x::StaticVector{N, T}, v::StaticVector{N, T}) where {F, N, T}
    return HyperHessians.hvp(f, x, (v,))[1]
end

end

# Prototype: symmetric second-order "jet" number for full-vector-mode Hessians
# (n <= chunk). Stores the gradient once and only the upper triangle of the
# Hessian block: 1 + N + N(N+1)/2 floats vs HyperDual{N,N}'s 1 + 2N + N^2.
module Jets

    struct Jet{N, M, T} <: Real   # M = N(N+1)/2
        v::T
        g::NTuple{N, T}
        h::NTuple{M, T}
    end

    @inline nupper(N) = N * (N + 1) ÷ 2

    # upper-triangle helpers, fully unrolled with literal indices
    @inline @generated function symouter(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T}
        ex = Expr(:tuple)
        for i in 1:N, j in i:N
            push!(ex.args, :(muladd(a[$i], b[$j], b[$i] * a[$j])))
        end
        return ex
    end
    @inline @generated function halfouter(a::NTuple{N, T}) where {N, T}
        ex = Expr(:tuple)
        for i in 1:N, j in i:N
            push!(ex.args, :(a[$i] * a[$j]))
        end
        return ex
    end

    @inline function chain(j::Jet{N, M, T}, f, f′, f′′) where {N, M, T}
        return Jet{N, M, T}(f, f′ .* j.g, muladd.(f′, j.h, f′′ .* halfouter(j.g)))
    end

    # arithmetic
    @inline Base.:+(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T} =
        Jet{N, M, T}(a.v + b.v, a.g .+ b.g, a.h .+ b.h)
    @inline Base.:-(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T} =
        Jet{N, M, T}(a.v - b.v, a.g .- b.g, a.h .- b.h)
    @inline Base.:-(a::Jet{N, M, T}) where {N, M, T} =
        Jet{N, M, T}(-a.v, .-a.g, .-a.h)
    @inline Base.:*(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T} =
        Jet{N, M, T}(
        a.v * b.v,
        muladd.(a.v, b.g, b.v .* a.g),
        muladd.(a.v, b.h, muladd.(b.v, a.h, symouter(a.g, b.g))),
    )
    @inline Base.inv(a::Jet) = (f = inv(a.v); chain(a, f, -f * f, 2 * f * f * f))
    @inline Base.:/(a::Jet{N, M, T}, b::Jet{N, M, T}) where {N, M, T} = a * inv(b)

    # real mixing
    @inline Base.:+(a::Jet{N, M, T}, r::Real) where {N, M, T} = Jet{N, M, T}(a.v + r, a.g, a.h)
    @inline Base.:+(r::Real, a::Jet) = a + r
    @inline Base.:-(a::Jet{N, M, T}, r::Real) where {N, M, T} = Jet{N, M, T}(a.v - r, a.g, a.h)
    @inline Base.:-(r::Real, a::Jet{N, M, T}) where {N, M, T} = Jet{N, M, T}(r - a.v, .-a.g, .-a.h)
    @inline Base.:*(a::Jet{N, M, T}, r::Real) where {N, M, T} = Jet{N, M, T}(a.v * r, a.g .* r, a.h .* r)
    @inline Base.:*(r::Real, a::Jet) = a * r
    @inline Base.:/(a::Jet{N, M, T}, r::Real) where {N, M, T} = Jet{N, M, T}(a.v / r, a.g ./ r, a.h ./ r)

    # unary rules (enough for ackley / rosenbrock and friends)
    @inline Base.sin(j::Jet) = ((s, c) = sincos(j.v); chain(j, s, c, -s))
    @inline Base.cos(j::Jet) = ((s, c) = sincos(j.v); chain(j, c, -s, -c))
    @inline Base.exp(j::Jet) = (e = exp(j.v); chain(j, e, e, e))
    @inline function Base.sqrt(j::Jet)
        f = sqrt(j.v)
        f′ = 1 / (2 * f)
        return chain(j, f, f′, -2 * f′^3)
    end
    @inline Base.abs2(j::Jet) = chain(j, j.v * j.v, 2 * j.v, 2 * one(j.v))
    @inline Base.literal_pow(::typeof(^), j::Jet, ::Val{2}) = abs2(j)

    Base.zero(::Type{Jet{N, M, T}}) where {N, M, T} =
        Jet{N, M, T}(zero(T), ntuple(_ -> zero(T), Val(N)), ntuple(_ -> zero(T), Val(M)))
    Base.one(::Type{Jet{N, M, T}}) where {N, M, T} =
        Jet{N, M, T}(one(T), ntuple(_ -> zero(T), Val(N)), ntuple(_ -> zero(T), Val(M)))
    Base.zero(::Jet{N, M, T}) where {N, M, T} = zero(Jet{N, M, T})
    Base.one(::Jet{N, M, T}) where {N, M, T} = one(Jet{N, M, T})
    Base.convert(::Type{Jet{N, M, T}}, r::Real) where {N, M, T} =
        Jet{N, M, T}(T(r), ntuple(_ -> zero(T), Val(N)), ntuple(_ -> zero(T), Val(M)))
    Base.convert(::Type{Jet{N, M, T}}, j::Jet{N, M, T}) where {N, M, T} = j
    Base.promote_rule(::Type{Jet{N, M, T}}, ::Type{S}) where {N, M, T, S <: Real} =
        Jet{N, M, promote_type(T, S)}

    # driver: full-vector-mode Hessian
    function seed!(duals::Vector{Jet{N, M, T}}, x) where {N, M, T}
        @inbounds for i in 1:N
            duals[i] = Jet{N, M, T}(
                x[i],
                ntuple(k -> T(k == i), Val(N)),
                ntuple(_ -> zero(T), Val(M)),
            )
        end
        return duals
    end

    function extract!(H, out::Jet{N, M, T}) where {N, M, T}
        k = 0
        @inbounds for i in 1:N, j in i:N
            k += 1
            H[i, j] = out.h[k]
            H[j, i] = out.h[k]
        end
        return H
    end

    function hessian!(H, f::F, x::Vector{T}, duals::Vector{Jet{N, M, T}}) where {F, N, M, T}
        seed!(duals, x)
        out = f(duals)
        return extract!(H, out)
    end

    make_duals(x::Vector{T}) where {T} =
        Vector{Jet{length(x), nupper(length(x)), T}}(undef, length(x))

end # module

# ---------------------------------------------------------------------------
using HyperHessians, DiffTests, BenchmarkTools, Random, InteractiveUtils
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

let
    Random.seed!(1234)
    for f in (DiffTests.ackley, DiffTests.rosenbrock_1), n in (2, 4, 8, 16)
        x = rand(n)
        H_jet = zeros(n, n)
        duals = Jets.make_duals(x)
        Jets.hessian!(H_jet, f, x, duals)

        H_hh = zeros(n, n)
        cfg = HyperHessians.HessianConfig(x, HyperHessians.Chunk{n}())  # vector mode
        HyperHessians.hessian!(H_hh, f, x, cfg)
        ok = isapprox(H_jet, H_hh; rtol = 1.0e-10)

        t_jet = @belapsed Jets.hessian!($H_jet, $f, $x, $duals)
        t_hh = @belapsed HyperHessians.hessian!($H_hh, $f, $x, $cfg)
        println(
            rpad(string(nameof(f)), 14), "n=", rpad(n, 4),
            "jet=", rpad(string(round(t_jet * 1.0e9, digits = 0)), 9), "ns   ",
            "HyperDual{n,n}=", rpad(string(round(t_hh * 1.0e9, digits = 0)), 9), "ns   ",
            "speedup=", round(t_hh / t_jet, digits = 2), "x   correct=", ok
        )
    end

    # SIMD check: instruction mix of one multiply at n=8
    T_jet = Jets.Jet{8, 36, Float64}
    T_hd = HyperHessians.HyperDual{8, 8, Float64}
    for (label, T) in [("Jet{8} mul", T_jet), ("HyperDual{8,8} mul", T_hd)]
        buf = IOBuffer()
        code_native(buf, *, (T, T); debuginfo = :none)
        is = [strip(l) for l in split(String(take!(buf)), '\n') if startswith(l, '\t')]
        println(
            rpad(label, 20), " instrs=", length(is),
            "  fma=", count(i -> occursin(r"^fmla|^fmadd|^vfmadd", i), is),
            "  scalar-fma=", count(i -> occursin(r"^fmadd|^vfmadd[0-9]+sd", i), is),
            "  calls=", count(i -> occursin(r"^bl\s|^call", i), is)
        )
    end
    println("sizeof Jet{8}: ", sizeof(T_jet), " B vs HyperDual{8,8}: ", sizeof(HyperHessians.HyperDual{8, 8, Float64}), " B")
end

# Runnable definitions for the Dual and scalar-HyperDual slides in slides/slides.md.
#
#     include("slides/duals.jl")     # from the repo root (or jld run slides/duals.jl)
#     ScalarDual.derivative(x -> x * sin(x * x), 2.0)
#     NestedDual.second_derivative(x -> x * sin(x * x), 2.0)
#     ChunkedDual.hessian(x -> x[1] * sin(x[2]), [1.0, 2.0])
#
# The deck redefines `Dual` as it generalizes (scalar -> Dual{T} -> Dual{T,N}),
# so each stage lives in its own module and they don't clobber each other.
# Lines marked `glue:` are needed to run but are not shown on the slides.
#
# Running this file directly (`julia slides/duals.jl`) checks every slide
# example against its exact value.

# ---- slide: Dual numbers · Implementation -----------------------------------
module ScalarDual

    struct Dual
        x::Float64   # value
        ε::Float64   # epsilon coefficient
    end
    Base.:+(a::Dual, b::Dual) = Dual(a.x + b.x, a.ε + b.ε)
    Base.:*(a::Dual, b::Dual) = Dual(a.x * b.x, a.x * b.ε + b.x * a.ε)  # product rule
    Base.sin(d::Dual) = Dual(sin(d.x), cos(d.x) * d.ε)          # chain rule
    derivative(f, x) = f(Dual(x, 1.0)).ε # seed ε = 1, run f, read ε

end # module

# ---- slides: Differentiation² + Jacobians (n passes) ------------------------
module NestedDual

    struct Dual{T}
        x::T
        ε::T
    end
    Base.:+(a::Dual, b::Dual) = Dual(a.x + b.x, a.ε + b.ε)
    Base.:*(a::Dual, b::Dual) = Dual(a.x * b.x, a.x * b.ε + b.x * a.ε)
    Base.sin(d::Dual) = Dual(sin(d.x), cos(d.x) * d.ε)
    Base.cos(d::Dual) = Dual(cos(d.x), -sin(d.x) * d.ε)
    Base.one(d::Dual) = Dual(one(d.x), zero(d.x))

    derivative(f, x) = f(Dual(x, one(x))).ε
    second_derivative(f, x) = derivative(y -> derivative(f, y), x)

    function jacobian(f, x)
        cols = []
        for i in eachindex(x)
            seed = zeros(length(x))
            seed[i] = 1.0 # direction eᵢ
            y = f(Dual.(x, seed))
            push!(cols, [d.ε for d in y])
        end
        return stack(cols) # m × n
    end

    function jacobian_fd(f, x; h = 1.0e-8)
        f0 = f(x)
        cols = []
        for i in eachindex(x)
            xh = copy(x)
            xh[i] += h
            push!(cols, (f(xh) - f0) / h)
        end
        return stack(cols) # m × n
    end

end # module

# ---- slides: Chunk mode + Hessian with dual numbers -------------------------
module ChunkedDual

    struct Dual{T, N}
        x::T
        ε::NTuple{N, T}
    end
    Base.:+(a::Dual, b::Dual) = Dual(a.x + b.x, a.ε .+ b.ε)
    Base.:*(a::Dual, b::Dual) = Dual(a.x * b.x, a.x .* b.ε .+ b.x .* a.ε)
    Base.sin(d::Dual) = Dual(sin(d.x), cos(d.x) .* d.ε)
    Base.cos(d::Dual) = Dual(cos(d.x), (-sin(d.x)) .* d.ε)

    # glue: the nested seeding in `hessian` builds Dual(x[i]::Dual, Tuple(seed))
    # with Float64 seeds next to a Dual value — promote the mismatched pieces,
    # lifting plain numbers into constant duals (zero ε).
    function Dual(x, ε::NTuple{N, Any}) where {N}
        T = promote_type(typeof(x), map(typeof, ε)...)
        return Dual{T, N}(convert(T, x), NTuple{N, T}(ε))
    end
    Base.convert(::Type{Dual{T, N}}, v::Real) where {T, N} =
        Dual{T, N}(convert(T, v), ntuple(_ -> zero(T), N))
    Base.promote_rule(::Type{Dual{T, N}}, ::Type{S}) where {T, N, S <: Real} =
        Dual{promote_type(T, S), N}
    # glue: a Dual acts as a scalar when broadcast against an ε tuple (a.x .* b.ε)
    Base.Broadcast.broadcastable(d::Dual) = Ref(d)

    function jacobian(f, x)                # H = I: whole Jacobian, one call to f
        n = length(x)
        ds = []
        for i in 1:n
            seed = zeros(n)
            seed[i] = 1.0      # row i of H = I
            push!(ds, Dual(x[i], Tuple(seed)))
        end
        y = f(ds)              # ONE call
        return stack([d.ε for d in y], dims = 1)   # m × n
    end

    function jacobian(f, x, N)             # register-sized chunks: n/N passes
        n = length(x)
        blocks = []
        for c in 0:N:(n - 1)           # one pass per chunk
            ds = []
            for i in 1:n
                seed = zeros(N)
                if c < i <= c + N  # chunk c of row i
                    seed[i - c] = 1.0
                end
                push!(ds, Dual(x[i], Tuple(seed)))
            end
            y = f(ds)              # J[:, c+1:c+N]
            push!(blocks, stack([d.ε for d in y], dims = 1))
        end
        return hcat(blocks...)     # m × n
    end

    gradient(f, x) = vec(jacobian(y -> [f(y)], x))

    hessian(f, x) = jacobian(y -> gradient(f, y), x)

end # module

# ---- slides: HyperDual Implementation + Hessian one entry per pass ----------
module ScalarHyperDual

    struct HyperDual
        x::Float64     # value
        ε1::Float64    # f′
        ε2::Float64    # f′, independent copy
        ε12::Float64   # f″
    end
    Base.:*(a::HyperDual, b::HyperDual) =
        HyperDual(
        a.x * b.x,
        a.x * b.ε1 + b.x * a.ε1,
        a.x * b.ε2 + b.x * a.ε2,
        a.x * b.ε12 + b.x * a.ε12 + a.ε1 * b.ε2 + a.ε2 * b.ε1
    )  # product rule
    function Base.sin(d::HyperDual)
        s, c = sincos(d.x)      # f′, f″ evaluated ONCE, as plain floats
        return HyperDual(s, c * d.ε1, c * d.ε2, c * d.ε12 - s * d.ε1 * d.ε2)
    end
    second_derivative(f, x) = f(HyperDual(x, 1.0, 1.0, 0.0)).ε12  # a = b = 1, c = 0

    function hessian(f, x)
        n = length(x)
        H = zeros(n, n)
        for i in 1:n, j in i:n   # symmetry: i ≤ j only
            ds = [
                HyperDual(
                        x[k], float(k == i),
                        float(k == j), 0.0
                    )
                    for k in 1:n
            ]
            H[i, j] = H[j, i] = f(ds).ε12  # one pass
        end
        return H
    end

end # module

# ---- slides: HyperDual Chunk mode + Chunk mode: the code --------------------
module ChunkedHyperDual

    struct HyperDual{N1, N2}
        x::Float64
        ε1::NTuple{N1, Float64}
        ε2::NTuple{N2, Float64}
        ε12::NTuple{N1, NTuple{N2, Float64}}
    end

    # glue: the 3-argument constructor the hessian code uses ("ε₁₂ seeded to zero")
    HyperDual(x, ε1::NTuple{N1, Float64}, ε2::NTuple{N2, Float64}) where {N1, N2} =
        HyperDual(x, ε1, ε2, ntuple(_ -> ntuple(_ -> 0.0, N2), N1))

    # glue: the chunked product/chain rules (the slide diff shows only sin's ε₁₂ line)
    Base.:*(a::HyperDual{N1, N2}, b::HyperDual{N1, N2}) where {N1, N2} =
        HyperDual(
        a.x * b.x,
        a.x .* b.ε1 .+ b.x .* a.ε1,
        a.x .* b.ε2 .+ b.x .* a.ε2,
        ntuple(
            i -> a.x .* b.ε12[i] .+ b.x .* a.ε12[i] .+
                a.ε1[i] .* b.ε2 .+ b.ε1[i] .* a.ε2, N1
        )
    )
    function Base.sin(d::HyperDual{N1, N2}) where {N1, N2}
        s, c = sincos(d.x)
        return HyperDual(
            s, c .* d.ε1, c .* d.ε2,
            ntuple(i -> c .* d.ε12[i] .- s .* (d.ε1[i] .* d.ε2), N1)
        )
    end

    seed(i, R) = Tuple(float(i == j) for j in R)

    function hessian(f, x)              # N₁ = N₂ = n
        n = length(x)
        ds = [
            HyperDual(x[i], seed(i, 1:n), seed(i, 1:n))
                for i in 1:n
        ]             # ε₁₂ seeded to zero
        v = f(ds)                       # ONE call
        return [v.ε12[i][j] for i in 1:n, j in 1:n]
    end

    # glue: the mirror helper the chunked variant calls; assumes N divides n
    function symmetrize!(H)             # mirror the computed I ≤ J blocks down
        for j in axes(H, 2), i in (j + 1):lastindex(H, 1)
            H[i, j] = H[j, i]
        end
        return H
    end

    function hessian(f, x, N)
        n = length(x); H = zeros(n, n)
        for s1 in 1:N:n, s2 in s1:N:n    # block pairs, I ≤ J
            I, J = s1:(s1 + N - 1), s2:(s2 + N - 1)
            ds = [
                HyperDual(x[i], seed(i, I), seed(i, J))
                    for i in 1:n
            ]
            v = f(ds)                    # one pass per pair
            H[I, J] .= [v.ε12[i][j] for i in 1:N, j in 1:N]
        end
        return symmetrize!(H)            # mirror I < J blocks
    end

end # module

# ---- smoke tests: julia slides/duals.jl -------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    f(x) = x * sin(x * x)
    F(x) = [x[1] * sin(x[2]), x[1] * x[2]]
    # the deck's Hessian example: f(x) = x[1] * sin(x[2] * x[3]) at [1, 2, 3]
    g(x) = x[1] * sin(x[2] * x[3])
    xg = [1.0, 2.0, 3.0]
    Hg = [
        0.0        3cos(6.0)              2cos(6.0)
        3cos(6.0)  -9sin(6.0)             cos(6.0) - 6sin(6.0)
        2cos(6.0)  cos(6.0) - 6sin(6.0)   -4sin(6.0)
    ]

    @assert ScalarDual.derivative(f, 2.0) ≈ sin(4.0) + 8cos(4.0)
    @assert NestedDual.derivative(f, 2.0) ≈ sin(4.0) + 8cos(4.0)
    @assert NestedDual.second_derivative(f, 2.0) ≈ 12cos(4.0) - 32sin(4.0)
    @assert NestedDual.jacobian(F, [1.0, 2.0]) ≈ [sin(2.0) cos(2.0); 2.0 1.0]
    @assert isapprox(NestedDual.jacobian_fd(F, [1.0, 2.0]), [sin(2.0) cos(2.0); 2.0 1.0]; atol = 1.0e-6)
    @assert ChunkedDual.jacobian(F, [1.0, 2.0]) ≈ [sin(2.0) cos(2.0); 2.0 1.0]
    @assert ChunkedDual.jacobian(F, [1.0, 2.0], 1) ≈ [sin(2.0) cos(2.0); 2.0 1.0]
    @assert ChunkedDual.gradient(g, xg) ≈ [sin(6.0), 3cos(6.0), 2cos(6.0)]
    @assert ChunkedDual.hessian(g, xg) ≈ Hg
    @assert ScalarHyperDual.second_derivative(f, 2.0) ≈ 12cos(4.0) - 32sin(4.0)
    @assert ScalarHyperDual.hessian(g, xg) ≈ Hg
    @assert ChunkedHyperDual.hessian(g, xg) ≈ Hg
    @assert ChunkedHyperDual.hessian(g, xg, 1) ≈ Hg
    let q(x) = (x[1] * sin(x[2] * x[3])) * x[4], x4 = [1.0, 2.0, 3.0, 0.5]
        @assert ChunkedHyperDual.hessian(q, x4, 2) ≈ ChunkedHyperDual.hessian(q, x4)
    end
    println("slides/duals.jl: all slide examples check out")
end

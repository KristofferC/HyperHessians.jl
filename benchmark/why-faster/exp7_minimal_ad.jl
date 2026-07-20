# Verify the minimal AD implementations that will appear on the intro slides.
module MiniDual
    struct Dual{T} <: Number
        v::T   # value
        ϵ::T   # coefficient of ε = the derivative
    end

    Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.ϵ + b.ϵ)
    Base.:-(a::Dual, b::Dual) = Dual(a.v - b.v, a.ϵ - b.ϵ)
    Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.ϵ * b.v + a.v * b.ϵ)
    Base.sin(d::Dual) = Dual(sin(d.v), cos(d.v) * d.ϵ)
    Base.cos(d::Dual) = Dual(cos(d.v), -sin(d.v) * d.ϵ)
    Base.exp(d::Dual) = Dual(exp(d.v), exp(d.v) * d.ϵ)
    Base.one(d::Dual) = Dual(one(d.v), zero(d.v))   # lets Duals nest

    derivative(f, x) = f(Dual(x, one(x))).ϵ
    second_derivative(f, x) = derivative(y -> derivative(f, y), x)
end

module MiniGrad
    struct Dual{N, T} <: Number
        v::T
        ∂::NTuple{N, T}   # one lane per input
    end

    Base.:+(a::Dual{N}, b::Dual{N}) where {N} = Dual(a.v + b.v, a.∂ .+ b.∂)
    Base.:-(a::Dual{N}, b::Dual{N}) where {N} = Dual(a.v - b.v, a.∂ .- b.∂)
    Base.:*(a::Dual{N}, b::Dual{N}) where {N} = Dual(a.v * b.v, b.v .* a.∂ .+ a.v .* b.∂)
    Base.sin(d::Dual) = Dual(sin(d.v), cos(d.v) .* d.∂)

    function gradient(f, x::Vector{T}) where {T}
        N = length(x)
        seed(i) = ntuple(j -> T(i == j), N)
        duals = [Dual(x[i], seed(i)) for i in 1:N]
        return f(duals).∂
    end
end

module MiniHyper
    struct HyperDual{T} <: Number
        v::T
        ϵ₁::T
        ϵ₂::T
        ϵ₁₂::T
    end

    Base.:+(a::HyperDual, b::HyperDual) =
        HyperDual(a.v + b.v, a.ϵ₁ + b.ϵ₁, a.ϵ₂ + b.ϵ₂, a.ϵ₁₂ + b.ϵ₁₂)
    Base.:*(a::HyperDual, b::HyperDual) = HyperDual(
        a.v * b.v,
        a.v * b.ϵ₁ + a.ϵ₁ * b.v,
        a.v * b.ϵ₂ + a.ϵ₂ * b.v,
        a.v * b.ϵ₁₂ + a.ϵ₁₂ * b.v + a.ϵ₁ * b.ϵ₂ + a.ϵ₂ * b.ϵ₁,
    )
    chain(h, f, f′, f′′) =
        HyperDual(f, f′ * h.ϵ₁, f′ * h.ϵ₂, f′ * h.ϵ₁₂ + f′′ * h.ϵ₁ * h.ϵ₂)
    Base.sin(h::HyperDual) = chain(h, sin(h.v), cos(h.v), -sin(h.v))

    second_derivative(f, x) = f(HyperDual(x, one(x), one(x), zero(x))).ϵ₁₂
end

let
    f = x -> x * x + sin(x)
    println("derivative(x -> x*x + sin(x), 1.0)        = ", MiniDual.derivative(f, 1.0))
    println("expected 2 + cos(1)                       = ", 2 + cos(1))
    println("second_derivative via nesting             = ", MiniDual.second_derivative(f, 1.0))
    println("expected 2 - sin(1)                       = ", 2 - sin(1))
    g = v -> v[1] * v[2] + sin(v[1])
    println("gradient(v -> v[1]v[2] + sin(v[1]), [1,2]) = ", MiniGrad.gradient(g, [1.0, 2.0]))
    println("expected (2 + cos(1), 1)                  = ", (2 + cos(1), 1.0))
    println("second_derivative via HyperDual           = ", MiniHyper.second_derivative(f, 1.0))
    # also check the x^2 lowering works (literal_pow -> x*x)
    println("derivative(x -> x^2 + sin(x), 1.0)        = ", MiniDual.derivative(x -> x^2 + sin(x), 1.0))
    # value/first-derivative agreement between Dual and HyperDual
    h = MiniHyper.HyperDual(1.0, 1.0, 1.0, 0.0)
    println("f(HyperDual(1,1,1,0)) = ", f(h))
end

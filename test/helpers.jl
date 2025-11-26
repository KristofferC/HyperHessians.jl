function ackley_stable(x::AbstractVector{T}) where {T}
    a, b, c = T(20.0), T(-0.2), T(2.0 * π)
    len_recip = T(inv(length(x)))
    sum_sqrs = sum(i -> i^2, x; init = zero(eltype(x)))
    sum_cos = sum(i -> cos(c * i), x; init = zero(eltype(x)))
    return (
        -a * exp(b * sqrt(len_recip * sum_sqrs)) -
            exp(len_recip * sum_cos) + a + exp(T(1))
    )
end

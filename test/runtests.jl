
using HyperHessians: HyperHessians, hessian, hessian!, Chunk, HessianConfig, HyperDual
using DiffTests
using ForwardDiff
using Test
using SIMD

@testset "scalar" begin
f(x) = exp(x) / sqrt(sin(x)^3 + cos(x)^3);
x = rand()
@test hessian(f, x) ≈  ForwardDiff.derivative(x->ForwardDiff.derivative(f, x), x)
end # testset

@testset "correctness" begin
for f in (DiffTests.rosenbrock_1,
          DiffTests.rosenbrock_2,
          DiffTests.rosenbrock_3,
          #DiffTests.rosenbrock_4,
          DiffTests.ackley,
          DiffTests.self_weighted_logit,
          DiffTests.nested_array_mul,)
    for n in (1, 4, HyperHessians.DEFAULT_CHUNK_THRESHOLD, HyperHessians.DEFAULT_CHUNK_THRESHOLD + 7)
        if n == 1 && (f == DiffTests.rosenbrock_4 || f == DiffTests.nested_array_mul)
            continue
        end
        x = rand(n)
        H  = ForwardDiff.hessian(f, x)

        for chunk in (1, n÷2, n)
            if chunk <= 0
                continue
            end
            @info "f=$f, n=$n, chunk=$chunk"

            cfg = HessianConfig(x, Chunk{chunk}())
            @test H ≈ hessian(f, x, cfg)
            @test H ≈ hessian(f, x, cfg)

            # cfg_threaded = HessianConfigThreaded(x, Chunk{chunk}())
            # @test H ≈ hessian(f, x, cfg_threaded)
            # @test H ≈ hessian(f, x, cfg_threaded)
        end
    end
end
end # testset

function ackley_stable(x::AbstractVector{T}) where {T}
    a, b, c = T(20.0), T(-0.2), T(2.0*π)
    len_recip = T(inv(length(x)))
    sum_sqrs = sum(i->i^2, x; init=zero(eltype(x)))
    sum_cos = sum(i->cos(c*i), x; init=zero(eltype(x)))
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + exp(T(1)))
end

@testset "Float32" begin
x = rand(Float32, 8)
@test hessian(ackley_stable, x) isa Matrix{Float32}
@test hessian(ackley_stable, x)  ≈ ForwardDiff.hessian(ackley_stable, x)
@test hessian(DiffTests.ackley, x) ≈ ForwardDiff.hessian(DiffTests.ackley, x)
end


using StaticArrays
@testset "StaticArrays" begin
x = rand(SVector{4})
f = DiffTests.rosenbrock_3
@test ForwardDiff.hessian(f, x) ≈ hessian(f, x)
@test hessian(f, x) isa MMatrix
@test_broken hessian(f, x) isa SMatrix
end # testset

@testset "No spurious promotions primitives" begin
h = HyperDual(0.8f0, Vec(0.7f0, 0.7f0), Vec(0.7f0, 0.7f0))
for (fsym, _, _) in HyperHessians.DIFF_RULES
    if fsym in (:asec, :acsc, :asecd)
        hv = inv(h)
    else
        hv = h
    end
    f = @eval $fsym
    try
        v = f(hv)
        @test v isa HyperDual{2, Float32}
    catch e
        e isa DomainError || rethrow()
    end
end
end #testset

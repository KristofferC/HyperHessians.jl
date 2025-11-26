using HyperHessians: HyperHessians, hessian, hessian!, hessiangradvalue, hessiangradvalue!, hvp, Chunk, HessianConfig, HyperDual, apply_scalar_rule
using DiffTests
using ForwardDiff
using Test
using SIMD

@testset "scalar" begin
    f(x) = exp(x) / sqrt(sin(x)^3 + cos(x)^3)
    x = rand()
    @test hessian(f, x) ≈ ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)
end # testset

@testset "correctness" begin
    for f in (
            DiffTests.rosenbrock_1,
            DiffTests.rosenbrock_2,
            DiffTests.rosenbrock_3,
            #DiffTests.rosenbrock_4,
            DiffTests.ackley,
            DiffTests.self_weighted_logit,
            DiffTests.nested_array_mul,
        )
        for n in (1, 4, 8, 8 + 7)
            if n == 1 && (f == DiffTests.rosenbrock_4 || f == DiffTests.nested_array_mul)
                continue
            end
            x = rand(n)
            H = ForwardDiff.hessian(f, x)

            for chunk in (1, n ÷ 2, n)
                if chunk <= 0
                    continue
                end
                @info "f=$f, n=$n, chunk=$chunk"

                cfg = HessianConfig(x, Chunk{chunk}())
                @test H ≈ hessian(f, x, cfg)
                @test H ≈ hessian(f, x, cfg)

            end
        end
    end
end # testset

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

@testset "Float32" begin
    x = rand(Float32, 8)
    @test hessian(ackley_stable, x) isa Matrix{Float32}
    @test hessian(ackley_stable, x) ≈ ForwardDiff.hessian(ackley_stable, x)
    @test hessian(DiffTests.ackley, x) ≈ ForwardDiff.hessian(DiffTests.ackley, x)
end

@testset "Hessian-vector products" begin
    f = DiffTests.ackley
    x = rand(7)
    v = rand(7)

    H = ForwardDiff.hessian(f, x)
    cfg_vec = HessianConfig(x, Chunk{length(x)}())
    cfg_chunk = HessianConfig(x, Chunk{4}())

    hv_vec = hvp(f, x, v, cfg_vec)
    hv_chunk = hvp(f, x, v, cfg_chunk)
    @test hv_vec ≈ H * v
    @test hv_chunk ≈ H * v
end

@testset "hessiangradvalue" begin
    f = DiffTests.ackley
    x = rand(8)
    H = zeros(length(x), length(x))
    G = zeros(length(x))
    cfg_chunk = HessianConfig(x, Chunk{4}())
    value = hessiangradvalue!(H, G, f, x, cfg_chunk)
    @test value ≈ f(x)
    @test G ≈ ForwardDiff.gradient(f, x)
    @test H ≈ ForwardDiff.hessian(f, x)

    H2 = similar(H)
    G2 = similar(G)
    cfg_vec = HessianConfig(x, Chunk{length(x)}())
    value_vec = hessiangradvalue!(H2, G2, f, x, cfg_vec)
    @test value_vec ≈ f(x)
    @test G2 ≈ ForwardDiff.gradient(f, x)
    @test H2 ≈ ForwardDiff.hessian(f, x)

    res = hessiangradvalue(f, x, cfg_vec)
    @test res.value ≈ f(x)
    @test res.gradient ≈ ForwardDiff.gradient(f, x)
    @test res.hessian ≈ ForwardDiff.hessian(f, x)
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

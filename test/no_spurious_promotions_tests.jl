@testset "No spurious promotions primitives" begin
    seed = ϵT{2, Float32}((0.7f0, 0.7f0))
    h = HyperDual(0.8f0, seed, seed)
    for (fsym, _, _) in HyperHessians.DIFF_RULES
        hv = fsym in (:asec, :acsc, :asecd) ? inv(h) : h
        f = @eval $fsym
        try
            v = f(hv)
            @test v isa HyperDual{2, 2, Float32}
        catch e
            e isa DomainError || rethrow()
        end
    end
end

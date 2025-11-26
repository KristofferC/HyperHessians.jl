@testset "No spurious promotions primitives" begin
    h = HyperDual(0.8f0, Vec(0.7f0, 0.7f0), Vec(0.7f0, 0.7f0))
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

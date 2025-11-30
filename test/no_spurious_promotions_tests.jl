module NoSpuriousPromotionsTests

using Test
using HyperHessians
using HyperHessians: HyperDual, HyperDualSIMD, ϵT, ϵT_SIMD

@testset "No spurious promotions primitives" begin
    for (seed, T, DualType) in (
            (ϵT{2, Float32}((0.7f0, 0.7f0)), HyperDual{2, 2, Float32}, HyperDual),
            (ϵT_SIMD{2, Float32}((0.7f0, 0.7f0)), HyperDualSIMD{2, 2, Float32}, HyperDualSIMD),
        )
        h = DualType(0.8f0, seed, seed)
        for (fsym, _, _) in HyperHessians.DIFF_RULES
            hv = fsym in (:asec, :acsc, :asecd) ? inv(h) : h
            f = @eval $fsym
            try
                v = f(hv)
                @test v isa T
            catch e
                e isa DomainError || rethrow()
            end
        end
    end
end

end # module

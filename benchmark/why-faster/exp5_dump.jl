# Dump full native/LLVM code for the report's code panels into dumps/.
# The mul_nested.s listing is the "two calls + 648-byte memcpy" exhibit;
# mul_hyperdual.s is the straight-line packed-FMA exhibit.
# Run with: julia --project=benchmark benchmark/why-faster/exp5_dump.jl
using HyperHessians, ForwardDiff, DiffTests, InteractiveUtils
let
    dir = joinpath(@__DIR__, "dumps")
    mkpath(dir)
    T_hd = HyperHessians.HyperDual{8, 8, Float64}
    T_d8 = ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, 8}, 8}

    open(joinpath(dir, "mul_hyperdual.s"), "w") do io
        code_native(io, *, (T_hd, T_hd); debuginfo = :none)
    end
    open(joinpath(dir, "mul_nested.s"), "w") do io
        code_native(io, *, (T_d8, T_d8); debuginfo = :none)
    end
    open(joinpath(dir, "sqrt_hyperdual.ll"), "w") do io
        code_llvm(io, sqrt, (T_hd,); debuginfo = :none)
    end
    open(joinpath(dir, "sqrt_nested.ll"), "w") do io
        code_llvm(io, sqrt, (T_d8,); debuginfo = :none)
    end
    open(joinpath(dir, "rosen_hyperdual.ll"), "w") do io
        code_llvm(io, DiffTests.rosenbrock_1, (Vector{T_hd},); debuginfo = :none)
    end
    open(joinpath(dir, "rosen_nested.ll"), "w") do io
        code_llvm(io, DiffTests.rosenbrock_1, (Vector{T_d8},); debuginfo = :none)
    end
    println("wrote dumps to ", dir)
end

# Decompose the per-element cost of sin/exp across number types, to show
# where derivative propagation dominates the scalar call. Run on both a
# NEON and an AVX-512 machine to see the lane-shape effect.
# Run with: julia --project=benchmark benchmark/why-faster/exp8_sin_case.jl
using HyperHessians, ForwardDiff, BenchmarkTools, Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

let
    Random.seed!(1)
    T_hd = HyperHessians.HyperDual{8, 8, Float64}
    T_d1 = ForwardDiff.Dual{Nothing, Float64, 8}                      # first order
    T_dd = ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, 8}, 8}
    n = 256
    x = rand(n) .+ 0.5
    for (label, T) in [
            ("Float64", Float64),
            ("Dual{F,8} 1st order", T_d1),
            ("nested Dual 8x8", T_dd),
            ("HyperDual{8,8}", T_hd),
        ]
        a = T.(x)
        o = similar(a)
        t_sin = @belapsed ($o .= sin.($a))
        t_exp = @belapsed ($o .= exp.($a))
        println(
            rpad(label, 22),
            " sin=", round(t_sin / n * 1.0e9, digits = 1), "ns",
            "  exp=", round(t_exp / n * 1.0e9, digits = 1), "ns   (per element)"
        )
    end
end

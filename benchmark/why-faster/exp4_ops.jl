# Per-element cost of single dual operations and whole-function evaluations at
# identical 81-float payload (HyperDual{8,8} vs nested Dual{Dual{Float64,8},8}).
# Run with: julia --project=benchmark benchmark/why-faster/exp4_ops.jl
using HyperHessians, ForwardDiff, DiffTests, BenchmarkTools, Random
let
    Random.seed!(1)
    T_hd = HyperHessians.HyperDual{8, 8, Float64}
    T_d8 = ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, 8}, 8}

    n = 256
    x = rand(n) .+ 0.5

    for (label, T) in [("HyperDual{8,8}", T_hd), ("nested Dual 8x8", T_d8)]
        a = T.(x)
        b = T.(x .+ 1)
        out = similar(a)
        t_mul = @belapsed ($out .= $a .* $b)
        t_div = @belapsed ($out .= $a ./ $b)
        t_exp = @belapsed ($out .= exp.($a))
        t_sqrt = @belapsed ($out .= sqrt.($a))
        t_sin = @belapsed ($out .= sin.($a))
        println(
            rpad(label, 20),
            " mul=", round(t_mul / n * 1.0e9, digits = 1), "ns",
            "  div=", round(t_div / n * 1.0e9, digits = 1), "ns",
            "  exp=", round(t_exp / n * 1.0e9, digits = 1), "ns",
            "  sqrt=", round(t_sqrt / n * 1.0e9, digits = 1), "ns",
            "  sin=", round(t_sin / n * 1.0e9, digits = 1), "ns  (per element)"
        )
    end

    f = DiffTests.rosenbrock_1
    for (label, T) in [("HyperDual{8,8}", T_hd), ("nested Dual 8x8", T_d8)]
        xd = T.(x)
        t = @belapsed $f($xd)
        println("rosenbrock_1 one eval ", rpad(label, 20), round(t * 1.0e6, digits = 2), " µs")
    end
end

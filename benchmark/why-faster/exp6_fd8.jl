using HyperHessians, ForwardDiff, DiffTests, BenchmarkTools, Random
let
    Random.seed!(1234)
    n = 256
    x = rand(n)
    H = zeros(n, n)
    count = Ref(0)
    for (name, f) in [("ackley", DiffTests.ackley), ("rosenbrock_1", DiffTests.rosenbrock_1)]
        fc = y -> (count[] += 1; f(y))
        cfg8c = ForwardDiff.HessianConfig(fc, x, ForwardDiff.Chunk{8}())
        ForwardDiff.hessian!(H, fc, x, cfg8c)
        count[] = 0
        ForwardDiff.hessian!(H, fc, x, cfg8c)
        evals = count[]

        cfg8 = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{8}())
        ForwardDiff.hessian!(H, f, x, cfg8)
        alloc = @allocated ForwardDiff.hessian!(H, f, x, cfg8)
        t = @belapsed ForwardDiff.hessian!($H, $f, $x, $cfg8)

        cfg_hh = HyperHessians.HessianConfig(x)
        HyperHessians.hessian!(H, f, x, cfg_hh)
        t_hh = @belapsed HyperHessians.hessian!($H, $f, $x, $cfg_hh)

        println(
            name, ": FD(chunk 8) evals=", evals, "  t=", round(t * 1.0e3, digits = 2),
            " ms  alloc=", alloc, " B   |  HH t=", round(t_hh * 1.0e3, digits = 2),
            " ms  speedup=", round(t / t_hh, digits = 2), "x"
        )
    end
end

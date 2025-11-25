using HyperHessians, ForwardDiff, DiffTests, BenchmarkTools, Test, Printf


struct Result
    f::Function
    n::Int
    time_fd::Float64
    timd_fh::Float64
end

function run_benchmark()
    results = Result[]
    for f in (DiffTests.ackley, DiffTests.rosenbrock_1) # DiffTests.VECTOR_TO_NUMBER_FUNCS
        @info f
        # ForwardDiff and HyperHessians should use the same default chunk size for these sizes
        for n in (1, 8, 128)
            x = rand(n)
            H_fd = similar(x, length(x), length(x))
            cfg_fd = ForwardDiff.HessianConfig(f, x)
            #ForwardDiff.hessian!(H_fd, f, x, cfg_fd)
            time_fd = @benchmark ForwardDiff.hessian!($H_fd, $f, $x, $cfg_fd)

            H_fh = similar(x, length(x), length(x))
            cfg_fh = HyperHessians.HessianConfig(f, x)
            # HyperHessians.hessian!(H_fh, f, x, cfg_fh)
            time_fh = @benchmark HyperHessians.hessian!($H_fh, $f, $x, $cfg_fh)

            push!(results, Result(f, n, minimum(time_fd.times), minimum(time_fh.times)))
            @test H_fd ≈ H_fh
        end
    end
    return results
end

function prettytime(t)
    if t < 1.0e3
        value, units = t, "ns"
    elseif t < 1.0e6
        value, units = t / 1.0e3, "μs"
    elseif t < 1.0e9
        value, units = t / 1.0e6, "ms"
    else
        value, units = t / 1.0e9, "s"
    end
    return string(@sprintf("%.3f", value), " ", units)
end


function print_results(io::IO, results)
    print(
        io, """
        | Function      | input length | Time ForwardDiff | Time HyperHessians | Speedup |
        | ------------- | ------------ | ---------------- | ----------------- | --------|
        """
    )

    for r in results
        ratio = round(r.time_fd / r.timd_fh, sigdigits = 2)
        println(
            io,
            "| `", r.f, "` | ", r.n, " | ", prettytime(r.time_fd), " | ", prettytime(r.timd_fh), " | ", ratio
        )
    end
    return
end

results = run_benchmark()
print_results(stdout, results)

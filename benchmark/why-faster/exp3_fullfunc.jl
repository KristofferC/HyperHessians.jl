# LLVM IR statistics for whole benchmark functions compiled on each dual type:
# vectorization (<2 x double> ops), memcpys, and out-of-line call sites.
# The nested-dual versions show non-inlined `dual_definition_retval` calls in
# the hot loop; the HyperDual versions are call-free straight-line code.
# Run with: julia --project=benchmark benchmark/why-faster/exp3_fullfunc.jl
using HyperHessians, ForwardDiff, DiffTests, InteractiveUtils

T_hd = HyperHessians.HyperDual{8, 8, Float64}
T_dd8 = ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, 8}, 8}

function llvm_stats(f, Ts)
    buf = IOBuffer()
    code_llvm(buf, f, Ts; debuginfo = :none)
    s = String(take!(buf))
    v2 = length(collect(eachmatch(r"<2 x double>", s)))
    v4 = length(collect(eachmatch(r"<4 x double>", s)))
    memcpy = length(collect(eachmatch(r"@llvm\.memcpy", s)))
    calls = [m.captures[1] for m in eachmatch(r"call[^@\n]*@(\"?j1?_[\w\.\#]+)", s)]
    nlines = count(==('\n'), s)
    return (; nlines, v2, v4, memcpy, ncalls = length(calls), calls = unique(calls))
end

for (label, fn, T) in [
        ("ackley HyperDual{8,8}", DiffTests.ackley, Vector{T_hd}),
        ("ackley nested 8x8", DiffTests.ackley, Vector{T_dd8}),
        ("rosenbrock HyperDual{8,8}", DiffTests.rosenbrock_1, Vector{T_hd}),
        ("rosenbrock nested 8x8", DiffTests.rosenbrock_1, Vector{T_dd8}),
    ]
    st = llvm_stats(fn, (T,))
    println(
        rpad(label, 28), " IRlines=", lpad(st.nlines, 6),
        "  <2xf64>=", lpad(st.v2, 5), "  <4xf64>=", lpad(st.v4, 4),
        "  memcpy=", lpad(st.memcpy, 4),
        "  outcalls=", st.ncalls
    )
    println("    calls: ", join(st.calls, ", "))
end

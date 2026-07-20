# Evaluation counts, allocations, and per-eval cost — chunk 8 pinned for both
# packages so ForwardDiff's nested dual and HyperDual carry the same 81-float
# payload. Run with: julia --project=benchmark benchmark/why-faster/exp1_counts.jl
using HyperHessians, ForwardDiff, DiffTests, BenchmarkTools, Random

Random.seed!(1234)
n = 256
x = rand(n)
f = DiffTests.ackley

const COUNT = Ref(0)

H = zeros(n, n)
chunk_fd = ForwardDiff.Chunk{8}()

# --- evaluation counts -------------------------------------------------
fc = y -> (COUNT[] += 1; f(y))
cfg_fd_c = ForwardDiff.HessianConfig(fc, x, chunk_fd)
ForwardDiff.hessian!(H, fc, x, cfg_fd_c) # warm
COUNT[] = 0
ForwardDiff.hessian!(H, fc, x, cfg_fd_c)
fd_evals = COUNT[]

cfg_hh = HyperHessians.HessianConfig(x)
HyperHessians.hessian!(H, fc, x, cfg_hh)
COUNT[] = 0
HyperHessians.hessian!(H, fc, x, cfg_hh)
hh_evals = COUNT[]

println("n = $n, chunk 8 both")
println("FD evals of f per hessian!: ", fd_evals)   # ⌈n/8⌉² = 1024
println("HH evals of f per hessian!: ", hh_evals)   # k(k+1)/2 = 528

# --- dual type sizes ---------------------------------------------------
cfg_fd = ForwardDiff.HessianConfig(f, x, chunk_fd)
DD = eltype(cfg_fd.gradient_config)   # nested dual fed to f
HD = eltype(cfg_hh.duals)
println("FD nested dual type: ", DD, "  sizeof = ", sizeof(DD), " bytes (", sizeof(DD) ÷ 8, " Float64s)")
println("HH hyperdual type:   ", HD, "  sizeof = ", sizeof(HD), " bytes (", sizeof(HD) ÷ 8, " Float64s)")
println("input working set per eval (both): ", n * sizeof(HD) / 1024, " KiB")

# --- allocations per hessian! ------------------------------------------
ForwardDiff.hessian!(H, f, x, cfg_fd)
a_fd = @allocated ForwardDiff.hessian!(H, f, x, cfg_fd)
HyperHessians.hessian!(H, f, x, cfg_hh)
a_hh = @allocated HyperHessians.hessian!(H, f, x, cfg_hh)
println("FD allocations per hessian!: ", a_fd, " bytes")
println("HH allocations per hessian!: ", a_hh, " bytes")

# --- per-eval cost of f on each dual type (identical payload) ----------
xdd = DD.(x)
xhd = HD.(x)
t_dd = @belapsed $f($xdd)
t_hd = @belapsed $f($xhd)
t_plain = @belapsed $f($x)
println("one f eval, plain Float64:          ", t_plain * 1.0e6, " µs")
println("one f eval, FD nested dual (8x8):   ", t_dd * 1.0e6, " µs")
println("one f eval, HH hyperdual (8x8):     ", t_hd * 1.0e6, " µs")

# --- totals: evals × per-eval cost should predict these; the residual is
# --- seeding/extraction/allocation overhead ----------------------------
t_fd = @belapsed ForwardDiff.hessian!($H, $f, $x, $cfg_fd)
t_hh = @belapsed HyperHessians.hessian!($H, $f, $x, $cfg_hh)
println("full FD hessian!: ", t_fd * 1.0e3, " ms   (evals*evalcost = ", fd_evals * t_dd * 1.0e3, " ms)")
println("full HH hessian!: ", t_hh * 1.0e3, " ms   (evals*evalcost = ", hh_evals * t_hd * 1.0e3, " ms)")
println("speedup: ", t_fd / t_hh)

# Collect machine-specific codegen and timing evidence for the why-faster
# report, in one self-describing text file that can be diffed across machines.
#
# Usage (from the repository root):
#
#     julia --project=benchmark benchmark/why-faster/collect.jl
#
# Writes benchmark/why-faster/results/<cpu>-<stock|patched>.txt
#
# Run it twice per machine:
#   1. with the registry ForwardDiff        -> "...-stock.txt"
#   2. after applying forwarddiff-inline.patch and Pkg.develop'ing the
#      patched clone (see README.md)        -> "...-patched.txt"
# The script detects which ForwardDiff is active by inspecting the generated
# code, so the same command produces the right file name in both cases.
#
# Takes a couple of minutes. Commit (or send back) the results/ files.

using HyperHessians, ForwardDiff, DiffTests, BenchmarkTools, Random, InteractiveUtils

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

const T_HD = HyperHessians.HyperDual{8, 8, Float64}
const T_DD = ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, 8}, 8}

function llvm_string(f, Ts)
    buf = IOBuffer()
    code_llvm(buf, f, Ts; debuginfo = :none)
    return String(take!(buf))
end

function native_string(f, Ts)
    buf = IOBuffer()
    code_native(buf, f, Ts; debuginfo = :none)
    return String(take!(buf))
end

function native_stats(s)
    instr = [strip(l) for l in split(s, '\n') if startswith(l, '\t')]
    n_total = length(instr)
    n_fma = count(i -> occursin(r"^fmla|^fmadd|^fmls|^vfmadd|^vfmsub|^vfnmadd", i), instr)
    n_call = count(i -> occursin(r"^bl\s|^callq|^call\s", i), instr)
    n_xmm = count(i -> occursin("xmm", i), instr)
    n_ymm = count(i -> occursin("ymm", i), instr)
    n_zmm = count(i -> occursin("zmm", i), instr)
    return (; n_total, n_fma, n_call, n_xmm, n_ymm, n_zmm)
end

# Detect whether the active ForwardDiff has the @inline patch: the stock
# version leaves out-of-line j_dual_definition_retval calls in nested-dual *.
fd_patched() = !occursin("dual_definition_retval", llvm_string(*, (T_DD, T_DD)))

function main()
    variant = fd_patched() ? "patched" : "stock"
    cpu = replace(lowercase(Sys.CPU_NAME), r"[^a-z0-9-]" => "-")
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    out = joinpath(outdir, "$cpu-$variant.txt")

    open(out, "w") do io
        println(io, "=== SECTION: info ===")
        println(io, "cpu: ", Sys.CPU_NAME)
        println(io, "arch: ", Sys.ARCH)
        println(io, "julia: ", VERSION)
        println(io, "threads: ", Threads.nthreads())
        println(io, "ForwardDiff: v", pkgversion(ForwardDiff), " at ", pkgdir(ForwardDiff))
        println(io, "variant: ", variant)
        println(io, "HyperDual type: ", T_HD, " (", sizeof(T_HD), " bytes)")
        println(io, "nested Dual type: ", T_DD, " (", sizeof(T_DD), " bytes)")

        # ---- per-op timings, equal payload -----------------------------
        println(io, "\n=== SECTION: per-op ns/element (n=256 vectors) ===")
        Random.seed!(1)
        n = 256
        x = rand(n) .+ 0.5
        for (label, T) in [("HyperDual{8,8}", T_HD), ("nested Dual 8x8", T_DD)]
            a = T.(x)
            b = T.(x .+ 1)
            o = similar(a)
            vals = (
                mul = @belapsed($o .= $a .* $b),
                div = @belapsed($o .= $a ./ $b),
                exp = @belapsed($o .= exp.($a)),
                sqrt = @belapsed($o .= sqrt.($a)),
                sin = @belapsed($o .= sin.($a)),
            )
            print(io, rpad(label, 18))
            for (k, v) in pairs(vals)
                print(io, " ", k, "=", round(v / n * 1.0e9, digits = 1), "ns")
            end
            println(io)
        end

        # ---- whole-function per-eval -----------------------------------
        println(io, "\n=== SECTION: one f eval (n=256) ===")
        for (fname, f) in [("ackley", DiffTests.ackley), ("rosenbrock_1", DiffTests.rosenbrock_1)]
            for (label, T) in [("HyperDual{8,8}", T_HD), ("nested Dual 8x8", T_DD)]
                xd = T.(x)
                println(io, rpad(fname, 14), rpad(label, 18), round(@belapsed($f($xd)) * 1.0e6, digits = 2), " µs")
            end
        end

        # ---- end-to-end hessian!, chunk 8 both -------------------------
        println(io, "\n=== SECTION: hessian! totals (n=256, chunk 8 both) ===")
        Random.seed!(1234)
        xh = rand(n)
        H = zeros(n, n)
        count = Ref(0)
        for (fname, f) in [("ackley", DiffTests.ackley), ("rosenbrock_1", DiffTests.rosenbrock_1)]
            fc = y -> (count[] += 1; f(y))
            cfgc = ForwardDiff.HessianConfig(fc, xh, ForwardDiff.Chunk{8}())
            ForwardDiff.hessian!(H, fc, xh, cfgc)
            count[] = 0
            ForwardDiff.hessian!(H, fc, xh, cfgc)
            evals = count[]

            cfg_fd = ForwardDiff.HessianConfig(f, xh, ForwardDiff.Chunk{8}())
            ForwardDiff.hessian!(H, f, xh, cfg_fd)
            alloc = @allocated ForwardDiff.hessian!(H, f, xh, cfg_fd)
            t_fd = @belapsed ForwardDiff.hessian!($H, $f, $xh, $cfg_fd)

            cfg_hh = HyperHessians.HessianConfig(xh)
            HyperHessians.hessian!(H, f, xh, cfg_hh)
            t_hh = @belapsed HyperHessians.hessian!($H, $f, $xh, $cfg_hh)

            println(
                io, rpad(fname, 14), "FD evals=", evals, " t=", round(t_fd * 1.0e3, digits = 2),
                "ms alloc=", alloc, "B | HH t=", round(t_hh * 1.0e3, digits = 2),
                "ms | speedup=", round(t_fd / t_hh, digits = 2), "x"
            )
        end

        # ---- native code stats + listings for the multiply kernels -----
        s_hd = native_string(*, (T_HD, T_HD))
        s_dd = native_string(*, (T_DD, T_DD))
        println(io, "\n=== SECTION: native mul stats ===")
        for (label, s) in [("HyperDual{8,8}", s_hd), ("nested Dual 8x8", s_dd)]
            st = native_stats(s)
            println(
                io, rpad(label, 18), " instrs=", st.n_total, " fma=", st.n_fma,
                " calls=", st.n_call, " xmm=", st.n_xmm, " ymm=", st.n_ymm, " zmm=", st.n_zmm
            )
        end

        println(io, "\n=== SECTION: asm mul HyperDual{8,8} ===")
        println(io, s_hd)
        println(io, "\n=== SECTION: asm mul nested Dual 8x8 ===")
        println(io, s_dd)
    end

    println("wrote ", out)
    return out
end

main()

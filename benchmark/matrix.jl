# HyperHessians vs ForwardDiff speedup matrix — the numbers behind the slide table.
#
# For every (operation, function, input size) cell:
#   1. ChunkPicker picks the fastest configuration for each package
#      (HyperHessians through its native backend, so the chunk sweep also
#       includes the Jet variants; ForwardDiff through DifferentiationInterface).
#   2. The two winning configurations are re-benchmarked with a bigger budget —
#      those times make the table. ForwardDiff runs through DifferentiationInterface
#      (which is also the only way it exposes HVPs); HyperHessians uses its native API.
#   3. Every HyperHessians result is checked against ForwardDiff before being recorded.
#
# Run (any machine):
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark benchmark/matrix.jl
#
# Results land in benchmark/results/matrix-<tag>.toml (raw data, including the
# picked configurations and full sweep timings) and benchmark/results/
# matrix-<tag>.md (ready-to-paste slide tables). <tag> defaults to the hostname.
#
# Env knobs:
#   MATRIX_TAG     results file tag (default: hostname)
#   BENCH_SECONDS  budget for the final measurement of each winner (default 1.0)
#   PICK_SECONDS   per-candidate budget in the configuration sweep (default 0.25)
#   REPORT_ONLY=1  regenerate the .md from an existing .toml without benchmarking

using HyperHessians
using ForwardDiff
using DiffTests
using ChunkPicker
using BenchmarkTools
import DifferentiationInterface as DI
using DifferentiationInterface: AutoForwardDiff
using LinearAlgebra
using Random
using Printf
using TOML
using Dates
import Pkg

const RESULTS_DIR = joinpath(@__DIR__, "results")
isdir(RESULTS_DIR) || mkdir(RESULTS_DIR)
const TAG = replace(get(ENV, "MATRIX_TAG", first(split(gethostname(), '.'))), r"[^A-Za-z0-9_-]" => "-")
const TOML_FILE = joinpath(RESULTS_DIR, "matrix-$TAG.toml")
const MD_FILE = joinpath(RESULTS_DIR, "matrix-$TAG.md")

const BENCH_SECONDS = parse(Float64, get(ENV, "BENCH_SECONDS", "1.0"))
const PICK_SECONDS = parse(Float64, get(ENV, "PICK_SECONDS", "0.25"))

# The serial comparison should be honest even if Julia was started with threads.
LinearAlgebra.BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# The benchmark grid
# ---------------------------------------------------------------------------
logsumexp(x) = log(sum(exp, x))

const FUNCS = [
    (name = "ackley", f = DiffTests.ackley, note = "dense, transcendental (exp/cos/sqrt)"),
    (name = "rosenbrock", f = DiffTests.rosenbrock_1, note = "banded Hessian, cheap polynomial loop"),
    (name = "logsumexp", f = logsumexp, note = "diagonal + rank-1 dense Hessian"),
    (name = "self_weighted_logit", f = DiffTests.self_weighted_logit, note = "fully coupled dense via dot(x,x)"),
]
const SIZES = [4, 16, 64, 256]
const OPS = [:hessian, :hvp]

geomean(v) = exp(sum(log, v) / length(v))

# ---------------------------------------------------------------------------
# Final measurement of the picked configurations
# ---------------------------------------------------------------------------
function hh_config(x, pick, op::Symbol, v)
    if op === :hessian
        return pick.kind === :jet ?
            HyperHessians.HessianConfig(x, HyperHessians.Jet; simd = pick.simd) :
            HyperHessians.HessianConfig(x, HyperHessians.Chunk{pick.chunk}(); simd = pick.simd)
    else
        return HyperHessians.HVPConfig(x, v, HyperHessians.Chunk{pick.chunk}(); simd = pick.simd)
    end
end

function bench_final(op::Symbol, f::F, x, v, hh_pick, fd_pick) where {F}
    n = length(x)
    fd_backend = AutoForwardDiff(chunksize = fd_pick.chunk)
    if op === :hessian
        H_fd = Matrix{Float64}(undef, n, n)
        prep = DI.prepare_hessian(f, fd_backend, x)
        DI.hessian!(f, H_fd, prep, fd_backend, x)
        t_fd = @belapsed DI.hessian!($f, $H_fd, $prep, $fd_backend, $x) seconds = BENCH_SECONDS

        H_hh = Matrix{Float64}(undef, n, n)
        cfg = hh_config(x, hh_pick, op, v)
        HyperHessians.hessian!(H_hh, f, x, cfg)
        t_hh = @belapsed HyperHessians.hessian!($H_hh, $f, $x, $cfg) seconds = BENCH_SECONDS

        ok = isapprox(H_fd, H_hh; rtol = 1.0e-8, atol = 1.0e-10)
        ok || @warn "Hessian mismatch" f n maxdiff = maximum(abs, H_fd .- H_hh)
        return t_fd, t_hh, ok
    else
        tx = (v,)
        ty = (similar(v),)
        prep = DI.prepare_hvp(f, fd_backend, x, tx)
        DI.hvp!(f, ty, prep, fd_backend, x, tx)
        t_fd = @belapsed DI.hvp!($f, $ty, $prep, $fd_backend, $x, $tx) seconds = BENCH_SECONDS

        hv = similar(v)
        cfg = hh_config(x, hh_pick, op, v)
        HyperHessians.hvp!(hv, f, x, v, cfg)
        t_hh = @belapsed HyperHessians.hvp!($hv, $f, $x, $v, $cfg) seconds = BENCH_SECONDS

        ok = isapprox(ty[1], hv; rtol = 1.0e-8, atol = 1.0e-10)
        ok || @warn "HVP mismatch" f n maxdiff = maximum(abs, ty[1] .- hv)
        return t_fd, t_hh, ok
    end
end

sweep_data(pick) = [
    Dict{String, Any}(
        "chunk" => t.chunk, "kind" => String(t.kind), "simd" => t.simd, "time" => t.time,
    ) for t in pick.timings
]

function run_cell(op::Symbol, func, n::Int)
    f = func.f
    Random.seed!(1234)
    x = rand(n)
    v = rand(n)

    hh_pick = pick_chunk(
        HyperHessiansBackend(), f, x;
        op, tangents = v, seconds = PICK_SECONDS, verbose = false,
    )
    fd_pick = pick_chunk(
        AutoForwardDiff(), f, x;
        op, tangents = v, seconds = PICK_SECONDS, verbose = false,
    )

    t_fd, t_hh, ok = bench_final(op, f, x, v, hh_pick, fd_pick)

    @printf(
        "  %-20s n=%-4d  FD %10s (%s)   HH %10s (%s)   speedup %.2fx%s\n",
        func.name, n, BenchmarkTools.prettytime(t_fd * 1.0e9), fd_pick.recommendation,
        BenchmarkTools.prettytime(t_hh * 1.0e9), hh_pick.recommendation,
        t_fd / t_hh, ok ? "" : "  MISMATCH",
    )

    return Dict{String, Any}(
        "op" => String(op), "func" => func.name, "n" => n,
        "speedup" => t_fd / t_hh,
        "check_passed" => ok,
        "fd" => Dict{String, Any}(
            "time" => t_fd,
            "chunk" => fd_pick.chunk,
            "config" => fd_pick.recommendation,
            "sweep" => sweep_data(fd_pick),
        ),
        "hh" => Dict{String, Any}(
            "time" => t_hh,
            "chunk" => hh_pick.chunk,
            "kind" => String(hh_pick.kind),
            "simd" => hh_pick.simd,
            "config" => hh_pick.recommendation,
            "sweep" => sweep_data(hh_pick),
        ),
    )
end

function machine_meta()
    # "1.4.5" for a registered dep, "1.4.5 @ 569af35" for a path-tracked one
    # (with "+dirty" appended if the checkout has uncommitted src changes).
    pkgver(name) = begin
        deps = [d for d in values(Pkg.dependencies()) if d.name == name]
        isempty(deps) && return "unknown"
        d = deps[1]
        ver = string(something(d.version, "dev"))
        d.is_tracking_path || return ver
        # the manifest's recorded version can be stale for path deps
        ver = string(get(TOML.parsefile(joinpath(d.source, "Project.toml")), "version", ver))
        rev = try
            readchomp(`git -C $(d.source) rev-parse --short HEAD`)
        catch
            return "$ver @ path"
        end
        dirty = !isempty(read(`git -C $(d.source) status --short --untracked-files=no`, String))
        return "$ver @ $rev" * (dirty ? "+dirty" : "")
    end
    return Dict{String, Any}(
        "date" => Dates.format(now(), dateformat"yyyy-mm-dd HH:MM"),
        "host" => gethostname(),
        "cpu" => Sys.cpu_info()[1].model,
        "arch" => string(Sys.ARCH),
        "os" => string(Sys.KERNEL),
        "julia" => string(VERSION),
        "nthreads" => Threads.nthreads(),
        "blas_threads" => 1,
        "bench_seconds" => BENCH_SECONDS,
        "pick_seconds" => PICK_SECONDS,
        "hyperhessians_rev" => pkgver("HyperHessians"),
        "forwarddiff_version" => pkgver("ForwardDiff"),
        "differentiationinterface_version" => pkgver("DifferentiationInterface"),
    )
end

# ---------------------------------------------------------------------------
# Markdown report: one speedup matrix per op, geomean margins, config matrix
# ---------------------------------------------------------------------------
cell_for(results, op, fname, n) =
    only(r for r in results if r["op"] == op && r["func"] == fname && r["n"] == n)

# Short label for a picked configuration, e.g. "J" (jet), "c8" (chunk 8),
# with an "s" suffix for the simd variants.
hh_label(hh) = (hh["kind"] == "jet" ? "J" : "c$(hh["chunk"])") * (hh["simd"] ? "s" : "")

function op_tables(io, results, op::String, funcnames)
    sizes = sort(unique(r["n"] for r in results if r["op"] == op))
    fmt(x) = @sprintf("%.2f×", x)

    println(io, "### ", op == "hessian" ? "Hessian" : "Hessian-vector product", " speedup (ForwardDiff time / HyperHessians time)\n")
    println(io, "| function | ", join(("n=$n" for n in sizes), " | "), " | geomean |")
    println(io, "| --- | ", join(("---:" for _ in sizes), " | "), " | ---: |")
    for fname in funcnames
        sp = [cell_for(results, op, fname, n)["speedup"] for n in sizes]
        # each cell carries the picked configs (hh / fd) as a muted annotation
        cells = map(sizes, sp) do n, s
            r = cell_for(results, op, fname, n)
            fmt(s) * " <span class=\"cfg\">$(hh_label(r["hh"]))/c$(r["fd"]["chunk"])</span>"
        end
        println(io, "| $fname | ", join(cells, " | "), " | ==", fmt(geomean(sp)), "== |")
    end
    colmeans = [geomean([cell_for(results, op, fname, n)["speedup"] for fname in funcnames]) for n in sizes]
    total = geomean([r["speedup"] for r in results if r["op"] == op])
    println(io, "| **geomean** | ", join(fmt.(colmeans), " | "), " | ==**", fmt(total), "**== |")

    println(io, "\nPicked configurations (HyperHessians / ForwardDiff chunk; J = Jet, cN = chunk N):\n")
    println(io, "| function | ", join(("n=$n" for n in sizes), " | "), " |")
    println(io, "| --- | ", join(("---" for _ in sizes), " | "), " |")
    for fname in funcnames
        labels = map(sizes) do n
            r = cell_for(results, op, fname, n)
            "$(hh_label(r["hh"])) / c$(r["fd"]["chunk"])"
        end
        println(io, "| $fname | ", join(labels, " | "), " |")
    end
    println(io)
    return
end

function report(data)
    meta, results = data["meta"], data["results"]
    funcnames = unique(r["func"] for r in results)
    io = IOBuffer()
    println(io, "# HyperHessians vs ForwardDiff — best-configuration speedup matrix\n")
    println(
        io,
        "$(meta["cpu"]) ($(meta["arch"])) · julia $(meta["julia"]) · " *
            "ForwardDiff $(meta["forwarddiff_version"]) via DifferentiationInterface " *
            "$(meta["differentiationinterface_version"]) · HyperHessians $(meta["hyperhessians_rev"]) · " *
            "single thread · $(meta["date"])\n",
    )
    println(
        io,
        "Each cell: both packages at their fastest ChunkPicker-picked configuration, " *
            "BenchmarkTools minimum time, results verified against each other.\n",
    )
    for op in ("hessian", "hvp")
        any(r["op"] == op for r in results) && op_tables(io, results, op, funcnames)
    end
    bad = [r for r in results if !r["check_passed"]]
    isempty(bad) || println(io, "**WARNING:** $(length(bad)) cell(s) failed the correctness check.\n")
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    if get(ENV, "REPORT_ONLY", "0") == "1"
        data = TOML.parsefile(TOML_FILE)
    else
        results = Dict{String, Any}[]
        for op in OPS
            printstyled("== $op\n"; bold = true)
            for func in FUNCS, n in SIZES
                push!(results, run_cell(op, func, n))
            end
        end
        data = Dict{String, Any}("meta" => machine_meta(), "results" => results)
        open(TOML_FILE, "w") do io
            TOML.print(io, data; sorted = true)
        end
        @info "wrote $TOML_FILE"
    end
    md = report(data)
    write(MD_FILE, md)
    @info "wrote $MD_FILE"
    print(md)
    return data
end

main()

# Fair performance comparison: HyperHessians.jl vs ForwardDiff.jl
#
# Run with:
#     julia --project=benchmark benchmark/compare.jl
#
# The benchmark results are cached to benchmark/results/data.jls. To re-draw the
# plots from cached data without re-running the benchmarks:
#     REPLOT=1 julia --project=benchmark benchmark/compare.jl
#
# Produces PNGs and a markdown summary in benchmark/results/.
#
# Fairness notes:
#   * Both packages get a preallocated config and output, and use the in-place
#     API (`hessian!`) so nothing but the differentiation work is timed.
#   * Each package uses its own default chunk size (the out-of-the-box
#     experience). The chunk sizes actually used are reported.
#   * The same input `x` (fixed seed) is fed to both and every result is checked
#     against ForwardDiff for correctness before being recorded.
#   * Minimum time over the benchmark samples is used (least noise).

using HyperHessians
using ForwardDiff
using DiffTests
using BenchmarkTools
using Printf
using Random
using LinearAlgebra
using Serialization
using CairoMakie
using DifferentiationInterface: prepare_hvp, hvp!, AutoForwardDiff

const RESULTS_DIR = joinpath(@__DIR__, "results")
isdir(RESULTS_DIR) || mkdir(RESULTS_DIR)
const DATA_FILE = joinpath(RESULTS_DIR, "data.jls")

# Time budget per individual benchmark (seconds). Lower for a quick run.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "BENCH_SECONDS", "1.0"))

# Keep everything except the explicit threading benchmark single-threaded, so
# the serial comparison is honest even when Julia is started with many threads.
LinearAlgebra.BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------
# Representative test functions with different Hessian characteristics
# ---------------------------------------------------------------------------
const FUNCS = [
    (f = DiffTests.ackley,              name = "ackley",              note = "dense, transcendental (exp/cos/sqrt)"),
    (f = DiffTests.rosenbrock_1,        name = "rosenbrock_1",        note = "banded Hessian, cheap polynomial loop"),
    (f = DiffTests.self_weighted_logit, name = "self_weighted_logit", note = "fully coupled dense via dot(x,x)"),
    (f = DiffTests.vec2num_3,           name = "vec2num_3",           note = "O(n^2) work per eval: norm(x'.*x)"),
]

# vec2num_3 does O(n^2) work per call, so cap its size to keep runtime sane.
sizes_for(name) = name == "vec2num_3" ? [2, 4, 8, 16, 32, 64, 128] :
    [2, 4, 8, 16, 32, 64, 128, 256]

const HVP_FUNCS = FUNCS[1:2]
const CHUNK_SIZES = [1, 2, 4, 8, 16, 32]
const CHUNK_N = 128

# Threading scaling: sweep ntasks on a large-enough input that a Hessian takes
# well over ~100µs serially (below that, the serial config wins). Capped at 8 —
# this machine's 8 performance cores — since spilling onto slower efficiency
# cores with a static equal split gives irregular, sub-linear returns.
const THREAD_FUNCS = FUNCS[1:2]
const THREAD_N = 512
const NTASKS = filter(<=(Threads.nthreads()), [1, 2, 4, 6, 8])

fd_chunk(n) = ForwardDiff.pickchunksize(n)
hh_chunk(x) = HyperHessians.chunksize(HyperHessians.Chunk(x))

# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------
function bench_hessian(f, n)
    Random.seed!(1234)
    x = rand(n)

    H_fd = Matrix{Float64}(undef, n, n)
    cfg_fd = ForwardDiff.HessianConfig(f, x)
    ForwardDiff.hessian!(H_fd, f, x, cfg_fd)
    t_fd = @belapsed ForwardDiff.hessian!($H_fd, $f, $x, $cfg_fd)

    H_hh = Matrix{Float64}(undef, n, n)
    cfg_hh = HyperHessians.HessianConfig(x)
    HyperHessians.hessian!(H_hh, f, x, cfg_hh)
    t_hh = @belapsed HyperHessians.hessian!($H_hh, $f, $x, $cfg_hh)

    isapprox(H_fd, H_hh; rtol = 1.0e-8, atol = 1.0e-10) ||
        @warn "Hessian mismatch" f n maxdiff = maximum(abs, H_fd .- H_hh)
    return (t_fd = t_fd, t_hh = t_hh)
end

function bench_hvp(f, n)
    Random.seed!(1234)
    x = rand(n)
    v = rand(n)

    hv_hh = similar(x)
    cfg_hh = HyperHessians.HVPConfig(x)
    HyperHessians.hvp!(hv_hh, f, x, v, cfg_hh)
    t_hh = @belapsed HyperHessians.hvp!($hv_hh, $f, $x, $v, $cfg_hh)

    backend = AutoForwardDiff()
    prep = prepare_hvp(f, backend, x, (v,))
    hv_fd = (similar(x),)
    hvp!(f, hv_fd, prep, backend, x, (v,))
    t_fd = @belapsed hvp!($f, $hv_fd, $prep, $backend, $x, ($v,))

    isapprox(hv_hh, hv_fd[1]; rtol = 1.0e-8, atol = 1.0e-10) || @warn "HVP mismatch" f n
    return (t_fd = t_fd, t_hh = t_hh)
end

function bench_chunks(f, n, chunks)
    Random.seed!(1234)
    x = rand(n)
    H = Matrix{Float64}(undef, n, n)

    ts = Float64[]
    for c in chunks
        cfg = HyperHessians.HessianConfig(x, HyperHessians.Chunk{c}())
        HyperHessians.hessian!(H, f, x, cfg)
        push!(ts, @belapsed HyperHessians.hessian!($H, $f, $x, $cfg))
    end
    return (ts = ts,)
end

# Threading scaling: serial baseline vs ThreadedHessianConfig for a sweep of
# ntasks. The independent chunk passes are what get parallelized.
function bench_threading(f, n, ntasks_list)
    Random.seed!(1234)
    x = rand(n)
    H = Matrix{Float64}(undef, n, n)

    cfg_serial = HyperHessians.HessianConfig(x)
    HyperHessians.hessian!(H, f, x, cfg_serial)
    H_ref = copy(H)
    t_serial = @belapsed HyperHessians.hessian!($H, $f, $x, $cfg_serial)

    ts = Float64[]
    for nt in ntasks_list
        cfg = HyperHessians.ThreadedHessianConfig(x, HyperHessians.Chunk(x); ntasks = nt)
        HyperHessians.hessian!(H, f, x, cfg)
        isapprox(H, H_ref; rtol = 1.0e-10) || @warn "threaded mismatch" f n nt
        push!(ts, @belapsed HyperHessians.hessian!($H, $f, $x, $cfg))
    end
    return (ntasks = collect(ntasks_list), ts = ts, t_serial = t_serial)
end

# ---------------------------------------------------------------------------
# Collect data (or load from cache)
# ---------------------------------------------------------------------------
if haskey(ENV, "REPLOT") && isfile(DATA_FILE)
    @info "Loading cached benchmark data from $DATA_FILE"
    data = deserialize(DATA_FILE)
    hess_data, hvp_data, chunk_data = data.hess, data.hvp, data.chunk
    thread_data = get(data, :thread, nothing)
else
    hess_data = Dict{String, Any}()
    @info "Benchmarking full Hessian"
    for F in FUNCS
        ns = sizes_for(F.name)
        t_fd = Float64[]; t_hh = Float64[]
        for n in ns
            @info "  hessian $(F.name) n=$n"
            r = bench_hessian(F.f, n)
            push!(t_fd, r.t_fd); push!(t_hh, r.t_hh)
        end
        hess_data[F.name] = (ns = ns, t_fd = t_fd, t_hh = t_hh)
    end

    hvp_data = Dict{String, Any}()
    @info "Benchmarking Hessian-vector products"
    for F in HVP_FUNCS
        ns = [8, 16, 32, 64, 128, 256]
        t_fd = Float64[]; t_hh = Float64[]
        for n in ns
            @info "  hvp $(F.name) n=$n"
            r = bench_hvp(F.f, n)
            push!(t_fd, r.t_fd); push!(t_hh, r.t_hh)
        end
        hvp_data[F.name] = (ns = ns, t_fd = t_fd, t_hh = t_hh)
    end

    chunk_data = Dict{String, Any}()
    @info "Benchmarking chunk-size sensitivity"
    for F in HVP_FUNCS
        @info "  chunk sweep $(F.name) n=$CHUNK_N"
        chunk_data[F.name] = bench_chunks(F.f, CHUNK_N, CHUNK_SIZES)
    end

    thread_data = nothing
    if Threads.nthreads() > 1
        thread_data = Dict{String, Any}()
        @info "Benchmarking threading scaling (nthreads=$(Threads.nthreads()), ntasks=$NTASKS)"
        for F in THREAD_FUNCS
            @info "  threading $(F.name) n=$THREAD_N"
            thread_data[F.name] = bench_threading(F.f, THREAD_N, NTASKS)
        end
    else
        @warn "Julia started with 1 thread; skipping threading scaling. Rerun with `julia --threads=auto`."
    end

    serialize(DATA_FILE, (; hess = hess_data, hvp = hvp_data, chunk = chunk_data, thread = thread_data))
    @info "Cached data to $DATA_FILE"
end

# ===========================================================================
# Plotting — custom CairoMakie theme (matches the report palette)
# ===========================================================================
const INK   = "#16202b"
const MUTED = "#5b6672"
const GROUND = RGBf(0.984, 0.988, 0.992)      # faint cool panel
# Per-function palette: steel blue, rust, green, violet
const FCOLOR = Dict(
    "ackley"              => "#2f6f9f",
    "rosenbrock_1"        => "#b1442f",
    "self_weighted_logit" => "#3f7d5a",
    "vec2num_3"           => "#8a5a9c",
)
const FMARK = Dict(
    "ackley"              => :circle,
    "rosenbrock_1"        => :rect,
    "self_weighted_logit" => :diamond,
    "vec2num_3"           => :utriangle,
)

set_theme!(
    Theme(
        fontsize = 15,
        figure_padding = 20,
        backgroundcolor = :white,
        Axis = (
            backgroundcolor = GROUND,
            xgridcolor = (INK, 0.07), ygridcolor = (INK, 0.07),
            xminorgridvisible = false, yminorgridvisible = false,
            topspinevisible = false, rightspinevisible = false,
            leftspinecolor = (INK, 0.25), bottomspinecolor = (INK, 0.25),
            xtickcolor = (INK, 0.35), ytickcolor = (INK, 0.35),
            xticklabelcolor = MUTED, yticklabelcolor = MUTED,
            xlabelcolor = INK, ylabelcolor = INK,
            xlabelfont = :bold, ylabelfont = :bold,
            titlecolor = INK, titlefont = :bold, titlealign = :left, titlesize = 17,
            titlegap = 8,
        ),
        Legend = (
            framevisible = true, framecolor = (INK, 0.12),
            backgroundcolor = (:white, 0.9), labelcolor = INK, padding = (10, 10, 6, 6),
            rowgap = 2, patchsize = (26, 14),
        ),
    ),
)

logticks(vals) = (Float64.(vals), string.(vals))

# Human-readable time-axis ticks. Values are in µs (the plotted unit).
function fmt_time_µs(v)
    s = v * 1.0e-6
    n, u = s < 1.0e-6 ? (s * 1.0e9, "ns") :
        s < 1.0e-3 ? (s * 1.0e6, "µs") :
        s < 1.0     ? (s * 1.0e3, "ms") : (s, "s")
    str = isapprox(n, round(n)) ? string(Int(round(n))) : @sprintf("%.1f", n)
    return "$str $u"
end
const STEP125_µs = sort([m * 10.0^e for e in -3:6 for m in (1, 2, 5)])
timeticks(vals) = (vals, fmt_time_µs.(vals))

# --- Figure 1: speedup (linear y from 0), full Hessian + HVP ---------------
let
    fig = Figure(size = (1020, 520), figure_padding = (16, 30, 14, 12))
    axh = Axis(
        fig[1, 1];
        xscale = log10, xlabel = "input length n", ylabel = "speedup  (× faster)",
        title = "Full Hessian", xticks = logticks([2, 4, 8, 16, 32, 64, 128, 256]),
    )
    axv = Axis(
        fig[1, 2];
        xscale = log10, xlabel = "input length n",
        title = "Hessian–vector product", xticks = logticks([8, 16, 32, 64, 128, 256]),
    )
    for F in FUNCS
        d = hess_data[F.name]
        scatterlines!(
            axh, Float64.(d.ns), d.t_fd ./ d.t_hh;
            color = FCOLOR[F.name], marker = FMARK[F.name], markersize = 11,
            linewidth = 2.4, label = F.name,
        )
    end
    for F in HVP_FUNCS
        d = hvp_data[F.name]
        scatterlines!(
            axv, Float64.(d.ns), d.t_fd ./ d.t_hh;
            color = FCOLOR[F.name], marker = FMARK[F.name], markersize = 11,
            linewidth = 2.4, label = F.name,
        )
    end
    for ax in (axh, axv)
        hlines!(ax, [1.0]; color = (INK, 0.45), linestyle = :dash, linewidth = 1.5)
        ylims!(ax, 0, nothing)
    end
    axislegend(axh; position = :lt, labelsize = 13)
    axislegend(axv; position = :rt, labelsize = 13)
    Label(
        fig[0, :], "Speedup of HyperHessians over ForwardDiff  (higher is better; dashed = parity)";
        color = INK, font = :bold, fontsize = 16, halign = :left, padding = (4, 0, 2, 0),
    )
    save(joinpath(RESULTS_DIR, "speedup.png"), fig; px_per_unit = 2)
end

# --- Figure 3: chunk-size tuning (HyperHessians only) ----------------------
let
    fig = Figure(size = (840, 560), figure_padding = (16, 26, 14, 16))
    ax = Axis(
        fig[1, 1];
        xscale = log2, yscale = log10,
        xlabel = "chunk size", ylabel = "time per Hessian",
        title = "HyperHessians chunk-size tuning  (n = $CHUNK_N)",
        xticks = logticks(CHUNK_SIZES), yticks = timeticks(STEP125_µs),
    )
    for F in HVP_FUNCS
        c = chunk_data[F.name]
        scatterlines!(
            ax, Float64.(CHUNK_SIZES), c.ts .* 1e6;
            color = FCOLOR[F.name], marker = FMARK[F.name], markersize = 11,
            linewidth = 2.4, label = F.name,
        )
    end
    axislegend(ax; position = :ct, labelsize = 13)
    Label(
        fig[2, 1],
        "Small chunks mean many passes; large chunks mean unwieldy dual numbers. The default (8) sits near the optimum.";
        color = MUTED, fontsize = 12.5, halign = :left, justification = :left,
        word_wrap = true, padding = (4, 0, 0, 0),
    )
    save(joinpath(RESULTS_DIR, "chunk_sensitivity.png"), fig; px_per_unit = 2)
end

# --- Figure 4: threading scaling -------------------------------------------
if thread_data !== nothing
    fig = Figure(size = (840, 560), figure_padding = (16, 26, 14, 16))
    ax = Axis(
        fig[1, 1];
        xlabel = "number of tasks (threads)", ylabel = "speedup vs serial  (×)",
        title = "Threading scaling  ($(FUNCS[1].name)/$(FUNCS[2].name), n = $THREAD_N)",
        xticks = logticks(NTASKS), xscale = log2,
    )
    # Sample y = x densely so the ideal line renders correctly on the log2 x-axis
    # (a single straight segment would bow upward between the endpoints).
    xs_ideal = 2.0 .^ range(0, log2(float(NTASKS[end])), length = 64)
    lines!(
        ax, xs_ideal, xs_ideal;
        color = (INK, 0.4), linestyle = :dash, linewidth = 1.5, label = "ideal (linear)",
    )
    for F in THREAD_FUNCS
        t = thread_data[F.name]
        keep = t.ntasks .<= NTASKS[end]
        scatterlines!(
            ax, Float64.(t.ntasks[keep]), (t.t_serial ./ t.ts)[keep];
            color = FCOLOR[F.name], marker = FMARK[F.name], markersize = 11,
            linewidth = 2.4, label = F.name,
        )
    end
    axislegend(ax; position = :lt, labelsize = 13)
    Label(
        fig[2, 1],
        "The k(k+1)/2 independent chunk passes run in parallel with ThreadedHessianConfig, timed against " *
        "the serial HessianConfig. Scaling is near-linear across the machine's 8 performance cores. " *
        "(Beyond them, the 16 efficiency cores add throughput only sub-linearly, so the sweep stops at 8.)";
        color = MUTED, fontsize = 12.5, halign = :left, justification = :left,
        word_wrap = true, padding = (4, 0, 0, 0),
    )
    save(joinpath(RESULTS_DIR, "threading_scaling.png"), fig; px_per_unit = 2)
end

# ===========================================================================
# Markdown summary
# ===========================================================================
prettytime(t) = t < 1e-6 ? @sprintf("%.1f ns", t * 1e9) :
    t < 1e-3 ? @sprintf("%.2f µs", t * 1e6) :
    t < 1.0  ? @sprintf("%.2f ms", t * 1e3) : @sprintf("%.2f s", t)

open(joinpath(RESULTS_DIR, "summary.md"), "w") do io
    println(io, "# HyperHessians vs ForwardDiff\n")
    println(io, "Julia $(VERSION), $(Sys.CPU_NAME), $(Threads.nthreads()) thread(s).\n")

    println(io, "## Full Hessian\n")
    println(io, "| function | n | chunk FD | chunk HH | ForwardDiff | HyperHessians | speedup |")
    println(io, "| --- | --- | --- | --- | --- | --- | --- |")
    for F in FUNCS
        d = hess_data[F.name]
        for (k, n) in enumerate(d.ns)
            println(io, "| `$(F.name)` | $n | $(fd_chunk(n)) | $(hh_chunk(rand(n))) | ",
                    "$(prettytime(d.t_fd[k])) | $(prettytime(d.t_hh[k])) | ",
                    @sprintf("%.2fx", d.t_fd[k] / d.t_hh[k]), " |")
        end
    end

    println(io, "\n## Hessian-vector product\n")
    println(io, "| function | n | ForwardDiff (DI) | HyperHessians | speedup |")
    println(io, "| --- | --- | --- | --- | --- |")
    for F in HVP_FUNCS
        d = hvp_data[F.name]
        for (k, n) in enumerate(d.ns)
            println(io, "| `$(F.name)` | $n | $(prettytime(d.t_fd[k])) | ",
                    "$(prettytime(d.t_hh[k])) | ", @sprintf("%.2fx", d.t_fd[k] / d.t_hh[k]), " |")
        end
    end

    println(io, "\n## HyperHessians chunk-size tuning (n=$CHUNK_N)\n")
    println(io, "| function | " * join(["chunk $c" for c in CHUNK_SIZES], " | ") * " |")
    println(io, "| --- | " * join(fill("---", length(CHUNK_SIZES)), " | ") * " |")
    for F in HVP_FUNCS
        c = chunk_data[F.name]
        println(io, "| `$(F.name)` | " * join(prettytime.(c.ts), " | ") * " |")
    end

    if thread_data !== nothing
        println(io, "\n## Threading scaling (n=$THREAD_N, nthreads=$(Threads.nthreads()))\n")
        println(io, "| function | serial | " * join(["$nt tasks" for nt in NTASKS], " | ") * " |")
        println(io, "| --- | --- | " * join(fill("---", length(NTASKS)), " | ") * " |")
        for F in THREAD_FUNCS
            t = thread_data[F.name]
            speeds = [@sprintf("%.1fx", t.t_serial / ti) for ti in t.ts]
            println(io, "| `$(F.name)` | $(prettytime(t.t_serial)) | " * join(speeds, " | ") * " |")
        end
    end
end

println("\n", read(joinpath(RESULTS_DIR, "summary.md"), String))
@info "Wrote plots and summary to $RESULTS_DIR"

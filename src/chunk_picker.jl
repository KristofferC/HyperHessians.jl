using BenchmarkTools
using Printf: @sprintf

const DEFAULT_CHUNK_SIZES = (1, 2, 4, 8, 16)

"""
    pick_chunksize(f, x; chunk_sizes=$DEFAULT_CHUNK_SIZES, samples=100, seconds=1)

Benchmark `hessian(f, x)` with various chunk sizes and display a comparison table.

Returns the chunk size with the best median time.

# Arguments
- `f`: Function to compute the Hessian of (must return a scalar)
- `x`: Example input vector

# Keyword Arguments
- `chunk_sizes`: Tuple of chunk sizes to test (default: $(DEFAULT_CHUNK_SIZES))
- `samples`: Number of samples for BenchmarkTools (default: 100)
- `seconds`: Max seconds per benchmark (default: 1)

# Example
```julia
using HyperHessians
f(x) = sum(x.^2) + prod(x)
x = rand(10)
best = HyperHessians.pick_chunksize(f, x)
```
"""
function pick_chunksize(
        f::F, x::AbstractVector{T};
        chunk_sizes = DEFAULT_CHUNK_SIZES,
    ) where {F, T}
    n = length(x)
    valid_sizes = filter(c -> c <= n, chunk_sizes)

    isempty(valid_sizes) && error("No valid chunk sizes for input of length $n")

    results = Vector{NamedTuple{(:chunk, :trial), Tuple{Int, BenchmarkTools.Trial}}}()

    println("Benchmarking hessian with $(length(valid_sizes)) chunk sizes...")
    println("Input length: $n, Element type: $T")
    println()

    for chunk in valid_sizes
        cfg = HessianConfig(x, Chunk{chunk}())
        H = similar(x, n, n)
        trial = @benchmark hessian!($H, $f, $x, $cfg)
        push!(results, (chunk = chunk, trial = trial))
    end

    # Sort by median time for display
    sort!(results, by = r -> median(r.trial).time)

    # Print table
    println("┌────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────┐")
    println("│ Chunk  │  Median     │  Mean       │  Min        │  Max        │ Allocs  │")
    println("├────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────┤")

    best_time = median(results[1].trial).time

    for (i, r) in enumerate(results)
        med = median(r.trial)
        mn = mean(r.trial)
        mi = minimum(r.trial)
        ma = maximum(r.trial)

        chunk_str = lpad(r.chunk, 4)
        med_str = lpad(prettytime(med.time), 9)
        mean_str = lpad(prettytime(mn.time), 9)
        min_str = lpad(prettytime(mi.time), 9)
        max_str = lpad(prettytime(ma.time), 9)
        alloc_str = lpad(med.allocs, 5)

        marker = i == 1 ? " ◀" : "  "
        println("│ $chunk_str   │ $med_str$marker │ $mean_str   │ $min_str   │ $max_str   │ $alloc_str   │")
    end

    println("└────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────┘")

    best_chunk = results[1].chunk
    println()
    println("Best chunk size: $best_chunk")

    return best_chunk
end

function prettytime(t)
    if t < 1.0e3
        return @sprintf("%.2f ns", t)
    elseif t < 1.0e6
        return @sprintf("%.2f μs", t / 1.0e3)
    elseif t < 1.0e9
        return @sprintf("%.2f ms", t / 1.0e6)
    else
        return @sprintf("%.2f s", t / 1.0e9)
    end
end

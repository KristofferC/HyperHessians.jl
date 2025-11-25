# HyperHessians.jl

HyperHessians.jl is a package to compute hessians using forward mode automatic differentiation.
It works similar to `ForwardDiff.hessian` but should have better run-time and compile-time performance in all cases.

There are some limitations compared to ForwardDiff.jl:
- Only support for basic numeric types (`Float64`, `Float32`, etc.).
- Not as many primitives implemented.

## Usage

### Basic
To compute the hessian of a function `f` w.r.t the vector `x` the quick and dirty way is to call
`HyperHessians.hessian(f, x)`:

```julia
julia> x = [1.0,2.0,3.0,4.0];

julia> HyperHessians.hessian(x->sum(cumprod(x)), x)
4×4 Matrix{Float64}:
  0.0  16.0  10.0  6.0
 16.0   0.0   5.0  3.0
 10.0   5.0   0.0  2.0
  6.0   3.0   2.0  0.0
```

This also works for scalar functions:

```julia
julia> f(x) = exp(x) / sqrt(sin(x)^3 + cos(x)^3);

julia> HyperHessians.hessian(f, 2.0)
82.55705026089272
```

When the input is a vector, the basic usage will, however, not give the best performance.

### Advanced

For best performance we want to do the following things:

1. Cache the input array of custom numbers that HyperHessians uses so they can be reused upon multiple calls to `hessian`.
2. Decide on a good "chunk size" which is roughly the size of the section of the hessian computed for every call to the function.
3. Pre-allocate the output hessian matrix.

Step 1 and 2 are done by creating a `HessianConfig` object:

```julia
x = rand(32); # input array
chunk_size = HyperHessians.Chunk{8}() # chosen chunk size
cfg = HyperHessians.HessianConfig(f, x, chunk_size) # creating the config object (tagged to f)
```

The larger the chunk size the larger part of the Hessian is computed on every call to `f` (if the chunk size is equal to
the input vector, the whole hessian is computed in one call to `f`).
However, with a larger chunk size the special numbers HyperHessians uses become larger and if they become too large this can lead to inefficient execution.

A choice of a chunk size is, therefore, a trade-off and the optimal one is likely to be dependent on the particular function getting differentiated.
A decent overall choice seems to be a chunk size of 8.
It is also in general a good idea to pick a chunk size as a multiple of 4 to use SIMD effectively.

The `chunk_size` argument can be left out and HyperHessians will try to determine a reasonable choice.

If the chunk size `c` is smaller than the input vector with length `n`, the function will be called `k = ceil(Int, n / c); k(k+1)÷2` times, each time computing a part of the hessian:

```julia
julia> mysum(x) = (@info "got called"; sum(x));

julia> x = rand(8); n = length(x); c = 4;

julia> cfg = HyperHessians.HessianConfig(mysum, x, HyperHessians.Chunk{c}());

julia> HyperHessians.hessian(mysum, x, cfg)
[ Info: got called
[ Info: got called
[ Info: got called

julia> k=ceil(Int, n / c); k*(k+1)÷2
3
```

Finally, it is also a good idea to allocate the output hessian and use the in-place `hessian!` function instead.
Putting it all together, the result would look something like this:

```julia
julia> x = rand(8);

julia> H = similar(x, 8, 8);

julia> cfg = HyperHessians.HessianConfig(f, x, HyperHessians.Chunk{8}());

julia> HyperHessians.hessian!(H, f, x, cfg)
```

## Performance

To get an estimate of the performance of HyperHessians we here benchmark it
against the ForwardDiff.jl package for two common benchmark functions [`rosenbrock`](https://github.com/JuliaDiff/DiffTests.jl/blob/32b82197f23dbb3c5b2035be1d11158a15d89855/src/DiffTests.jl#L76-L84)
and [`ackley`](https://github.com/JuliaDiff/DiffTests.jl/blob/32b82197f23dbb3c5b2035be1d11158a15d89855/src/DiffTests.jl#L101-L112).
The results can be reproduced with `benchmark/fdiff.jl`.

| Function      | input length | Time ForwardDiff | Time HyperHessians | Speedup |
| ------------- | ------------ | ---------------- | ----------------- | --------|
| `ackley` | 1 | 54.053 ns | 36.725 ns | 1.5
| `ackley` | 8 | 751.448 ns | 461.276 ns | 1.6
| `ackley` | 128 | 2.085 ms | 957.194 μs | 2.2
| `rosenbrock_1` | 1 | 24.666 ns | 7.436 ns | 3.3
| `rosenbrock_1` | 8 | 967.438 ns | 436.330 ns | 2.2
| `rosenbrock_1` | 128 | 3.175 ms | 1.031 ms | 3.1

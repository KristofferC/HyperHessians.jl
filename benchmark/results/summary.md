# HyperHessians vs ForwardDiff

Julia 1.12.2, arrowlake-s, 24 thread(s).

## Full Hessian

| function | n | chunk FD | chunk HH | ForwardDiff | HyperHessians | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| `ackley` | 2 | 2 | 2 | 60.7 ns | 28.9 ns | 2.10x |
| `ackley` | 4 | 4 | 4 | 229.4 ns | 59.8 ns | 3.84x |
| `ackley` | 8 | 8 | 8 | 889.1 ns | 228.4 ns | 3.89x |
| `ackley` | 16 | 8 | 8 | 6.12 µs | 1.60 µs | 3.83x |
| `ackley` | 32 | 11 | 8 | 41.38 µs | 8.13 µs | 5.09x |
| `ackley` | 64 | 11 | 8 | 293.23 µs | 49.60 µs | 5.91x |
| `ackley` | 128 | 12 | 8 | 2.09 ms | 344.03 µs | 6.08x |
| `ackley` | 256 | 12 | 8 | 16.19 ms | 2.57 ms | 6.30x |
| `rosenbrock_1` | 2 | 2 | 2 | 47.7 ns | 19.2 ns | 2.48x |
| `rosenbrock_1` | 4 | 4 | 4 | 157.2 ns | 31.7 ns | 4.96x |
| `rosenbrock_1` | 8 | 8 | 8 | 1.07 µs | 206.4 ns | 5.17x |
| `rosenbrock_1` | 16 | 8 | 8 | 8.32 µs | 1.61 µs | 5.18x |
| `rosenbrock_1` | 32 | 11 | 8 | 54.43 µs | 8.81 µs | 6.18x |
| `rosenbrock_1` | 64 | 11 | 8 | 407.35 µs | 56.73 µs | 7.18x |
| `rosenbrock_1` | 128 | 12 | 8 | 3.29 ms | 405.21 µs | 8.11x |
| `rosenbrock_1` | 256 | 12 | 8 | 25.93 ms | 3.07 ms | 8.45x |
| `self_weighted_logit` | 2 | 2 | 2 | 78.3 ns | 43.5 ns | 1.80x |
| `self_weighted_logit` | 4 | 4 | 4 | 204.0 ns | 101.9 ns | 2.00x |
| `self_weighted_logit` | 8 | 8 | 8 | 874.8 ns | 292.8 ns | 2.99x |
| `self_weighted_logit` | 16 | 8 | 8 | 6.20 µs | 2.09 µs | 2.96x |
| `self_weighted_logit` | 32 | 11 | 8 | 32.51 µs | 11.24 µs | 2.89x |
| `self_weighted_logit` | 64 | 11 | 8 | 224.87 µs | 73.04 µs | 3.08x |
| `self_weighted_logit` | 128 | 12 | 8 | 2.16 ms | 546.99 µs | 3.95x |
| `self_weighted_logit` | 256 | 12 | 8 | 16.73 ms | 4.19 ms | 3.99x |
| `vec2num_3` | 2 | 2 | 2 | 143.4 ns | 84.3 ns | 1.70x |
| `vec2num_3` | 4 | 4 | 4 | 1.04 µs | 366.1 ns | 2.83x |
| `vec2num_3` | 8 | 8 | 8 | 12.33 µs | 3.22 µs | 3.83x |
| `vec2num_3` | 16 | 8 | 8 | 177.22 µs | 34.81 µs | 5.09x |
| `vec2num_3` | 32 | 11 | 8 | 1.99 ms | 456.88 µs | 4.35x |
| `vec2num_3` | 64 | 11 | 8 | 44.44 ms | 9.56 ms | 4.65x |
| `vec2num_3` | 128 | 12 | 8 | 871.73 ms | 184.18 ms | 4.73x |

## Hessian-vector product

| function | n | ForwardDiff (DI) | HyperHessians | speedup |
| --- | --- | --- | --- | --- |
| `ackley` | 8 | 294.4 ns | 82.6 ns | 3.56x |
| `ackley` | 16 | 1.16 µs | 286.0 ns | 4.06x |
| `ackley` | 32 | 3.03 µs | 976.1 ns | 3.10x |
| `ackley` | 64 | 11.09 µs | 3.65 µs | 3.04x |
| `ackley` | 128 | 50.78 µs | 14.06 µs | 3.61x |
| `ackley` | 256 | 199.70 µs | 55.37 µs | 3.61x |
| `rosenbrock_1` | 8 | 169.6 ns | 88.9 ns | 1.91x |
| `rosenbrock_1` | 16 | 782.9 ns | 369.2 ns | 2.12x |
| `rosenbrock_1` | 32 | 2.69 µs | 1.42 µs | 1.90x |
| `rosenbrock_1` | 64 | 9.96 µs | 5.57 µs | 1.79x |
| `rosenbrock_1` | 128 | 39.87 µs | 22.73 µs | 1.75x |
| `rosenbrock_1` | 256 | 157.82 µs | 88.85 µs | 1.78x |

## HyperHessians chunk-size tuning (n=128)

| function | chunk 1 | chunk 2 | chunk 4 | chunk 8 | chunk 16 | chunk 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `ackley` | 4.69 ms | 1.36 ms | 547.99 µs | 341.55 µs | 1.29 ms | 2.12 ms |
| `rosenbrock_1` | 2.68 ms | 913.64 µs | 523.18 µs | 403.49 µs | 2.21 ms | 4.01 ms |

## Threading scaling (n=512, nthreads=24)

| function | serial | 1 tasks | 2 tasks | 4 tasks | 6 tasks | 8 tasks |
| --- | --- | --- | --- | --- | --- | --- |
| `ackley` | 19.87 ms | 1.0x | 1.9x | 3.7x | 5.3x | 6.9x |
| `rosenbrock_1` | 23.92 ms | 1.0x | 1.9x | 3.8x | 5.5x | 7.1x |

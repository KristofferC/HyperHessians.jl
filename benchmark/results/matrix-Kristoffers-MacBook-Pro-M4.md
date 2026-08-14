# HyperHessians vs ForwardDiff — best-configuration speedup matrix

Apple M4 Pro (aarch64) · julia 1.12.6 · ForwardDiff 1.4.5 via DifferentiationInterface 0.7.21 · HyperHessians 0.3.0 @ fc3cd80+dirty · single thread · 2026-08-08 15:39

Each cell: both packages at their fastest ChunkPicker-picked configuration, BenchmarkTools minimum time, results verified against each other.

### Hessian speedup (ForwardDiff time / HyperHessians time)

| function | n=4 | n=16 | n=64 | n=256 | geomean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ackley | 2.52× <span class="cfg">Js/c4</span> | 2.30× <span class="cfg">c4s/c16</span> | 3.26× <span class="cfg">c8s/c8</span> | 3.63× <span class="cfg">c8s/c16</span> | ==2.88×== |
| rosenbrock | 2.86× <span class="cfg">c4s/c4</span> | 3.43× <span class="cfg">c4s/c8</span> | 3.94× <span class="cfg">c4s/c8</span> | 4.45× <span class="cfg">c6s/c6</span> | ==3.62×== |
| logsumexp | 2.64× <span class="cfg">Js/c4</span> | 2.37× <span class="cfg">J/c8</span> | 2.34× <span class="cfg">c6/c8</span> | 2.79× <span class="cfg">c6/c8</span> | ==2.53×== |
| self_weighted_logit | 1.74× <span class="cfg">J/c4</span> | 2.16× <span class="cfg">J/c16</span> | 2.36× <span class="cfg">c8s/c4</span> | 2.38× <span class="cfg">c8s/c16</span> | ==2.14×== |
| **geomean** | 2.40× | 2.52× | 2.90× | 3.22× | ==**2.74×**== |

Picked configurations (HyperHessians / ForwardDiff chunk; J = Jet, cN = chunk N):

| function | n=4 | n=16 | n=64 | n=256 |
| --- | --- | --- | --- | --- |
| ackley | Js / c4 | c4s / c16 | c8s / c8 | c8s / c16 |
| rosenbrock | c4s / c4 | c4s / c8 | c4s / c8 | c6s / c6 |
| logsumexp | Js / c4 | J / c8 | c6 / c8 | c6 / c8 |
| self_weighted_logit | J / c4 | J / c16 | c8s / c4 | c8s / c16 |

### Hessian-vector product speedup (ForwardDiff time / HyperHessians time)

| function | n=4 | n=16 | n=64 | n=256 | geomean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ackley | 1.52× <span class="cfg">c4s/c4</span> | 1.97× <span class="cfg">c16s/c16</span> | 1.78× <span class="cfg">c16s/c8</span> | 1.94× <span class="cfg">c16s/c8</span> | ==1.80×== |
| rosenbrock | 2.31× <span class="cfg">c4s/c4</span> | 1.73× <span class="cfg">c16s/c2</span> | 1.86× <span class="cfg">c8s/c2</span> | 2.10× <span class="cfg">c16s/c2</span> | ==1.99×== |
| logsumexp | 1.59× <span class="cfg">c4s/c4</span> | 1.56× <span class="cfg">c16s/c16</span> | 1.39× <span class="cfg">c16s/c8</span> | 1.49× <span class="cfg">c16s/c8</span> | ==1.50×== |
| self_weighted_logit | 1.44× <span class="cfg">c4/c4</span> | 1.42× <span class="cfg">c16/c4</span> | 1.21× <span class="cfg">c8/c4</span> | 1.29× <span class="cfg">c12/c4</span> | ==1.34×== |
| **geomean** | 1.69× | 1.66× | 1.54× | 1.67× | ==**1.64×**== |

Picked configurations (HyperHessians / ForwardDiff chunk; J = Jet, cN = chunk N):

| function | n=4 | n=16 | n=64 | n=256 |
| --- | --- | --- | --- | --- |
| ackley | c4s / c4 | c16s / c16 | c16s / c8 | c16s / c8 |
| rosenbrock | c4s / c4 | c16s / c2 | c8s / c2 | c16s / c2 |
| logsumexp | c4s / c4 | c16s / c16 | c16s / c8 | c16s / c8 |
| self_weighted_logit | c4 / c4 | c16 / c4 | c8 / c4 | c12 / c4 |


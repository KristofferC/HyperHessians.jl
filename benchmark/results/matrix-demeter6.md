# HyperHessians vs ForwardDiff — best-configuration speedup matrix

AMD EPYC 9354 32-Core Processor (x86_64) · julia 1.12.6 · ForwardDiff 1.4.5 via DifferentiationInterface 0.7.21 · HyperHessians 0.3.0 @ fc3cd80+dirty · single thread · 2026-08-08 13:41

Each cell: both packages at their fastest ChunkPicker-picked configuration, BenchmarkTools minimum time, results verified against each other.

### Hessian speedup (ForwardDiff time / HyperHessians time)

| function | n=4 | n=16 | n=64 | n=256 | geomean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ackley | 2.47× <span class="cfg">Js/c4</span> | 2.31× <span class="cfg">J/c16</span> | 2.86× <span class="cfg">c16s/c11</span> | 3.61× <span class="cfg">c16s/c16</span> | ==2.77×== |
| rosenbrock | 4.27× <span class="cfg">c4s/c4</span> | 2.90× <span class="cfg">c4s/c16</span> | 4.11× <span class="cfg">c8s/c16</span> | 4.85× <span class="cfg">c8s/c16</span> | ==3.96×== |
| logsumexp | 1.98× <span class="cfg">c4/c4</span> | 1.82× <span class="cfg">J/c16</span> | 1.90× <span class="cfg">c4/c16</span> | 2.02× <span class="cfg">c16s/c16</span> | ==1.93×== |
| self_weighted_logit | 1.59× <span class="cfg">c4s/c4</span> | 1.94× <span class="cfg">J/c16</span> | 1.77× <span class="cfg">c16s/c13</span> | 2.58× <span class="cfg">c16s/c16</span> | ==1.94×== |
| **geomean** | 2.40× | 2.20× | 2.51× | 3.09× | ==**2.53×**== |

Picked configurations (HyperHessians / ForwardDiff chunk; J = Jet, cN = chunk N):

| function | n=4 | n=16 | n=64 | n=256 |
| --- | --- | --- | --- | --- |
| ackley | Js / c4 | J / c16 | c16s / c11 | c16s / c16 |
| rosenbrock | c4s / c4 | c4s / c16 | c8s / c16 | c8s / c16 |
| logsumexp | c4 / c4 | J / c16 | c4 / c16 | c16s / c16 |
| self_weighted_logit | c4s / c4 | J / c16 | c16s / c13 | c16s / c16 |

### Hessian-vector product speedup (ForwardDiff time / HyperHessians time)

| function | n=4 | n=16 | n=64 | n=256 | geomean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ackley | 1.93× <span class="cfg">c4s/c4</span> | 2.35× <span class="cfg">c16s/c16</span> | 2.00× <span class="cfg">c16s/c13</span> | 2.07× <span class="cfg">c16s/c16</span> | ==2.08×== |
| rosenbrock | 2.14× <span class="cfg">c4s/c4</span> | 1.93× <span class="cfg">c4s/c16</span> | 1.93× <span class="cfg">c4s/c13</span> | 1.84× <span class="cfg">c4s/c16</span> | ==1.96×== |
| logsumexp | 1.63× <span class="cfg">c4s/c4</span> | 2.62× <span class="cfg">c16s/c16</span> | 2.61× <span class="cfg">c16s/c22</span> | 2.87× <span class="cfg">c16s/c16</span> | ==2.38×== |
| self_weighted_logit | 1.19× <span class="cfg">c4/c4</span> | 1.36× <span class="cfg">c16/c16</span> | 1.10× <span class="cfg">c16s/c11</span> | 1.18× <span class="cfg">c16/c16</span> | ==1.20×== |
| **geomean** | 1.68× | 2.00× | 1.83× | 1.90× | ==**1.85×**== |

Picked configurations (HyperHessians / ForwardDiff chunk; J = Jet, cN = chunk N):

| function | n=4 | n=16 | n=64 | n=256 |
| --- | --- | --- | --- | --- |
| ackley | c4s / c4 | c16s / c16 | c16s / c13 | c16s / c16 |
| rosenbrock | c4s / c4 | c4s / c16 | c4s / c13 | c4s / c16 |
| logsumexp | c4s / c4 | c16s / c16 | c16s / c22 | c16s / c16 |
| self_weighted_logit | c4 / c4 | c16 / c16 | c16s / c11 | c16 / c16 |


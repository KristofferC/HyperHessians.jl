@layout title
@eyebrow HyperHessians.jl
# Forward mode AD specialized for second order derivatives
@chips Kristoffer Carlsson | [:github: @KristofferC](https://github.com/KristofferC) | [:mail: kristoffer.carlsson@juliahub.com](mailto:kristoffer.carlsson@juliahub.com) | JuliaCon 2026

---

@layout: center

!big Theory

---

@eyebrow We do not talk about reverse mode...
# Forward vs. Reverse Mode

- This talk is about forward mode AD
- "Reverse mode" better when number of input ≫  number of outputs (e.g. gradients) and
- Applications like machine learning often have a huge number of inputs -> reverse mode better
- For physics, quite common with equal input and outputs (n unknowns, n equations)
- Fancier stuff like "forward over reverse" etc.

---

# Finite differences



::: cols
::: panel Taylor, with a step $h$
$$ f(x_0 + h) = f(x_0) + h\,f'(x_0) + \tfrac{h^2}{2}\,f''(x_0) + \cdots $$
$$ f'(x_0) = \frac{f(x_0+h) - f(x_0)}{h} + \mathcal{O}(h) $$
:::

- Big $h$: the dropped Taylor terms dominate-
- Small $h$: the subtraction cancels in floating point.
- Have to call `f` twice
:: col
```julia> | `f(x) = x sin(x²)` at `x = 2`, error vs exact f′
julia> f(x) = x * sin(x * x);

julia> fd(f, x, h) = (f(x + h) - f(x)) / h;

julia> exact = sin(4.0) + 8cos(4.0);

julia> fd(f, 2.0, 1e-2)  - exact
0.08441453688334022       # truncation

julia> fd(f, 2.0, 1e-14) - exact
-0.14247963371404015      # cancellation
```
:::

<div id="fderrchart" style="margin-top: 10px;"></div>

---

@eyebrow Dual numbers
# Definition and operators

- Complex numbers: $z = a + bi, \quad i^2=-1, \quad \operatorname{Im} [z] = b$
- Dual numbers: $d = x + h\varepsilon, \quad \varepsilon^2 = 0,  \quad \varepsilon[d] = h$

$$ f(d) = f(x + h\,\varepsilon) = f(x) + h f'(x)\,\varepsilon + \underbrace{\tfrac{h^2}{2} f''(x)\,\varepsilon^2 + \cdots}_{=\ 0\ \textbf{exactly}} $$
$$ f'(x) = \varepsilon[f(x + h\,\varepsilon)] / h $$

- Seed $h = 1$, read the derivative off the $\varepsilon$ slot: $f'(x) = \varepsilon[ f(x + \varepsilon)$ ].
- Primitives need one rule each: $\ \sin(d) = \sin(x) + h\cos(x)\,\varepsilon$.
- $d_1 d_2 = x_1 x_2 + (x_1 h_2 + x_2 h_1)\,\varepsilon$ — the product rule falls out ($h_1 h_2\,\varepsilon^2 = 0$).

---

@eyebrow Dual numbers
# Implementation

```julia size="14" title="Forward mode AD in Julia"
struct Dual
    x::Float64   # value
    ε::Float64   # epsilon coefficient
end
Base.:+(a::Dual, b::Dual) = Dual(a.x + b.x, a.ε + b.ε)
Base.:*(a::Dual, b::Dual) = Dual(a.x * b.x, a.x * b.ε + b.x * a.ε)  # product rule
Base.sin(d::Dual)         = Dual(sin(d.x), cos(d.x) * d.ε)          # chain rule
derivative(f, x) = f(Dual(x, 1.0)).ε # seed ε = 1, run f, read ε
```

```julia> size="14"
julia> f(x) = x * sin(x * x);

julia> derivative(f, 2.0)
-5.985951462216824

julia> sin(4) + 8cos(4)
-5.985951462216824
```

- Data layout: @fig{cells v f1} where @fig{ cells v = value | f1 = $\varepsilon$ coefficient}


---

@eyebrow Second order
# Differentiation^2
@kicker `second_derivative(f, x) = derivative(y -> derivative(f, y), x)` — `derivative` is just Julia code

- Nested `Dual` numbers

```diff2 size="16" title="A bit of generalization"
-struct Dual
-    x::Float64
-    ε::Float64
-end
+struct Dual{T}
+    x::T
+    ε::T
+end
-derivative(f, x) = f(Dual(x, 1.0)).ε
+derivative(f, x) = f(Dual(x, one(x))).ε
+Base.one(d::Dual) = Dual(one(d.x), zero(d.x))
+Base.cos(d::Dual) = Dual(cos(d.x), -sin(d.x) * d.ε)
```
```julia> size="14"
julia> second_derivative(f, x) = derivative(y -> derivative(f, y), x);

julia> second_derivative(f, 2.0)
16.37395639949036

julia> 12cos(4) - 32sin(4)
16.37395639949036
```

- Data layout: @fig{cells v f1 f1 f3} @fig{cells v f1 = $(f,\, f')$ | f1 f3 = $(f',\, f'')$}, $f'$ computed **twice**


---

@eyebrow Many inputs, many outputs
# Jacobians: $f\colon \mathbb{R}^n \to \mathbb{R}^m$

$$ f(\mathbf{x} + \mathbf{h}\,\varepsilon) = f(\mathbf{x}) + J(\mathbf{x})\,\mathbf{h}\,\varepsilon, \qquad \mathbf{x},\, \mathbf{h} \in \mathbb{R}^n, \quad f(\mathbf{x}) \in \mathbb{R}^m, \quad J(\mathbf{x}) \in \mathbb{R}^{m \times n},\ \ J_{kj} = \frac{\partial f_k}{\partial x_j} $$

One pass is a **Jacobian-vector product**. Seed the identity, column by column: $\mathbf{h} = \mathbf{e}_i$ picks out $J\mathbf{e}_i$ — column $i$.

::: cols
```julia size="15" title="n passes, one per column"
function jacobian(f, x)
    cols = []
    for i in eachindex(x)
        seed = zeros(length(x))
        seed[i] = 1.0 # direction eᵢ
        y = f(Dual.(x, seed))
        push!(cols, [d.ε for d in y])
    end
    return stack(cols) # m × n
end
```
:: col
```julia size="15" title="finite differences: same loop, n + 1 evaluations"
function jacobian_fd(f, x; h = 1e-8)
    f0 = f(x)
    cols = []
    for i in eachindex(x)
        xh = copy(x)
        xh[i] += h
        push!(cols, (f(xh) - f0) / h)
    end
    return stack(cols) # m × n
end
```
:::

---

@eyebrow More efficient
# Chunk mode: N columns per pass

$$ f(\mathbf{x} + H\,\varepsilon) = f(\mathbf{x}) + J(\mathbf{x})\,H\,\varepsilon, \qquad H \in \mathbb{R}^{n \times N} \text{ — one direction per column}, \quad H = I \;\Rightarrow\; JH = J $$

```diff2 size="15" title="scalar partial → chunk of partials"
-struct Dual{T}
-    x::T
-    ε::T
-end
+struct Dual{T,N}
+    x::T
+    ε::NTuple{N,T}
+end
-Base.:*(a::Dual, b::Dual) =
-    Dual(a.x * b.x,
-         a.x * b.ε + b.x * a.ε)
+Base.:*(a::Dual, b::Dual) =
+    Dual(a.x * b.x,
+         a.x .* b.ε .+ b.x .* a.ε)
-Base.sin(d::Dual) = Dual(sin(d.x), cos(d.x) * d.ε)
+Base.sin(d::Dual) = Dual(sin(d.x), cos(d.x) .* d.ε)
```


- Seed the identity all at once ($H = I$, so $N = n$) → whole Jacobian in **one pass**.
- `sin` still calls `cos` **once** — expensive rule work is amortized across all $N$ directions.
- Cost per op grows with $N$, a fat `Dual{N}` spills registers → cap $N$ and sweep $n/N$ passes.
- Data layout: @fig{cells v f1*8} where @fig{cells v = value | f1*8 = $\varepsilon$-tuple, $N = 8$ partials}

---

@eyebrow More efficient
# Chunk mode: the code

::: cols
```julia size="14" title="H = I: whole Jacobian, one call to f"
function jacobian(f, x)
    n = length(x)
    ds = []
    for i in 1:n
        seed = zeros(n)
        seed[i] = 1.0      # row i of H = I
        push!(ds, Dual(x[i], Tuple(seed)))
    end
    y = f(ds)              # ONE call
    return stack([d.ε for d in y], dims = 1)   # m × n
end
```
@pills good:primal computed once | bad:large input -> fat duals

```julia> size="14"
julia> F(x) = [x[1] * sin(x[2]), x[1] * x[2]];

julia> jacobian(F, [1.0, 2.0])
2×2 Matrix{Float64}:
 0.909297  -0.416147
 2.0        1.0
```

:: col
```julia size="14" title="register-sized chunks: n/N passes"
function jacobian(f, x, N)
    n = length(x)
    blocks = []
    for c in 0:N:n-1           # one pass per chunk
        ds = []
        for i in 1:n
            seed = zeros(N)
            if c < i <= c + N  # chunk c of row i
                seed[i-c] = 1.0
            end
            push!(ds, Dual(x[i], Tuple(seed)))
        end
        y = f(ds)              # J[:, c+1:c+N]
        push!(blocks, stack([d.ε for d in y], dims = 1))
    end
    return hcat(blocks...)     # m × n
end
```
@pills good:lean duals | bad:primal recomputed n/N times
:::

---

@eyebrow Second order
# Hessian with dual numbers
@kicker Nest the chunked dual — the Hessian is the Jacobian of the gradient

::: cols
```julia size="14" title="gradient, then jacobian of the gradient"
gradient(f, x) = vec(jacobian(y -> [f(y)], x))

hessian(f, x) = jacobian(y -> gradient(f, y), x)
```
```julia> size="14"
julia> f(x) = x[1] * sin(x[2]);

julia> hessian(f, [1.0, 2.0])
2×2 Matrix{Float64}:
  0.0       -0.416147
 -0.416147  -0.909297
```
:: col
- The seeds nest exactly like the scalar case.
- One number carries **four groups**: $(f, \nabla_1)$, then $(\nabla_2, H \text{ row})$ per outer direction — $(1+N_1)(1+N_2)$ slots.
- $\nabla f$ is computed **twice** — the scalar redundancy again, now $n$ wide.
  ~ a symmetric *jet* number ($\varepsilon^3 = 0$) stores nothing twice — skipped today, see the backup slides
- Memory layout, $N_1 = N_2 = 2$: @fig{cells v f1*2 | f2 f3*2 | f2 f3*2} — flat, in field order: $(f, \nabla_1)$, then each $\varepsilon$ slot $(\nabla_2^{(j)}, H_{j,:})$
:::

---

@eyebrow HyperDual numbers
# Definition and operators
@kicker Don't nest — give one number two first-order directions and their cross term

- Dual numbers, one direction: $d = x + h\varepsilon, \quad \varepsilon^2 = 0$ — first order only.
- HyperDual numbers: $d = x + a\,\varepsilon_1 + b\,\varepsilon_2 + c\,\varepsilon_1\varepsilon_2, \quad \varepsilon_1^2 = \varepsilon_2^2 = 0, \quad \varepsilon_1\varepsilon_2 \neq 0$

$$ f(d) = f(x) + f'(x)\,\delta + \tfrac{f''(x)}{2}\,\delta^2 + \cdots, \qquad \delta = a\,\varepsilon_1 + b\,\varepsilon_2 + c\,\varepsilon_1\varepsilon_2 $$

$$ \delta^2 = 2ab\,\varepsilon_1\varepsilon_2, \quad \delta^3 = 0\ \textbf{exactly} \;\Rightarrow\; f(d) = f(x) + a f'(x)\,\varepsilon_1 + b f'(x)\,\varepsilon_2 + \bigl(c\,f'(x) + ab\,f''(x)\bigr)\,\varepsilon_1\varepsilon_2 $$

- That last slot **is** the second-order chain rule — one rule per primitive, $f'$ and $f''$ as plain numbers: $\ \sin(d) = \sin x + a\cos x\,\varepsilon_1 + b\cos x\,\varepsilon_2 + (c\cos x - ab\sin x)\,\varepsilon_1\varepsilon_2$.
- Seed $a = b = 1$, $c = 0$, read the $\varepsilon_1\varepsilon_2$ slot: $\ f''(x) = \varepsilon_1\varepsilon_2[\,f(x + \varepsilon_1 + \varepsilon_2)\,]$.
- $d_1 d_2 = x_1x_2 + (x_1a_2 + x_2a_1)\,\varepsilon_1 + (x_1b_2 + x_2b_1)\,\varepsilon_2 + (x_1c_2 + x_2c_1 + a_1b_2 + a_2b_1)\,\varepsilon_1\varepsilon_2$ — the product rule falls out again.

---

@eyebrow HyperDual numbers
# Implementation

```julia size="14" title="second-order forward mode in Julia"
struct HyperDual
    x::Float64     # value
    ε1::Float64    # f′
    ε2::Float64    # f′, independent copy
    ε12::Float64   # f″
end
Base.:*(a::HyperDual, b::HyperDual) =
    HyperDual(a.x * b.x,
              a.x * b.ε1 + b.x * a.ε1,
              a.x * b.ε2 + b.x * a.ε2,
              a.x * b.ε12 + b.x * a.ε12 + a.ε1 * b.ε2 + a.ε2 * b.ε1)  # product rule
function Base.sin(d::HyperDual)
    s, c = sincos(d.x)      # f′, f″ evaluated ONCE, as plain floats
    HyperDual(s, c * d.ε1, c * d.ε2, c * d.ε12 - s * d.ε1 * d.ε2)
end
second_derivative(f, x) = f(HyperDual(x, 1.0, 1.0, 0.0)).ε12  # a = b = 1, c = 0
```

```julia> size="14"
julia> f(x) = x * sin(x * x);

julia> second_derivative(f, 2.0)
16.37395639949036

julia> 12cos(4) - 32sin(4)
16.37395639949036
```

- Data layout: @fig{cells v e1 e2 e12} where @fig{cells v = value | e1 = $f'$ | e2 = $f'$ again | e12 = $f''$} — flat, no nesting, no rules-of-rules.

---

@eyebrow HyperDual numbers
# Chunk mode: an $N_1 \times N_2$ block of $H$ per pass
@kicker An off-diagonal Hessian block mixes directions from two different chunks — they must ride through $f$ independently

$$ d = x + \textstyle\sum_i a_i\,\varepsilon_{1,i} + \sum_j b_j\,\varepsilon_{2,j}, \qquad [\varepsilon_{1,i}\,\varepsilon_{2,j}]\ f(d) = \frac{\partial^2 f}{\partial x_i\,\partial x_j} = H[i,j] $$

```diff2 size="15" title="scalar partials → a chunk per ε, a block in ε₁ε₂"
-struct HyperDual
-    x::Float64
-    ε1::Float64
-    ε2::Float64
-    ε12::Float64
-end
+struct HyperDual{N1,N2}
+    x::Float64
+    ε1::NTuple{N1,Float64}
+    ε2::NTuple{N2,Float64}
+    ε12::NTuple{N1,NTuple{N2,Float64}}
+end
-    HyperDual(s, c * d.ε1, c * d.ε2,
-              c * d.ε12 - s * d.ε1 * d.ε2)
+    HyperDual(s, c .* d.ε1, c .* d.ε2,
+              ntuple(i -> c .* d.ε12[i] .- s .* (d.ε1[i] .* d.ε2), N1))
```

+ $\varepsilon_1$ carries chunk $I$, $\varepsilon_2$ carries chunk $J$, the $\varepsilon_1\varepsilon_2$ slots are an $N_1 \times N_2$ **block** of $H$.
+ `HyperDual{N1,N2}`: $1 + N_1 + N_2 + N_1 N_2$ slots — `{8,8}` is 81 floats, register-sized.
+ Symmetry: only block pairs $I \le J$ — $\tfrac{k(k+1)}{2}$ evaluations for $k$ chunks, not $k^2$.
+ Asymmetric chunks come free: seed $\varepsilon_2$ with a tangent $v$ → `HyperDual{N,1}` computes $Hv$ without forming $H$.

---

@eyebrow HyperDual numbers
# Chunk mode: the code

::: cols
```julia size="14" title="N₁ = N₂ = n: whole Hessian, one call to f"
seed(i, R) = Tuple(float(i == j) for j in R)

function hessian(f, x)              # N₁ = N₂ = n
    n = length(x)
    ds = [HyperDual(x[i], seed(i, 1:n), seed(i, 1:n))
          for i in 1:n]             # ε₁₂ seeded to zero
    v = f(ds)                       # ONE call
    return [v.ε12[i][j] for i in 1:n, j in 1:n]
end
```
@pills good:primal & rules once | bad:n² slots per number

```julia> size="14"
julia> f(x) = x[1] * sin(x[2]);

julia> hessian(f, [1.0, 2.0])
2×2 Matrix{Float64}:
  0.0       -0.416147
 -0.416147  -0.909297
```

:: col
```julia size="14" title="register-sized chunks: one block pair per pass"
function hessian(f, x, N)
    n = length(x); H = zeros(n, n)
    for (I, J) in block_pairs(n, N)  # ranges, I ≤ J only
        ds = [HyperDual(x[i], seed(i, I), seed(i, J))
              for i in 1:n]
        v = f(ds)                    # one pass per pair
        H[I, J] .= [v.ε12[i][j] for i in 1:N, j in 1:N]
    end
    return symmetrize!(H)            # mirror I < J blocks
end
```
@pills good:lean numbers | good:symmetry: k(k+1)/2 passes | bad:primal recomputed per pair
:::

---

@eyebrow Second order, done right
# Same four groups of data — stored together, or interleaved
@kicker Both hold exactly the same 81 floats — the arithmetic works on the groups

@fig lane-legend

::: panel `HyperDual{8,8}` — each group contiguous: an 8-wide group is one AVX-512 register
@fig lane-hh
:::

::: panel `Jet{8}` — gradient + upper triangle only, symmetry stored once: 45 floats
@fig lane-jet
:::

::: panel nested `Dual` 8×8 — one gradient float interleaved into every Hessian row
@fig lane-fd
:::

?> The Hessian update multiplies gradient chunk 2 against every Hessian row. Top: that chunk is one register load. Bottom: the same 8 floats sit 72 bytes apart — every operation gathers and scatters, and the lone value float rides along in every lane.

---

@layout: center

!big Implementations

---

# ForwardDiff.jl

- The "GOAT" (Greatest of ALl Time)
-


---

@layout: center

!big Benchmarking



---

@layout: center

!big Questions?

@chips github.com/KristofferC/HyperHessians.jl

---

@layout: center

!big Backup

@chips jets: second order without nesting | $\varepsilon^3 = 0$ | skipped for time

---

@eyebrow Backup · jets
# One variable needs one epsilon: $\varepsilon^3 = 0$
@kicker A *jet*: keep one more Taylor power instead of nesting two first-order numbers

$$ t = x + a\,\varepsilon + c\,\varepsilon^2, \qquad \varepsilon^3 = 0 $$

$$ f(t) = f(x) + f'(x)\,(a\varepsilon + c\varepsilon^2) + \tfrac{f''(x)}{2}\,a^2\varepsilon^2 \;\xrightarrow{\ a=1,\ c=0\ }\; f(x) + f'(x)\,\varepsilon + \tfrac{f''(x)}{2}\,\varepsilon^2 $$

+ Three slots $\bigl(f,\ f',\ f''/2\bigr)$ — one pass, exact, **nothing stored twice**.
+ Multiply: $\ t_1 t_2 = x_1x_2 + (x_1a_2 + x_2a_1)\,\varepsilon + (x_1c_2 + x_2c_1 + a_1a_2)\,\varepsilon^2$.
+ Rules use $f'$, $f''$ directly: $\ \sin(t) = \sin x + a\cos x\,\varepsilon + \bigl(c\cos x - \tfrac{a^2}{2}\sin x\bigr)\,\varepsilon^2$.

---

@eyebrow Backup · jets
# The scalar jet in code
@kicker Same drill as the five-line forward mode — rules transcribed straight from the algebra

```julia title="three slots, no nesting"
struct Jet
    v::Float64   # f
    e::Float64   # ε  coefficient
    h::Float64   # ε² coefficient (= f''/2 after seeding)
end
Base.:*(a::Jet, b::Jet) = Jet(a.v * b.v,
                              a.v * b.e + b.v * a.e,
                              a.v * b.h + b.v * a.h + a.e * b.e)
function Base.sin(t::Jet)
    s, c = sincos(t.v)                  # f', f'' evaluated ONCE, as plain floats
    Jet(s, c * t.e, c * t.h - s * t.e^2 / 2)
end
second_derivative(f, x) = 2 * f(Jet(x, 1.0, 0.0)).h
```
@pills good:one pass → (f, f′, f″) = (−1.5136…, −5.98595…, 16.37395…) | good:no rules-of-rules | good:flat struct of 3 floats

---

@eyebrow Backup · jets
# Count the flops
@kicker A number type that tallies every `*` and `+` — generic code means we can just pass it through

::: cols
```julia size="14" title="the meter"
const FLOPS = Ref(0)

struct Flop <: Real
    x::Float64
end
Base.:*(a::Flop, b::Flop) = (FLOPS[] += 1; Flop(a.x * b.x))
Base.:+(a::Flop, b::Flop) = (FLOPS[] += 1; Flop(a.x + b.x))

flops(g) = (FLOPS[] = 0; g(); FLOPS[])
```
:: col
```julia> size="14" | one second-order multiply, same seeds
julia> d = Dual(Dual(Flop(2.0), Flop(1.0)),
                Dual(Flop(1.0), Flop(0.0)));

julia> flops(() -> d * d)
14

julia> t = Jet(Flop(2.0), Flop(1.0), Flop(0.0));

julia> flops(() -> t * t)
9
```
@pills bad:nested: 9 muls + 5 adds | good:jet: 6 muls + 3 adds
:::

---

@eyebrow Backup · jets
# One ε per variable — the chunk-mode move, again
@kicker Same generalization as `Dual` → `Dual{N}`, but now the directions meet: pairwise products survive, triples die

$$ t = x + \textstyle\sum_i a_i\,\varepsilon_i + \sum_{i \le j} c_{ij}\,\varepsilon_i\varepsilon_j, \qquad \varepsilon_i\varepsilon_j\varepsilon_k = 0 $$

$$ f\bigl(\mathbf{x} + \textstyle\sum_i \varepsilon_i\,\mathbf{e}_i\bigr) = f + \sum_i \partial_i f\,\varepsilon_i + \tfrac12 \sum_{i,j} H_{ij}\,\varepsilon_i\varepsilon_j $$

+ First order: $\varepsilon_i\varepsilon_j = 0$ killed the cross terms. Now they *are* the payload: slot $\varepsilon_i\varepsilon_j$ reads $H_{ij}$.
+ $\varepsilon_i\varepsilon_j = \varepsilon_j\varepsilon_i$ folds the two halves of the sum together — symmetry lands in storage for free, only $i \le j$ kept.
+ Multiply: the scalar `a.e * b.e` becomes $a_i b_j + a_j b_i$ per slot — the same rule, once per pair.

---

@eyebrow Backup · jets
# All of the Hessian in one pass — until it gets fat
@kicker Seed every variable with its own $\varepsilon_i$ — one evaluation, exact, whole Hessian

::: cols
What one number carries:

- value — $1$ slot
- gradient — $n$ slots ($\varepsilon_i$)
- Hessian upper triangle — $\tfrac{n(n+1)}{2}$ slots: $\varepsilon_i\varepsilon_j$ with $i \le j$, symmetry stored **once**

This is `Jet{N}` in HyperHessians — at $n = 4$ it is 15 floats against 25 for a two-epsilon number, and it wins for small $n$.
:: col
| n | slots | bytes |
| --- | --- | --- |
| 4 | 15 | 120 B |
| 16 | 153 | 1.2 KB |
| 64 | 2145 | 17 KB |
| 256 | ==33153== | ==260 KB== |
?> quadratic in n: at n = 256 one *number* is ~260 KB — no register, no cache line, no chance
:::

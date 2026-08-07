Slide 1:

Dual mode:

- Finite difference
- show taylor expansion: f(x + h) = f(x) + h * f'(x) + (h^2/2) * f''(x) + O(h^3)
- solve for f'(x): f'(x) = (f(x + h) - f(x)) / h - (h/2) * f''(x) - ...
                         = (f(x + h) - f(x)) / h + O(h)
  (point: truncation error O(h) -- must pick h; too small and the subtraction
   cancels in floating point, too big and the dropped Taylor terms bite)

- define dual number: d = x + h * epsilon where epsilon^2 = 0
- show f(d) = f(x + h * epsilon) = f(x) + h * f'(x) * epsilon + (h^2/2) * f''(x) * epsilon^2 + ...
  every term from epsilon^2 on is exactly zero -> f(d) = f(x) + h * f'(x) * epsilon
  (point: truncation is exact, no step size to choose, unlike finite differences)
- seed h = 1, read derivative off the epsilon coefficient: f'(x) = epsilon-part of f(x + epsilon)
- for example sin(d) = sin(x) + h * cos(x) * epsilon
- show d_1 * d_2 = (x_1 + h_1 * epsilon) * (x_2 + h_2 * epsilon)
                 = x_1 * x_2 + (x_1 * h_2 + x_2 * h_1) * epsilon   # h_1 * h_2 * epsilon^2 term = 0
  (point: product rule falls out of the algebra)

data layout figure:
[orange][dark orange]

----

Slide 2, micro implementation  in julia of a forward mode for scalar

- the number type: value + epsilon coefficient

    struct Dual
        v::Float64   # value
        e::Float64   # epsilon coefficient
    end

- a few operations, each one line of the algebra from slide 1:

    Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.e + b.e)
    Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.e + b.v * a.e)  # product rule
    Base.sin(d::Dual)         = Dual(sin(d.v), cos(d.v) * d.e)          # chain rule

- derivative: seed epsilon = 1, run f, read the epsilon part

    derivative(f, x) = f(Dual(x, 1.0)).e

- example:

    julia> f(x) = x * sin(x * x)

    julia> derivative(f, 2.0)
    -5.985951462216824       # = sin(4) + 8cos(4)

  (point: ~5 lines total, works on unmodified generic code; real packages add
   promotion rules Dual + Float64 etc., more rules, and vectors of epsilons)



---

Slide 3, second derivatives by nesting

- derivative() is just julia code, so differentiate it with itself:

    second_derivative(f, x) = derivative(y -> derivative(f, y), x)

- one catch: our Dual has Float64 fields, so a Dual can't hold a Dual.
  loosen the fields to any T and seed generically (diff):

    -struct Dual
    -    v::Float64
    -    e::Float64
    +struct Dual{T}
    +    v::T
    +    e::T
     end

    -derivative(f, x) = f(Dual(x, 1.0)).e
    +derivative(f, x) = f(Dual(x, one(x))).e
    +Base.one(d::Dual) = Dual(one(d.v), zero(d.v))   # so the inner seed nests
    +Base.cos(d::Dual) = Dual(cos(d.v), -sin(d.v) * d.e)

  (the +/*/sin rules from slide 2 keep working: they never mention Float64.
   but note the cos rule became *required*: sin's rule calls cos(d.v), and
   under nesting d.v is itself a Dual -- rules get differentiated too)

- the math: nesting = two independent epsilons, eps1^2 = eps2^2 = 0 but
  eps1*eps2 != 0:

    f(x + eps1 + eps2) = f(x) + f'(x) * (eps1 + eps2) + f''(x) * eps1*eps2

  f''(x) = the eps1*eps2 coefficient

- example (same f as slide 2):

    julia> second_derivative(f, 2.0)
    16.37395639949036        # = 6x cos(x^2) - 4x^3 sin(x^2) at x = 2

- what the nested number actually carries: ((f, f'), (f', f'')) -- four slots,
  f' stored and computed TWICE
  (point: nesting works out of the box, that is the magic of generic code;
   the redundancy is the thread we pull later -- flat second-order numbers
   (slides 6-8) store nothing twice and go straight for the answer)
- Hessian = the same nesting one level up: H = jacobian(gradient(f)) --
  after the next two slides we have both pieces

---

Slide 4, extend to jacobian, insert code

- f: R^n -> R^m now; input is a vector of duals x + h * epsilon with seed
  direction h in R^n. Taylor per output component F_k:

    F_k(x + h * epsilon) = F_k(x) + (sum_j dF_k/dx_j * h_j) * epsilon   # eps^2 = 0 as before

  stack the m components -> one pass computes a jacobian-vector product:

    f(x + h * epsilon) = f(x) + J(x) h * epsilon,     J_kj = dF_k/dx_j

- to get the whole matrix, seed with the identity: h = e_i picks out
  J e_i = column i of J. n passes, one per column:

    [ f(x + e_1 eps) ... f(x + e_n eps) ]  epsilon parts  ->  [ J e_1 ... J e_n ] = J

    function jacobian(f, x::Vector)
        cols = map(eachindex(x)) do i
            d = [Dual(x[j], i == j ? 1.0 : 0.0) for j in eachindex(x)]
            [y.e for y in f(d)]
        end
        return stack(cols)   # m x n
    end

- example:

    julia> f(x) = [x[1] * x[2], sin(x[1])]

    julia> jacobian(f, [2.0, 3.0])
    2×2 Matrix{Float64}:
      3.0        2.0         # [ x[2]      x[1] ]
     -0.416147  -0.0         # [ cos(x[1])  0   ]   (-0.0: cos(2)*0.0, harmless)

- gradient is just the m = 1 row of this: seed the same way, one output
  (point: one full evaluation of f per *input* dimension -- n passes, primal
   recomputed every pass; cost scales with n regardless of m)
  (point: real packages instead make e a *vector* of epsilons and push a chunk
   of input directions through in one pass, sharing the primal work -- that
   layout choice is where the performance story starts)

-- slide 5 chunk mode

Show the diff of extending the partials to a tuple to compute multiple columns at the same time

- e goes from one epsilon to N of them; every rule just broadcasts (* -> .*):

    -struct Dual
    -    v::Float64
    -    e::Float64
    +struct Dual{N}
    +    v::Float64
    +    e::NTuple{N,Float64}    # N partials ride along
     end

    -Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.e + b.e)
    +Base.:+(a::Dual{N}, b::Dual{N}) where N = Dual(a.v + b.v, a.e .+ b.e)

    -Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.e + b.v * a.e)
    +Base.:*(a::Dual{N}, b::Dual{N}) where N = Dual(a.v * b.v, a.v .* b.e .+ b.v .* a.e)

    -Base.sin(d::Dual) = Dual(sin(d.v), cos(d.v) * d.e)
    +Base.sin(d::Dual) = Dual(sin(d.v), cos(d.v) .* d.e)

- jacobian in ONE pass: identity seed all at once, epsilon tuple of output k
  is row k of J

    function jacobian(f, x::Vector)
        n = length(x)
        d = [Dual(x[j], ntuple(i -> Float64(i == j), n)) for j in 1:n]
        return stack([y.e for y in f(d)]; dims = 1)   # m x n
    end

    julia> jacobian(f, [2.0, 3.0])   # same J as slide 3, one call to f
    2×2 Matrix{Float64}:
      3.0        2.0
     -0.416147  -0.0

  (point: primal computed once, shared by all n columns; sin still calls cos
   *once* and broadcasts it over the N partials -- expensive rule work is
   amortized across directions)
  (point: cost per operation grows with N, and a fat Dual{N} stops fitting in
   registers -- so real packages cap N at a *chunk* (ForwardDiff: ~8-12) and
   sweep n/N passes; picking the chunk layout is the performance knob)

---

-- slide 6 flat second-order number, scalar case (same drill as slide 1)

- the nested dual carried ((f, f'), (f', f'')) -- f' twice. for ONE variable
  a single epsilon is enough if we keep one more power: eps^3 = 0 (not eps^2)

    t = x + a*eps + c*eps^2,     eps^3 = 0     # a "jet": truncated taylor series

- insert into taylor, u = a*eps + c*eps^2:

    u^2 = a^2 * eps^2         # cross/higher terms all carry eps^3 = die
    u^3 = 0                   # series terminates exactly, again

    f(x + u) = f(x) + f'(x) * u + (f''(x)/2) * u^2
             = f(x) + a f'(x) eps + (c f'(x) + (a^2/2) f''(x)) eps^2

- seed a = 1, c = 0:

    f(t) = f(x) + f'(x) eps + (f''(x)/2) eps^2

  three slots (f, f', f''/2): one pass, exact, NOTHING stored twice

- multiplication rule, collect coefficients of t1 * t2:

    value:  x1*x2
    eps:    x1*a2 + x2*a1              # first-order product rule
    eps^2:  x1*c2 + x2*c1 + a1*a2      # second-order cross term

- primitive rules use f, f', f'' directly, evaluated once as plain scalars:

    sin(t) = sin(x) + a cos(x) eps + (c cos(x) - (a^2/2) sin(x)) eps^2

---

-- slide 7 micro implementation of the scalar jet (same drill as slide 2)

- three slots, straight from the slide 6 algebra:

    struct Jet
        v::Float64   # f
        e::Float64   # eps coefficient
        h::Float64   # eps^2 coefficient (= f''/2 after seeding)
    end

    Base.:+(a::Jet, b::Jet) = Jet(a.v + b.v, a.e + b.e, a.h + b.h)

    Base.:*(a::Jet, b::Jet) = Jet(a.v * b.v,
                                  a.v * b.e + b.v * a.e,          # product rule
                                  a.v * b.h + b.v * a.h + a.e * b.e)

    function Base.sin(t::Jet)
        s, c = sincos(t.v)               # f', f'' evaluated ONCE, as plain floats
        Jet(s, c * t.e, c * t.h - s * t.e^2 / 2)
    end

- seed (a, c) = (1, 0), read the eps^2 slot (times 2):

    second_derivative(f, x) = 2 * f(Jet(x, 1.0, 0.0)).h

- example (same f; one pass carries f, f', f''/2 together):

    julia> t = f(Jet(2.0, 1.0, 0.0));

    julia> t.v, t.e, 2 * t.h
    (-1.5136049906158564, -5.985951462216824, 16.37395639949036)

  matches slide 2 (f'), slide 3 (f''), analytic f -- one evaluation, 3 floats

  (point: vs slide 3 nesting: same answers, but no Dual-of-Dual, no
   rules-of-rules -- sincos runs once on machine floats and feeds every slot,
   and unlike the nested number nothing is computed twice)

---

-- slide 8 multivariate jet -> too fat -> chunk it -> eps1/eps2

- multivariate: seed ALL n directions at once, keep every second-order product
  (eps_i eps_j eps_k = 0). the number carries

    value              1 slot
    gradient           n slots            (eps_i)
    hessian, i <= j    n(n+1)/2 slots     (eps_i eps_j, symmetric: upper triangle once)

  whole hessian in ONE pass -- this is HyperHessians' Jet{N}, great for small n
  (n = 4: 15 floats vs 25 for the two-epsilon version)

- but the slot count is quadratic in n:

    n      1 + n + n(n+1)/2
    4      15
    16     153
    64     2145
    256    33153        # ~260 KB per *number* -- no register, no cache line

- so chunk, like slide 5. but a hessian block H[I,J] mixes a direction i from
  chunk I with a direction j from chunk J -- two INDEPENDENT sets of
  directions must ride through f together:

    h = x + sum_i a_i eps1_i + sum_j b_j eps2_j + (eps1*eps2 block)
    coefficient of eps1_i * eps2_j in f(h)  =  d2f/dx_i dx_j  =  H[i, j]

  THIS is why eps2 exists: eps1 carries chunk I, eps2 carries chunk J, the
  eps1*eps2 slots are an N1 x N2 block of mixed partials
  -> HyperDual{N1, N2}: 1 + N1 + N2 + N1*N2 slots ({8,8}: 81 floats, register-sized)

- sweep block pairs, and symmetry means only i <= j: k(k+1)/2 evaluations for
  k chunks instead of k^2 (n = 256, chunk 8: 528 vs 1024 -- the 1.94x from the
  benchmark report)
  (point: scalar case collapses to slide 6 -- both chunks are the same
   direction, e1 = e2, which is exactly why the jet needs no eps2)
  (point: asymmetric chunks come free: seed eps2 with a single tangent vector
   v -> HyperDual{N, 1} computes the hessian-vector product H*v without ever
   forming H)

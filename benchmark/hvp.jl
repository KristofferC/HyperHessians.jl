using HyperHessians, DiffTests, BenchmarkTools, LinearAlgebra

const f = DiffTests.rosenbrock_1
const n = 16
const x = rand(n)
const v = rand(n)

const cfg = HyperHessians.HessianConfig(x)
const hv = similar(x)
const H = similar(x, n, n)

function hess_vp!(H, hv, f::F, x, v, cfg) where {F}
    HyperHessians.hessian!(H, f, x, cfg)
    return mul!(hv, H, v)
end

@info "HVP"
@btime HyperHessians.hvp!($hv, $f, $x, $v, $cfg);

@info "Hessian! + mul!"
@btime hess_vp!($H, $hv, $f, $x, $v, $cfg);


nothing

using HyperHessians, DiffTests, BenchmarkTools, LinearAlgebra

const f = DiffTests.rosenbrock_1
const n = 16
const x = rand(n)
const v = rand(n)

const cfg = HyperHessians.HessianConfig(x) # for full Hessian benchmark
const cfg_dir = HyperHessians.DirectionalHVPConfig(x)
const hv = similar(x)
const hv_dir = similar(x)
const H = similar(x, n, n)

function hess_vp!(H, hv, f::F, x, v, cfg) where {F}
    HyperHessians.hessian!(H, f, x, cfg)
    return mul!(hv, H, v)
end

@info "HVP"
@btime HyperHessians.hvp!($hv, $f, $x, $v); # directional default

@info "HVP (directional cfg)"
@btime HyperHessians.hvp!($hv_dir, $f, $x, $v, $cfg_dir);

@info "Hessian! + mul!"
@btime hess_vp!($H, $hv, $f, $x, $v, $cfg);


nothing

using HyperHessians, DiffTests, BenchmarkTools, LinearAlgebra, DifferentiationInterface, ForwardDiff


function hess_vp!(H, hv, f::F, x, v, cfg) where {F}
    HyperHessians.hessian!(H, f, x, cfg)
    return mul!(hv, H, v)
end

function do_benchmark(n)
    @info "n = $n"
    f = DiffTests.rosenbrock_1
    x = rand(n)
    v = rand(n)
    v_bundle = (rand(n), rand(n), rand(n))

    cfg = HyperHessians.HessianConfig(x) # for full Hessian benchmark
    cfg_dir = HyperHessians.DirectionalHVPConfig(x)
    cfg_bundle = HyperHessians.DirectionalHVPConfig(x, v_bundle)
    hv = similar(x)
    hv_dir = similar(x)
    hv_bundle = (similar(x), similar(x), similar(x))
    H = similar(x, n, n)


    @info "HyperHessians HVP single"
    @btime HyperHessians.hvp!($hv_dir, $f, $x, $v, $cfg_dir)
    @info "HyperHessians HVP bundle"
    @btime HyperHessians.hvp!($hv_bundle, $f, $x, $v_bundle, $cfg_bundle)

    @info "ForwardDiff HVP single"
    prep_single = DifferentiationInterface.prepare_hvp(f, AutoForwardDiff(), x, (v,))
    @btime DifferentiationInterface.hvp!($f, ($hv,), $prep_single, AutoForwardDiff(), $x, ($v,))

    @info "ForwardDiff HVP bundle"
    prep_bundle = DifferentiationInterface.prepare_hvp(f, AutoForwardDiff(), x, v_bundle)
    @btime DifferentiationInterface.hvp!($f, $hv_bundle, $prep_bundle, AutoForwardDiff(), $x, $v_bundle)

    @info "Naive"
    return @btime hess_vp!($H, $hv, $f, $x, $v, $cfg)
end

for n in (8, 16, 128)
    do_benchmark(n)
end

nothing

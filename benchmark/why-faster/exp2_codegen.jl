# Generated-code statistics for single dual operations at identical payload:
# HyperDual{8,8} (81 floats) vs nested Dual{Dual{Float64,8},8} (81 floats).
# Counts native instructions, FMA instructions, and out-of-line calls.
# Run with: julia --project=benchmark benchmark/why-faster/exp2_codegen.jl
using HyperHessians, ForwardDiff, InteractiveUtils

T_hd8 = HyperHessians.HyperDual{8, 8, Float64}
T_dd8 = ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, 8}, 8}

function native_stats(f, Ts)
    buf = IOBuffer()
    code_native(buf, f, Ts; debuginfo = :none)
    s = String(take!(buf))
    lines = split(s, '\n')
    instr = [strip(l) for l in lines if occursin(r"^\t", l)]
    n_total = length(instr)
    n_fma = count(i -> occursin(r"^fmla|^fmadd|^vfmadd|^vfmsub|^fmls", i), instr)
    n_ldst = count(i -> occursin(r"^ld|^st|^vmov|^mov", i), instr)
    return (; n_total, n_fma, n_ldst)
end

function llvm_calls(f, Ts)
    buf = IOBuffer()
    code_llvm(buf, f, Ts; debuginfo = :none)
    s = String(take!(buf))
    calls = collect(eachmatch(r"call.*@([\"\w\.]+)", s))
    return [c.captures[1] for c in calls]
end

for (label, op, Ts) in [
        ("mul HyperDual{8,8}", *, (T_hd8, T_hd8)),
        ("mul nested Dual 8x8", *, (T_dd8, T_dd8)),
        ("add HyperDual{8,8}", +, (T_hd8, T_hd8)),
        ("add nested Dual 8x8", +, (T_dd8, T_dd8)),
        ("exp HyperDual{8,8}", exp, (T_hd8,)),
        ("exp nested Dual 8x8", exp, (T_dd8,)),
        ("sin HyperDual{8,8}", sin, (T_hd8,)),
        ("sin nested Dual 8x8", sin, (T_dd8,)),
        ("sqrt HyperDual{8,8}", sqrt, (T_hd8,)),
        ("sqrt nested Dual 8x8", sqrt, (T_dd8,)),
        ("div HyperDual{8,8}", /, (T_hd8, T_hd8)),
        ("div nested Dual 8x8", /, (T_dd8, T_dd8)),
    ]
    st = native_stats(op, Ts)
    calls = llvm_calls(op, Ts)
    println(
        rpad(label, 26), " instrs=", lpad(st.n_total, 5),
        "  fma=", lpad(st.n_fma, 4),
        "  ld/st/mov=", lpad(st.n_ldst, 5),
        "  calls=", isempty(calls) ? "none" : join(unique(calls), ",")
    )
end

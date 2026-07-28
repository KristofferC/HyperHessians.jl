module HyperHessiansForwardDiffExt

# Mixing HyperDual (e.g. from hessian seeding) with ForwardDiff.Dual (e.g.
# from a derivative computed inside the function being differentiated) is
# ambiguous out of the box, since both packages define methods against
# ::Real. Resolve the ambiguities by letting the Dual wrap the HyperDual,
# just like ForwardDiff wraps any other Real: forward to the ForwardDiff
# method with the HyperDual passed as a generic Real scalar.

using HyperHessians: HyperDual
using ForwardDiff: ForwardDiff, Dual

for f in (:+, :-, :*, :/, :^, :atan, :hypot, :log, :<, :<=, :(==), :isless)
    @eval begin
        @inline Base.$f(h::HyperDual, d::Dual) = invoke(Base.$f, Tuple{Real, typeof(d)}, h, d)
        @inline Base.$f(d::Dual, h::HyperDual) = invoke(Base.$f, Tuple{typeof(d), Real}, d, h)
    end
end
@inline Base.mod(h::HyperDual, d::Dual) = invoke(Base.mod, Tuple{Real, typeof(d)}, h, d)
@inline Base.rem(h::HyperDual, d::Dual) = invoke(Base.rem, Tuple{Real, typeof(d)}, h, d)
@inline ForwardDiff.NaNMath.pow(h::HyperDual, d::Dual) = invoke(ForwardDiff.NaNMath.pow, Tuple{Real, typeof(d)}, h, d)
@inline ForwardDiff.NaNMath.pow(d::Dual, h::HyperDual) = invoke(ForwardDiff.NaNMath.pow, Tuple{typeof(d), Real}, d, h)

Base.promote_rule(::Type{HyperDual{N1, N2, T}}, ::Type{Dual{Ty, V, N}}) where {N1, N2, T, Ty, V, N} =
    Dual{Ty, promote_type(HyperDual{N1, N2, T}, V), N}

# muladd: all mixed HyperDual/Dual argument combinations. ForwardDiff defines
# its ternary ops against every type in AMBIGUOUS_TYPES (not just Real), so
# the remaining non-HyperDual/non-Dual slot must be split over the same set
# for the mixed methods to be strictly more specific than all of them.
const _OTHERS = Tuple(T for T in ForwardDiff.AMBIGUOUS_TYPES if T <: Real)
for pat in Iterators.product((:H, :D, :O), (:H, :D, :O), (:H, :D, :O))
    (:H in pat && :D in pat) || continue
    names = (:x, :y, :z)
    ivk = [pat[i] === :H ? Real : :(typeof($(names[i]))) for i in 1:3]
    for Q in (:O in pat ? _OTHERS : (nothing,))
        sig = [:($(names[i])::$(pat[i] === :H ? HyperDual : pat[i] === :D ? Dual : Q)) for i in 1:3]
        @eval @inline Base.muladd($(sig...)) = invoke(Base.muladd, Tuple{$(ivk...)}, x, y, z)
    end
end

end # module

module HyperHessiansForwardDiffExt

# Mixing HyperDual or Jet (e.g. from hessian seeding) with ForwardDiff.Dual
# (e.g. from a derivative computed inside the function being differentiated)
# is ambiguous out of the box, since both packages define methods against
# ::Real. Resolve the ambiguities by letting the Dual wrap the second-order
# number, just like ForwardDiff wraps any other Real: forward to the
# ForwardDiff method with the HyperDual/Jet passed as a generic Real scalar.

using HyperHessians: HyperDual, Jet
using ForwardDiff: ForwardDiff, Dual

for H in (HyperDual, Jet)
    for f in (:+, :-, :*, :/, :^, :atan, :hypot, :log, :<, :<=, :(==), :isless)
        @eval begin
            @inline Base.$f(h::$H, d::Dual) = invoke(Base.$f, Tuple{Real, typeof(d)}, h, d)
            @inline Base.$f(d::Dual, h::$H) = invoke(Base.$f, Tuple{typeof(d), Real}, d, h)
        end
    end
    @eval begin
        @inline Base.mod(h::$H, d::Dual) = invoke(Base.mod, Tuple{Real, typeof(d)}, h, d)
        @inline Base.rem(h::$H, d::Dual) = invoke(Base.rem, Tuple{Real, typeof(d)}, h, d)
        @inline ForwardDiff.NaNMath.pow(h::$H, d::Dual) = invoke(ForwardDiff.NaNMath.pow, Tuple{Real, typeof(d)}, h, d)
        @inline ForwardDiff.NaNMath.pow(d::Dual, h::$H) = invoke(ForwardDiff.NaNMath.pow, Tuple{typeof(d), Real}, d, h)
    end
end

Base.promote_rule(::Type{HyperDual{N1, N2, T, S}}, ::Type{Dual{Ty, V, N}}) where {N1, N2, T, S, Ty, V, N} =
    Dual{Ty, promote_type(HyperDual{N1, N2, T, S}, V), N}
Base.promote_rule(::Type{Jet{NJ, M, T, S}}, ::Type{Dual{Ty, V, N}}) where {NJ, M, T, S, Ty, V, N} =
    Dual{Ty, promote_type(Jet{NJ, M, T, S}, V), N}

# muladd: all mixed HyperDual-or-Jet/Dual argument combinations. ForwardDiff
# defines its ternary ops against every type in AMBIGUOUS_TYPES (not just
# Real), so the remaining non-HyperDual/non-Dual slot must be split over the
# same set for the mixed methods to be strictly more specific than all of them.
const _OTHERS = Tuple(T for T in ForwardDiff.AMBIGUOUS_TYPES if T <: Real)
for H in (HyperDual, Jet)
    for pat in Iterators.product((:H, :D, :O), (:H, :D, :O), (:H, :D, :O))
        (:H in pat && :D in pat) || continue
        names = (:x, :y, :z)
        ivk = [pat[i] === :H ? Real : :(typeof($(names[i]))) for i in 1:3]
        for Q in (:O in pat ? _OTHERS : (nothing,))
            sig = [:($(names[i])::$(pat[i] === :H ? H : pat[i] === :D ? Dual : Q)) for i in 1:3]
            @eval @inline Base.muladd($(sig...)) = invoke(Base.muladd, Tuple{$(ivk...)}, x, y, z)
        end
    end
end

end # module

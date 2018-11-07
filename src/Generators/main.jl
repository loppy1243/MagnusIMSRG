module Generators
using ManyBody
using ..MagnusIMSRG: SIGNAL_OPS, E_DENOM_ATOL, TwoBodyARRAYOP, isocc, isunocc, FUNCOP, H

SIGNAL_OPS && include("../signalops.jl")

let EMIT_ZERO_WARNING_1=true, EMIT_ZERO_WARNING_2=true
global white
function white(Ω::TwoBodyARRAYOP, h::TwoBodyARRAYOP)
    E, f, Γ = h

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    function _b1(I, J)
        d = Δ(I..., J...)
        mask = isunocc(I)*isocc(J)
        if !iszero(mask) && abs(d) < E_DENOM_ATOL
            if EMIT_ZERO_WARNING_1
                @warn("One-body energy denominator is zero! Will not warn again.")
                EMIT_ZERO_WARNING_1 = false
            end
            zero(d)
        elseif iszero(mask)
            zero(d)
        else
            mask*f[I, J] / d
        end
    end

    function _b2(I, J)
        d = Δ(I..., J...)
        mask = all(isunocc, I)*all(isocc, J)
        if !iszero(mask) && abs(d) < E_DENOM_ATOL
            if EMIT_ZERO_WARNING_2
                @warn("Two-body energy denominator is zero! Will not warn again.")
                EMIT_ZERO_WARNING_2 = false
            end
            zero(d)
        elseif iszero(mask)
            zero(d)
        else
            4 \ mask*Γ[I, J] / d
        end
    end

    b1 = FUNCOP(1)() do I, J
#        i, j = inner(I), inner(J)
        _b1(I, J) - conj(_b1(J, I))
    end
    b2 = FUNCOP(2)() do I, J
        _b2(I, J) - conj(_b2(J, I))
    end

    (zero(E), tabulate(b1), tabulate(b2))
end end

end # module Generators

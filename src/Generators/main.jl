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

    function _b1(i, j)
        d = Δ(i, j)
        if abs(d) < E_DENOM_ATOL
            if EMIT_ZERO_WARNING_1
                @warn("One-body energy denominator is zero! Will not warn again.")
                EMIT_ZERO_WARNING_1 = false
            end
            zero(d)
        else
            f[i, j] / d
        end
    end

    function _b2(i, j, k, l)
        d = Δ(i, j, k, l)
        if abs(d) < E_DENOM_ATOL
            if EMIT_ZERO_WARNING_2
                @warn("Two-body energy denominator is zero! Will not warn again.")
                EMIT_ZERO_WARNING_2 = false
            end
            zero(d)
        else
            4 \ Γ[i, j, k, l] / d
        end
    end

    f_ret = Array{ELTYPE}(undef, DIM, DIM)
    Γ_ret = Array{ELTYPE}(undef, DIM, DIM, DIM, DIM)

    for i in PARTS, j in HOLES
        f_ret[index(i), index(j)] = _b1(i, j) - conj(_b1(j, i))
    end

    for i in PARTS, j in PARTS, k in HOLES, l in HOLES
        Γ_ret[index(i), index(j), index(k), index(l)] = _b2(i, j, k, l) - conj(_b2(k, l, i, j))
    end

    (zero(E), ARRAYOP(1)(b1), ARRAYOP(2)(b2))
end end

end # module Generators

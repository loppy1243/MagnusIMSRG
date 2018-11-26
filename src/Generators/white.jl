let EMIT_ZERO_WARNING_1=true, EMIT_ZERO_WARNING_2=true
global white
function white(h::IMArrayOp{2})
    E, f, Γ = h.parts

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    function f′(i, j)
        d = Δ(i, j)
        if abs(d) < ENG_DENOM_ATOL
            if EMIT_ZERO_WARNING_1
                @warn("One-body energy denominator is zero! Will not warn again.")
                EMIT_ZERO_WARNING_1 = false
            end
            zero(d)
        else
            f[i, j] / d
        end
    end

    function Γ′(i, j, k, l)
        d = Δ(i, j, k, l)
        if abs(d) < ENG_DENOM_ATOL
            if EMIT_ZERO_WARNING_2
                @warn("Two-body energy denominator is zero! Will not warn again.")
                EMIT_ZERO_WARNING_2 = false
            end
            zero(d)
        else
            4 \ Γ[i, j, k, l] / d
        end
    end

    ret = zero(h)
    ret_f = ret.parts[1]; ret_Γ = ret.parts[2]

    for i in PARTS, j in HOLES
        ret_f[i, j] = f′(i, j) - f′(j, j)
    end
    for i in PARTS, j in PARTS, k in HOLES, l in HOLES
        ret_Γ[i, j, k, l] = Γ′(i, j, k, l) - Γ′(k, l, i, j)
    end

    ret
end end

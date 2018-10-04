module Generators
using ManyBody
using ..TwoBodyARRAYOP, ..isocc, ..isunocc, ..FUNCOP

module OnceMacro
export @once
const _DONE = Dict{Symbol, Bool}()
macro once(expr)
    key = esc(gensym())
    quote
        if !get(_DONE, $key, false)
            $(esc(expr))
            nothing
        else
            _DONE[$key] = true
        end
    end
end

function __init()__
    for k in keys(_DONE)
        _DONE[k] = false
    end
end
end # module OnceMacro
using .OnceMacro: @once

function white(Ω::TwoBodyARRAYOP, h0::TwoBodyARRAYOP)
    E0, f, Γ = H(Ω, h0)

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    function _b1(I, J)
        mask = isunocc(I)*isocc(J)
        x = mask*f[I, J]
        d = Δ(I..., J...)
        if iszero(d)
            @once @warn("One-body energy denominator is zero! Will not warn again.")
            d
        else
            x/d
        end
    end
    function _b2(I, J)
        mask = all(isunocc, I)*all(isocc, J)
        x = mask * Γ[I, J] / 4
        d = Δ(I..., J...)
        if iszero(d)
            @once @warn("Two-body energy denominator is zero! Will not warn again.")
            d
        else
            x/d
        end
    end

    b1 = FUNCOP(1)() do I, J
#        i, j = inner(I), inner(J)
        _b1(I, J) - conj(_b1(J, I))
    end
    b2 = FUNCOP(2)() do I, J
        _b2(I, J) - conj(_b2(J, I))
    end

    (zero(E0), tabulate(b1), tabulate(b2))
end

end # module Generators

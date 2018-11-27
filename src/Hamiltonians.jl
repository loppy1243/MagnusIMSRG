module Hamiltonians
using ManyBody

import ..SIGNAL_OPS

SIGNAL_OPS && include("signalops.jl")

function impairing(ref, δ, g)
    E = sum(occ(ref)) do p
        δ*(p.level-1) - g/4*isocc(ref, flipspin(p))
    end

    f(p, q) = (p == q)*(δ*(p.level-1) - g/2*isocc(ref, flipspin(p)))

    function Γ(p, q, r, s)
        mask = (p.level == q.level)*(r.level == s.level) #=
            =# * spinup(p)*spindown(q)*spindown(s)*spinup(r)

        -g*mask
    end
    Γ_AS(p, q, r, s) = 2\(Γ(p, q, r, s) + Γ(q, p, s, r) - Γ(p, q, s, r) - Γ(q, p, r, s))

    (E, f, Γ_AS)
end

end # module Hamiltonians

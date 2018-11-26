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

        -g/2*mask
    end

    (E, f, Γ)
end

end # module Hamiltonians

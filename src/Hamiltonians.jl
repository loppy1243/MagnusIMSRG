module Hamiltonians

import ..SIGNAL_OPS

SIGNAL_OPS && include("signalops.jl")

function impairing2(ref, δ, g)
    self_int = g/4*nocc(ref)

    E = sum(occ(ref)) do i
        δ*(i.level - 1)
    end - self_int

    f = (p, q) -> (p == q)*(δ*(p.level-1) - self_int)

    Γ = function(p, q, r, s)
        mask = (p.level == q.level)*(r.level == s.level) #=
            =# * spinup(p)*spinup(r)*spindown(q)*spindown(s)

        -g/2*mask
    end

    (E, f, Γ)
end

end # module Hamiltonians

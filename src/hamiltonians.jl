export im_pairingH

module Hamiltonians
using IMOperators; IMOps = IMOperators
using Parameters: @unpack
using ..choosetol, ..factorial

function H(Ω, h0, params)
    @debug "Entering H"
    @unpack comm, H_ATOL, H_RTOL, H_BATCHSIZE = params
    @debug "H term" n=0
    prev_tot = zero(h0)
    prev_ad = h0

    # First term
    tot = 1/factorial(Float64, 0) * prev_ad

    n = 1
    while IMOps.norm(tot - prev_tot) > choosetol(H_ATOL, H_RTOL*IMOps.norm(tot))
        @debug "H term" n
        prev_tot = tot
        tot += sum(n:n+H_BATCHSIZE-1) do i
            1/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += H_BATCHSIZE
    end

    tot
end

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

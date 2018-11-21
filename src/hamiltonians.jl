export im_pairingH

module Hamiltonians

function H(Ω, h0)
    @debug "Entering H"
    @debug "H term" n=0
    prev_tot = ZERO_OP
    prev_ad = h0

    # First term
    tot = 1/factorial(Float64, 0) * prev_ad

    n = 1
    while norm(tot - prev_tot) > max(H_ATOL, H_RTOL*norm(tot))
        @debug "H term" n
        prev_tot = tot
        tot += sum(n:n+H_BATCHSIZE-1) do i
            1/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += H_BATCHSIZE
    end

    tot
end

function im_pairing(g)
    E = sum(occ(REFSTATE)) do i
        LEVEL_SPACING*(level(i) - 1)
    end - g/2*FERMILEVEL

    f = FUNCOP(1)() do (p,), (q,)
        (p == q)*(LEVEL_SPACING*(level(p)-1) - g/2*FERMILEVEL)
    end

    Γ = FUNCOP(2)() do (p, q), (r, s)
        mask = (level(p) == level(q))*(level(r) == level(s)) #=
            =# * spinup(p)*spinup(r)*spindown(q)*spindown(s)

        -g/2*mask
    end

    (E, tabulate(f), tabulate(Γ))
end

end # module Hamiltonians

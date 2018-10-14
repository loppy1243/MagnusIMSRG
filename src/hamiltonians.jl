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

function im_pairingH(g)
    E0 = 2sum(occ(REFSTATE)) do i
        LEVEL_SPACING*(level(i) - 1)
    end - g/2*FERMILEVEL^2

    f = FUNCOP(1)() do (p,), (q,)
        (p == q)*(LEVEL_SPACING*(level(p)-1) - g/2*FERMILEVEL)
    end

    Γ = FUNCOP(2)() do (p, q), (r, s)
        mask = (level(p) == level(q))*(level(r) == level(s)) #=
            =# * spinup(p)*spinup(r)*spindown(q)*spindown(s)

        -g/2*mask
    end

    (E0, tabulate(f), tabulate(Γ))
end

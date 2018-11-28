function wegner(h)
    E, f, Γ = h.parts
    h_od = zero(h); E_od, f_od, Γ_od = h_od.parts

    for p in PARTS, h in HOLES
        f_od[p, h] = f[p, h] + conj(f[h, p])

        for p′ in PARTS, h′ in HOLES
            Γ_od[p, p′, h, h′] = Γ[p, p′, h, h′] + conj(Γ[h, h′, p, p′])
        end
    end

    comm(h - h_od, h_od)
end

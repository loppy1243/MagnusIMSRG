function mbpt2(op::IMArrayOp{2})
    @localgetparams HOLES, PARTS
    E, f, Γ = op.parts

    ret = zero(E[])
    for i in HOLES, j in HOLES, a in PARTS, b in PARTS
        ret += 4 \ Γ[a, b, i, j]^2 / (f[i, i] + f[j, j] - f[a, a] - f[b, b])
    end

    ret
end

function mbpt2(op::TwoBodyARRAYOP)
    _, f, Γ = op

    sum(Iterators.product(HOLES, HOLES, PARTS, PARTS)) do (i, j, a, b)
        4 \ Γ[a, b, i, j]^2 / (f[i, i] + f[j, j] + f[a, a] + f[b, b])
    end
end

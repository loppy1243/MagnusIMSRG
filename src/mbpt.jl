mbpt2(op::TwoBodyARRAYOP) = mbpt2((op[1], op[2].rep, op[3].rep))
function mbpt2(op)
    _, f, Γ = op

    sum(Iterators.product(HOLES, HOLES, PARTS, PARTS)) do X
        i, j, a, b = index.(X)
        4 \ Γ[a, b, i, j]^2 / (f[i, i] + f[j, j] - f[a, a] - f[b, b])
    end
end

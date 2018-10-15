import LinearAlgebra

const TwoBodyARRAYOP = Tuple{ELTYPE, ARRAYOP(1), ARRAYOP(2)}

const ZERO_OP = (zero(ELTYPE), zero(ARRAYOP(1)), zero(ARRAYOP(2)))

nbody(A, n) = A[n+1]

Base.:+(a::TwoBodyARRAYOP, b::TwoBodyARRAYOP) = a .+ b
Base.:*(a::Number, b::TwoBodyARRAYOP) = a .* b
Base.:*(a::TwoBodyARRAYOP, b::Number) = a .* b
Base.:/(a::TwoBodyARRAYOP, b::Number) = a ./ b
Base.:\(a::Number, b::TwoBodyARRAYOP) = a .\ b
Base.:-(a::TwoBodyARRAYOP) = .-a
Base.:+(a::TwoBodyARRAYOP) = a
Base.:-(a::TwoBodyARRAYOP, b::TwoBodyARRAYOP) = a .- b
function norm(op)
    E0, f, Γ = op
    abs(E0) + LinearAlgebra.norm(f.rep) + LinearAlgebra.norm(Γ.rep)
end

function to_mbop(op, T=ELTYPE, MB=MBBASIS)
    E0, f, Γ = op

    FunctionOperator{MB, T}() do X, Y
        b0 = E0*(X'Y)
        b1 = sum(matrixiter(f)) do ((p,), (q,))
            NA = normord(Operators.A(p', q))
            f[p, q]*(X'NA(Y))
        end
        b2 = sum(matrixiter(Γ)) do (I, J)
            p, q = I; r, s = J
            NA = normord(Operators.A(p', q', s, r))
            Γ[I, J]*(X'NA(Y))
        end

        b0 + b1 + b2
    end
end

function mbdiag(op, MB=MBBASIS)
    E0, f, Γ = op

    ret = fill(E0, dim(MB))

    for ((p,), (q,)) in matrixiter(f)
        NA = normord(Operators.A(p', q))
        for X in MB
            ret[index(X)] += f[p, q]*(X'NA(X))
        end
    end
    for (I, J) in matrixiter(Γ)
        p, q = I; r, s = J
        NA = normord(Operators.A(p', q', s, r))
        for X in MB
            ret[index(X)] += Γ[I, J]*(X'NA(X))
        end
    end

    ret
end

function randop(T=ELTYPE, B=SPBASIS)
    arrop(N) = ArrayOperator{Bases.Product{N, NTuple{N, B}}, T, Array{T, 2N}}

    d = dim(B)
    E0 = rand(T)
    f = rand(T, d, d)
    Γ = rand(T, d, d, d, d)

    (E0, arrop(1)(f), arrop(2)(Γ))
end

function hermiticize(op::TwoBodyARRAYOP)
    E0, f, Γ = op
    Γc_rep = PermutedDimsArray(Γ.rep, [3, 4, 1, 2])

    (E0, ARRAYOP(1)(2\(f.rep + f.rep')), ARRAYOP(2)(2\(Γ.rep + Γc_rep)))
end

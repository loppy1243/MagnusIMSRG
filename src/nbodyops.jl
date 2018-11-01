export to_mbop, nbody

import LinearAlgebra
using LinearAlgebra: I

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
    sqrt(abs(E0)^2 + LinearAlgebra.norm(f.rep)^2 + LinearAlgebra.norm(Γ.rep)^2)
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

## This does not work
#function to_mbop2(op, T=ELTYPE, MB=MBBASIS)
#    E0, f, Γ = op
#    d = dim(MB)
#
#    ret = E0*Array{T}(I, d, d)
#    for p in SPBASIS, q in SPBASIS, r in SPBASIS, s in SPBASIS
#        NA1 = normord(Operators.A(p', q))
#        NA2 = normord(Operators.A(p', q', s, r))
#        for X in MB, Y in MB
#            Y′ = deepcopy(Y)
#            ret[index(X), index(Y)] +=
#                d^2 \ f[p, q]*(X'applyop!(NA1, Y)) + Γ[p, q, r, s]*(X'applyop!(NA2, Y′))
#        end
#    end
#
#    MBARRAYOP(ret)
#end

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

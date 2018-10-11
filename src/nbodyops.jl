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

function to_mbop(op)
    E0, f, Γ = op

    MBFUNCOP() do X, Y
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

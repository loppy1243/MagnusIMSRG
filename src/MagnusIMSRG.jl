module MagnusIMSRG

using ManyBody
import ManyBody.Operators
using OrdinaryDiffEq
using Combinatorics: levicivita
using JuliaUtil: cartesian_pow, bernoulli

const SPBASIS = Bases.Pairing{4}
const REFSTATE = RefStates.Fermi(Bases.Pairing{4}, 2)
const MBBASIS = Bases.Paired{2, 4}
#const ELTYPE = Float64
const ARRAYOP{N} = F64ArrayOperator{Bases.Product{N, NTuple{N, SPBASIS}}}
const FUNCOP{N} = F64FunctionOperator{Bases.Product{N, NTuple{N, SPBASIS}}}
const MBFUNCOP = F64FunctionOperator{MBBASIS}

const LEVEL_SPACING = 1.0

const TwoBodyARRAYOP = Tuple{ARRAYOP{0}, ARRAYOP{1}, ARRAYOP{2}}

isocc(x) = ManyBody.isocc(REFSTATE, x)
isunocc(x) = ManyBody.isocc(REFSTATE, x)
normord(x) = ManyBody.normord(REFSTATE, x)

nbody(A, n) = A[n-1]

comm2(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) =
    (_comm0(BT, A, B), _comm1(A, B), _comm2(A, B))

_comm0(A, B) = fill(_comm0_2_2(nbody(A, 2), nbody(B, 2)))
function _comm1(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    FUNCOP{1}() do i, j
        _comm1_1_2(A1, B2, i, j) + _comm1_2_2(A2, B2, i, j) - _comm1_1_2(B1, A2, i, j)
    end |> tabulate
end
function _comm2(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    FUNCOP{2}() do I, J
        _comm2_1_2(A1, B2, I, J) + _comm2_2_2(A2, B2, I, J) + _comm2_1_2(B1, A2, I, J)
    end |> tabulate
end


_comm0_2_2(A, B) = 4 \ sum(matrixiter(A)) do X
    I, J = X

    isocc(I)*isunocc(J)*(A[I, J]*B[J, I] - B[I, J]*A[J, I])
end

_comm1_1_2(A, B, i, j) = sum(matrixiter(A)) do X
    a, b = X

    (isocc(a) - isocc(b))*A[a, b]*B[b, i, a, j]
end
_comm1_2_2(A, B, i, j) = 2 \ sum(cartesian_pow(SPBASIS, Val{3})) do I
    a, b, c = I
    isunocc(a)*isunocc(b)*isocc(c) + isocc(a)*isocc(b)*isunocc(c) #=
 =# * (A[c, i, a, b]*B[a, b, c, j] - B[c, i, a, b]*A[a, b, c, j])
end

function _comm2_1_2(A, B, I, J)
    i, j = I; k, l = J
    4 \ sum(SPBASIS) do a
        A[i, a]*B[a, j, k, l] - A[j, a]*B[a, i, k, l] #=
     =# - A[a, k]*B[i, j, a, l] + A[a, l]*B[i, j, a, k]
    end
end
function _comm2_2_2(A, B, I, J)
    i, j = I; k, l = J
    4 \ sum(cartesian_pow(SPBASIS, Val{2})) do I
       a, b = I
       2 \ (1-isocc(a)-isocc(b))*(A[i, j, a, b]*B[a, b, k, l] - B[i, j, a, b]*A[a, b, k, l]) #=
    =# + (isocc(a)-isocc(b))*(A[a, i, b, k]*B[b, j, a, l] - A[a, j, b, k]*B[b, i, a, l] #=
                           =# - A[a, i, b, l]*B[a, j, a, k] + A[a, j, b, l]*B[b, i, a, k])
    end
end

function white(Ω::TwoBodyARRAYOP, H0::TwoBodyARRAYOP)
    E0, f, Γ = H(Ω, H0)

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    _b1(i, j) = isunocc(i)*isocc(j)*f[i, j] / Δ(i, j)
    _b2(I, J) = 4 \ isunocc(I)*isocc(J) * Γ[I, J] / Δ(I..., J...)

    b1 = FUNCOP{1}() do I, J
        _b1(I..., J...) - conj(_b1(J..., I...))
    end
    b2 = FUNCOP{2}() do I, J
        _b2(I..., J...) - conj(_b2(J..., I...))
    end

    (zero(E0), tabulate(b1), tabulate(b2))
end

const comm = comm2
const generator = white

function dΩ(Ω, params, s)
    H0, tol, batchsize = params
    prev_tot = (0.0, zero(ARRAYOP{1}), zero(ARRAYOP{2}))
    prev_ad = generator(Ω, H0)

    # Do first term
    tot = bernoulli(Float64, 0)/factorial(Float64(0)) .* prev_ad

    n = 1
    while sum(vecnorm.(tot - prev_tot)) tol
        prev_tot = tot
        tot += sum(n:n+batchsize-1) do i
            bernoulli(Float64, i)/factorial(Float64(i)) .* (prev_ad = comm(Ω, prev_ad))
        end
        n += batchsize
    end

    tot
end

function im_pairingH(g)
    E0 = sum(occ(REFSTATE)) do i
        LEVEL_SPACING*(level(i) - 1) - g/4*isocc(flipspin(i))
    end
    f = FUNCOP{1}() do p, q
        p != q && return 0
        2 \ (LEVEL_SPACING*(level(p)-1) - isocc(p)*g/2)
    end
    Γ = FUNCOP{2}() do I, J
        sgn1 = index(I) |> Tuple |> collect |> sortperm |> levicivita
        sgn2 = index(J) |> Tuple |> collect |> sortperm |> levicivita
        X = Bases.Slater{SPBASIS}(I...)
        Y = Bases.Slater{SPBASIS}(J...)
        g/2/4*sgn1*sgn2 * sum(cartesian_pow(1:nlevels(SPBASIS), Val{2})) do Z
            p, q, r, s = SPBASIS.((Z[1], Z[1], Z[2], Z[2]),
                                  (SPINUP, SPINDOWN, SPINUP, SPINDOWN))
            X'Operators.A(p', q', s, r)(Y)
        end
    end

    (E0, f, Γ)
end

function to_mbop(op)
    E0, f, Γ = op

    MBFUNCOP() do Y, X
        b0 = E0*(Y'X)
        b1 = sum(matrixiter(f)) do Z
            p, q = Z[1][1], Z[2][1]
            NA = normord(Operators.A(p', q))
            f[p, q]*(Y'NA(X))
        end
        b2 = sum(matrixiter(Γ)) do Z
            I, J = Z
            p, q = I; r, s = J
            NA = normord(Operators.A(p', q', r, s))
            Γ[I, J]*(Y'NA(X))
        end

        b0 + b1 + b2
    end
end

const ALG = Euler()
function solve(h)
end

end # module MagnusIMSRG

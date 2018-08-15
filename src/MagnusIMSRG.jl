module MagnusIMSRG

using ManyBody
using OrdinaryDiffEq
using JuliaUtil: cartesian_pow, bernoulli

const SPBASIS = Bases.Index{Bases.Pairing{4}}
const REFSTATE = RefStates.Fermi{2, Bases.Pairing{4}}
const MBBASIS = Bases.Paired{2, 4}
const ELTYPE = Float64
const ARRAYOP{N} = F64ArrayOperator{N, SPBASIS}
const FUNCOP{N} = F64FunctionOperator{N, SPBASIS}
const MBFUNCOP = F64FunctionOperator{1, MBBASIS}

const LEVEL_SPACING = 1.0

const TwoBodyARRAYOP = Tuple{ARRAYOP{0}, ARRAYOP{1}, ARRAYOP{2}}

isocc(x) = ManyBody.isocc(REFSTATE, x)

nbody(A, n) = A[n-1]
nbodyrep(A, n) = rep(nbody(A, n))

comm2(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) =
    (_comm0(BT, A, B), tabulate(_comm1(A, B)), tabulate(_comm2(A, B)))
_comm0(A, B) = ARRAYOP{0}(fill(_comm0_2_2(nbodyrep(A, 2), nbodyrep(B, 2))))
_comm1(A, B) = FUNCOP{1}() do i, j
    _comm1_1_2(nbodyrep(A, 1), nbodyrep(B, 2), i, j) #=
 =# + _comm1_2_2(nbodyrep(A, 2), nbodyrep(B, 2), i, j) #=
 =# - _comm1_1_2(nbodyrep(B, 1), nbodyrep(A, 2), i, j)
end
_comm2(A, B) = FUNCOP{2}() do i, j, k, l
    _comm2_1_2(nbodyrep(A, 1), nbodyrep(B, 2), i, j, k, l) #=
 =# + _comm2_2_2(nbodyrep(A, 2), nbodyrep(B, 2), i, j, k, l) #=
 =# + _comm2_1_2(nbodyrep(B, 1), nbodyrep(A, 2), i, j, k, l)
end

_comm0_2_2(A, B) = 4 \ sum(cartesian_pow(SPBASIS, Val{4})) do I
    a, b, c, d = I
    isocc(a)*isocc*(b)*(~isocc(c))*(~isocc(d)) #=
 =# * (A[a, b, c, d]*B[c, d, a, b] - B[a, b, c, d]*A[c, d, a, b])
end

_comm1_1_2(A, B, i, j) = sum(cartesian_pow(SPBASIS, Val{2})) do I
    a, b = I
    (isocc(a) - isocc(b))*A[a, b]*B[b, i, a, j]
end

_comm1_2_2(A, B, i, j) = 2 \ sum(cartesian_pow(SPBASIS, Val{3})) do I
    a, b, c = I
    (~isocc(a))*(~isocc(b))*isocc(c) + isocc(a)*isocc(b)*(~isocc(c)) #=
 =# * (A[c, i, a, b]*B[a, b, c, j] - B[c, i, a, b]*A[a, b, c, j])
end

_comm2_1_2(A, B, i, j, k, l) = 4 \ sum(SPBASIS) do a
    A[i, a]*B[a, j, k, l] - A[j, a]*B[a, i, k, l] #=
 =# - A[a, k]*B[i, j, a, l] + A[a, l]*B[i, j, a, k]
end

_comm2_2_2(A, B, i, j, k, l) = 4 \ sum(cartesian_pow(SPBASIS, Val{2})) do I
    a, b = I
    2 \ (1-isocc(a)-isocc(b)) * (A[i, j, a, b]*B[a, b, k, l] - B[i, j, a, b]*A[a, b, k, l]) #=
 =# + (isocc(a)-isocc(b))*(A[a, i, b, k]*B[b, j, a, l] - A[a, j, b, k]*B[b, i, a, l] #=
                        =# - A[a, i, b, l]*B[a, j, a, k] + A[a, j, b, l]*B[b, i, a, k])
end

function white(Ω::TwoBodyARRAYOP, H0::TwoBodyARRAYOP)
    _, f, Γ = H(Ω, H0)

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    _b1(i, j) = ~(isocc(i))*isocc(j)*f[i, j] / Δ(i, j)
    _b2(i, j, k, l) = 4 \ (~isocc(i))*(~isocc(j))*isocc(k)*isocc(l) #=
                       =# * Γ[i, j, k, l] / Δ[i, j, k, l]

    b1 = FUNCOP{1}() do i, j
        _b1(i, j) - conj(_b1(j, i))
    end
    b2 = FUNCOP{2}() do i, j, k, l
        _b2(i, j, k, l) - conj(_b2(l, k, j, i))
    end

    (zero(ARRAYOP{0}), tabulate(b1), tabulate(b2))
end

const comm = comm2
const generator = white

function dΩ(Ω, params, s)
    H0, tol, batchsize = params
    prev_tot = (0.0im, zero(ARRAYOP{1}), zero(ARRAYOP{2}))
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
    E0_val = sum(occ(REFSTATE)) do i
        LEVEL_SPACING*(level(i) - 1) - g/4*isocc(flipspin(i))
    end
    E0 = FUNCOP{0}(() -> E0_val)
    f = FUNCOP{1}() do p, q
        p != q && return 0
        2 \ (LEVEL_SPACING*(level(p)-1) - isocc(p)*g/2)
    end
    Γ = FUNCOP{2}() do p, q, r, s
        mask = (level(p) == level(r)) #=
            =# * (level(q) == level(s)) #=
            =# * spinup(p)*spinup(r) #=
            =# * spindown(q)*spindown(s)

        -g/2*mask
    end

    (E0, f, Γ)
end

function to_mbop(op)
    E0, f, Γ = op

    MBFUNCOP() do Y, X
        b0 = E0[]*(Y'X)
        b1 = sum(cartesian_pow(SPBASIS, Val{2})) do I
            p, q = I
            sgn, NA = normord(A(p', q))
            f[p, q]*sgn*(Y'NA(X))
        end
        b2 = sum(cartesian_pow(SPBASIS, Val{4})) do I
            p, q, r, s = I
            sgn, NA = normord(A(p', q))
            val3 = NA(X)
            val = Γ[p, q, r, s]
            val2 = val*sgn*(Y'val3)
            if !iszero(val)
                @show inner(Y)'inner(val3)
#                print(map(inner, I), " --> ")
#                println(sgn, ", ", val, ", ", val2)
            end
            val2
        end
        println()

        b0 + b1 + b2
    end
end

const ALG = Euler()
function solve(h)
end

end # module MagnusIMSRG

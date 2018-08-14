module MagnusIMSRG

using ManyBody
using Loppy.Util: cartesian_pow, bernoulli

const SPBASIS = Bases.Index{Bases.Pairing{4}}
const REFSTATE = RefStates.Fermi{2, Bases.Pairing{4}}
const MBBASIS = Bases.Paired{2, 4}

const CArrOperator{N} = Operator{N, SPBASIS, <:Array{Complex}}

isocc(x) = ManyBody.isocc(REFSTATE, x)

nbody(A, n) = A[n-1]

## How to structure this? We want tabulate() to propragate through the calculation after it is
## called *once*.
comm2(A, B) = (_comm0(A, B), _comm1(A, B),  _comm2(A, B))
_comm0(A, B) = _comm0_2_2(nbody(A, 2), nbody(B, 2))
_comm1(A, B, i, j) =
    _comm1_1_2(nbody(A, 1), nbody(B, 2), i, j) + _comm1_2_2(nbody(A, 2), nbody(B, 2), i, j) #=
 =# - _comm1_1_2(nbody(B, 1), nbody(A, 2), i, j)
_comm2(A, B, i, j, k, l) =
    _comm2_1_2(nbody(A, 1), nbody(B, 2), i, j, k, l) #=
 =# + _comm2_2_2(nbody(A, 2), nbody(B, 2), i, j, k, l) #=
 =# + _comm2_1_2(nbody(B, 1), nbody(A, 2), i, j, k, l)

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

function white(Ω, H0)
    _, f, Γ = H(Ω, H0)

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    _b1(i, j) = ~(isocc(i))*isocc(j)*f[i, j] / Δ(i, j)
    _b2(i, j, k, l) = 4 \ (~isocc(i))*(~isocc(j))*isocc(k)*isocc(l) #=
                       =# * Γ[i, j, k, l] / Δ[i, j, k, l]

    b1 = @Operator(SPBASIS) do i, j
        _b1(i, j) - conj(_b1(j, i))
    end
    b2 = @Operator(SPBASIS) do i, j, k, l
        _b2(i, j, k, l) - conj(_b2(l, k, j, i))
    end

    (0, b1, b2)
end

const comm = comm2
const generator = white

# Add methods for Operator addition, etc.
function dΩ(Ω, params, s)
    H0, tol, batchsize = params
    tot = (0.0im, zero(CArrayOperator{1}), zero(CArrayOperator{2}))
    prev_ad = generator(Ω, H0) .|> tabulate

    # Do first term

    n = 1
    while any(vecnorm.(tot - prev_tot) .> tol)
        prev_tot = tot
        tot += sum(n:n+batchsize-1) do i
            bernoulli(Float64, i)/factorial(Float64(i))*(prev_ad = comm(Ω, prev_ad))
        end
        n += batchsize
    end

    tot
end

end # module MagnusIMSRG

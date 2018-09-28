module MagnusIMSRG

using ManyBody
import ManyBody.Operators
import LinearAlgebra
import JuliaUtil
#using OrdinaryDiffEq
using Base.Cartesian: @nloops
using Combinatorics: levicivita, combinations
using JuliaUtil: bernoulli, @every

#### Workaround b/c no internet :(
#JuliaUtil.fbinom(a, b) =
#    factorial(Float64, a) / (factorial(Float64, b)*factorial(Float64, a-b))

### Parameters ###############################################################################
const Ω_RTOL = 0.1
const Ω_ATOL = 0.0
const Ω_BATCHSIZE = 5
const H_RTOL = 0.1
const H_ATOL = 0.0
const INT_RTOL = 0.1
const INT_ATOL = 0.0
const H_BATCHSIZE = 5
const S_BIG_STEP = 1.0
const S_SMALL_STEP = 0.1
const MAX_INT_ITERS = 100
##############################################################################################

const SPBASIS = Bases.Pairing{4}
const IXBASIS = indextype(SPBASIS)
const REFSTATE = RefStates.Fermi(Bases.Pairing{4}, 2)
const MBBASIS = Bases.Paired{2, 4}
const ELTYPE = Float64
const ARRAYOP(N) = F64ArrayOperator{Bases.Product{N, NTuple{N, SPBASIS}}, 2N}
const FUNCOP(N) = F64FunctionOperator{Bases.Product{N, NTuple{N, SPBASIS}}}
const MBFUNCOP = F64FunctionOperator{MBBASIS}

const DIM = dim(SPBASIS)
const LEVEL_SPACING = 1.0
const FERMILEVEL = fermilevel(REFSTATE)
const ZERO_OP = (zero(ELTYPE), zero(ARRAYOP(1)), zero(ARRAYOP(2)))

const TwoBodyARRAYOP = Tuple{ELTYPE, ARRAYOP(1), ARRAYOP(2)}

isocc(x) = ManyBody.isocc(REFSTATE, x)
isunocc(x) = ManyBody.isocc(REFSTATE, x)
normord(x) = ManyBody.normord(REFSTATE, x)

nbody(A, n) = A[n+1]

comm2(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) =
    (_comm0(A, B), _comm1(A, B), _comm2(A, B))

_comm0(A, B) = _comm0_2_2(nbody(A, 2), nbody(B, 2))
function _comm1(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    FUNCOP(1)() do i, j
        i, j = inner(i), inner(j)
        _comm1_1_2(A1, B2, i, j) + _comm1_2_2(A2, B2, i, j) - _comm1_1_2(B1, A2, i, j)
    end |> tabulate
end
function _comm2(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    ARRAYOP(2)(_comm2_1_2(A1, B2) + _comm2_2_2(A2, B2) + _comm2_1_2(B1, A2))
end

_comm0_2_2(A, B) = 4 \ sum(matrixiter(A)) do X
    I, J = X

    isocc(I)*isunocc(J)*(A[I, J]*B[J, I] - B[I, J]*A[J, I])
    ## tr(A*B)
end

let MASK(a, b) = isocc(a) - isocc(b),
    DENSITY_MAT = (b=basis(SPBASIS); MASK.(b, b))
global _comm1_1_2
function _comm1_1_2(A, B)
    A′ = copy(reshape(DENSITY_MAT .* A.rep, DIM^2))
    B′ = copy(reshape(PermutedDimsArray(B.rep, [1, 3, 2, 4]), DIM, DIM^3))

    A'*B'
end end
### Pointwise version
#_comm1_1_2(A, B, i, j) = sum(matrixiter(A)) do X
#    a, b = map(inner, X)
#
#    (isocc(a) - isocc(b))*A[a, b]*B[b, i, a, j]
#end

function _comm1_2_2(A, B, i, j)
    tot = zero(ELTYPE)
    @nloops 3 a (_ -> SPBASIS) begin
        tot += isunocc(a_1)*isunocc(a_2)*isocc(a_3) + isocc(a_1)*isocc(a_2)*isunocc(a_3) #=
            =# * (A[a_3, i, a_1, a_2]*B[a_1, a_2, a_3, j] #=
               =# - B[a_3, i, a_1, a_2]*A[a_1, a_2, a_3, j])
    end

    tot / 2
end

function _comm2_1_2(A, B)
    B′ = reshape(B.rep, DIM, DIM^3)
    C = A.rep * B′

    A′ = permutedims(A.rep, [2, 1])
    B′ .= reshape(PermutedDimsArray(B.rep, [3, 2, 1, 4]), DIM, DIM^3)
    D = A' * B′

    ret = Array{ELTYPE}(undef, DIM, DIM, DIM, DIM)

    for I in CartesianIndices(ret)
        i, j, k, l = Tuple(I)
        LI = LinearIndices((DIM, DIM, DIM))

        ret[I] = 4 \ (C[i, LI[i, j, k]] - C[j, LI[i, k, l]]
                      - D[k, LI[i, j, l]] + D[l, LI[j, i, k]])
    end

    ret

## Pointwise version
#    i, j = I; k, l = J
#    4 \ sum(SPBASIS) do a
#        A[i, a]*B[a, j, k, l] - A[j, a]*B[a, i, k, l] #=
#     =# - A[a, k]*B[i, j, a, l] + A[a, l]*B[i, j, a, k]
#    end
end

let MASK1(a, b) = 1 - isocc(a) - isocc(b),
    MASK2(a, b) = isocc(a) - isocc(b),
    DENSITY_MAT_1 = (b=basis(SPBASIS); MASK1.(b, b)),
    DENSITY_MAT_2 = (b=basis(SPBASIS); MASK2.(b, b))
global _comm2_2_2
function _comm2_2_2(A, B)
    to_mat(x) = reshape(x, DIM^2, DIM^2)

    A′ = Array{ELTYPE}(undef, DIM^2, DIM^2)
    B′ = Array{ELTYPE}(undef, DIM^2, DIM^2)

    A′ .= to_mat(A.rep)
    B′ .= to_mat(B.rep .* DENSITY_MAT_1)
    C′ = A′*B′

    A′ .= to_mat(A.rep .* DENSITY_MAT_1)
    B′ .= to_mat(B.rep)
    C′ .-= B′*A′
    
    C′ ./= 2

    A′ .= PermutedDimsArray(A.rep, [2, 4, 1, 3]) |> to_mat
    B′ .= PermutedDimsArray(B.rep, [3, 1, 2, 4]).*DENSITY_MAT_2 |> to_mat
    D′ = A′*B′

    A′ .= PermutedDimsArray(A.rep, [1, 3, 2, 4]).*DENSITY_MAT_2 |> to_mat
    B′ .= PermutedDimsArray(B.rep, [3, 1, 2, 4]) |> to_mat
    D′ .+= B′*A′

    ret = reshape(C′, DIM, DIM, DIM, DIM)

    for I in CartesianIndices(ret)
        a, b, c, d = Tuple(I)
        LI = LinearIndices((DIM, DIM))

        ret[a, b, c, d] = (D′[LI[a, c], LI[b, d]] - D′[LI[a, d], LI[c, b]]) / 4
    end

    ret

## Pointwise version
#    i, j = I; k, l = J
#
#    tot = zero(ELTYPE)
#    @nloops 2 a (_ -> SPBASIS) begin
#       tot += 2 \ (1-isocc(a_1)-isocc(a_2))*(A[i, j, a_1, a_2]*B[a_1, a_2, k, l] #=
#                =# - B[i, j, a_1, a_2]*A[a_1, a_2, k, l]) #=
#           =# + (isocc(a_1)-isocc(a_2))*(A[a_1, i, a_2, k]*B[a_2, j, a_1, l] #=
#                                      =# - A[a_1, j, a_2, k]*B[a_2, i, a_1, l] #=
#                                                            # \___/ typo?
#                                      =# - A[a_1, i, a_2, l]*B[a_1, j, a_1, k] #=
#                                      =# + A[a_1, j, a_2, l]*B[a_2, i, a_1, k])
#    end
#
#    tot / 4
end end

function white(Ω::TwoBodyARRAYOP, h0::TwoBodyARRAYOP)
    E0, f, Γ = H(Ω, h0)

    Δ(i, k) = f[i, i] - f[k, k] + Γ[i, k, i, k]
    Δ(i, j, k, l) =
        f[i, i] + f[j, j] - f[k, k] - f[l, l] + Γ[k, l, k, l] + Γ[i, j, i, j] #=
     =# - Γ[i, k, i, k] - Γ[j, l, j, l] - Γ[i, l, i, l] - Γ[j, k, j, k]

    function _b1(i, j)
        x = isunocc(i)*isocc(j)*f[i, j]
        d = Δ(i, j)
        iszero(d) ? d : x/d
    end
    function _b2(I, J)
        x = 4 \ isunocc(I)*isocc(J) * Γ[I, J]
        d = Δ(I..., J...)
        iszero(d) ? d : x/d
    end

    b1 = FUNCOP(1)() do I, J
        i, j = inner(I), inner(J)
        _b1(i, j) - conj(_b1(j, i))
    end
    b2 = FUNCOP(2)() do I, J
        _b2(I, J) - conj(_b2(J, I))
    end

    (zero(E0), tabulate(b1), tabulate(b2))
end

const comm = comm2
const generator = white

Base.:+(a::TwoBodyARRAYOP, b::TwoBodyARRAYOP) = a .+ b
Base.:*(a::Number, b::TwoBodyARRAYOP) = a .* b
Base.:*(a::TwoBodyARRAYOP, b::Number) = a .* b
Base.:/(a::TwoBodyARRAYOP, b::Number) = a ./ b
Base.:\(a::Number, b::TwoBodyARRAYOP) = a .\ b
Base.:-(a::TwoBodyARRAYOP) = .-a
Base.:+(a::TwoBodyARRAYOP) = a
## Workaround so we can use isapprox()
Base.:-(a::TwoBodyARRAYOP, b::TwoBodyARRAYOP) = a .- b
function norm(op)
    E0, f, Γ = op
    abs(E0) + LinearAlgebra.norm(f.rep) + LinearAlgebra.norm(Γ.rep)
end

factorial(T::Type{<:Number}, n::Integer) = prod(one(T):convert(T, n))

function dΩ(Ω, h0)
    @info "dΩ term" n=0
    prev_tot = ZERO_OP
    prev_ad = generator(Ω, h0)
    tot = bernoulli(Float64, 0)/factorial(Float64, 0) * prev_ad

    n = 1
    while norm(tot - prev_tot) > max(Ω_ATOL, Ω_RTOL*norm(tot))
        @info "dΩ term" n
        prev_tot = tot
        tot += sum(n:n+Ω_BATCHSIZE-1) do i
            bernoulli(Float64, i)/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += Ω_BATCHSIZE
    end

    tot
end

function im_pairingH(g)
    E0 = 2sum(occ(REFSTATE)) do i
        LEVEL_SPACING*(level(i) - 1)
    end - g/2*FERMILEVEL^2

    f = FUNCOP(1)() do p, q
        p, q = inner(p), inner(q)
        (p == q)*(LEVEL_SPACING*(level(p)-1) - g/2*FERMILEVEL)
    end

    Γ = FUNCOP(2)() do I, J
        p, q = I; r, s = J
        mask = (level(p) == level(q))*(level(r) == level(s)) #=
            =# * spinup(p)*spinup(r)*spindown(q)*spindown(s)

        -g/2*mask
    end

    (E0, tabulate(f), tabulate(Γ))
end

function to_mbop(op)
    E0, f, Γ = op

    MBFUNCOP() do X, Y
        b0 = E0*(X'Y)
        b1 = sum(matrixiter(f)) do Z
            p, q = Z[1][1], Z[2][1]
            NA = normord(Operators.A(p', q))
            f[p, q]*(X'NA(Y))
        end
        b2 = sum(matrixiter(Γ)) do Z
            I, J = Z
            p, q = I; r, s = J
            NA = normord(Operators.A(p', q', s, r))
            Γ[I, J]*(X'NA(Y))
        end

        b0 + b1 + b2
    end
end

function H(Ω, h0)
    prev_tot = ZERO_OP
    prev_ad = comm(Ω, h0)

    # First term
    tot = 1/factorial(Float64, 0) * prev_ad

    n = 1
    while norm(tot - prev_tot) > max(H_ATOL, H_RTOL*norm(tot))
        prev_tot = tot
        tot += sum(n:n+H_BATCHSIZE-1) do i
            1/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += H_BATCHSIZE
    end

    tot
end

function solve(h0; max_int_iters=MAX_INT_ITERS)
    s = 0.0
    n = 0
    Ω = ZERO_OP
    h_prev = ZERO_OP
    h = h0

    while norm(h - h_prev) > max(INT_ATOL, INT_RTOL*norm(h))
        @info "Integration iter" n
        if n >= max_int_iters
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end

        h_prev = h
        Ω += dΩ(Ω, h0) * S_SMALL_STEP
        s += S_SMALL_STEP
        h = H(Ω, h0)
        n += 1
    end

    h
end

end # module MagnusIMSRG

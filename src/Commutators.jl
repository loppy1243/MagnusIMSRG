module Commutators
using ManyBody, ..IMOperators
using Base.Cartesian

using LinearAlgebra: tr
using Parameters: @unpack
import ..SIGNAL_OPS, ..@getparams

SIGNAL_OPS && include("signalops.jl")

@getparams DIM, SPBASIS, ELTYPE, REFSTATE

isocc(a) = ManyBody.isocc(REFSTATE, a)
isunocc(a) = ManyBody.isunocc(REFSTATE, a)

### Occupation Matrices ######################################################################
# hole-hole or -(particle-particle)
const OMAT_HmH = [isocc(a)-isocc(b) for a in SPBASIS, b in SPBASIS]
const OMAT_HPmHP = [isocc(a)*isunocc(b) - isocc(b)*isunocc(a)
                     for a in SPBASIS, b in SPBASIS]
# particle*particle or -(hole*hole)
const OMAT_1mHmH = [1-isocc(a)-isocc(b) for a in SPBASIS, b in SPBASIS]
# particle*particle*hole + hole*hole*particle
const OMAT_PPHpHHP = [isunocc(a)*isunocc(b)*isocc(c) + isocc(a)*isocc(b)*isunocc(c)
                      for a in SPBASIS, b in SPBASIS, c in SPBASIS]
# hole*hole*particle*particle
const OMAT_HHPP = [isocc(a)*isocc(b)*isunocc(c)*isunocc(d)
                   for a in SPBASIS, b in SPBASIS, c in SPBASIS, d in SPBASIS]
##############################################################################################
### In this file....
#   Primed variables -- different view on same data
#   Greek variables  -- copy

cartesian_sum(f, N, itr) = cartesian_sum(f, Val{N}, itr)
@generated cartesian_sum(f, ::Type{Val{N}}, itr) where N = quote
    fst = first(itr)
    ret = zero(@ncall($N, f, _->fst))
    @nloops $N x _->itr begin
        ret += @ncall($N, f, x)
    end
    ret
end

function comm2(A::IMArrayOp{2}, B::IMArrayOp{2})
    matrix(x) = reshape(x, DIM, DIM)
    tensor(x) = reshape(x, DIM, DIM, DIM, DIM)

    A0, A1, A2 = A.parts; B0, B1, B2 = B.parts
    orig_size1 = promote_shape(A1, B1); orig_size2 = promote_shape(A2, B2)

    A1 = matrix(A1); A2 = tensor(A2)
    B1 = matrix(B1); B2 = tensor(B2)
    args = (A1, A2, B1, B2)

    C0 = _comm0(args...); C1 = _comm1(args...); C2 = _comm2(args...)

    IMArrayOp(C0, reshape(C1, orig_size1), reshape(C2, orig_size2))
end
function comm2_pw(A::IMArrayOp{2}, B::IMArrayOp{2})
    A0, A1, A2 = A.parts; B0, B1, B2 = B.parts
    args = (A1, A2, B1, B2)

    IMArrayOp(_comm0_pw(args...), _comm1_pw(args...), _comm2_pw(args...))
end

_comm0(A1, A2, B1, B2) = _comm0_1_1(A1, B1) + _comm0_2_2(A2, B2)
_comm0_pw(A1, A2, B1, B2) = _comm0_1_1_pw(A1, B1) + _comm0_2_2_pw(A2, B2)

_comm1(A1, A2, B1, B2) =
    _comm1_1_1(A1, B1) + _comm1_1_2(A1, B2) + _comm1_2_2(A2, B2) - _comm1_1_2(B1, A2)
_comm1_pw(A1, A2, B1, B2) = tabulate(Array{ELTYPE}, 2, SPBASIS) do i, j
    _comm1_1_1_pw(A1, B1, i, j) + _comm1_1_2_pw(A1, B2, i, j) #=
    =#+ _comm1_2_2_pw(A2, B2, i, j) - _comm1_1_2_pw(B1, A2, i, j)
end

_comm2(A1, A2, B1, B2) = _comm2_1_2(A1, B2) + _comm2_2_2(A2, B2) - _comm2_1_2(B1, A2)
_comm2_pw(A1, A2, B1, B2) = tabulate(Array{ELTYPE}, 4, SPBASIS) do i, j, k, l
    _comm2_1_2_pw(A1, B2, i, j, k, l) + _comm2_2_2_pw(A2, B2, i, j, k, l) #=
    =#- _comm2_1_2_pw(B1, A2, i, j, k, l)
end

function _comm0_1_1(A, B)
    matrix(x) = reshape(x, DIM, DIM)

    α = matrix(A.*transpose(OMAT_HPmHP))
    B′ = matrix(B)

    tr(α*B′)
end
_comm0_1_1_pw(A, B) = cartesian_sum(2, SPBASIS) do i, j
    isocc(i)*isunocc(j)*(A[i, j]*B[j, i] - B[i, j]*A[j, i])
end

function _comm0_2_2(A, B)
    matrix(x) = reshape(x, DIM^2, DIM^2)

    α = matrix(A.*OMAT_HHPP)
    B′ = matrix(B)
    C = α*B′ / 4

    A′ = matrix(A)
    β = matrix(B.*OMAT_HHPP)
    C .-= β*A′ ./ 4

    tr(C)
end

_comm0_2_2_pw(A, B) = 4 \ cartesian_sum(4, SPBASIS) do i, j, k, l
    isocc(i)*isocc(j)*isunocc(k)*isunocc(l) #=
    =#* (A[i, j, k, l]*B[k, l, i, j] - B[i, j, k, l]*A[k, l, i, j])
end

_comm1_1_1(A, B) = A*B - B*A
_comm1_1_1_pw(A, B, i, j) = sum(SPBASIS) do a
    A[i, a]*B[a, j] - B[i, a]*A[a, j]
end

function _comm1_1_2(A, B)
    vector(x) = reshape(x, DIM^2)
    matrix22(x) = reshape(x, DIM^2, DIM^2)
    matrix11(x) = reshape(x, DIM, DIM)

    α = vector(A.*OMAT_HPmHP)

    # [b, i, a, j] -> [i, j, a, b]
    B′ = matrix22(PermutedDimsArray(B, [2, 4, 3, 1]))
    # [i, j, a, b]*[a, b] = [i, j]
    matrix11(B′*α)
end
_comm1_1_2_pw(A, B, i, j) = cartesian_sum(2, SPBASIS) do a, b
    isocc(a)*isunocc(b)*(A[a, b]*B[b, i, a, j] - A[b, a]*B[a, i, b, j])
end

function _comm1_2_2(A, B)
    matrix13(x) = reshape(x, DIM, DIM^3)
    matrix31(x) = reshape(x, DIM^3, DIM)

    # [c, i, a, b] -> [i, a, b, c]
    A′ = matrix13(PermutedDimsArray(A, [2, 3, 4, 1]))
    β = matrix31(B.*OMAT_PPHpHHP)
    # [i, a, b, c]*[a, b, c, j] = [i, j]
    C = A′*β

    α = β
    α .= matrix31(A.*OMAT_PPHpHHP)
    # [c, i, a, b] -> [i, a, b, c]
    B′ = matrix13(PermutedDimsArray(B, [2, 3, 4, 1]))
    # [i, a, b, c]*[a, b, c, j] = [i, j]
    C .-= B′*α

    C ./= 2
end
_comm1_2_2_pw(A, B, i, j) = 2 \ cartesian_sum(3, SPBASIS) do a, b, c
    (isunocc(a)*isunocc(b)*isocc(c) + isocc(a)*isocc(b)*isunocc(c)) #=
    =#* (A[c, i, a, b]*B[a, b, c, j] - B[c, i, a, b]*A[a, b, c, j])
end

function _comm2_1_2(A, B)
    matrix13(x) = reshape(x, DIM, DIM^3)
    matrix31(x) = reshape(x, DIM^3, DIM)
    tensor(x) = reshape(x, DIM, DIM, DIM, DIM)

    B′ = matrix13(B)
    # [i, a]*[a, j, k, l] = [i, j, k, l]
    C = A*B′

    # [i, j, a, l] -> [i, j, l, a]
    B′ = matrix31(PermutedDimsArray(B, [1, 2, 4, 3]))
    # [i, j, l, a]*[a, k] = [i, j, l, k]
    D = B′*A

    C′ = tensor(C); D′ = tensor(D)
    ret = similar(C′)
    for I in CartesianIndices(ret)
        i, j, k, l = Tuple(I)

        #           P()              P(i,j)           P()              P(k,l)
        ret[I] = C′[i, j, k, l] - C′[j, i, k, l] - D′[i, j, l, k] + D′[i, j, k, l]
    end

    ret

end
_comm2_1_2_pw(A, B, i, j, k, l) = sum(SPBASIS) do a
    prod1(i, j, k, l) = A[i, a]*B[a, j, k, l]
    prod2(i, j, k, l) = A[a, k]*B[i, j, a, l]

    prod1(i, j, k, l) - prod1(j, i, k, l) - prod2(i, j, k, l) + prod2(i, j, l, k)
end

function _comm2_2_2(A, B)
    matrix(x) = reshape(x, DIM^2, DIM^2)
    tensor(x) = reshape(x, DIM, DIM, DIM, DIM)

    A′ = matrix(A)
    β = matrix(B.*OMAT_1mHmH)
    C = A′*β

    α = matrix(A.*OMAT_1mHmH)
    B′ = matrix(B)
    C .-= B′*α
    C ./= 2

    # [i, b, a, l] -> [a, b, i, l]
    α .= matrix(PermutedDimsArray(A, [3, 2, 1, 4].*OMAT_HmH))
    # [a, j, k, b] -> [j, k, a, b]
    β .= matrix(PermutedDimsArray(B, [2, 3, 1, 4]))
    # [j, k, a, b]*[a, b, i, l] = [j, k, i, l]
    D = β*α

    C′ = tensor(C); D′ = tensor(D)
    for I in CartesianIndices(C′)
        i, j, k, l = Tuple(I)

        #           P()              P(i,j)           P(k,l)           P(i,j)P(k,l)
        C′[I] += D′[j, k, i, l] - D′[i, k, j, l] - D′[j, l, i, k] + D′[i, l, j, k]
    end

    C′
end
_comm2_2_2_pw(A, B, i, j, k, l) = cartesian_sum(2, SPBASIS) do a, b
    prod3(i, j, k, l) = A[i, b, a, l]*B[a, j, k, b]

    2 \ (1-isocc(a)-isocc(b))*(A[i, j, a, b]*B[a, b, k, l] - B[i, j, a, b]*A[a, b, k, l]) #=
    =#+ (isocc(b)-isocc(a))*(prod3(i, j, k, l) - prod3(j, i, k, l) - prod3(i, j, l, k) #=
         =#+ prod3(j, i, l, k))
end

end # module Commutators

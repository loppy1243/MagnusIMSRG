using Base.Cartesian

### Density Matrices #########################################################################
# hole*particle or -(particle*hole)
const DMAT_HmH = [isocc(a)-isocc(b) for a in SPBASIS, b in SPBASIS]
# particle*particle or -(hole*hole)
const DMAT_1mHmH = [1-isocc(a)-isocc(b) for a in SPBASIS, b in SPBASIS]
# particle*particle*hole + hole*hole*particle
const DMAT_PPHpHHP = [isunocc(a)*isunocc(b)*isocc(c) + isocc(a)*isocc(b)*isunocc(c)
                      for a in SPBASIS, b in SPBASIS, c in SPBASIS]
##############################################################################################

comm2(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) =
    (_comm0(A, B), _comm1(A, B), _comm2(A, B))
comm2_pw(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) =
    (_comm0_pw(A, B), _comm1_pw(A, B), _comm2_pw(A, B))

_comm0(A, B) = _comm0_1_1(nbody(A, 1), nbody(B, 1)) + _comm0_2_2(nbody(A, 2), nbody(B, 2))
_comm0_pw(A, B) =
    _comm0_1_1(nbody(A, 1), nbody(B, 1)) + _comm0_2_2_pw(nbody(A, 2), nbody(B, 2))

function _comm1(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    ARRAYOP(1)(_comm1_1_2(A1, B2) + _comm1_2_2(A2, B2) - _comm1_1_2(B1, A2))
end
function _comm1_pw(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    ret = Array{ELTYPE}(undef, DIM, DIM)

    for i in SPBASIS, j in SPBASIS
        ret[index(i), index(j)] =
            _comm1_1_2_pw(A1, B2, i, j) + _comm1_2_2_pw(A2, B2, i, j) #=
         =# - _comm1_1_2_pw(B1, A2, i, j)
    end

    ARRAYOP(1)(ret)
end

function _comm2(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    ARRAYOP(2)(_comm2_1_2(A1, B2) + _comm2_2_2(A2, B2) - _comm2_1_2(B1, A2))
end
function _comm2_pw(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    ret = Array{ELTYPE}(undef, DIM, DIM, DIM, DIM)
    @nloops 4 i (_ -> SPBASIS) begin
        @nref(4, ret, d -> index(i_d)) =
            @ncall(4, _comm2_1_2_pw, A1, B2, i) + @ncall(4, _comm2_2_2_pw, A2, B2, i) #=
         =# - @ncall(4, _comm2_1_2_pw, B1, A2, i)
    end

    ARRAYOP(2)(ret)
end

_comm0_1_1(A, B) = sum(matrixiter(A)) do X
    i, j = X

    (isocc(i)-isunocc(j))*A[i, j]*B[j, i]
end
const _comm0_1_1_pw = _comm0_1_1

_comm0_2_2(A, B) = 4 \ sum(matrixiter(A)) do X
    I, J = X

    all(isocc, I)*all(isunocc, J)*(A[I, J]*B[J, I] - B[I, J]*A[J, I])
end
const _comm0_2_2_pw = _comm0_2_2

function _comm1_1_2(A, B)
    A′ = reshape(A.rep.*DMAT_HmH, DIM^2)
    B′ = reshape(permutedims(B.rep, [2, 4, 3, 1]), DIM^2, DIM^2)

    reshape(B′*A′, DIM, DIM)
end
_comm1_1_2_pw(A, B, i, j) = sum(matrixiter(A)) do X
    a, b = X

    (isocc(a)-isocc(b))*A[a, b]*B[b, i, a, j]
end

function _comm1_2_2(A, B)
    A′ = reshape(PermutedDimsArray(A.rep, [2, 3, 4, 1]), DIM, DIM^3) |> copy
    B′ = reshape(B.rep.*DMAT_PPHpHHP, DIM^3, DIM)
    C′ = A′*B′

    # NOTE: here we have A -> B′ and B -> A′
    B′ .= reshape(A.rep.*DMAT_PPHpHHP, DIM^3, DIM)
    A′ .= reshape(PermutedDimsArray(B.rep, [2, 3, 4, 1]), DIM, DIM^3)
    C′ .-= A′*B′

    C′ ./= 2

    reshape(C′, DIM, DIM)

end
function _comm1_2_2_pw(A, B, i, j)
    tot = zero(ELTYPE)
    @nloops 3 a (_ -> SPBASIS) begin
        tot += (isunocc(a_1)*isunocc(a_2)*isocc(a_3) + isocc(a_1)*isocc(a_2)*isunocc(a_3)) #=
            =# * (A[a_3, i, a_1, a_2]*B[a_1, a_2, a_3, j] #=
               =# - B[a_3, i, a_1, a_2]*A[a_1, a_2, a_3, j])
    end

    tot / 2
end

function _comm2_1_2(A, B)
    B′ = reshape(B.rep, DIM, DIM^3) |> copy
    C′ = A.rep * B′

#    B′ .= reshape(permutedims(B.rep, [1, 2, 4, 3]), DIM^3, DIM)
#    D′ = B′*A.rep
    B′ .= reshape(PermutedDimsArray(B.rep, [3, 1, 2, 4]), DIM, DIM^3)
    D′ = transpose(B′)*A.rep

    # Type unstable?
    C′ = reshape(C′, DIM, DIM, DIM, DIM)
    D′ = reshape(D′, DIM, DIM, DIM, DIM)

    ret = Array{ELTYPE}(undef, DIM, DIM, DIM, DIM)

    for I in CartesianIndices(ret)
        i, j, k, l = Tuple(I)

        ret[I] = 4 \ (C′[i, j, k, l] - C′[j, i, k, l] - D′[i, j, l, k] + D′[i, j, k, l])
    end

    ret

end
_comm2_1_2_pw(A, B, i, j, k, l) = 4 \ sum(SPBASIS) do a
    prod1(i, j, k, l) = A[i, a]*B[a, j, k, l]
    prod2(i, j, k, l) = A[a, k]*B[i, j, a, l]

    prod1(i, j, k, l) - prod1(j, i, k, l) - prod2(i, j, k, l) + prod2(i, j, l, k)
end

function _comm2_2_2(A, B)
    to_mat(x) = reshape(x, DIM^2, DIM^2)

    A′ = Array{ELTYPE}(undef, DIM^2, DIM^2)
    B′ = Array{ELTYPE}(undef, DIM^2, DIM^2)

    A′ .= to_mat(A.rep)
    B′ .= to_mat(B.rep .* DMAT_1mHmH)
    C′ = A′*B′

    A′ .= to_mat(A.rep .* DMAT_1mHmH)
    B′ .= to_mat(B.rep)
    C′ .-= B′*A′
    C′ ./= 8

    A′ .= PermutedDimsArray(A.rep, [1, 3, 2, 4]).*DMAT_HmH |> to_mat
    B′ .= PermutedDimsArray(B.rep, [4, 2, 3, 1]) |> to_mat
    D′ = B′*A′

    # Type instability?
    C′ = reshape(C′, DIM, DIM, DIM, DIM)
    D′ = reshape(D′, DIM, DIM, DIM, DIM)

    for I in CartesianIndices(C′)
        i, j, k, l = Tuple(I)

        C′[I] += 4 \ (D′[l, j, i, k] - D′[l, i, j, k] - D′[k, j, i, l] + D′[k, i, j, l])
    end

    C′
end
function _comm2_2_2_pw(A, B, i, j, k, l)
    tot = zero(ELTYPE)
    for a in SPBASIS, b in SPBASIS
        prod3(i, j, k, l) = A[a, i, b, k]*B[b, j, a, l]

        tot += 2 \ (1-isocc(a)-isocc(b))*(A[i, j, a, b]*B[a, b, k, l] #=
                 =# - B[i, j, a, b]*A[a, b, k, l]) #=
            =# + (isocc(a)-isocc(b)) #=
            =# * (prod3(i, j, k, l) - prod3(j, i, k, l) - prod3(i, j, l, k) #=
               =# + prod3(j, i, l, k))
    end

    tot / 4
end

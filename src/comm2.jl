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

comm2(A, B) =
    (_comm0(A, B), _comm1(A, B), _comm2(A, B))
comm2_pw(A, B) =
    (_comm0_pw(A, B), _comm1_pw(A, B), _comm2_pw(A, B))

_comm0(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) = _comm0_1_1(nbody(A, 1).rep, nbody(B, 1).rep) + _comm0_2_2(nbody(A, 2).rep, nbody(B, 2).rep)
_comm0_pw(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP) =
    _comm0_1_1(nbody(A, 1).rep, nbody(B, 1).rep) + _comm0_2_2_pw(nbody(A, 2).rep, nbody(B, 2).rep)
_comm0(A, B) = _comm0_1_1(nbody(A, 1), nbody(B, 1)) + _comm0_2_2(nbody(A, 2), nbody(B, 2))
_comm0_pw(A, B) =
    _comm0_1_1(nbody(A, 1), nbody(B, 1)) + _comm0_2_2_pw(nbody(A, 2), nbody(B, 2))

function _comm1(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP)
    A0, A1, A2 = A
    B0, B1, B2 = B
    A1, A2, B1, B2 = map(x -> x.rep, (A1, A2, B1, B2))

    ARRAYOP(1)(_comm1_1_1(A1, B1) + _comm1_1_2(A1, B2) + _comm1_2_2(A2, B2) #=
            =# - _comm1_1_2(B1, A2))
end
function _comm1(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    _comm1_1_1(A1, B1) + _comm1_1_2(A1, B2) + _comm1_2_2(A2, B2) - _comm1_1_2(B1, A2)
end

function _comm1_pw(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP)
    A0, A1, A2 = A
    B0, B1, B2 = B
    A1, A2, B1, B2 = map(x -> x.rep, (A1, A2, B1, B2))

    ret = Array{ELTYPE}(undef, DIM, DIM)

    for i in SPBASIS, j in SPBASIS
        ret[index(i), index(j)] =
            _comm1_1_1_pw(A1, B1, i, j) + _comm1_1_2_pw(A1, B2, i, j) #=
         =# + _comm1_2_2_pw(A2, B2, i, j) - _comm1_1_2_pw(B1, A2, i, j)
    end

    ARRAYOP(1)(ret)
end

function _comm2(A::TwoBodyARRAYOP, B::TwoBodyARRAYOP)
    A0, A1, A2 = A
    B0, B1, B2 = B
    A1, A2, B1, B2 = map(x -> x.rep, (A1, A2, B1, B2))

    ARRAYOP(2)(_comm2_1_2(A1, B2) + _comm2_2_2(A2, B2) - _comm2_1_2(B1, A2))
end
function _comm2(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B

    _comm2_1_2(A1, B2) + _comm2_2_2(A2, B2) - _comm2_1_2(B1, A2)
end
function _comm2_pw(A, B)
    A0, A1, A2 = A
    B0, B1, B2 = B
    A1, A2, B1, B2 = map(x -> x.rep, (A1, A2, B1, B2))

    ret = Array{ELTYPE}(undef, DIM, DIM, DIM, DIM)
    @nloops 4 i (_ -> SPBASIS) begin
        @nref(4, ret, d -> index(i_d)) =
            @ncall(4, _comm2_1_2_pw, A1, B2, i) + @ncall(4, _comm2_2_2_pw, A2, B2, i) #=
         =# - @ncall(4, _comm2_1_2_pw, B1, A2, i)
    end

    ARRAYOP(2)(ret)
end

_comm0_1_1(A, B) = sum(Iterators.product(SPBASIS, SPBASIS)) do X
    i, j = X

    (isocc(i)-isocc(j))*A[index(i), index(j)]*B[index(j), index(i)]
end
const _comm0_1_1_pw = _comm0_1_1

_comm0_2_2(A, B) = 4 \ sum(Iterators.product(SPBASIS, SPBASIS, SPBASIS, SPBASIS)) do X
    i, j, k, l = X

    all(isocc, (i, j))*all(isunocc, (k, l)) #=
 =# * (A[index(i), index(j), index(k), index(l)]*B[index(k), index(l), index(i), index(j)]
       - B[index(i), index(j), index(k), index(l)]*A[index(j), index(l), index(i), index(j)])
end
const _comm0_2_2_pw = _comm0_2_2

_comm1_1_1(A, B) = A*B - B*A
_comm1_1_1_pw(A, B, i, j) = ((i, j) = index.((i, k)); sum(SPBASIS) do a
    a = index(a)
    A[i, a]*B[a, j] - B[i, a]*A[a, j]
end)

function _comm1_1_2(A, B)
    A′ = reshape(A.*DMAT_HmH, DIM^2)
    B′ = reshape(permutedims(B, [2, 4, 3, 1]), DIM^2, DIM^2)

    reshape(B′*A′, DIM, DIM)
end
_comm1_1_2_pw(A, B, i, j) = sum(Iterators.product(SPBASIS, SPBASIS)) do X
    a, b = index.(X)
    i, j = index.((i, j))

    (isocc(a)-isocc(b))*A[a, b]*B[b, i, a, j]
end

function _comm1_2_2(A, B)
    A′ = reshape(PermutedDimsArray(A, [2, 3, 4, 1]), DIM, DIM^3) |> copy
    B′ = reshape(B.*DMAT_PPHpHHP, DIM^3, DIM)
    C′ = A′*B′

    # NOTE: here we have A -> B′ and B -> A′
    B′ .= reshape(A.*DMAT_PPHpHHP, DIM^3, DIM)
    A′ .= reshape(PermutedDimsArray(B, [2, 3, 4, 1]), DIM, DIM^3)
    C′ .-= A′*B′

    C′ ./= 2

    reshape(C′, DIM, DIM)

end
function _comm1_2_2_pw(A, B, i, j)
    tot = zero(ELTYPE)
    i, j = index.((i, j))
    @nloops 3 a (_ -> SPBASIS) begin
        tot += (isunocc(a_1)*isunocc(a_2)*isocc(a_3) + isocc(a_1)*isocc(a_2)*isunocc(a_3)) #=
            =# * (A[index(a_3), i, index(a_1), index(a_2)]*B[index(a_1), index(a_2), index(a_3), j] #=
               =# - B[index(a_3), i, index(a_1), index(a_2)]*A[index(a_1), index(a_2), index(a_3), j])
    end

    tot / 2
end

function _comm2_1_2(A, B)
    B′ = reshape(B, DIM, DIM^3) |> copy
    C′ = A * B′

#    B′ .= reshape(permutedims(B, [1, 2, 4, 3]), DIM^3, DIM)
#    D′ = B′*A
    B′ .= reshape(PermutedDimsArray(B, [3, 1, 2, 4]), DIM, DIM^3)
    D′ = transpose(B′)*A

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
_comm2_1_2_pw(A, B, i, j, k, l) = ((i, j, k, l) = index.((i, j, k, l)); 4 \ sum(SPBASIS) do a
    a = index(a)
    prod1(i, j, k, l) = A[i, a]*B[a, j, k, l]
    prod2(i, j, k, l) = A[a, k]*B[i, j, a, l]

    prod1(i, j, k, l) - prod1(j, i, k, l) - prod2(i, j, k, l) + prod2(i, j, l, k)
end)

function _comm2_2_2(A, B)
    to_mat(x) = reshape(x, DIM^2, DIM^2)

    A′ = Array{ELTYPE}(undef, DIM^2, DIM^2)
    B′ = Array{ELTYPE}(undef, DIM^2, DIM^2)

    A′ .= to_mat(A)
    B′ .= to_mat(B .* DMAT_1mHmH)
    C′ = A′*B′

    A′ .= to_mat(A .* DMAT_1mHmH)
    B′ .= to_mat(B)
    C′ .-= B′*A′
    C′ ./= 8

    A′ .= PermutedDimsArray(A, [1, 3, 2, 4]).*DMAT_HmH |> to_mat
    B′ .= PermutedDimsArray(B, [4, 2, 3, 1]) |> to_mat
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
    i, j, k, l = index.((i, j, k, l))
    for a in SPBASIS, b in SPBASIS
        prod3(i, j, k, l) = A[index(a), i, b, k]*B[b, j, index(a), l]

        tot += 2 \ (1-isocc(index(a))-isocc(b))*(A[i, j, index(a), b]*B[index(a), b, k, l] #=
                                      =# - B[i, j, index(a), b]*A[index(a), b, k, l]) #=
            =# + (isocc(a)-isocc(b)) #=
              =# * (prod3(i, j, k, l) - prod3(j, i, k, l) - prod3(i, j, l, k) #=
                 =# + prod3(j, i, l, k))
    end

    tot / 4
end

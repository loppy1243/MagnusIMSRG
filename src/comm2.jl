using Base.Cartesian

### Gotta move these...
### Density Matrices #########################################################################
# hole*particle or -(particle*hole)
const DMAT_HmH = [isocc(a)-isocc(b) for a in SPBASIS, b in SPBASIS]
# particle*particle or -(hole*hole)
const DMAT_1mHmH = [1-isocc(a)-isocc(b) for a in SPBASIS, b in SPBASIS]
# particle*particle*hole + hole*hole*particle
const DMAT_PPHpHHP = [isunocc(a)*isunocc(b)*isocc(c) + isocc(a)*isocc(b)*isunocc(c)
                      for a in SPBASIS, b in SPBASIS, c in SPBASIS]
##############################################################################################

vector(N, A::AbstractVector) = A
vector(N, A::AbstractArray) = if N == 1
    reshape(A, DIM^2)
elseif N == 2
    reshape(A, DIM^4)
else
    throw(MethodError(vector, (N, A)))
end

matrix(N, A::AbstractMatrix) = A
matrix(N, A::AbstractArray) = if N == 1
    reshape(A, DIM, DIM)
elseif N == 2
    reshape(A, DIM^2, DIM^2)
else
    throw(MethodError(matrix, (N, A)))
end

tensor(N, A::AbstractArray{<:Any, 4}) = A
tensor(N, A::AbstractArray) = if N == 2
    reshape(A, DIM, DIM, DIM, DIM)
else
    throw(MethodError(tensor, (N, A)))
end

cartesian_sum(f, N, itr) = cartesian_sum(f, Val{N}, itr)
@generated cartesian_sum(f, ::Type{Val{N}}, itr) = quote
    fst = first(itr)
    ret = zero(@ncall($N, f, _->fst))
    @nloops $N x _->itr begin
        ret += @ncall($N, f, x)
    end
    ret
end

comm2(A::IMArrayOp{2}, B::IMArrayOp{2}) =
    IMArrayOp(_comm0(A, B), _comm1(A, B), _comm2(A, B))
comm2_pw(A::IMArrayOp{2}, B::IMArrayOp) =
    IMArrayOp(_comm0_pw(A, B), _comm1_pw(A, B), _comm2_pw(A, B))

_comm0(A, B)    = _comm0_1_1(A.parts[1], B.parts[1]) + _comm0_2_2(A.parts[2], B.parts[2])
_comm0_pw(A, B) = _comm0_1_1(A.parts[1], B.parts[1]) + _comm0_2_2_pw(B.parts[2], B.parts[2])

function _comm1(A, B)
    A0, A1, A2 = A.parts
    B0, B1, B2 = B.parts

    _comm1_1_1(A1, B1) + _comm1_1_2(A1, B2) + _comm1_2_2(A2, B2) - _comm1_1_2(B1, A2)
end
function _comm1(A, B)
    A0, A1, A2 = A.parts
    B0, B1, B2 = B.parts

    _comm1_1_1(A1, B1) + _comm1_1_2(A1, B2) + _comm1_2_2(A2, B2) - _comm1_1_2(B1, A2)
end

function _comm1_pw(A, B)
    A0, A1, A2 = A.parts
    B0, B1, B2 = B.parts

    ret = Array{ELTYPE}(undef, DIM, DIM)

    tabulate(Array{ELTYPE}, 2, SPBASIS) do I
        _comm1_1_1_pw(A1, B1, I...) + _comm1_1_2_pw(A1, B2, I...) #=
        =#+ _comm1_2_2_pw(A2, B2, I...) - _comm1_1_2_pw(B1, A2, I...)
    end
end

function _comm2(A, B)
    A0, A1, A2 = A.parts
    B0, B1, B2 = B.parts

    _comm2_1_2(A1, B2) + _comm2_2_2(A2, B2) - _comm2_1_2(B1, A2)
end
function _comm2_pw(A, B)
    A0, A1, A2 = A.parts
    B0, B1, B2 = B.parts

    tabulate(Array{ELTYPE}, 4, SPBASIS) do I
        _comm2_1_2_pw(A1, B2, I...) + _comm2_2_2_pw(A2, B2, I...) #=
        =#- _comm2_1_2_pw(B1, A2, I...)
    end
end

_comm0_1_1(A, B) = cartesian_sum(2, SPBASIS) do i, j
    (isocc(i)-isocc(j))*A[i, j]*B[j, i]
end
#function _comm0_1_1(A, B)
#    A = matrix(1, A) |> copy
#    B = matrix(1, B)
#
#    A .*= DMAT_HmH
#    tr(A*B)
#end
const _comm0_1_1_pw = _comm0_1_1

_comm0_2_2(A, B) = 4 \ cartesian_sum(4, SPBASIS) do i, j, k, l
    all(isocc, (i, j))*all(isunocc, (k, l)) #=
    =# * (A[i, j, k, l]*B[k, l, i, j] - B[i, j, k, l]*A[k, l, i, j])
end
#function _comm0_1_1(A, B)
#    A = matrix(2, A) |> copy; B = matrix(2, B) |> copy
#    A .= DMAT_HHPP
#    C = A*B
#    A .= matrix(2, A)
#    B .= DMAT_HHPP
#    C .-= B*A
#
#    C
#end
const _comm0_2_2_pw = _comm0_2_2

function _comm1_1_1(A, B)
    A = matrix(1, A); B = matrix(1, B)
    A*B - B*A
end
_comm1_1_1_pw(A, B, i, j) = sum(SPBASIS) do a
    A[i, a]*B[a, j] - B[i, a]*A[a, j]
end

function _comm1_1_2(A, B)
    A = matrix(1, A) |> copy
    A .= DMAT_HmH
    A = vector(1, A)

    B = tensor(2, B)
    B = matrix(2, permutedims(B, [2, 4, 3, 1]))

    matrix(1, B*A)
end
_comm1_1_2_pw(A, B, i, j) = cartesian_sum(2, SPBASIS) do a, b
    (isocc(A)-isocc(b))*A[a, b]*B[b, i, a, j]
end

function _comm1_2_2(A, B)
    A = tensor(2, A); B = tensor(2, B)
    A′ = reshape(permutedims(A, [2, 3, 4, 1]), DIM, DIM^3)
    B′ = reshape(B.*DMAT_PPHpHHP, DIM^3, DIM)
    C′ = A′*B′

    # NOTE: here we have A -> B′ and B -> A′
    B′ .= reshape(A.*DMAT_PPHpHHP, DIM^3, DIM)
    A′ .= reshape(PermutedDimsArray(B, [2, 3, 4, 1]), DIM, DIM^3)
    C′ .-= A′*B′

    C′ ./= 2
end
#function _comm1_2_2(A, B)
#    A = matrix(2, A); B = matrix(2, B)
#    B′ = B.*DMAT_PPHpHHP
#    C = A*B′
#
#    A′ = A.*DMAT_PPHpHHP
#    C .-= B*A′
#
#    sum(C, dims=1)
#end
_comm1_2_2_pw(A, B, i, j) = 2 \ cartesian_sum(3, SPBASIS) do a, b, c
    (isunocc(a)*isunicc(b)*isocc(c) + isocc(a)*isocc(b)*isocc(c)) #=
    =# * (A[c, i, a, b]*B[a, b, c, j] - B[c, i, a, b]*A[a, b, c, j])
end

function _comm2_1_2(A, B)
    A = matrix(1, A); B = tensor(2, B)
    B′ = reshape(B, DIM, DIM^3)
    C′ = A*B′

#    B′ .= reshape(permutedims(B, [1, 2, 4, 3]), DIM^3, DIM)
#    D′ = B′*A
    B′ .= reshape(PermutedDimsArray(B, [3, 1, 2, 4]), DIM, DIM^3)
    D′ = transpose(B′)*A

    # Type unstable?
    C′ = tensor(4, C′)
    D′ = tensor(4, D′)

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

### Update Line ##########################################################################

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

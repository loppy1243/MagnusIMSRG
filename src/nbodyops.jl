module IMOperators
export IMArrayOp, DenseIMArrayOP, imrank, hconj, mbop

#import LinearAlgebra
#using LinearAlgebra: I

const _X{T} = Union{T, Array{T, 0}}

struct IMArrayOp{N, T, AS<:NTuple{N, AbstractArray{T}}}
    b0::Array{T, 0}
    bs::AS

    function IMArrayOp{N, T, AS}(b0::_X{T}, bs::AS) where
                      {N, T, AS<:NTuple{N, AbstractArray{T}}}
        for (i, b) in enumerate(bs)
            @assert ndims(b) == 2i
        end
        new(fill(b0), bs)
    end
end
Base.getproperty(op::IMArrayOp, s::Symbol) =
    if s === :parts
        Indexer(op)
    else
        Base.getfield(op, s)
    end

function IMArrayOp{N, T, AS}(b0::_X{T}, bs::Vararg{AbstractArray{T}, N}) where
                  {N, T, AS<:NTuple{N, AbstractArray{T}}}
    bs::AS
    IMArrayOp{N, T, AS}(b0, bs)
end
IMArrayOp{N, T}(b0::_X{T}, bs::Vararg{AbstractArray{T}, N}) where {N, T} =
    IMArrayOp{N, T, typeof(bs)}(b0, bs)
IMArrayOp{N}(b0::_X{T}, bs::Vararg{AbstractArray{T}, N}) where T =
    IMArrayOp{N, _X{T}, typeof(bs)}(b0, bs)
IMArrayOp(b0::_X{T}, bs::AbstractArray{T}...) where T =
    IMArrayOp{length(bs), T, typeof(bs)}(b0, bs)
(O::Type{<:IMArrayOp})(args::Tuple) = O(args...)

IMArrayOp{N, T, AS}(::UndefInitializer, dims::Vararg{<:Any, N}) where
         {N, T, AS<:NTuple{N, AbstractArray{T}}} =
    IMArrayOp{N, T, AS}(zero(T), Tuple(A(undef, ds) for (A, ds) in zip(AS.types, dims)))
(O::Type{<:IMArrayOp})(::UndefInitializer, dims...) =
    O(zero(T), Tuple(A(undef, ds) for (A, ds) in zip(arraytypes(O), dims)))

const DenseIMArrayOp{N, T, AS<:NTuple{N, Array{T}}} = IMArrayOp{N, T, AS}
arraytypes(O::Type{IMArrayOp{<:Any, <:Any, AS}}) where
           AS<:(NTuple{<:Any, AbstractArray{T}} where T) = AS
arraytypes(O::Type{IMArrayOp{N, T}}) = Tuple(arraytype(O, n) for n = 1:imrank(O))
arraytype(O::Type{<:DenseIMArrayOp, n}) = Array{eltype(O), 2n}

imrank(::Type{<:IMArrayOp{N}}) where N = N
imrank(op::IMArrayOp) = imrank(typeof(op))
imrank(A::Type{<:AbstractArray}) = div(ndims(a), 2)
imrank(a::AbstractArray) = imrank(typeof(a))
Base.eltype(::Type{<:IMArrayOp{<:Any, T}}) where T = T

# Make into method on Base.size?
size(op::IMArrayOp) = map(size, op.bs)

### Indexing
##############################################################################################
_imoptype(O::Type{<:IMArrayOp}) = O
_imoptype(op::IMArrayOp) = _imoptype(typeof(op))
struct Indexer{O<:IMArrayOp}; op::O end
@inline function Base.getindex(I::Indexer, i::Int)
    @boundscheck i <= imrank(I.op)
    i == 0 ? I.op.b0 : I.op.bs[i]
end
@inline Base.getindex(I::Indexer, ixs::AbstractArray) = Tuple(I[i] for i in ixs)
Base.firstindex(I::Indexer) = 0
Base.lastindex(I::Indexer) = imrank(_imoptype(I))

### Iteration
##############################################################################################
Base.IteratorEltype(::Type{<:Indexer}) = Base.EltypeUnknown()
Base.IteratorSize(IT::Type{<:Indexer}) = Base.HasLength()

function Base.iterate(I::Indexer, st=1)
    st > imrank(I.op) && return nothing
    st == 1 ? (I.op.b0, st+1) : (I.op.bs[st], st+1)
end
Base.length(I::Index) = imrank(_imoptype(I)) + 1

### Operations
##############################################################################################

function Base.map(f, ops::IMArrayOp{N}...)
    @assert all(==(size(ops[1])), ops)

    T = promote_type(eltype(dest), map(eltype, ops))
    ret = IMArrayOp{N, T}(undef, size(op[1])...)

    map!(f, ret, ops)
end

function Base.map!(f, dest::IMArrayOp{N}, srcs::IMArrayOp{N}...) where N
    @assert all(==(size(dest)), srcs)

    @assert all(map(size, dest.parts) .== map(size, src.parts))
    for bs in zip(dest.parts, map(op -> op.parts, srcs)...)
        map!(f, bs[1], bs[2:end]...)
    end

    dest
end

trunc(op::IMArrayOp, N) = if N >= imrank(op)
    op
else
    IMArrayOP{N, eltype(op), typeof(op.bs[1:N])}(op.b0, op.bs[1:N])
end

for op in (:+, :-)
    @eval begin
        Base.$op(a::IMArrayOp{N}, b::IMArrayOp{N}) where N = map($op, a, b)
        Base.$op(a::IMArrayOp) = map($op, a)
    end
end
for op in (:*, :/)
    @eval Base.$op(a::IMArrayOp, b::Number) = map(x -> $op(x, b), a)
end
for op in (:*, :\)
    @eval Base.$op(a::Number, b::IMArrayOp) = map(x -> $op(a, x), b)
end

"""
    hconj(x)

Hermitian conjugate of x.
"""
hconj(x::Number) = conj(x)
function hconj(x::AbstractArray{<:Any, 0})
    ret = similar(x)
    ret[] = hconj(x[])
    ret
end
# Note that we must have `ndims(x) % 2 == 0`
function hconj(x::AbstractArray)
    d = ndims(a)
    d2 = div(d, 2)
    conj.(PermutedDimsArray(x, [d2+1:d... 1:d2...]))
end
hconj(x::IMArrayOp) = typeof(x)(map(hconj, x.parts)...)
Base.ctranspose(op::IMArrayOp) = hconj(op)

norm(op::IMArrayOp) = sqrt(sum(x -> sum(x.^2), op.parts))

Base.similar(O::Type{<:IMArrayOp}, dims...) = similar(O, eltype(O), dims...)
Base.similar(O::Type{<:IMArrayOp{N}}, T::Type, dims::Vararg{<:Any, N}) where N =
    similar(O, T, dims)
Base.similar(O::Type{<:IMArrayOp{N}}, T::Type, dims::NTuple{N}) where N =
    O(fill(T), map(similar, arraytypes(O), dims))
Base.similar(op::IMArrayOp) = typeof(op)(undef)
Base.similar(op::IMArrayOp, T::Type) = IMArrayOp(map(x -> similar(x, T), op.parts)...)
Base.similar(op::IMArrayOp, dims...) = similar(typeof(op), dims...)
Base.similar(op::IMArrayOp, T::Type, dims...) = similar(typeof(op), T, dims...)

ManyBody.tabulate(fs, O::Type{<:IMArrayOp}, B::AbstractBasis) =
    tabulate(fs, O, Tuple(B for _=1:imrank(O)))
ManyBody.tabulate(fs, O::Type{<:IMArrayOp}, Bs::Tuple) =
    O(fill(fs[1]), map(tabulate, fs[2:end], arraytypes(O), Bs) |> Tuple)
function ManyBody.tabulate!(fs, op::IMArrayOp, Bs::Tuple)
    op.parts[0][] = fs[1]
    foreach(tabulate!, fs[2:end], op.parts[1:end], Bs)
    op
end

### Update Line ##############################################################################

mbop(op, B) = mbop(op, B, B)
function mbop(op::IMArrayOp{2}, B1, B2)
    T = eltype(op)
    E, f, Γ = op.parts

    tabulate(typeof(op.parts[1]), B1, B2) do X, Y
        SPB = spbasis(Y)

        ret = E*overlap(X, Y)
        for p in SPB, q in SPB
            NA = normord(Operators.@A(p', q))
            ret += f[p, q] * NA(X, Y)
        end
       
        for p in SPB, q in SPB, r in SPB, s in SPB
            NA = normord(Operators.@A(p', q', s, r))
            ret += Γ[p, q, r, s] * NA(X, Y)
        end

        b0 + b1 + b2
    end
end

randimop(T, dims...) = IMArrayOp(rand(T), map(ds -> rand(T, ds), dims)...)

end # module IMOperators

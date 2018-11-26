module IMOperators
export IMArrayOp, DenseIMArrayOp, imrank, hconj, mbop, randimop
using ManyBody

import ..SIGNAL_OPS

SIGNAL_OPS && include("signalops.jl")

const _X{T} = Union{T, Array{T, 0}}

struct IMArrayOp{N, T, AS<:NTuple{N, AbstractArray{T}}}
    b0::Array{T, 0}
    bs::AS

    IMArrayOp{N, T, AS}(b0::T, bs::AS) where {N, T, AS<:NTuple{N, AbstractArray{T}}} =
        new(fill(b0), bs)
    IMArrayOp{N, T, AS}(b0::Array{T, 0}, bs::AS) where {N, T, AS<:NTuple{N, AbstractArray{T}}} =
        new(b0, bs)
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
IMArrayOp{N}(b0::_X{T}, bs::Vararg{AbstractArray{T}, N}) where {N, T} =
    IMArrayOp{N, _X{T}, typeof(bs)}(b0, bs)
IMArrayOp(b0::_X{T}, bs::AbstractArray{T}...) where T =
    IMArrayOp{length(bs), T, typeof(bs)}(b0, bs)
(O::Type{<:IMArrayOp})(args::Tuple) = O(args...)

IMArrayOp{N, T, AS}(::UndefInitializer, dims::Vararg{<:Any, N}) where
         {N, T, AS<:NTuple{N, AbstractArray{T}}} =
    IMArrayOp{N, T, AS}(zero(T), Tuple(A(undef, ds) for (A, ds) in zip(AS.types, dims)))

const DenseIMArrayOp{N, T, AS<:NTuple{N}} = IMArrayOp{N, T, AS}

arraytypes(O::Type{<:IMArrayOp{<:Any, <:Any, AS}}) where AS = Tuple(AS.types)

imrank(::Type{<:IMArrayOp{N}}) where N = N
imrank(op::IMArrayOp) = imrank(typeof(op))
imrank(A::Type{<:AbstractArray}) = div(ndims(a), 2)
imrank(a::AbstractArray) = imrank(typeof(a))
Base.eltype(::Type{<:IMArrayOp{<:Any, T}}) where T = T

# Make into method on Base.size?
size(op::IMArrayOp) = map(Base.size, op.bs)

Base.zero(O::Type{<:IMArrayOp}) = O(zero(eltype(O)), map(zero, arraytypes(O)))
function Base.zero(op::IMArrayOp)
    z = zero(eltype(op))

    ret = similar(op)
    for part in ret.parts
        fill!(part, z)
    end

    ret
end
Base.iszero(op::IMArrayOp) = all(iszero, op)

### Indexing
##############################################################################################
struct Indexer{O<:IMArrayOp}; op::O end
_imoptype(::Type{<:Indexer{O}}) where O<:IMArrayOp = O
_imoptype(I::Indexer) = _imoptype(typeof(I))

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
Base.IteratorSize(::Type{<:Indexer}) = Base.HasLength()
Base.length(I::Indexer) = imrank(_imoptype(I)) + 1
function Base.iterate(I::Indexer, st=0)
    st > imrank(I.op) && return nothing
    st == 0 ? (I.op.b0, st+1) : (I.op.bs[st], st+1)
end

Base.IteratorEltype(::Type{<:IMArrayOp}) = Base.HasEltype()
Base.IteratorSize(::Type{<:IMArrayOp}) = Base.HasLength()
Base.length(op::IMArrayOp) = sum(length, op.parts)
function Base.iterate(op::IMArrayOp, (part, inner)=(0, (op.parts[0],)))
    if (X = iterate(inner...)) === nothing
        part == imrank(op) && return nothing
        part += 1
        X = iterate(op.parts[part])
    end

    (X[1], (part, (op.parts[part], X[2])))
end

### Operations
##############################################################################################

function Base.map(f, ops::IMArrayOp{N}...) where N
    sz = IMOperators.size(ops[1])
    @assert all(ops) do op
        IMOperators.size(op) == sz
    end

    T = promote_type(map(eltype, ops)...)
    ret = similar(ops[1], T)

    map!(f, ret, ops...)
end

function Base.map!(f, dest::IMArrayOp{N}, srcs::IMArrayOp{N}...) where N
    @assert all(srcs) do op
        IMOperators.size(op) == IMOperators.size(dest)
    end

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
    @eval Base.$op(a::IMArrayOp{N}, b::IMArrayOp{N}) where N = map($op, a, b)
end
Base.:+(a::IMArrayOp) = a
Base.:-(a::IMArrayOp) = map(-, a)
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
    d = ndims(x); @assert d % 2 == 0
    d2 = div(d, 2)
    conj.(PermutedDimsArray(x, [d2+1:d..., 1:d2...]))
end
hconj(x::IMArrayOp) = typeof(x)(map(hconj, x.parts)...)
Base.adjoint(op::IMArrayOp) = hconj(op)

norm(op::IMArrayOp) = sqrt(sum(x -> sum(x.^2), op.parts))

Base.similar(O::Type{<:IMArrayOp}, dims...) = similar(O, eltype(O), dims...)
Base.similar(O::Type{<:IMArrayOp{N}}, T::Type, dims::Vararg{<:Any, N}) where N =
    similar(O, T, dims)
Base.similar(O::Type{<:IMArrayOp{N}}, T::Type, dims::NTuple{N}) where N =
    O(fill(T), map(similar, arraytypes(O), dims))
Base.similar(op::IMArrayOp) = typeof(op)(map(similar, op.parts)...)
Base.similar(op::IMArrayOp, T::Type) = IMArrayOp(map(x -> similar(x, T), op.parts)...)
Base.similar(op::IMArrayOp, dims...) = similar(typeof(op), dims...)
Base.similar(op::IMArrayOp, T::Type, dims...) = similar(typeof(op), T, dims...)

ManyBody.tabulate(fs, O::Type{<:IMArrayOp}, Bs::Tuple{Type{<:AbstractArray}, Vararg{Any}}) =
    tabulate(fs, O, (Bs,))
function ManyBody.tabulate(fs, O::Type{<:IMArrayOp}, Bs::Tuple)
    As = map(x -> x[1]{eltype(O)}, Bs)
    Bs = map(x -> Base.tail(x), Bs)
    IMArrayOp(fill(fs[1]), map(tabulate, fs[2:end], As, Bs)...)
end

function ManyBody.tabulate!(fs, op::IMArrayOp, Bs::Tuple)
    op.parts[0][] = fs[1]
    foreach(tabulate!, fs[2:end], op.parts[1:end], Bs)
    op
end

mbop(op, B) = mbop(op, B, B)
function mbop(op::IMArrayOp{2}, B1, B2)
    T = eltype(op)
    E, f, Γ = op.parts

    tabulate(Array{eltype(op)}, B1, B2) do X, Y
        SPB = spbasis(Y)

        ret = E*overlap(X, Y)
        for p in SPB, q in SPB
            sgn, NA = normord(Operators.@A(p', q))
            ret += sgn*f[p, q]*NA(X, Y)
        end
       
        for p in SPB, q in SPB, r in SPB, s in SPB
            sgn, NA = normord(Operators.@A(p', q', s, r))
            ret += sgn*Γ[p, q, r, s]*NA(X, Y)
        end

        ret
    end
end

randimop(T, dims...) = IMArrayOp(rand(T), map(ds -> rand(T, ds), dims)...)

function Base.show(io::IO, mime::MIME"text/plain", op::IMArrayOp)
    println(typeof(op))
    for i = 0:imrank(op)
        print("$i-Body Part: ")
        show(io, mime, op.parts[i])
        println()
    end
end

end # module IMOperators

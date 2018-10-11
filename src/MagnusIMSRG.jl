module MagnusIMSRG

using ManyBody
import ManyBody.Operators
import JuliaUtil
#using OrdinaryDiffEq
using Combinatorics: levicivita, combinations
using JuliaUtil: bernoulli, @every

#### Workaround b/c no internet :(
#JuliaUtil.fbinom(a, b) =
#    factorial(Float64, a) / (factorial(Float64, b)*factorial(Float64, a-b))

### Parameters ###############################################################################
const Ω_RTOL = 0.1
const Ω_ATOL = 0.0
const Ω_BATCHSIZE = 5
const H_RTOL = 0.0
const H_ATOL = 0.005
const INT_RTOL = 0.0
const INT_ATOL = 0.0001
const H_BATCHSIZE = 5
const S_BIG_STEP = 1.0
const S_SMALL_STEP = 0.01
const MAX_INT_ITERS = 100
##############################################################################################

const SPBASIS = Bases.Pairing{4}
const spbasis = basis(SPBASIS)
const IXBASIS = indextype(SPBASIS)
const REFSTATE = RefStates.Fermi(SPBASIS, 2)
const MBBASIS = Bases.Paired{2, 4}
const ELTYPE = Float64
const ARRAYOP(N) = F64ArrayOperator{Bases.Product{N, NTuple{N, SPBASIS}}, 2N}
const FUNCOP(N) = F64FunctionOperator{Bases.Product{N, NTuple{N, SPBASIS}}}
const MBFUNCOP = F64FunctionOperator{MBBASIS}

const DIM = dim(SPBASIS)
const LEVEL_SPACING = 1.0
const FERMILEVEL = fermilevel(REFSTATE)

isocc(x) = ManyBody.isocc(REFSTATE, x)
isunocc(x) = ManyBody.isocc(REFSTATE, x)
normord(x) = ManyBody.normord(REFSTATE, x)

include("nbodyops.jl")
include("comm2.jl")
include("hamiltonians.jl")
include("Generators/main.jl"); import .Generators

const comm = comm2
const generator = Generators.white

factorial(T::Type{<:Number}, n::Integer) = prod(one(T):convert(T, n))

function dΩ(Ω, h0)
    @debug "dΩ term" n=0
    prev_tot = ZERO_OP
    prev_ad = generator(Ω, h0)
    tot = bernoulli(Float64, 0)/factorial(Float64, 0) * prev_ad

    n = 1
    while norm(tot - prev_tot) > max(Ω_ATOL, Ω_RTOL*norm(tot))
        @debug "dΩ term" n
        prev_tot = tot
        tot += sum(n:n+Ω_BATCHSIZE-1) do i
            bernoulli(Float64, i)/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += Ω_BATCHSIZE
    end

    tot
end

function solve(h0; max_int_iters=MAX_INT_ITERS)
    s = 0.0
    n = 0
    Ω = ZERO_OP
    h_prev = ZERO_OP
    h = h0

    while @show(norm(h - h_prev)) > max(INT_ATOL, INT_RTOL*norm(h))
        @debug "Integration iter" n
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

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

##############################################################################################
### Flags ####################################################################################
const SIGNAL_OPS = false
##############################################################################################
### Parameters ###############################################################################
const E_DENOM_ATOL = 1e-5
const Ω_RTOL = 0.0
const Ω_ATOL = 0.01
const Ω_BATCHSIZE = 5
const H_RTOL = 0.0
const H_ATOL = 0.01
const INT_RTOL = 1e-8
#const INT_ATOL = 1e-3
const INT_DIV_RTHRESH = 1.0
const H_BATCHSIZE = 5
const S_BIG_STEP = 1.0
const S_SMALL_STEP = 1.0
const MAX_INT_ITERS = 100
##############################################################################################

const SPBASIS = Bases.Pairing{4}
const IXBASIS = indextype(SPBASIS)
const REFSTATE = RefStates.Fermi(SPBASIS, 2)
const MBBASIS = Bases.Paired{2, 4}
const ELTYPE = Float64
const ARRAYOP(N) = F64ArrayOperator{Bases.Product{N, NTuple{N, SPBASIS}}, 2N}
const FUNCOP(N) = F64FunctionOperator{Bases.Product{N, NTuple{N, SPBASIS}}}
const MBFUNCOP = F64FunctionOperator{MBBASIS}
const MBARRAYOP = F64ArrayOperator{MBBASIS, 2}

const DIM = dim(SPBASIS)
const LEVEL_SPACING = 1.0
const FERMILEVEL = fermilevel(REFSTATE)

const HOLES = ManyBody.holes(REFSTATE)
const PARTS = ManyBody.parts(REFSTATE)
isocc(x) = ManyBody.isocc(REFSTATE, x)
isunocc(x) = ManyBody.isunocc(REFSTATE, x)
normord(x) = ManyBody.normord(REFSTATE, x)

SIGNAL_OPS && include("signalops.jl")
include("nbodyops.jl")
include("comm2.jl")
include("hamiltonians.jl")
include("Generators/main.jl"); import .Generators
include("mbpt.jl")

const comm = comm2
const generator = Generators.white

factorial(T::Type{<:Number}, n::Integer) = prod(one(T):convert(T, n))

function dΩ(Ω, h)
    @debug "Entering dΩ"
    @debug "dΩ term" n=0
    prev_tot = ZERO_OP
    prev_ad = generator(Ω, h)
    tot = bernoulli(Float64, 0)/factorial(Float64, 0) * prev_ad

    n = 1
    while norm(tot - prev_tot) > max(Ω_ATOL, Ω_RTOL*norm(tot))
        prev_tot = tot
        tot += sum(n:n+Ω_BATCHSIZE-1) do i
            bernoulli(Float64, i)/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += Ω_BATCHSIZE
    end

    tot
end

solve(h0; kws...) = solve((xs...,) -> nothing, h0; kws...)
function solve(cb, h0; max_int_iters=MAX_INT_ITERS, ds=S_SMALL_STEP, print_info=true)
    s = 0.0
    n = 0
    Ω = ZERO_OP
    h_prev = ZERO_OP
    h = h0

    while (ratio = abs((dE_2 = mbpt2(h))/nbody(h, 0))) > INT_RTOL
        print_info && _solve_print_info(n, max_int_iters, nbody(h, 0), dE_2, ratio)
        cb(s, Ω, h, dE_2)

        if n >= max_int_iters
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if ratio > INT_DIV_RTHRESH
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end

        h_prev = h
        Ω += dΩ(Ω, h) * ds
        s += ds
        h = H(Ω, h0)
        n += 1
    end

    Ω
end
function _solve_print_info(n, max_int_iters, E, dE_2, r)
    sigdigs = 5

    decdigs(x) = sigdigs - ndigits(trunc(Int, x))
    nzdecdig(x) = ceil(Int, -log10(abs(x-trunc(x))))

    r_decdigs = nzdecdig(INT_RTOL)
    E_decdigs = decdigs(E)

    n = lpad(n, ndigits(max_int_iters))

    r = round(r, digits=r_decdigs)
    r = rpad(r, ndigits(trunc(Int, r))+1+r_decdigs, '0')

    E = round(E, sigdigits=sigdigs)
    E = rpad(E, (E<0)+sigdigs+1, '0')

    dE_2 = round(dE_2, digits=E_decdigs)
    dE_2 = rpad(dE_2, (dE_2<0)+ndigits(trunc(Int, dE_2))+1+E_decdigs, '0')

    println("$n: E = $E,  dE(2) = $dE_2,  Ratio = $r")
end

end # module MagnusIMSRG

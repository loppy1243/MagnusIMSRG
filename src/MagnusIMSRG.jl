module MagnusIMSRG

using ManyBody
using Parameters, OrdinaryDiffEq
using JuliaUtil: bernoulli
#using Sundials: CVODE_BDF


### Flags ####################################################################################
##############################################################################################
const SIGNAL_OPS = false

### Parameters ###############################################################################
##############################################################################################
#@with_kw struct HyParams
#    ENG_DENOM_ATOL  ::Float64                  = 1e-5
#
#    Ω_RTOL          ::Float64                  = 0.0
#    Ω_ATOL          ::Float64                  = 0.01
#    Ω_BATCHSIZE     ::Float64                  = 5
#
#    H_RTOL          ::Float64                  = 0.0
#    H_ATOL          ::Float64                  = 0.01
#    H_BATCHSIZE     ::Int                      = 5
#
#    INT_RTOL        ::Float64                  = 1e-8
#    INT_ATOL        ::Float64                  = 1e-3
#    INT_DIV_RTHRESH ::Float64                  = 1.0
#    MAX_INT_ITERS   ::Int                      = 100
#
#    ELTYPE          ::Type                     = Float64
#    MBBASIS         ::Type                     = Bases.Paired{4, 4}
#    SPBASIS         ::Type                     = spbasis(MBBASIS)
#    DIM             ::Int                      = dim(spbasis)
#    REFSTATE        ::Union{RefState, MBBasis} = RefStates.Fermi{SPBASIS}(2)
#    HOLES           ::Vector{SPBASIS}          = holes(REFSTATE)
#    PARTS           ::Vector{SPBASIS}          = parts(REFSTATE)
#end

@with_kw struct Params
    S_LARGE_STEP    ::Float64                  = 1.0
    S_SMALL_STEP    ::Float64                  = 0.1
    TRUNC_LEVEL     ::Int                      = 2
    COMMUTATOR      ::Symbol                   = :comm2
    GENERATOR       ::Symbol                   = :white
    ENG_DENOM_ATOL  ::Float64                  = 1e-5

    Ω_RTOL          ::Float64                  = 0.0
    Ω_ATOL          ::Float64                  = 0.01
    Ω_BATCHSIZE     ::Float64                  = 5

    H_RTOL          ::Float64                  = 0.0
    H_ATOL          ::Float64                  = 0.01
    H_BATCHSIZE     ::Int                      = 5

    INT_RTOL        ::Float64                  = 1e-8
    INT_ATOL        ::Float64                  = 1e-3
    INT_DIV_RTHRESH ::Float64                  = 1.0
    MAX_INT_ITERS   ::Int                      = 100
    PRINT_INFO      ::Bool                     = true


    ELTYPE          ::Type                     = Float64
    MBBASIS         ::Type                     = Bases.Paired{4, 4}
    SPBASIS         ::Type                     = spbasis(supbasis(MBBASIS))
    DIM             ::Int                      = dim(SPBASIS)
    REFSTATE        ::Union{RefState, MBBasis} = RefStates.Fermi{SPBASIS}(2)
    HOLES           ::Vector                   = holes(REFSTATE)
    PARTS           ::Vector                   = parts(REFSTATE)
end

const PARAMS = Params()
include("getparams.jl")

SIGNAL_OPS && include("signalops.jl")

include("util.jl")
include("IMOperators.jl");      using .IMOperators; const IMOps = IMOperators
include("Commutators.jl");     import .Commutators
include("Hamiltonians.jl");    import .Hamiltonians
include("Generators/main.jl"); import .Generators
include("mbpt.jl")

const comm = getfield(Commutators, PARAMS.COMMUTATOR)
const gen = getfield(Generators, PARAMS.GENERATOR)

factorial(T::Type{<:Number}, n::Integer) = prod(one(T):convert(T, n))

function dΩ(Ω, h)
    @debug "Entering dΩ"
    @localgetparams Ω_ATOL, Ω_RTOL, Ω_BATCHSIZE
    @debug "dΩ term" n=0

    prev_tot = zero(Ω)
    prev_ad = gen(h)
    tot = bernoulli(Float64, 0)/factorial(Float64, 0) * prev_ad

    n = 1
    while IMOps.norm(tot - prev_tot) > choosetol(Ω_ATOL, Ω_RTOL*IMOps.norm(tot))
        prev_tot = tot
        tot += sum(n:n+Ω_BATCHSIZE-1) do i
            bernoulli(Float64, i)/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += Ω_BATCHSIZE
    end

    tot
end

function H(Ω, h0)
    @debug "Entering H"
    @localgetparams H_ATOL, H_RTOL, H_BATCHSIZE
    @debug "H term" n=0
    prev_tot = zero(h0)
    prev_ad = h0

    # First term
    tot = 1/factorial(Float64, 0) * prev_ad

    n = 1
    while IMOps.norm(tot - prev_tot) > choosetol(H_ATOL, H_RTOL*IMOps.norm(tot))
        @debug "H term" n
        prev_tot = tot
        tot += sum(n:n+H_BATCHSIZE-1) do i
            1/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += H_BATCHSIZE
    end

    tot
end

solve(h0) = solve((xs...,) -> nothing, h0)
function solve(cb, h0::IMArrayOp; kws...)
    @localgetparams(ELTYPE, INT_ATOL, INT_RTOL, MAX_INT_ITERS, INT_DIV_ATOL,
                    INT_DIV_RTOL, SPBASIS, PRINT_INFO)

    s = 0.0
    n = 0
    Ω = zero(h0)
    h = h0
    dE_2 = mbpt2(h)

    while abs(dE_2) > choosetol(INT_ATOL, INT_RTOL*abs(h.parts[0][]))
        ratio = dE_2/h.parts[0][]
        if n >= max_int_iters
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if abs(dE_2) > choosetol(INT_DIV_ATOL, INT_DIV_RTOL*h.parts[0][])
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end
        PRINT_INFO && _solve_print_info(n, max_int_iters, h.parts[0][], dE_2, ratio)
        cb(s, Ω, h, dE_2)

        Ω += dΩ(Ω, h) * ds
        s += ds
        h = H(Ω, h0)
        dE_2 = mbpt2(h)
        n += 1
    end
    PRINT_INFO && _solve_print_info(n, max_int_iters, h.parts[0][], dE_2, ratio)
    cb(s, Ω, h, dE_2)

    Ω
end

const ALG = AutoTsit5(Rosenbrock23())
solve_nomagnus(h0) = solve_nomagnus((xs...,) -> nothing, h0)
function solve_nomagnus(cb, h0)
    @localgetparams(ELTYPE, INT_ATOL, INT_RTOL, MAX_INT_ITERS, INT_DIV_ATOL,
                    INT_DIV_RTOL, SPBASIS, PRINT_INFO)
    s = 0.0
    n = 0
    h = h0
    dE_2 = mbpt2(h)

    dH(h, s) = comm(gen(h), h)
    integrator = ODEProblem(dH, h, (s, s+S_LARGE_STEP)) |> x -> init(x, ALG)

    while abs(dE_2) > choosetol(INT_ATOL, INT_RTOL*abs(h.parts[0][]))
        ratio = dE_2/h.parts[0][]
        if n >= max_int_iters
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if abs(dE_2) > choosetol(INT_DIV_ATOL, INT_DIV_RTOL*h.parts[0][])
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end
        PRINT_INFO && _solve_print_info(n, max_int_iters, h.parts[0][], dE_2, ratio)
        cb(s, Ω, h, dE_2)

        solve!(integrator)
        h = integrator.sol[end]
        dE_2 = mbpt2(h)
        s += S_LARGE_STEP; add_tstop!(integrator, s+S_LARGE_STEP)
        n += 1
    end
    PRINT_INFO && _solve_print_info(n, max_int_iters, h.parts[0][], dE_2, ratio)
    cb(s, Ω, h, dE_2)

    h
end

function _solve_print_info(n, max_int_iters, E, dE_2, r)
    try
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
    catch
        println(stderr, "ERROR: Failed to print info.")
    end
end

end # module MagnusIMSRG

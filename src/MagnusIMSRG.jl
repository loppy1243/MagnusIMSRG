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

include("parameters.jl")
const PARAMS = Params()
include("getparams.jl")

SIGNAL_OPS && include("signalops.jl")

include("util.jl")
include("IMOperators.jl");      using .IMOperators; const IMOps = IMOperators
include("Commutators.jl");     import .Commutators
const comm = getfield(Commutators, PARAMS.COMMUTATOR)
include("Hamiltonians.jl");    import .Hamiltonians
include("Generators/main.jl"); import .Generators
const gen = getfield(Generators, PARAMS.GENERATOR)
include("mbpt.jl")


factorial(T::Type{<:Number}, n::Integer) = prod(one(T):convert(T, n))

@getparams Ω_ATOL, Ω_RTOL, Ω_BATCHSIZE
function dΩ(Ω, h)
    @debug "Entering dΩ"
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

@getparams H_ATOL, H_RTOL, H_BATCHSIZE
function H(Ω, h0)
    @debug "Entering H"
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
function solve(cb, h0::IMArrayOp)
    @localgetparams(ELTYPE, INT_ATOL, INT_RTOL, MAX_INT_ITERS, INT_DIV_ATOL,
                    INT_DIV_RTOL, SPBASIS, PRINT_INFO, S_SMALL_STEP)

    s = 0.0
    n = 0
    Ω = zero(h0)
    h = h0
    dE_2 = mbpt2(h)

    while abs(dE_2) > choosetol(INT_ATOL, INT_RTOL*abs(h.parts[0][]))
        @show 
        ratio = dE_2/h.parts[0][]
        if n >= MAX_INT_ITERS
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if abs(dE_2) > choosetol(INT_DIV_ATOL, INT_DIV_RTOL*h.parts[0][])
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end
        PRINT_INFO && _solve_print_info(n, h.parts[0][], dE_2, ratio, MAX_INT_ITERS, INT_RTOL)
        cb(s, Ω, h, dE_2)

        Ω += dΩ(Ω, h) * S_SMALL_STEP
        s += S_SMALL_STEP
        h = H(Ω, h0)
        dE_2 = mbpt2(h)
        n += 1
    end
    PRINT_INFO && _solve_print_info(n, h.parts[0][], dE_2, dE_2/h.parts[0][],
                                    MAX_INT_ITERS, INT_RTOL)
    cb(s, Ω, h, dE_2)

    Ω
end

const ALG = RK4()
solve_nomagnus(h0) = solve_nomagnus((xs...,) -> nothing, h0)
function solve_nomagnus(cb, h0::IMArrayOp)
    @localgetparams(ELTYPE, INT_ATOL, INT_RTOL, MAX_INT_ITERS, INT_DIV_ATOL,
                    INT_DIV_RTOL, SPBASIS, PRINT_INFO, S_LARGE_STEP)
    s = 0.0
    n = 0
    h = similar(h0); vec(h) .= vec(h0)
    dE_2 = mbpt2(h)

    function dH(v, _, s)
        vec(h) .= v
        comm(gen(h), h) |> vec |> collect
    end
    integrator = ODEProblem(dH, collect(vec(h0)), (s, s+S_LARGE_STEP)) |> x -> init(x, ALG)

    while abs(dE_2) > choosetol(INT_ATOL, INT_RTOL*abs(h.parts[0][]))
        ratio = dE_2/h.parts[0][]
        if n >= MAX_INT_ITERS
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if abs(dE_2) > choosetol(INT_DIV_ATOL, INT_DIV_RTOL*h.parts[0][])
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end
        PRINT_INFO && _solve_print_info(n, h.parts[0][], dE_2, ratio, MAX_INT_ITERS, INT_RTOL)
        cb(s, nothing, h, dE_2)

        solve!(integrator)
        vec(h) .= integrator.sol[end]
        dE_2 = mbpt2(h)
        s += S_LARGE_STEP; add_tstop!(integrator, s+S_LARGE_STEP)
        n += 1
    end
    PRINT_INFO && _solve_print_info(n, h.parts[0][], dE_2, dE_2/h.parts[0][],
                                    MAX_INT_ITERS, INT_RTOL)
    cb(s, nothing, h, dE_2)

    h
end

function _solve_print_info(n, E, dE_2, r, MAX_INT_ITERS, INT_RTOL)
    try
        sigdigs = 5

        decdigs(x) = sigdigs - ndigits(trunc(Int, x))
        nzdecdig(x) = ceil(Int, -log10(abs(x-trunc(x))))

        r_decdigs = nzdecdig(INT_RTOL)
        E_decdigs = decdigs(E)

        n = lpad(n, ndigits(MAX_INT_ITERS))

        r = round(r, digits=r_decdigs)
        r = rpad(r, ndigits(trunc(Int, r))+1+r_decdigs, '0')

        E = round(E, sigdigits=sigdigs)
        E = rpad(E, (E<0)+sigdigs+1, '0')

        dE_2 = round(dE_2, digits=E_decdigs)
        dE_2 = rpad(dE_2, (dE_2<0)+ndigits(trunc(Int, dE_2))+1+E_decdigs, '0')

        println("$n: E = $E,  dE(2) = $dE_2,  Ratio = $r")
    catch ex
        ex isa InterruptException && rethrow()
        println(stderr, "ERROR: Failed to print info.")
    end
end

end # module MagnusIMSRG

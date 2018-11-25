module MagnusIMSRG

using ManyBody
import ManyBody.Operators
import JuliaUtil
using Parameters, OrdinaryDiffEq
using JuliaUtil: bernoulli, @every
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
    S_LARGE_STEP         ::Float64  = 1.0
    S_SMALL_STEP         ::Float64  = 0.1
    TRUNC_LEVEL          ::Int      = 2
    H                    ::Function
    H_PARAMS             ::Union{Tuple, NamedTuple}
    comm                 ::Function = comm2
    gen                  ::Function = Generators.white
#    HYPARAMS             ::HyParams = HyParams()
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

    ELTYPE          ::Type                     = Float64
    MBBASIS         ::Type                     = Bases.Paired{4, 4}
    SPBASIS         ::Type                     = spbasis(MBBASIS)
    DIM             ::Int                      = dim(spbasis)
    REFSTATE        ::Union{RefState, MBBasis} = RefStates.Fermi{SPBASIS}(2)
    HOLES           ::Vector{SPBASIS}          = holes(REFSTATE)
    PARTS           ::Vector{SPBASIS}          = parts(REFSTATE)
end

SIGNAL_OPS && include("signalops.jl")
include("nbodyops.jl")
include("comm2.jl")
include("hamiltonians.jl")
include("Generators/main.jl"); import .Generators
include("mbpt.jl")


factorial(T::Type{<:Number}, n::Integer) = prod(one(T):convert(T, n))

choosetol(tols...) = minimum(tol in tols if !iszero(tol))

function dΩ(Ω, params, h)
    @debug "Entering dΩ"
    @unpack gen, comm, Ω_ATOL, Ω_RTOL, Ω_BATCHSIZE = params
    @debug "dΩ term" n=0

    prev_tot = zero(Ω)
    prev_ad = gen(h)
    tot = bernoulli(Float64, 0)/factorial(Float64, 0) * prev_ad

    n = 1
    while norm(tot - prev_tot) > choosetol(Ω_ATOL, Ω_RTOL*norm(tot))
        prev_tot = tot
        tot += sum(n:n+Ω_BATCHSIZE-1) do i
            bernoulli(Float64, i)/factorial(Float64, i) * (prev_ad = comm(Ω, prev_ad))
        end
        n += Ω_BATCHSIZE
    end

    tot
end

solve(H, h_params; kws...) = solve((xs...,) -> nothing, H, h_params; kws...)
function solve(cb, H, H_PARAMS; kws...)
    params = Params(; H=H, H_PARAMS=H_PARAMS, kws...)
    @unpack TRUNC_LEVEL, ELTYPE, INT_ATOL, INT_RTOL, MAX_INT_ITERS, INT_DIV_ATOL, INT_DIV_RTOL #=
         =# = params
    ZERO_OP = zero(DenseIMArrayOp{TRUNC_LEVEL, ELTYPE})

    s = 0.0
    n = 0
    Ω = ZERO_OP
    h_prev = ZERO_OP
    h = tabulate(H(H_PARAMS...), DenseIMArrayOp{2}, SPBASIS)
    dE_2 = mbpt2(h)

    while abs(dE_2) > choosetol(INT_ATOL, INT_RTOL*abs(h.parts[0]))
        ratio = dE_2/h.parts[0]
        if n >= max_int_iters
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if abs(dE_2) > choosetol(INT_DIV_ATOL, INT_DIV_RTOL*h.parts[0])
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end
        print_info && _solve_print_info(n, max_int_iters, h.parts[0], dE_2, ratio)
        cb(s, Ω, h, dE_2)

        h_prev = h
        Ω += dΩ(Ω, h) * ds
        s += ds
        h = H(Ω, h0)
        dE_2 = mbpt2(h)
        n += 1
    end
    print_info && _solve_print_info(n, max_int_iters, nbody(h, 0), dE_2, ratio)
    cb(s, Ω, h, dE_2)

    Ω
end

const ALG = AutoTsit5(Rosenbrock23())
function solve_nomagnus(cb, h0; max_int_iters=MAX_INT_ITERS, ds=S_SMALL_STEP, Ds=S_LARGE_STEP, print_info=true)
    tovec(op) = vcat(op[1], op[2][:], op[3][:])
    fromvec(v) = (v[1], reshape(v[2:1+DIM^2], DIM, DIM), reshape(v[2+DIM^2:1+DIM^2+DIM^4], DIM, DIM, DIM, DIM))
    s = 0.0
    n = 0
    h = tovec((h0[1], h0[2].rep, h0[3].rep))
    dE_2 = mbpt2(h0)

    function dh(h, _, s)
        h = fromvec(h)
        comm(generator(h), h) |> tovec
    end
    integrator = ODEProblem(dh, h, (s, s+Ds)) |> x -> init(x, ALG, dt=ds)
    
    while (ratio = abs(dE_2/h[1])) > INT_RTOL
        if n >= max_int_iters
            @warn "Iteration maximum exceeded in solve()" n s
            break
        end
        if ratio > INT_DIV_RTHRESH
            @warn "Divergence threshold exceeded in solve()" n s ratio
            break
        end
        print_info && _solve_print_info(n, max_int_iters, h[1], dE_2, ratio)
        cb(s, h, dE_2)

        solve!(integrator)
        s += Ds; add_tstop!(integrator, s+Ds)
        h = integrator.sol[end]
        dE_2 = mbpt2(fromvec(h))
        n += 1
    end
    print_info && _solve_print_info(n, max_int_iters, h[1], dE_2, ratio)
    cb(s, h, dE_2)

    fromvec(h)
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
        println("ERROR: Failed to print info.")
    end
end

end # module MagnusIMSRG

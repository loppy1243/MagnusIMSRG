@with_kw struct Params
    S_LARGE_STEP    ::Float64                  = 1.0
    S_SMALL_STEP    ::Float64                  = 0.1
    TRUNC_LEVEL     ::Int                      = 2
    COMMUTATOR      ::Symbol                   = :comm2
    GENERATOR       ::Symbol                   = :white
    ENG_DENOM_ATOL  ::Float64                  = 1e-5

    Ω_RTOL          ::Float64                  = 0.0
    Ω_ATOL          ::Float64                  = 0.01
    Ω_BATCHSIZE     ::Int                      = 5

    H_RTOL          ::Float64                  = 0.0
    H_ATOL          ::Float64                  = 0.01
    H_BATCHSIZE     ::Int                      = 5

    INT_RTOL        ::Float64                  = 1e-8
    INT_ATOL        ::Float64                  = 1e-3
    INT_DIV_ATOL    ::Float64                  = 0.0
    INT_DIV_RTOL    ::Float64                  = 1.0
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

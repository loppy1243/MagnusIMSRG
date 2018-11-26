using Test
using ManyBody, MagnusIMSRG, MagnusIMSRG.IMOperators
const IMOps = IMOperators

include("comm.jl"); commtest(atol=1e-5)
include("impairing.jl"); impairingtest(atol=1e-5)

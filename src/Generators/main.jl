module Generators
using ManyBody, ..IMOperators

import ..comm, ..SIGNAL_OPS, ..@getparams

SIGNAL_OPS && include("../signalops.jl")

@getparams ENG_DENOM_ATOL, HOLES, PARTS

include("white.jl")
include("wegner.jl")

end # module Generators

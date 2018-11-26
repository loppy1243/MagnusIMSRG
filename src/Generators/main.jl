module Generators
using ManyBody, ..IMOperators

import ..SIGNAL_OPS, ..@getparams

SIGNAL_OPS && include("../signalops.jl")

@getparams ENG_DENOM_ATOL, HOLES, PARTS

include("white.jl")

end # module Generators

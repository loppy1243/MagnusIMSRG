choosetol(tols...) = all(iszero, tols) ? 0 : minimum(tol for tol in tols if !iszero(tol))

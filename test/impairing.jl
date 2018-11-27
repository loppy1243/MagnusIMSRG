using Combinatorics: permutations, levicivita

import MagnusIMSRG.Hamiltonians: impairing
import ManyBody.Hamiltonians: pairing

impairingtest(; atol, fullmatrix) = @testset "IMPairing Hamiltonian" begin
    @debug "Testing Correctness"
    SPB = Bases.Pairing{4}
    DIM = dim(SPB)
    REF = RefStates.Fermi{SPB}(2)

    MB = if fullmatrix
        Bases.MBPairing{4, 4}
    else
        Bases.Paired{4, 4}
    end
    h_true = tabulate(pairing(1, 0.5), Array{Float64}, 2, MB)
    h_im = tabulate(impairing(REF, 1, 0.5), IMArrayOp{2, Float64},
                    (Array, 2, SPB), (Array, 4, SPB))
    h_mb = mbop(h_im, REF, MB, MB)
    @test all(x -> abs(x) < atol, h_mb - h_true)

    @debug "Testing anti-symmetry"
    arr = reshape(h_im.parts[2], DIM, DIM, DIM, DIM)
    for I in CartesianIndices(arr)
        for perm in permutations([1, 2, 3, 4])
            J = CartesianIndex(Tuple(I)[perm])
            @test arr[I] == levicivita(perm)*arr[J]
        end
    end
end

import MagnusIMSRG.Hamiltonians: impairing
import ManyBody.Hamiltonians: pairing

impairingtest(; atol, fullmatrix) = @testset "IMPairing Hamiltonian" begin
    SPB = Bases.Pairing{4}
    REF = RefStates.Fermi{SPB}(2)

    MB = if fullmatrix
        Bases.Paired{4, 4}
    else
        Bases.MBPairing{4, 4}
    end
    h_true = tabulate(pairing(1, 0.5), Array{Float64}, 2, MB)
    h_im = tabulate(impairing(REF, 1, 0.5), IMArrayOp{2, Float64},
                    (Array, 2, SPB), (Array, 4, SPB))
    h_mb = mbop(h_im, REF, MB, MB)
    @test all(x -> abs(x) < atol, h_mb - h_true)
end

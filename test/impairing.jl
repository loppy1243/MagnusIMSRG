import MagnusIMSRG.Hamiltonians: impairing
import ManyBody.Hamiltonians: pairing

impairingtest(; atol) = @testset "IMPairing Hamiltonian" begin
    SPB = Bases.Pairing{4}
    MB = Bases.Paired{4, 4}
    REF = RefStates.Fermi{SPB}(2)

    h_true = tabulate(pairing(1, 0.5), Array{Float64}, 2, MB)
    h_im = tabulate(impairing(REF, 1, 0.5), IMArrayOp{2, Float64},
                    (Array, 2, SPB), (Array, 4, SPB))
    h_mb = mbop(h_im, MB, MB)

    @test all(x -> abs(x) < atol, h_mb - h_true)
end

import MagnusIMSRG.Hamiltonians: impairing
import ManyBody.Hamiltonians: pairing

impairingtest(; atol, fullmatrix) = @testset "IMPairing H" begin
    SPB = Bases.Pairing{4}
    DIM = dim(SPB)
    REF = RefStates.Fermi{SPB}(2)

    MB = if fullmatrix
        Bases.MBPairing{4, 4}
    else
        Bases.Paired{4, 4}
    end
    h_im = tabulate(impairing(REF, 1, 0.5), IMArrayOp{2, Float64},
                    (Array, 2, SPB), (Array, 4, SPB))

    @testset "Correctness" begin
        h_true = tabulate(pairing(1, 0.5), Array{Float64}, 2, MB)
        h_mb = mbop(h_im, REF, MB, MB)
        @test all(x -> abs(x) < atol, h_mb - h_true)
    end

    arr = reshape(h_im.parts[2], DIM, DIM, DIM, DIM)
    @testset "Hermiticity" begin
        for I in CartesianIndices(arr)
            i, j, k, l = Tuple(I)
            @test arr[i, j, k, l] == conj(arr[k, l, i, j])
        end
    end
    @testset "Anti-symmetry" begin
        for I in CartesianIndices(arr)
            i, j, k, l = Tuple(I)
            @test arr[i, j, k, l] == -arr[i, j, l, k] == arr[j, i, l, k] == -arr[j, i, k, l]
        end
    end
end

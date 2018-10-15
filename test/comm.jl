using MagnusIMSRG: DIM, comm2, comm2_pw, randop, ZERO_OP, hermiticize

commtest(; atol) = @testset "Commutators" begin
    op_1 = randop()
    op_2 = randop()

    mat_E, mat_f, mat_Γ = comm2(op_1, op_2)
    pw_E, pw_f, pw_Γ = comm2_pw(op_1, op_2)

    @test abs(mat_E - pw_E) < atol
    @test all(abs.(mat_f.rep .- pw_f.rep) .< atol)
    @test all(abs.(mat_Γ.rep .- pw_Γ.rep) .< atol)

    op_1 = hermiticize(op_1)
    op_2 = hermiticize(op_2)

    mat_E, mat_f, mat_Γ = comm2(op_1, op_2)

    @test abs(mat_E) < atol
    @test all(abs.(mat_f.rep + mat_f.rep') .< atol)
    @test all(abs.(mat_Γ.rep + PermutedDimsArray(mat_Γ.rep, [3, 4, 1, 2])) .< atol)

    op_1 = ZERO_OP

    mat_E, mat_f, mat_Γ = comm2(op_1, op_2)

    @test abs(mat_E) < atol
    @test all(abs.(mat_f.rep) .< atol)
    @test all(abs.(mat_Γ.rep) .< atol)
end

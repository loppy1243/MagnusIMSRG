using MagnusIMSRG: DIM, comm2, comm2_pw, ARRAYOP, to_mbop

commtest(; atol) = @testset "Commutators" begin
    E0_1 = rand(Float64)
    E0_2 = rand(Float64)
    f_1 = rand(Float64, DIM, DIM)
    f_2 = rand(Float64, DIM, DIM)
    Γ_1 = rand(Float64, DIM, DIM, DIM, DIM)
    Γ_2 = rand(Float64, DIM, DIM, DIM, DIM)

    op_1 = (E0_1, ARRAYOP(1)(f_1), ARRAYOP(2)(Γ_1))
    op_2 = (E0_2, ARRAYOP(1)(f_2), ARRAYOP(2)(Γ_2))

    mbop_1 = to_mbop(op_1) |> tabulate
    mbop_2 = to_mbop(op_2) |> tabulate

    mat_E, mat_f, mat_Γ = comm2(op_1, op_2)
    pw_E, pw_f, pw_Γ = comm2_pw(op_1, op_2)

    @test abs(mat_E - pw_E) < atol
    @test all(abs.(mat_f.rep .- pw_f.rep) .< atol)
    @test all(abs.(mat_Γ.rep .- pw_Γ.rep) .< atol)
end

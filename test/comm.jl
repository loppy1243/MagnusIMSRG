using MagnusIMSRG: DIM, comm, ARRAYOP, to_mbop

function commtest()
    E0_1 = rand(Float64)
    f_1 = ARRAYOP(1)(rand(Float64, DIM, DIM))
    Γ_1 = ARRAYOP(2)(rand(Float64, DIM, DIM, DIM, DIM))
    op_1 = (E0_1, f_1, Γ_1)
    mbop_1 = to_mbop(op_1) |> tabulate

    E0_2 = rand(Float64)
    f_2 = ARRAYOP(1)(rand(Float64, DIM, DIM))
    Γ_2 = ARRAYOP(2)(rand(Float64, DIM, DIM, DIM, DIM))
    op_2 = (E0_2, f_2, Γ_2)
    mbop_2 = to_mbop(op_2) |> tabulate

    op_comm = comm(op_1, op_2) |> to_mbop |> tabulate
    mbop_comm = mbop_1.rep*mbop_2.rep - mbop_2.rep*mbop_1.rep

    x = op_comm.rep .== mbop_comm
    display(op_comm.rep)
    display(mbop_comm)
    display(x)
    @test all(x)
end

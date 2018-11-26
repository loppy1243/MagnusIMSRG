using MagnusIMSRG.IMOperators
using MagnusIMSRG.Commutators: comm2, comm2_pw

commtest(; atol) = @testset "Commutators" begin
    MagnusIMSRG.@localgetparams SPBASIS
    fdims = fulldims(SPBASIS)
    dims(N) = Tuple(d for _=1:N for d in fdims)

    op1 = randimop(Float64, dims(2), dims(4))
    op2 = randimop(Float64, dims(2), dims(4))

    within_tol(x) = all(y -> abs(y) < atol, x)

    mat_comm = comm2(op1, op2)
    pw_comm = comm2_pw(op1, op2)
    @test within_tol(mat_comm - pw_comm)

    op1 = (op1 + op1')/2
    op2 = (op2 + op2')/2
    mat_comm = comm2(op1, op2)
    @test within_tol((mat_comm + mat_comm')/2)

    op1 = zero(op1)
    mat_comm = comm2(op1, op2)
    @test within_tol(mat_comm)
end

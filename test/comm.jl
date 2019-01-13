using MagnusIMSRG.IMOperators
using MagnusIMSRG.Commutators: comm2, comm2_pw

const TWO_BODY_COMMS = [comm2, comm2_pw]

commtest(; atol) = @testset "Commutators" begin
    MagnusIMSRG.@localgetparams SPBASIS
    dims(N) = Tuple(d for _=1:N for d in fulldims(SPBASIS))

    op1 = randimop(Float64, dims(2), dims(4))
    op2 = randimop(Float64, dims(2), dims(4))

    test_within_tol(x) = @testset "$(n-1)-body piece" for (n, part) in enumerate(x.parts)
        @test all(abs.(x) .< atol)
    end

    @testset "$(TWO_BODY_COMMS[i]) vs. $(TWO_BODY_COMMS[j])" #=
 =# for i in eachindex(TWO_BODY_COMMS), j = i+1:lastindex(TWO_BODY_COMMS)
        commA_val = TWO_BODY_COMMS[i](op1, op2)
        commB_val = TWO_BODY_COMMS[j](op1, op2)

        test_within_tol(commA_val - commB_val)
    end

    op1 = (op1 + op1')/2
    op2 = (op2 + op2')/2
    @testset "Hermiticiy ($comm_func)" for comm_func in TWO_BODY_COMMS
        comm_val = comm_func(op1, op2)
        test_within_tol((comm_val + comm_val')/2)
    end

    op1 = zero(op1)
    @testset "Zero ($comm_func)" for comm_func in TWO_BODY_COMMS
        test_within_tol(comm_func(op1, op2))
    end
end

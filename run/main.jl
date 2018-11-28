module RunMagnusIMSRG
using ManyBody, MagnusIMSRG, MagnusIMSRG.IMOperators
using Plots; gr()

using MagnusIMSRG: @getparams, @setparams
import MagnusIMSRG.Hamiltonians: impairing
using LinearAlgebra: eigvals

@setparams(MAX_INT_ITERS = 65)
@getparams SPBASIS, ELTYPE, REFSTATE, MBBASIS

Plots.default(legend=false, dpi=200, grid=false)

function run(; magnus=true)
    mbop(op) = IMOperators.mbop(op, REFSTATE, MBBASIS, MBBASIS)

    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    exact_eigs = eigvals(mbop(h0)) |> sort
    E∞ = exact_eigs[argmin(abs.(exact_eigs.-h0.parts[0][]))]
    D = length(exact_eigs)

    ss = Float64[]
    Es = Float64[]
    dEs = Float64[]
    eigss = Float64[]

    solve = magnus ? MagnusIMSRG.solve : MagnusIMSRG.solve_nomagnus
    solve(h0) do s, Ω, h, dE
        E = h.parts[0][]
        new_eigs = mbop(h) |> eigvals |> sort

        push!(ss, s)
        push!(Es, E)
        push!(dEs, dE)
        eigss = isempty(eigss) ? transpose(new_eigs) : vcat(eigss, transpose(new_eigs))
    end

    E_plt = hline([E∞], title="Zero-body Evolution", label="Exact", ylabel="Energy",
                  linestyle=:dash, color=:black, legend=true)
    plot!(E_plt, ss, [Es Es.+dEs],
          label=["E" "+dE(2)"], color=[:green :red], markershape=[:circle :square],
          markersize=1)
    plot!(E_plt, ss, eigss[:, 1],
          label=["Eigenval"], color=:blue, markershape=:xcross, markersize=1)

    eig_diff_plt = plot(ss, eigss.-transpose(exact_eigs),
                        label=round.(exact_eigs, digits=4), markershape=:circle,
                        markersize=2, legend=:topright, legendfontsize=5,
                        xlabel="Flow Parameter", ylabel="Difference",
                        title="Eigenvalue Evolution")

    plt = plot(layout=(2, 1), E_plt, eig_diff_plt)
    gui(plt)

    ss, Es, dEs, exact_eigs, eigss, plt
end

function run_bare(magnus=true)
    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    solve = magnus ? MagnusIMSRG.solve : MagnusIMSRG.solve_nomagnus
    solve(h0)
end

end # module RunMagnusIMSRG

RunMagnusIMSRG.run(magnus=false)

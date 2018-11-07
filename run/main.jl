module RunMagnusIMSRG

using MagnusIMSRG, ManyBody
using Plots; gr()
using LinearAlgebra: eigvals

Plots.default(legend=false, dpi=200, grid=false)

function run()
    h0 = im_pairingH(0.5)
    exact_eigs = eigvals((h0 |> to_mbop |> tabulate).rep) |> sort
    E∞ = exact_eigs[argmin(abs.(exact_eigs.-nbody(h0, 0)))]
    D = length(exact_eigs)

    ss = []
    Es = []
    eigss = []

    E_plt = hline([E∞], title="Zero-body Evolution", label="Exact", ylabel="Energy",
                  linestyle=:dash, color=:black, legend=true)
    plot!(E_plt, 2, label=["E" "+dE(2)"], color=[:green :red], markershape=[:circle :square])
    eig_diff_plt = plot(D, label=round.(exact_eigs, digits=4), markershape=:circle,
                        markersize=2, legend=:bottomleft, legendfontsize=5,
                        xlabel="Flow Parameter", ylabel="Difference",
                        title="Eigenvalue Evolution")
    plt = plot(layout=(2, 1), E_plt, eig_diff_plt)

    MagnusIMSRG.solve(h0; max_int_iters=20, ds=0.5) do s, Ω, h, dE
        E = nbody(h, 0)
        new_eigs = tabulate(to_mbop(h)).rep |> eigvals |> sort

        push!(ss, s)
        push!(Es, E)
        eigss = vcat(eigss, new_eigs)
        
        push!(E_plt, 2, s, E)
        push!(E_plt, 3, s, E+dE)
        push!(eig_diff_plt, s, new_eigs.-exact_eigs)
        gui(plt)
    end

    ss, Es, exact_eigs, eigss, plt
end

function run_bare()
    MagnusIMSRG.solve(im_pairingH(0.5); max_int_iters=20, ds=0.5, print_info=false)
    nothing
end
    
matgen() = MagnusIMSRG.im_pairingH(0.5) |> x -> MagnusIMSRG.to_mbop(x) |> tabulate
matgen2() = MagnusIMSRG.im_pairingH(1.0) |> x -> MagnusIMSRG.to_mbop2(x)

end # module RunMagnusIMSRG

RunMagnusIMSRG.run()

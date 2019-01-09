module RunMagnusIMSRG
using ManyBody, MagnusIMSRG, MagnusIMSRG.IMOperators
using Plots, LaTeXStrings; gr()

using MagnusIMSRG: @getparams, @setparams
import MagnusIMSRG.Hamiltonians: impairing
using LinearAlgebra: eigvals

@setparams(MAX_INT_ITERS = 300, INT_RTOL = 1e-8)
@getparams SPBASIS, ELTYPE, REFSTATE, MBBASIS, PARTS, HOLES

Plots.default(legend=false, dpi=200, grid=false)

#function run(; magnus=true)
#    mbop(op) = IMOperators.mbop(op, REFSTATE, MBBASIS, MBBASIS)
#
#    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
#                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
#    exact_eigs = eigvals(mbop(h0)) |> sort
#    E∞ = exact_eigs[argmin(abs.(exact_eigs.-h0.parts[0][]))]
#    D = length(exact_eigs)
#
#    ss = Float64[]
#    Es = Float64[]
#    dEs = Float64[]
#    eigss = Float64[]
#
#    solve = magnus ? MagnusIMSRG.solve : MagnusIMSRG.solve_nomagnus
#    solve(h0) do s, Ω, h, dE
#        E = h.parts[0][]
#        new_eigs = mbop(h) |> eigvals |> sort
#
#        push!(ss, s)
#        push!(Es, E)
#        push!(dEs, dE)
#        eigss = isempty(eigss) ? transpose(new_eigs) : vcat(eigss, transpose(new_eigs))
#    end
#
#    E_plt = hline([E∞], title="Zero-body Evolution", label="Exact", ylabel="Energy",
#                  linestyle=:dash, color=:black, legend=true)
#    plot!(E_plt, ss, [Es Es.+dEs],
#          label=["E" "+dE(2)"], color=[:green :red], markershape=[:circle :square],
#          markersize=1)
#    plot!(E_plt, ss, eigss[:, 1],
#          label=["Eigenval"], color=:blue, markershape=:xcross, markersize=1)
#
#    eig_diff_plt = plot(ss, eigss.-transpose(exact_eigs),
#                        label=round.(exact_eigs, digits=4), markershape=:circle,
#                        markersize=2, legend=:topright, legendfontsize=5,
#                        xlabel="Flow Parameter", ylabel="Difference",
#                        title="Eigenvalue Evolution")
#
#    plt = plot(layout=(2, 1), E_plt, eig_diff_plt)
#    gui(plt)
#
#    ss, Es, dEs, exact_eigs, eigss, plt
#end

function run_bare(magnus=true)
    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    solve = magnus ? MagnusIMSRG.solve : MagnusIMSRG.solve_nomagnus
    solve(h0)
end

plot!(f; kws...) = plot!(Plots.current(), f; kws...)
function plot!(plt, f; file, magnus=true, kws...)
    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    solve = magnus ? MagnusIMSRG.solve : MagnusIMSRG.solve_nomagnus

    data = solve(f, h0)
    Plots.plot!(plt, data...; kws...)
    savefig(file)
end
plot(f; kws...) = plot!(Plots.plot(), f; kws...)

function offdiag(magnus=true)
    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    solve = magnus ? MagnusIMSRG.solve : MagnusIMSRG.solve_nomagnus

    f_od_norms = []
    Γ_od_norms = []
    ss = []
    solve(h0) do s, Ω, h, dE
        E, f, Γ = h.parts

        f_od_norm_sq = zero(ELTYPE)
        Γ_od_norm_sq = zero(ELTYPE)
        for p in PARTS, h in HOLES
            f_od_norm_sq += abs(f[p, h] + conj(f[h, p]))^2

            for p′ in PARTS, h′ in HOLES
                Γ_od_norm_sq += abs(Γ[p, p′, h, h′] + conj(Γ[h, h′, p, p′]))^2
            end
        end

        push!(f_od_norms, sqrt(f_od_norm_sq))
        push!(Γ_od_norms, sqrt(Γ_od_norm_sq))
        push!(ss, s)
    end

    plt_lin = plot(title="Off-diagonal Norms", xlabel="Flow Parameter", ylabel="Norm", legend=true)
#    plot!(ss, f_od_norms, label=L"||f_{\mathrm{od}}||", #=markershape=:circle,=# markersize=1)
    plot!(ss, Γ_od_norms, label=L"||\Gamma_{\mathrm{od}}||", #=markershape=:square,=# markersize=1)

    plt_log = plot(ss, log.(Γ_od_norms))
    savefig(plot(layout=(2,1), plt_lin, plt_log), "data/od_norms.pdf")
end

function offdiag(s, Ω, h, dE)
    E, f, Γ = h.parts

    f_od_norm_sq = zero(ELTYPE)
    Γ_od_norm_sq = zero(ELTYPE)
    for p in PARTS, h in HOLES
        f_od_norm_sq += abs(f[p, h] + conj(f[h, p]))^2

        for p′ in PARTS, h′ in HOLES
            Γ_od_norm_sq += abs(Γ[p, p′, h, h′] + conj(Γ[h, h′, p, p′]))^2
        end
    end

    (s, sqrt(Γ_od_norm_sq))
end
ratio(s, Ω, h, dE) = (s, dE/h.parts[0][])
logratio(s, Ω, h, dE) = (s, log10(abs(dE/h.parts[0][])))

function plot_energy()
    energy(s, Ω, h, dE) = (s, [h.parts[0][] h.parts[0][]+dE])

    mbop(op) = IMOperators.mbop(op, REFSTATE, MBBASIS, MBBASIS)

    # Redundancy...
    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    exact_eigs = eigvals(mbop(h0)) |> sort
    E∞ = exact_eigs[argmin(abs.(exact_eigs.-h0.parts[0][]))]

    hline([E∞], legend=true, label="Exact", linestyle=:dash, color=:black)
    plot!(energy, file="data/energy.pdf", labels=["E" "E+dE"], title="Energy Evolution",
          xlabel="Flow Parameter", ylabel="Energy")
end

function plot_eigvals()
    # Redundancy...
    h0 = tabulate(impairing(REFSTATE, 1, 0.5), IMArrayOp{2, ELTYPE},
                  (Array, 2, SPBASIS), (Array, 4, SPBASIS))
    mbop(x) = IMOperators.mbop(x, REFSTATE, MBBASIS, MBBASIS)
    eig0s = eigvals(mbop(h0)) |> sort

    function eigenvals(s, Ω, h, dE)
        eigs = eigvals(mbop(h)) |> sort
        (s, transpose(eigs.-eig0s))
    end

    plot(eigenvals, file="data/eigenvals.pdf", title="Eigenvalue Evolution",
         xlabel="Flow Parameter", ylabel=L"\lambda(s)-\lambda(s=0)")
end


#RunMagnusIMSRG.plot_energy()
RunMagnusIMSRG.plot_eigvals()
#RunMagnusIMSRG.plot(ratio, file="data/ratios.pdf", title="Convergence Ratio",
#                    xlabel="Flow Parameter", ylabel=L"\mathrm{d}E/E")
#RunMagnusIMSRG.plot(logratio, file="data/logratios.pdf", title="Convergence Ratio",
#                    xlabel="Flow Parameter", ylabel=L"\log|\mathrm{d}E/E|")
end # module RunMagnusIMSRG

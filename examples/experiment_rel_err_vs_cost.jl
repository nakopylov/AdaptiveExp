module ExperimentRelativeErrorVsCost

include("../src/AdaptiveExp.jl")
using .AdaptiveExp
using LinearAlgebra
using ArbNumerics, Readables
using Plots, LaTeXStrings, Latexify
pgfplotsx()

function normalize(A::AbstractMatrix, type::Number = 1)
    return A / opnorm(A, type)
end

function rel_err(A::AbstractMatrix, X::AbstractMatrix, ref::AbstractMatrix)
    return opnorm(ref - X, 1) / (opnorm(ref, 1) * opnorm(A, 1))
end

function estimate_error(A, tol; force_diag = false)
    ref = Float64.(exp(A) * ArbFloat(1))
    X, cost, label = expadapt(A, tol; force_diag = force_diag, method_info = true)
    error = rel_err(A, X, ref)
    return [error, cost, label]
end

function run_err_vs_cost_experiment(; force_diag = false)
    dim = 101
    n_repeats = 100
    norms = (10.0 .^ (2.0:-1:-3.0))
    tols = 10.0 .^ (0.0:-1.0:-16)

    if force_diag
        lut = diag_lut
    else
        lut = nondiag_lut
    end
    figs = []
    mms = []
    for NORM in trunc.(norms, digits = 8)
        costs = zero(tols)
        errs = zeros(2, length(tols))
        ms = []
        for _ in 1:n_repeats
            rn = Int(floor(dim / 2))
            if iseven(dim)
                left = -rn + 1
            else
                left = -rn
            end
            D = diagm(0 => collect((-left):rn))
            R = rand(-1:1, dim, dim)
            R .+= D
            A = NORM * normalize(R)
            tol_cost = t -> lookup_method(t, NORM, lut)[end]
            costs += tol_cost.(tols)

            err_cost = t -> estimate_error(A, t; force_diag = force_diag)
            tmp = reduce(hcat, err_cost.(tols))

            errs = errs + tmp[1:2, :]
            ms = tmp[3, :]
            push!(mms, ms)
        end

        costs ./= n_repeats
        errs ./= n_repeats

        fig = plot(costs, tols,
            linetype = :steppost,
            linewidth = 2, line = :dash,
            label = L"tol",
            title = L"|| hA ||_{1}=%$NORM",
            xlabel = L"cost,\ [$C$]",
            ylabel = L"|| w_{\alpha}(h A)-e^{h A} ||_{1} / (|| hA ||_{1} \cdot || e^{hA} ||_{1})",
            yscale = :log10,
            ylims = 10.0 .^ [-17, 0],
            yticks = 10.0 .^ [-16, -8, -12, -4],
            legend = false)

        refm, refs, refc = lookup_method(1e-16, NORM, matlab_lut)
        vline!([refc], annotations = ([0.99 * float(refc)], [1e-8], [refm.label]),
            line = :dot, linecolor = :black)

        plot!(errs[2, :], errs[1, :],
            linetype = :steppost,
            linewidth = 2,
            label = L"relerr",
            legend = false,
            annotations = (errs[2, :], errs[1, :] .* 5, ms)
        )

        push!(figs, fig)
        display(fig)
    end
end

run_err_vs_cost_experiment(; force_diag = true)
run_err_vs_cost_experiment(; force_diag = false)

end
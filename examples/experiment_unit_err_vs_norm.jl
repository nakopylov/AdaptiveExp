module ExperimentUnitaryErrorVsNorm

include("../src/AdaptiveExp.jl")
using .AdaptiveExp
using LinearAlgebra
using SkewLinearAlgebra
using Plots, LaTeXStrings, Latexify
pgfplotsx()

function normalize(A::AbstractMatrix, type::Number = 1)
    return A / opnorm(A, type)
end

function unit_err(X)
    return opnorm(adjoint(X) * X - I, 1) + eps()
end

function run_unit_err_experiment(A::AbstractMatrix, norms, tol = 1e-16, ref_errs = nothing)
    fig = plot(; title = L"tol=%$tol",
        xlabel = L"|| A ||_{1}",
        ylabel = L"|| r_{m,m}^{T}(A) J r_{m,m}(A) - J ||_{1} + \varepsilon")
    if !isnothing(ref_errs)
        plot!(norms, ref_errs, label = L"ref", line = :dash, linewidth = 5)
    end
    errs = []
    costs = []
    labels = []
    for n in norms
        C = n * A
        X, cst, lbl = expadapt(C, tol; force_diag = true, method_info = true)
        err = unit_err(X)
        push!(errs, err)
        push!(costs, latexify(rationalize(cst)))
        push!(labels, lbl)
    end
    plot!(norms, errs,
        linewidth = 1.5,
        xscale = :log10, yscale = :log10,
        ylims = 10.0 .^ [-17, -7],
        yticks = 10.0 .^ [-16, -8, -12, -4])
    annotate!(norms[1:4:end], (errs ./ 10)[1:4:end], costs[1:4:end])
    annotate!(norms[1:4:end], (errs .* 10)[1:4:end], labels[1:4:end])
    return fig
end

function get_ref_errs(A, norms)
    errs = Float64[]
    for n in norms
        err = unit_err(exp(n * A))
        push!(errs, err)
    end
    return errs
end

dim = 53
rn = Int(floor(dim / 2))
D = diagm(0 => collect((-rn):rn))
A = [[zero(D) D];
     [-D zero(D)]]
A = SkewHermitian(normalize(A))

norms = 10.0 .^ (-4.0:0.25:3.0)
tols = 10.0 .^ [-4, -8, -16]

figs = []
for tol in tols
    ref_errs = get_ref_errs(A, norms)
    f = run_unit_err_experiment(A, norms, tol, ref_errs)
    push!(figs, f)
end
display(plot(figs...))

dim = 101
B = rand(dim, dim)
B = B + B'
C = rand(dim, dim)
C = C - C'
A = normalize(1im * B + C)

figs = []
for tol in tols
    ref_errs = get_ref_errs(A, norms)
    f = run_unit_err_experiment(A, norms, tol, ref_errs)
    push!(figs, f)
end
display(plot(figs...))

end # module
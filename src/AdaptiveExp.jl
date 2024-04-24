module AdaptiveExp

export expadapt
export t2, t4, t8, t15, t21
export pr2_1, pr4_2, pr6_3, pr6_4, pr8_4, pr8_5, pr12_8
export pr2_2, pr3_3, pr4_4, pr5_5, pr6_6, pr7_7, pr8_8, pr9_9, pr13_13, pr12_12
export estimate_cost, lookup_method
export matlab_lut, diag_lut, nondiag_lut

using LinearAlgebra
using SkewLinearAlgebra
using LaTeXStrings

abstract type ExpApproximant end

include("coefs_diag_pade.jl")
include("coefs_sastre.jl")
include("coefs_superdiag_pade.jl")
include("coefs_taylor.jl")

function expadapt(A::AbstractMatrix)
    return expadapt(A, 1e-12; force_diag = false, method_info = false)
end

function expadapt(A::AbstractMatrix, tol;
        force_diag = false, method_info = false)
    if force_diag
        res = expadapt(A, tol, diag_lut; method_info = method_info)
    else
        res = expadapt(A, tol, nondiag_lut; method_info = method_info)
    end
    return res
end

function expadapt(A::AbstractMatrix, tol::Number, lut;
        method_info = false)
    n = LinearAlgebra.checksquare(A)

    tmp = similar(A)
    tmp .= A
    nA = opnorm(A, 1)

    method, s, total_cost = lookup_method(tol, nA, lut)
    rmul!(tmp, 1 / 2.0^s)
    X = expadapt(tmp, method)

    if s > 0
        for _ in 1:s
            mul!(tmp, X, X)
            X, tmp = tmp, X
        end
    end

    if method_info
        return [X, total_cost, method.label]
    else
        return X
    end
end

function expadapt(A::Union{Hermitian, SkewHermitian}, method::ExpApproximant, tol::Number;
        force_diag = true, method_info = false)
    if isa(A, SkewHermitian)
        # hack to skip ERROR: MethodError: no method matching strides(::SkewLinearAlgebra.SkewHermitian{Float64, Matrix{Float64}})
        A = Matrix(A)
    end
    expadapt(A, method, tol, diag_lut; method_info = method_info)
end

function expadapt(A::Union{Hermitian, SkewHermitian}, tol::Number;
        force_diag = true, method_info = false)
    if isa(A, SkewHermitian)
        # hack to skip ERROR: MethodError: no method matching strides(::SkewLinearAlgebra.SkewHermitian{Float64, Matrix{Float64}})
        A = Matrix(A)
    end
    expadapt(A, tol, diag_lut; method_info = method_info)
end

function expadapt(A::AbstractMatrix, method::ExpApproximant, tol::Number;
        force_diag = false, method_info = false)
    if force_diag
        expadapt(A, method, tol, diag_lut; method_info = method_info)
    else
        expadapt(A, method, tol, nondiag_lut; method_info = method_info)
    end
end

function expadapt(A::AbstractMatrix, method::ExpApproximant, tol::Number, lut;
        method_info = false)
    n = LinearAlgebra.checksquare(A)

    tmp = similar(A)
    tmp .= A
    nA = opnorm(A, 1)

    _, theta = select_method(tol, lut, method)
    s = 0
    while nA > theta
        rmul!(tmp, 1 / 2)
        nA /= 2
        s += 1
    end

    X = expadapt(tmp, method)

    if s > 0
        for _ in 1:s
            mul!(tmp, X, X)
            X, tmp = tmp, X
        end
    end

    total_cost = method.cost + s
    if method_info
        return [X, total_cost]
    else
        return X
    end
end

function expadapt(A::AbstractMatrix, method::PartitionedPadeExp)
    n = LinearAlgebra.checksquare(A)

    As = vcat([I(n)], [A^k for k in range(1, method.polyorder)])
    X = sum(method.polycoefs .* As)

    for rc in method.ratcoefs
        nm = rc[1, :]
        dn = rc[2, :]
        tmp1 = zero(A)
        tmp2 = zero(A)
        for k in eachindex(nm)
            tmp1 .+= nm[k] * As[k]
            tmp2 .+= dn[k] * As[k]
        end
        X .+= (tmp2 \ tmp1)

        tmp1 .= 0.0 * tmp1
        tmp2 .= 0.0 * tmp2
    end

    return X
end

function expadapt(A::AbstractMatrix, method::PartitionedDiagPadeExp)
    n = LinearAlgebra.checksquare(A)

    As = vcat([I(n)], [A^k for k in range(1, method.polyorder)])
    X = Matrix(zero(A))

    for rc in method.ratcoefs
        nm = rc[1, :]
        dn = rc[2, :]
        tmp1 = Matrix(zero(A))
        tmp2 = Matrix(zero(A))
        for k in eachindex(nm)
            tmp1 .+= nm[k] * As[k]
            tmp2 .+= dn[k] * As[k]
        end
        X .+= (tmp2 \ tmp1)

        tmp1 .= 0.0 * tmp1
        tmp2 .= 0.0 * tmp2
    end

    return X
end

function expadapt(A::AbstractMatrix, method::TaylorExp)
    n = LinearAlgebra.checksquare(A)
    As = vcat([I(n)], [A^k for k in range(1, length(method.polycoefs) - 1)])
    X = sum(method.polycoefs .* As)
    return X
end

function expadapt(A::AbstractMatrix, method::TaylorExp8)
    n = LinearAlgebra.checksquare(A)
    A2 = A * A
    A4 = A2 * (method.x[1] * A + method.x[2] * A2)
    A8 = (method.x[3] * A2 + A4) *
         (method.x[4] * I(n) + method.x[5] * A + method.x[6] * A2 + method.x[7] * A4)
    X = method.y[0 + 1] * I(n) + method.y[1 + 1] * A + method.y[2 + 1] * A2 + A8
    return X
end

function expadapt(A::AbstractMatrix, method::TaylorExp12)
    n = LinearAlgebra.checksquare(A)

    A2 = A * A
    A3 = A2 * A

    B1 = method.A[0 + 1, 1] * I(n) + method.A[1 + 1, 1] * A + method.A[2 + 1, 1] * A2 +
         method.A[3 + 1, 1] * A3
    B2 = method.A[0 + 1, 2] * I(n) + method.A[1 + 1, 2] * A + method.A[2 + 1, 2] * A2 +
         method.A[3 + 1, 2] * A3
    B3 = method.A[0 + 1, 3] * I(n) + method.A[1 + 1, 3] * A + method.A[2 + 1, 3] * A2 +
         method.A[3 + 1, 3] * A3
    B4 = method.A[0 + 1, 4] * I(n) + method.A[1 + 1, 4] * A + method.A[2 + 1, 4] * A2 +
         method.A[3 + 1, 4] * A3
    A6 = B3 + B4^2
    X = B1 + (B2 + A6) * A6
    return X
end

function expadapt(A::AbstractMatrix, method::TaylorExp18)
    n = LinearAlgebra.checksquare(A)

    A2 = A * A
    A3 = A2 * A
    A6 = A3 * A3

    B1 = method.a[1 + 0] * I(n) + method.a[1 + 1] * A + method.a[1 + 2] * A2 +
         method.a[1 + 3] * A3
    B2 = method.B[0 + 1, 1] * I(n) + method.B[1 + 1, 1] * A + method.B[2 + 1, 1] * A2 +
         method.B[3 + 1, 1] * A3 + method.B[5, 1] * A6
    B3 = method.B[0 + 1, 2] * I(n) + method.B[1 + 1, 2] * A + method.B[2 + 1, 2] * A2 +
         method.B[3 + 1, 2] * A3 + method.B[5, 2] * A6
    B4 = method.B[0 + 1, 3] * I(n) + method.B[1 + 1, 3] * A + method.B[2 + 1, 3] * A2 +
         method.B[3 + 1, 3] * A3 + method.B[5, 3] * A6
    B5 = method.B[0 + 1, 4] * I(n) + method.B[1 + 1, 4] * A + method.B[2 + 1, 4] * A2 +
         method.B[3 + 1, 4] * A3 + method.B[5, 4] * A6

    A9 = B1 * B5 + B4
    X = B2 + (B3 + A9) * A9
    return X
end

function expadapt(A::AbstractMatrix, method::Sastre22)
    n = LinearAlgebra.checksquare(A)

    A2 = A * A
    Y02 = A2 * (method.c[16] * A2 + method.c[15] * A)
    Y12 = (Y02 + method.c[14] * A2 + method.c[13] * A) *
          (Y02 + method.c[12] * A2 + method.c[11] * I(n)) + method.c[10] * Y02
    Y22 = (Y12 + method.c[9] * A2 + method.c[8] * A) *
          (Y12 + method.c[7] * Y02 + method.c[6] * A) + method.c[5] * Y12 +
          method.c[4] * Y02 +
          method.c[3] * A2 + method.c[2] * A + method.c[1] * I(n)
    return Y22
end

function expadapt(A::AbstractMatrix, method::Sastre23)
    n = LinearAlgebra.checksquare(A)

    A2 = A * A
    A3 = A2 * A
    Y03 = A3 * (method.c[1] * A3 + method.c[2] * A2 + method.c[3] * A)
    Y13 = (Y03 + method.c[4] * A3 + method.c[5] * A2 + method.c[6] * A) *
          (Y03 + method.c[7] * A3 + method.c[8] * A2) + method.c[9] * Y03 +
          method.c[10] * A3 + method.c[11] * A2
    Y23 = (Y13 + method.c[12] * A3 + method.c[13] * A2 + method.c[14] * A) *
          (Y13 + method.c[15] * Y03 + method.c[16] * A) +
          method.c[17] * Y13 + method.c[18] * Y03 + method.c[19] * A3 + method.c[20] * A2 +
          A + I(n)
    return Y23
end

function expadapt(A::AbstractMatrix, method::Pade1313)
    n = LinearAlgebra.checksquare(A)

    A2 = A * A
    A4 = A2 * A2
    A6 = A2 * A4

    U = A * (A6 *
         (method.ratcoefs[14] * A6 + method.ratcoefs[12] * A4 + method.ratcoefs[10] * A2) +
         method.ratcoefs[8] * A6 + method.ratcoefs[6] * A4 + method.ratcoefs[4] * A2 +
         method.ratcoefs[2] * I)
    V = A6 *
        (method.ratcoefs[13] * A6 + method.ratcoefs[11] * A4 + method.ratcoefs[9] * A2) +
        method.ratcoefs[7] * A6 + method.ratcoefs[5] * A4 + method.ratcoefs[3] * A2 +
        method.ratcoefs[1] * I
    X = (V .- U) \ (V .+ U)
    return X
end

include("lut.jl")

end # module AdaptiveExp

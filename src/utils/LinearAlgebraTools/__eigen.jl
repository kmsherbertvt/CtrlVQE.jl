import TemporaryArrays: @temparray

import LinearAlgebra # For eigen!
import LinearAlgebra: Hermitian

"""
    eigen!(Λ::Vector, U::Matrix, A::Matrix)

Diagonalize a matrix with minimal allocations.

# Parameters
- Λ: Vector where eigenvalues are written.
- U: Matrix where eigenvectors are written. `U[:,i]` corresponds to `Λ[i]`.
- A: An (abstract) matrix to diagonalize - assumed to be Hermitian.

# Returns
This function explicitly returns nothing.

"""
function eigen!(Λ::Vector, U::Matrix, A::AbstractMatrix)
    N = size(A,1)
    N == 1 && return __eigensolve__1!(Λ,U,A)
    N == 2 && return __eigensolve__2!(Λ,U,A)

    # No other recourse.
    F = Complex{real(eltype(A))}
    inplace = @temparray(F, size(A), :eigen, :inplace)
    inplace .= A
    ΛU = LinearAlgebra.eigen!(Hermitian(inplace))

    Λ .= ΛU.values
    U .= ΛU.vectors
    return
end

function __eigensolve__1!(Λ, U, A)
    a = only(A)
    Λ .= a
    U .= 1
    return
end

function __eigensolve__2!(Λ, U, A)
    a, c, b, d = A

    Λ[1] = (a+d)/2 - sqrt(((a-d)/2)^2 + abs2(b))
    Λ[2] = (a+d)/2 + sqrt(((a-d)/2)^2 + abs2(b))

    if abs(b) < eps(real(eltype(A)))
        U .= 0
        U[1,1] = 1
        U[2,2] = 1
        return
    end

    U[1,1] = (1 + abs2(b)/(d - Λ[1])^2)^(-1/2)
    U[2,1] = U[1,1] * -c / (d-Λ[1])
    U[2,2] = (1 + abs2(b)/(a - Λ[2])^2)^(-1/2)
    U[1,2] = U[2,2] * -b / (a-Λ[2])
    return
end

#= TODO:

Maybe some magic with Polynomials and \, like we did in AnalyticSolutions?
I'd like to see when if ever that method is faster/more compact than eigen!.

In any case, we should definitely do eigensolve_3 and _4 also.

=#
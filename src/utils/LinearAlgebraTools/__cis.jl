using .LinearAlgebraTools: eigen!

import TemporaryArrays: @temparray

import LinearAlgebra: Diagonal

"""
    cis_type(x)

Promote the number type of `x` to a complex float (compatible with cis operations).

The argument `x` may be a number, an array of numbers, or a number type itself.

"""
function cis_type(x)
    F = real(eltype(x))
    return F <: Integer ? ComplexF64 : Complex{F}
end

"""
    cis!(A::AbstractMatrix, x=1)

Calculates ``\\exp(ixA)`` for a Hermitian matrix `A`.

The name comes from the identity `exp(ix) = Cos(x) + I Sin(x)`.

Note that this method mutates `A` itself to the calculated exponential.
Therefore, `A` must have a complex float type, and it must not be an immutable view.
For example, even though `A` must be Hermitian for this method to work correctly,
    it can't actually be a `LinearAlgebra.Hermitian` view.

```jldoctests
julia> A = ComplexF64[0 1; 1 0];

julia> LAT.cis!(A, π/4) * √2
2×2 Matrix{ComplexF64}:
 1.0+0.0im  0.0+1.0im
 0.0+1.0im  1.0+0.0im

```

"""
function cis!(A::AbstractMatrix{<:Complex{<:AbstractFloat}}, x::Number=1)
    F = Complex{real(eltype(A))}
    Λ = @temparray(F, (size(A,1),), :cis, :values)
    U = @temparray(F, size(A), :cis, :vectors)

    eigen!(Λ, U, A)

    Λ .= exp.((im*x) .* Λ)
    left = @temparray(F, size(A), :cis, :left)
    left = mul!(left, U, Diagonal(Λ))

    return mul!(A, left, U')                # NOTE: OVERWRITES INPUT
end
using .LinearAlgebraTools: MatrixList, kron

import TemporaryArrays: @temparray

import LinearAlgebra: mul!

"""
    rotate!(R, x)

Apply the rotation `R` to the object `x`, mutating `x`.

Generally, `R` is a unitary (or orthogonal) matrix.
If `x` is a vector, `rotate!` computes ``x ← Rx``.
If `x` is a matrix, `rotate!` computes ``x ← RxR'``.

You may also pass `R` as a `MatrixList`,
    which is interpreted as a rotation with a factorized tensor structure.
In other words, if `r̄` is a `MatrixList`,
    `rotate!(r̄, x)` is equivalent to `rotate!(kron(r̄), x)`,
    except that the former has a more efficient implementation.

Since this method mutates `x`,
    the number type of `x` must be sufficiently expressive.
For example, if `R` is a unitary matrix, `x` had better be a vector of complex floats.

```jldoctests
julia> LAT.rotate!([0 1; 1 0], [1; 0]) # X|0⟩
2-element Vector{Int64}:
 0
 1

julia> LAT.rotate!([0 1; 1 0], [1 0; 0 -1]) # XZX'
2×2 Matrix{Int64}:
 -1  0
  0  1

```

"""
function rotate! end


function rotate!(R::AbstractMatrix{F_}, x::AbstractArray{F}) where {F_, F}
    # NOTE: Throws method-not-found error if x is not at least as rich as R.
    return rotate!(promote_type(F_, F), R, x)
end

function rotate!(::Type{F}, R::AbstractMatrix{F_}, x::AbstractVector{F}) where {F_, F}
    temp = @temparray(F, size(x), :rotate, :vector)
    x .= mul!(temp, R, x)
    return x
end

function rotate!(::Type{F}, R::AbstractMatrix{F_}, A::AbstractMatrix{F}) where {F_, F}
    left = @temparray(F, size(A), :rotate, :matrix)
    left = mul!(left, R, A)
    return mul!(A, left, R')
end

function rotate!(r̄::MatrixList{F_}, x::AbstractArray{F}) where {F_, F}
    # NOTE: Throws method-not-found error if x is not at least as rich as R.
    return rotate!(promote_type(F_, F), r̄, x)
end

function rotate!(::Type{F},
    r̄::MatrixList{F_},
    x::AbstractVector{F}
) where {F_, F}
    N = length(x)
    for i in axes(r̄,3)
        r = @view(r̄[:,:,i])                     # CURRENT ROTATOR
        m = size(r,1)
        x_ = transpose(reshape(x, (N÷m,m)))     # CREATE A PERMUTED VIEW
        temp = @temparray(F, (m,N÷m), :rotate, :vector, :product)
        temp = mul!(temp, r, x_)                # APPLY THE CURRENT OPERATOR
        x .= vec(temp)                          # COPY RESULT TO ORIGINAL STATE
    end
    return x
end

function rotate!(::Type{F},
    r̄::MatrixList{F_},
    A::AbstractMatrix{F}
) where {F_, F}
    # TODO (mid): Write this with tensor algebra
    return rotate!(kron(r̄), A)
end
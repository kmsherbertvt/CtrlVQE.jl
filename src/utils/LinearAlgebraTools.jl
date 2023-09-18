import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using LinearAlgebra: kron!, eigen, Diagonal, Hermitian, mul!

"""
    VectorList{T}

Semantic alias for `Matrix` which explicitly represents a distinct vector in each column.

Think of a `VectorList` `v̄` as a list where each `v[i] = v̄[:,i]` is a vector.

"""
const VectorList{T} = AbstractArray{T,2}

"""
    MatrixList{T}

Semantic alias for a 3d array which explicitly represents a list of matrices.

Think of a `MatrixList` `Ā` as a list where each `A[i] = Ā[:,:,i]` is a matrix.

"""
const MatrixList{T} = AbstractArray{T,3}

"""
    kron(v̄::VectorList; result=nothing)

The kronecker product of each vector in `v̄`.

Ordering: [x1 x2] ⊗ [y1 y2] = [x1⋅y1  x1⋅y2  x2⋅y1  x2⋅y2]

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function kron(v̄::VectorList{F}; result=nothing) where {F}
    isnothing(result) && (result = Vector{F}(undef, size(v̄,1)^size(v̄,2)))

    op  = array(F, (1,), LABEL); op[1] = one(F)
    tgt = nothing
    for i in axes(v̄,2)
        shape = (length(op)*size(v̄,1),)
        tgt = array(F, shape, LABEL)
        kron!(tgt, op, @view(v̄[:,i]))
        op = tgt
    end
    result .= tgt

    return result
end

"""
    kron(Ā::MatrixList; result=nothing)

The kronecker product of each matrix in `Ā`.

"""
function kron(Ā::MatrixList{F}; result=nothing) where {F}
    if result === nothing
        totalsize = (size(Ā,1)^size(Ā,3), size(Ā,2)^size(Ā,3))
        result = Matrix{F}(undef, totalsize)
    end

    op  = array(F, (1,1), LABEL); op[1] = one(F)
    tgt = nothing
    for i in axes(Ā,3)
        shape = size(op) .* (size(Ā,1), size(Ā,2))
        tgt = array(F, shape, LABEL)
        kron!(tgt, op, @view(Ā[:,:,i]))
        op = tgt
    end
    result .= tgt

    return result
end

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

"""
function cis!(A::AbstractMatrix{<:Complex{<:AbstractFloat}}, x::Number=1)
    Λ, U = eigen(Hermitian(A))              # TODO (lo): UNNECESSARY ALLOCATIONS

    F = Complex{real(eltype(Λ))}
    diag = array(F, size(Λ), LABEL)
    diag .= exp.((im*x) .* Λ)

    left = array(F, size(A), LABEL)
    left = mul!(left, U, Diagonal(diag))

    return mul!(A, left, U')                # NOTE: OVERWRITES INPUT
end

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

"""
function rotate! end


function rotate!(R::AbstractMatrix{F_}, x::AbstractArray{F}) where {F_, F}
    # NOTE: Throws method-not-found error if x is not at least as rich as R.
    return rotate!(promote_type(F_, F), R, x)
end

function rotate!(::Type{F}, R::AbstractMatrix{F_}, x::AbstractVector{F}) where {F_, F}
    temp = array(F, size(x), LABEL)
    x .= mul!(temp, R, x)
    return x
end

function rotate!(::Type{F}, R::AbstractMatrix{F_}, A::AbstractMatrix{F}) where {F_, F}
    left = array(F, size(A), LABEL)
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
        temp = array(F, (m,N÷m), LABEL)
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





"""
    braket(x1::AbstractVector, A::AbstractMatrix, x2::AbstractVector)

Compute the braket ⟨x1|A|x2⟩.

"""
function braket(x1::AbstractVector, A::AbstractMatrix, x2::AbstractVector)
    F = promote_type(eltype(x1), eltype(A), eltype(x2))
    covector = array(F, size(x2), (LABEL, :braket))
    covector .= x2
    covector = rotate!(A, covector)
    return x1' * covector
end

"""
    braket(x1::AbstractVector, ā::MatrixList, x2::AbstractVector)

Compute the braket ⟨x1|kron(ā)|x2⟩, but somewhat more efficiently.

"""
function braket(
    x1::AbstractVector,
    ā::MatrixList{F_},
    x2::AbstractVector,
) where {F_}
    F = promote_type(eltype(x1), F_, eltype(x2))
    covector = array(F, size(x2), (LABEL, :braket))
    covector .= x2
    covector = rotate!(ā, covector)
    return x1' * covector
end

"""
    expectation(A::AbstractMatrix, x::AbstractVector)

Compute the braket ``⟨x|A|x⟩``.

"""
expectation(A::AbstractMatrix, x::AbstractVector) = braket(x, A, x)

"""
    expectation(ā::MatrixList, x::AbstractVector)

Compute the braket `⟨x|kron(ā)|x⟩`, but somewhat more efficiently.

"""
expectation(ā::MatrixList, x::AbstractVector) = braket(x, ā, x)




"""
    basisvector(N::Int, i::Int)

A vector of `N` Bools, all zero except for index `i`, which is one.

"""
function basisvector(N::Int, i::Int)
    e = zeros(Bool, N)
    e[i] = 1
    return e
end
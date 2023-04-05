using LinearAlgebra: kron!, eigen, Diagonal, Hermitian, mul!
using Memoization: @memoize

@memoize function _TEMPARRAY(::F, shape::Tuple, index=nothing) where {F<:Number}
    # NOTE: `index` just gives a means of making distinct arrays of the same type and shape
    return Array{F}(undef, shape)
end

function _TEMPARRAY(F::Type{<:Number}, shape::Tuple, index=nothing)
    return _TEMPARRAY(zero(F), shape, index)
end

# List{T} = Union{AbstractVector{T}, Tuple{Vararg{T}}}
List{T} = AbstractVector{T}

function kron(v̄::List{<:AbstractVector{F}}) where {F}
    op  = _TEMPARRAY(F, (1,)); op[1] = one(F)
    tgt = nothing
    for i in eachindex(v̄)
        shape = (length(op)*length(v̄[i]),)
        tgt = _TEMPARRAY(F, shape)
        kron!(tgt, op, v̄[i])
        op = tgt
    end
    return copy(tgt)
end

function kron(Ā::List{<:AbstractMatrix{F}}) where {F}
    op  = _TEMPARRAY(F, (1,1)); op[1] = one(F)
    tgt = nothing
    for i in eachindex(Ā)
        shape = size(op) .* size(Ā[i])
        tgt = _TEMPARRAY(F, shape)
        kron!(tgt, op, Ā[i])
        op = tgt
    end
    return copy(tgt)
end

function cis_type(x)
    F = real(eltype(x))
    return F <: Integer ? ComplexF64 : Complex{F}
end

function cis!(A::AbstractMatrix{<:Complex{<:AbstractFloat}}, x::Number=1)
    # NOTE: calculates exp(𝑖xA), aka Cos(xA) + I Sin(xA), hence cis
    # NOTE: A must not be a restrictive view
    # NOTE: A must be Hermitian (in character, not in type)
    Λ, U = eigen(Hermitian(A))              # TODO: UNNECESSARY ALLOCATIONS

    F = Complex{real(eltype(Λ))}
    diag = _TEMPARRAY(F, size(Λ))
    diag .= exp.((im*x) .* Λ)

    left = _TEMPARRAY(F, size(A))
    left = mul!(left, U, Diagonal(diag))

    return mul!(A, left, U')                # NOTE: OVERWRITES INPUT
end


function rotate!(R::AbstractMatrix, x::AbstractVector)
    if eltype(x) !== promote_type(eltype(R), eltype(x))
        # TODO: I'd prefer to enforce this by dispatch, but I can't think of how.
        error("Type of `x` does not support rotation by `R`.")
    end

    temp = _TEMPARRAY(eltype(x), size(x))
    x .= mul!(temp, R, x)
    return x
end

function rotate!(R::AbstractMatrix, A::AbstractMatrix)
    if eltype(A) !== promote_type(eltype(R), eltype(A))
        # TODO: I'd prefer to enforce this by dispatch, but I can't think of how.
        error("Type of `A` does not support rotation by `R`.")
    end

    left = _TEMPARRAY(eltype(A), size(A))
    left = mul!(left, R, A)
    return mul!(A, left, R')
end

function rotate!(r̄::List{<:AbstractMatrix{F}}, x::AbstractVector) where {F}
    if eltype(x) !== promote_type(F, eltype(x))
        # TODO: I'd prefer to enforce this by dispatch, but I can't think of how.
        error("Type of `x` does not support rotation by `R`.")
    end

    N = length(x)
    for r in r̄
        m = size(r,1)
        x_ = transpose(reshape(x, (N÷m,m)))     # CREATE A PERMUTED VIEW
        temp = _TEMPARRAY(eltype(x), (m,N÷m))
        temp = mul!(temp, r, x_)                # APPLY THE CURRENT OPERATOR
        x .= vec(temp)                          # COPY RESULT TO ORIGINAL STATE
    end
    return x
end


function rotate!(r̄::List{<:AbstractMatrix{F}}, A::AbstractMatrix) where {F}
    if eltype(A) !== promote_type(F, eltype(A))
        # TODO: I'd prefer to enforce this by dispatch, but I can't think of how.
        error("Type of `x` does not support rotation by `R`.")
    end

    # TODO: Write this with tensor algebra
    return rotate!(kron(r̄), A)
end





function braket(x1::AbstractVector, A::AbstractMatrix, x2::AbstractVector)
    F = promote_type(eltype(x1), eltype(A), eltype(x2))
    covector = _TEMPARRAY(F, size(x2))
    covector = mul!(covector, A, x2)
    return x1' * covector
end

expectation(A::AbstractMatrix, x::AbstractVector) = braket(x, A, x)

# TODO: Tensor implementations for expectation, braket.


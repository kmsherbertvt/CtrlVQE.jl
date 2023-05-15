using LinearAlgebra: kron!, eigen, Diagonal, Hermitian, mul!

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

const VectorList{T} = Array{T,2}
const MatrixList{T} = Array{T,3}

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

function cis_type(x)
    F = real(eltype(x))
    return F <: Integer ? ComplexF64 : Complex{F}
end

function cis!(A::AbstractMatrix{<:Complex{<:AbstractFloat}}, x::Number=1)
    # NOTE: calculates exp(𝑖xA), aka Cos(xA) + I Sin(xA), hence cis
    # NOTE: A must not be a restrictive view
    # NOTE: A must be Hermitian (in character, not in type)
    Λ, U = eigen(Hermitian(A))              # TODO (mid): UNNECESSARY ALLOCATIONS

    F = Complex{real(eltype(Λ))}
    diag = array(F, size(Λ), LABEL)
    diag .= exp.((im*x) .* Λ)

    left = array(F, size(A), LABEL)
    left = mul!(left, U, Diagonal(diag))

    return mul!(A, left, U')                # NOTE: OVERWRITES INPUT
end


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





function braket(x1::AbstractVector, A::AbstractMatrix, x2::AbstractVector)
    F = promote_type(eltype(x1), eltype(A), eltype(x2))
    covector = array(F, size(x2), (LABEL, :braket))
    covector .= x2
    covector = rotate!(A, covector)
    return x1' * covector
end

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

expectation(A::AbstractMatrix, x::AbstractVector) = braket(x, A, x)
expectation(ā::MatrixList, x::AbstractVector) = braket(x, ā, x)




function basisvector(N::Int, i::Int)
    e = zeros(Bool, N)
    e[i] = 1
    return e
end
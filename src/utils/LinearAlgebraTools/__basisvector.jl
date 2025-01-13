import LinearAlgebra: I

"""
    basisvector(N::Int, i::Int)

Construct a length-`N` basis vector for index `i`.

Returns a vector of type `Bool`.

"""
function basisvector(N::Int, i::Int)
    return basisvector!(Vector{Bool}(undef,N), i)
end

"""
    basisvector!(e::AbstractVector, i::Int)

Write `e` into a basis vector for index `i`.

"""
function basisvector!(e::AbstractVector, i::Int)
    e .= 0
    e[i] = 1
    return e
end

"""
    basisvectors(N::Int)

Construct a matrix of length-`N` basis vectors, aka an identity matrix.

"""
function basisvectors(N::Int)
    return Matrix{Bool}(I, N,N)
end

"""
    basisvectors!(I::AbstractMatrix)

Write each column of `Im` into a basis vector.

In other words, write `Im` into an identity matrix.

"""
function basisvectors!(Im::AbstractMatrix)
    R = size(Im,1)
    offset = 0              # TRACKS STARTING INDEX FOR EACH COLUMN
    Im .= 0
    for c in axes(Im,2)
        Im[offset+c] = 1    # USE VECTOR INDEXING FOR FASTEST SPEED
        offset += R
    end
    return Im
end
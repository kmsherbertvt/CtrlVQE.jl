"""
    basisvector(N::Int, i::Int; result=nothing)

Construct a length-`N` basis vector for index `i`.

The result is written to `result` if provided. Otherwise returns a vector of type `Bool`.

    basisvector(::Type{T}, N::Int, i::Int)

Construct a length-`N` basis vector of type `T` for index `i`.

"""
function basisvector(N::Int, i::Int; result=nothing)
    isnothing(result) && (result = Array{Bool}(undef, N))
    result .= 0
    result[i] = 1
    return result
end

function basisvector(::Type{T}, N::Int, i::Int) where {T}
    return basisvector(N, i; result=Array{T}(undef, N))
end

"""
    basisvectors(N::Int)

Construct a matrix of length-`N` basis vectors, aka an identity matrix.

The result is written to `result` if provided. Otherwise returns a matrix of type `Bool`.

    basisvector(::Type{T}, N::Int, i::Int)

Construct a size-`N` identity matrix of type `T`.

"""
function basisvectors(N::Int; result=nothing)
    isnothing(result) && (result = Array{Bool}(undef, N, N))
    offset = 0                  # TRACKS STARTING INDEX FOR EACH COLUMN
    result .= 0
    for c in axes(result,2)
        result[offset+c] = 1    # USE VECTOR INDEXING FOR FASTEST SPEED
        offset += N
    end
    return result
end

function basisvectors(::Type{T}, N::Int) where {T}
    return basisvectors(N; result=Array{T}(undef, N, N))
end

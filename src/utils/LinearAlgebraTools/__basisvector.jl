"""
    basisvector(N::Int, i::Int; result=nothing)

Construct a length-`N` basis vector for index `i`.

The result is written to `result` if provided. Otherwise returns a vector of type `Bool`.

    basisvector(::Type{T}, N::Int, i::Int)

Construct a length-`N` basis vector of type `T` for index `i`.

```jldoctests
julia> LAT.basisvector(4, 2)
4-element Vector{Bool}:
 0
 1
 0
 0

julia> LAT.basisvector(ComplexF64, 4, 2)
4-element Vector{ComplexF64}:
 0.0 + 0.0im
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```

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

    basisvectors(::Type{T}, N::Int)

Construct a size-`N` identity matrix of type `T`.

```jldoctests
julia> LAT.basisvectors(4)
4×4 Matrix{Bool}:
 1  0  0  0
 0  1  0  0
 0  0  1  0
 0  0  0  1

julia> LAT.basisvectors(ComplexF64, 4)
4×4 Matrix{ComplexF64}:
 1.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  1.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  1.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  1.0+0.0im
```

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

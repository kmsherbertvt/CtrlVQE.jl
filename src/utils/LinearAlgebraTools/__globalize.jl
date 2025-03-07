import .LinearAlgebraTools: kron, basisvectors

import TemporaryArrays: @temparray

"""
    globalize(op::AbstractMatrix, n::Int, q::Int; result=nothing)

Extend a local operator `op` acting on qubit `q`,
    into the global Hilbert space of `n` qubits.

The array is stored in `result` if provided.

```jldoctests
julia> LAT.globalize([0 1; 1 0], 2, 1) # X⊗I
4×4 Matrix{Int64}:
 0  0  1  0
 0  0  0  1
 1  0  0  0
 0  1  0  0

julia> LAT.globalize([0 1; 1 0], 2, 2) # I⊗X
4×4 Matrix{Int64}:
 0  1  0  0
 1  0  0  0
 0  0  0  1
 0  0  1  0

```

"""
function globalize(
    op::AbstractMatrix{F}, n::Int, q::Int;
    result=nothing,
) where {F}
    m = size(op,1)
    N = m^n

    isnothing(result) && (result = Matrix{F}(undef, N, N))
    ops = @temparray(F, (m,m,n), :globalize)
    for p in 1:n
        if p == q
            ops[:,:,p] .= op
        else
            basisvectors(m; result=@view(ops[:,:,p]))
        end
    end
    return kron(ops; result=result)
end

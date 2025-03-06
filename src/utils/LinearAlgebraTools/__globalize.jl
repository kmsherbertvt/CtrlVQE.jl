import TemporaryArrays: @temparray

import .LinearAlgebraTools: kron, basisvectors

"""
    globalize(op::AbstractMatrix, n::Int, q::Int; result=nothing)

Extend a local operator `op` acting on qubit `q`,
    into the global Hilbert space of `n` qubits.

The array is stored in `result` if provided.

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

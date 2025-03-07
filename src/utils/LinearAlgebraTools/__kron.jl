import .LinearAlgebraTools: MatrixList, VectorList

import TemporaryArrays: @temparray

import LinearAlgebra: kron!

"""
    function kron(Ā; result=nothing)

Construct the Kronecker product of each element of Ā.

# Parameters
- Ā: either a `MatrixList` (to construct a large matrix)
    or a `VectorList` (to construct a large vector)

Ordering: [x1; x2] ⊗ [y1; y2] = [x1⋅y1l  x1⋅y2;  x2⋅y1;  x2⋅y2]

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

```jldoctests
julia> LAT.kron([1 0; 0 1]) # |0⟩⊗|1⟩
4-element Vector{Int64}:
 0
 1
 0
 0

julia> LAT.kron([0 1; 1 0;;; 0 1; 1 0]) # X⊗X
4×4 Matrix{Int64}:
 0  0  0  1
 0  0  1  0
 0  1  0  0
 1  0  0  0

```

"""
function kron end

function kron(v̄::VectorList{F}; result=nothing) where {F}
    isnothing(result) && (result = Vector{F}(undef, size(v̄,1)^size(v̄,2)))

    op  = @temparray(F, (1,), :kron, :vector); op[1] = one(F)
    tgt = nothing
    for i in axes(v̄,2)
        shape = (length(op)*size(v̄,1),)
        tgt = @temparray(F, shape, :kron, :vector)
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

    op  = @temparray(F, (1,1), :kron, :matrix); op[1] = one(F)
    tgt = nothing
    for i in axes(Ā,3)
        shape = size(op) .* (size(Ā,1), size(Ā,2))
        tgt = @temparray(F, shape, :kron, :matrix)
        kron!(tgt, op, @view(Ā[:,:,i]))
        op = tgt
    end
    result .= tgt

    return result
end
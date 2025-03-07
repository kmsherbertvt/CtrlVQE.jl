import .LinearAlgebraTools: MatrixList, rotate!

import TemporaryArrays: @temparray

"""
    braket(x1, A, x2)

Compute the braket ``⟨x1|A|x2⟩``.

# Parameters
- A: the operator to measure. Usually an (abstract) matrix.
    When `A` is a `MatrixList`, the operator is a Kronecker product of each matrix.
- x1, x2: the states, both (abstract) vectors, to measure with respect to.
    Note that `x1` should be passed as a vector, NOT as a dual vector.

```jldoctests
julia> LAT.braket([1; 0], [0 1; 1 0], [0; 1]) # ⟨0|X|1⟩
1
```

"""
function braket(
    x1::AbstractVector,
    A::Union{AbstractMatrix,MatrixList},
    x2::AbstractVector,
)
    F = promote_type(eltype(x1), eltype(A), eltype(x2))
    covector = @temparray(F, size(x2), :braket)
    covector .= x2
    covector = rotate!(A, covector)
    return x1' * covector
end

"""
    expectation(A, x)

Compute the braket ``⟨x|A|x⟩``.

# Parameters
- A: the operator to measure. Usually a matrix.
    When `A` is a `MatrixList`, the operator is a Kronecker product of each matrix.
- x: the state (a vector) to measure with respect to.

```
julia> LAT.expectation([1 0; 0 -1], [0; 1]) # ⟨1|Z|1⟩
-1
```

"""
function expectation(A::Union{AbstractMatrix,MatrixList}, x::AbstractVector)
    return braket(x, A, x)
end
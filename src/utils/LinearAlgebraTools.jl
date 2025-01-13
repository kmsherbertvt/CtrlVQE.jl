"""
    LinearAlgebraTools

Implement some frequently-used linear-algebraic operations.

Consider this to be a shallow extension of Julia's standard `LinearAlgebra` module,
    taking advantage of `Allocations` whenever possible.

Note that this module does not export anything.
This is intentional - the risk of name collisions is very high with this module,
    so I *insist* that no one ever writes `using LinearAlgebraTools`.

"""
module LinearAlgebraTools
    """
        VectorList{T}

    Semantic alias for a 2d array representing a distinct vector in each column.

    This is philosophically (but not operationally) distinct from a `Matrix`,
        which may imply some additional structure relating columns.

    """
    const VectorList{T} = AbstractArray{T,2}

    """
        MatrixList{T}

    Semantic alias for a 3d array representing a distinct matrix for each final index.

    In other words, think of a `MatrixList` `Ā` semantically as a list `A`
        where each `A[i] = Ā[:,:,i]` is a matrix.

    """
    const MatrixList{T} = AbstractArray{T,3}

    include("LinearAlgebraTools/__kron.jl")
        using .LinearAlgebraTools: kron
    include("LinearAlgebraTools/__rotate.jl")
        using .LinearAlgebraTools: rotate!
    include("LinearAlgebraTools/__eigen.jl")
        using .LinearAlgebraTools: eigen!
    include("LinearAlgebraTools/__cis.jl")
        using .LinearAlgebraTools: cis!, cis_type
    include("LinearAlgebraTools/__braket.jl")
        using .LinearAlgebraTools: braket, expectation
    include("LinearAlgebraTools/__basisvector.jl")
        using .LinearAlgebraTools: basisvector, basisvector!
        using .LinearAlgebraTools: basisvectors, basisvectors!
end
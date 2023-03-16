#= cache of temp arrays, indexed by (type, shape, index) =#

function kron(::AbstractVector{<:AbstractMatrix})::AbstractMatrix end
function kron(::AbstractVector{<:AbstractVector})::AbstractVector end

function exponentiate!(A::AbstractMatrix)::AbstractMatrix end
    # NOTE: mutates and returns A

function rotate!(U::AbstractMatrix, ψ::AbstractVector)::AbstractVector end
    # NOTE: mutates and returns ψ
function rotate!(U::AbstractMatrix, A::AbstractMatrix)::AbstractMatrix end
    # NOTE: mutates and returns A
    # NOTE: These two should exploit diagonal nature if Diagonal is passed as U. Verify.
function rotate!(u::AbstractVector{<:AbstractMatrix}, ψ::AbstractVector)::AbstractVector end
    # NOTE: This method is a tensorapply, assuming each u acts on a single body of ψ. I don't know offhand how to write it when bodies aren't necessarily the same size; I might have to impose each single body is equal size for sensible pre-allocation? That would be annoying.
function rotate!(u::AbstractVector{<:AbstractMatrix}, A::AbstractMatrix)::AbstractMatrix end
    # NOTE: Also want to write tensorapply on a matrix. Not sure how easy it it.

function expectation(A::AbstractMatrix, ψ::AbstractVector)::Number end
function braket(ψ1::AbstractVector, A::AbstractMatrix, ψ2::AbstractVector)::Number end
    # NOTE: Also take vector of single-body operators, use tensorapply.

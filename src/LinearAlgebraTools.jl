#= cache of temp arrays, indexed by (type, shape, index) =#
𝑖 = im
    # NOTE: MUCH more semantic imaginary unit. My keyboard is awesome. But VSCode doesn't like it; I'll probably get rid of this after awhile.
function kron(::AbstractVector{<:AbstractMatrix})::AbstractMatrix end
    # NOTE: must also be able to kron vectors. Might need separate method.
function propagator(H::AbstractMatrix, τ::Real)::AbstractMatrix end
    # NOTE: A bit much physics for a `LinearAlgebraTools` module. Maybe `ComputeBox` is more semantic. Anyway, this method should use cached temp arrays with eigen! to efficiently take exp(-𝑖τH). Okay maybe actually we move the -𝑖τ to Devices and call this `exponentiate`.
function rotate!(U::AbstractMatrix, ψ::AbstractVector)::AbstractVector end
    # NOTE: mutates and returns ψ
function rotate!(U::AbstractMatrix, A::AbstractMatrix)::AbstractMatrix end
    # NOTE: mutates and returns A
    # NOTE: These two should exploit diagonal nature if Diagonal is passed as U. Verify.
function rotate!(u::AbstractVector{<:AbstractMatrix}, ψ::AbstractVector)::AbstractVector end
    # NOTE: This method is a tensorapply, assuming each u acts on a single body of ψ. I don't know offhand how to write it when bodies aren't necessarily the same size; I might have to impose each single body is equal size for sensible pre-allocation? That would be annoying.

    # NOTE: we could write tensorapply on a matrix, but I don't _think_ we will want it anymore? Wanted it before for ligand operator, but I don't think we lose much by leaving single-body operations for the drive phase.

function compose(U1::AbstractMatrix, U2::AbstractMatrix)::AbstractMatrix end
    # NOTE: Just a semantic way of saying U1 * U2, permitting LAT optimizations if we deem it appropriate. But, pre-allocation doesn't help here since we have to return a new array... Might get rid of this.
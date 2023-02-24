using Memoization: @memoize
#= NOTE:

We are using `Memoization` to reconcile the following two considerations:
1. Simple-as-possible interface.
2. Don't re-calculate things you've already calculated.

The `Memoization` package is a bit beyond our control, generating two more problems:
1. Not every use-case will want every function call cached. Doing so is a waste of space.
2. If the state of any argument in a method changes, its cached value is no longer valid.

After toying with ideas for a number of more customized cache implementations,
    I've decided, to simply *not* memoize any function depending on an absolute time.
    I think that more or less solves the worst parts of problem 1.

    By chance, it seems like relative time only appears without absolute time in a context where the relative time could be interpreted as an absolute time (ie. propagation of a static hamiltonian), which means we actually just don't cache any times at all... Huh.

As it happens, it also solves problem 2 in the short term,
    because, at present, static device parameters are considered fixed.
So, the changing state of the device would only actually impact time-dependent methods.

BUT

If ever we implement a device with "tunable couplings",
    such that time-independent parameters of a device are changed on `Parameter.bind(⋅)`,
    the implementation of `Parameter.bind` should CLEAR the cache:

    Memoization.empty_all_caches!()

Alternatively, selectively clear caches for affected functions via:

    Memoization.empty_cache!(fn)

I don't know if it's possible to selectively clear cached values for specific methods.
If it can be done, it would require obtaining the actual `IdDict`
    being used as a cache for a particular function,
    figuring out exactly how that cache is indexed,
    and manually removing elements matching your targeted method signature.

=#

#= TODO: Go through and add @memoize to as many functions as appropriate.
    IdDict should be fine.
=#

using LinearAlgebra: Eigen, eigen
import ..Basis, ..Locality, ..Temporality

"""
NOTE: Implements `Parameter` interface.
"""
abstract type Device end




# METHODS NEEDING TO BE IMPLEMENTED
nqubits(::Device)::Int = error("Not Implemented")
nstates(::Device, q::Int)::Int = error("Not Implemented")

localidentityoperator(::Device, q::Int)::AbstractMatrix = error("Not Implemented")
localloweringoperator(::Device, q::Int)::AbstractMatrix = error("Not Implemented")


function localstatichamiltonian(::Device,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
)::AbstractMatrix
    return error("Not Implemented")
end

function localdrivenhamiltonian(::Device,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
    t::Real,
)::AbstractMatrix
    return error("Not Implemented")
end

function mixedstatichamiltonian(::Device,
    ā::AbstractVector{AbstractMatrix},
)::AbstractMatrix
    return error("Not Implemented")
end

function mixeddrivenhamiltonian(::Device,
    ā::AbstractVector{AbstractMatrix},
    t::Real,
)::AbstractMatrix
    return error("Not Implemented")
end








# UTILITIES

function globalize(device::Device, op::AbstractMatrix, q::Int)
    return LinearAlgebraTools.kron(
        p == q ? op : localidentityoperator(device, p) for p in 1:nqubits(device)
    )
end

nstates(device::Device) = prod(nstates(device,q) for q in 1:nqubits(device))






# BASIS ROTATIONS

function diagonalize(::Type{Basis.Dressed}, device::Device)
    H0 = hamiltonian(Temporality.Static, device)
    return eigen(H0)
    # TODO: Move code for Utils.dressedbasis to here.
end

function diagonalize(basis::Type{<:Basis.LocalBasis}, device::Device)
    ΛU = [diagonalize(basis, device, q) for q in 1:nqubits(device)]
    Λ = LinearAlgebraTools.kron(ΛU[q].values  for q in 1:nqubits(device))
    U = LinearAlgebraTools.kron(ΛU[q].vectors for q in 1:nqubits(device))
    return Eigen(Λ, U)
end

function diagonalize(::Type{Basis.Occupation}, device::Device, q::Int)
    I = localidentityoperator(device, q)
    return eigen(I)
end

function diagonalize(::Type{Basis.Coordinate}, device::Device, q::Int)
    a = localloweringoperator(device, q)
    Q = (a + a') / eltype(a)(√2)
    return eigen(Q)
end

function diagonalize(::Type{Basis.Momentum}, device::Device, q::Int)
    a = localloweringoperator(device, q)
    P = (a - a') / eltype(a)(√2)
    return eigen(P)
end



function basisrotation(
    src::Type{<:Basis.AbstractBasis},
    tgt::Type{<:Basis.AbstractBasis},
    device::Device,
)
    Λ0, U0 = diagonalize(src, device)
    Λ1, U1 = diagonalize(tgt, device)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` basis.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` basis.
    return LinearAlgebraTools.compose(U1', U0)
end

function basisrotation(
    src::Type{<:Basis.LocalBasis},
    tgt::Type{<:Basis.LocalBasis},
    device::Device,
    q::Int,
)
    Λ0, U0 = diagonalize(src, device, q)
    Λ1, U1 = diagonalize(tgt, device, q)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` basis.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` basis.
    return LinearAlgebraTools.compose(U1', U0)
end

function localbasisrotations(
    src::Type{<:Basis.LocalBasis},
    tgt::Type{<:Basis.LocalBasis},
    device::Device,
)
    return [basisrotation(src, tgt, device, q) for q in 1:nqubits(device)]
end





# OPERATORS



function identity(device::Device)
    return LinearAlgebraTools.kron(
        localidentityoperator(device, q) for q in 1:nqubits(device)
    )
end

function algebra(
    device::Device,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = basisrotation(Basis.Occupation, basis, device)
    ā = []
    for q in 1:nqubits(device)
        a0 = localloweringoperator(device, q)
        a = globalize(device, a0, q)
        a = LinearAlgebraTools.rotate!(U, a)
        push!(ā, a)
    end
    return ā
end

function localalgebra(
    device::Device,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    ā = []
    for q in 1:nqubits(device)
        U = basisrotation(Basis.Occupation, basis, device, q)
        a0 = localloweringoperator(device, q)
        a = LinearAlgebraTools.rotate!(U, a0)
        push!(ā, a)
    end
    return ā
end



#= `hamiltonian` METHODS =#

function hamiltonian(
    device::Device,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return sum((
        hamiltonian(Temporality.Static, device, basis),
        hamiltonian(Temporality.Driven, device, t, basis),
    ))
end

function hamiltonian(::Type{Temporality.Static},
    device::Device,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return sum((
        hamiltonian(Locality.Local, Temporality.Static, device, basis),
        hamiltonian(Locality.Mixed, Temporality.Static, device, basis),
    ))
end

function hamiltonian(::Type{Temporality.Static},
    device::Device,
    ::Type{Basis.Dressed},
)
    Λ, U = diagonalize(Basis.Dressed, device)
    return Diagonal(Λ)
end

function hamiltonian(::Type{Temporality.Driven},
    device::Device,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return sum((
        hamiltonian(Locality.Local, Temporality.Driven, device, t, basis),
        hamiltonian(Locality.Mixed, Temporality.Driven, device, t, basis),
    ))
end

function hamiltonian(::Type{Locality.Local},
    device::Device,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return sum((
        hamiltonian(Locality.Local, Temporality.Static, device, basis),
        hamiltonian(Locality.Local, Temporality.Driven, device, t, basis),
    ))
end

function hamiltonian(::Type{Locality.Mixed},
    device::Device,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return sum((
        hamiltonian(Locality.Mixed, Temporality.Static, device, basis),
        hamiltonian(Locality.Mixed, Temporality.Driven, device, t, basis),
    ))
end






function hamiltonian(::Type{Locality.Local}, ::Type{Temporality.Static},
    device::Device,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    ā = algebra(device, basis)
    return sum(localstatichamiltonian(device, ā, q) for q in 1:nqubits(device))
end

function hamiltonian(::Type{Locality.Local}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    ā = algebra(device, basis)
    return sum(localdrivenhamiltonian(device, ā, q, t) for q in 1:nqubits(device))
end

function hamiltonian(::Type{Locality.Mixed}, ::Type{Temporality.Static},
    device::Device,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    ā = algebra(device, basis)
    return mixedstatichamiltonian(device, ā)
end

function hamiltonian(::Type{Locality.Mixed}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    ā = algebra(device, basis)
    return mixeddrivenhamiltonian(device, ā, t)
end





function localhamiltonians(
    device::Device,
    t::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    h = localhamiltonians(Temporality.Static, device, basis)
    v = localhamiltonians(Temporality.Driven, device, t, basis)
    return [h[q] + v[q] for q in 1:nqubits(device)]
end

function localhamiltonians(::Type{Temporality.Static},
    device::Device,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    ā = localalgebra(device, basis)
    return [localstatichamiltonian(device, ā, q) for q in 1:nqubits(device)]
end

function localhamiltonians(::Type{Temporality.Driven},
    device::Device,
    t::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    ā = localalgebra(device, basis)
    return [localdrivenhamiltonian(device, ā, q, t) for q in 1:nqubits(device)]
end



#= `propagator` METHODS =#

function propagator(
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    H = hamiltonian(device, t, basis)
    return LinearAlgebraTools.propagator(H, τ)
end

function propagator(::Type{Temporality.Static},
    device::Device,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    H = hamiltonian(Temporality.Static, device, basis)
    return LinearAlgebraTools.propagator(H, τ)
end

function propagator(::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    H = hamiltonian(Temporality.Driven, device, t, basis)
    return LinearAlgebraTools.propagator(H, τ)
end

function propagator(::Type{Locality.Local},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis},
)
    H = hamiltonian(Locality.Local, device, t, basis)
    return LinearAlgebraTools.propagator(H, τ)
end
# SAME METHOD BUT FOR DIFFERENT BASIS TYPES
function propagator(::Type{Locality.Local},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    u = localpropagators(device, t, τ, basis)
    return LinearAlgebraTools.kron(u[q] for q in 1:nqubits(device))
end

function propagator(::Type{Locality.Mixed},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    H = hamiltonian(Locality.Mixed, device, t, basis)
    return LinearAlgebraTools.propagator(H, τ)
end

function propagator(::Type{Locality.Local}, ::Type{Temporality.Static},
    device::Device,
    τ::Real,
    basis::Type{<:AbstractBasis},
)
    H = hamiltonian(Locality.Local, Temporality.Static, device)
    return LinearAlgebraTools.propagator(H, τ)
end
# SAME METHOD BUT FOR DIFFERENT BASIS TYPES
function propagator(::Type{Locality.Local}, ::Type{Temporality.Static},
    device::Device,
    τ::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    u = localpropagators(Temporality.Static, device, τ, basis)
    return LinearAlgebraTools.kron(u[q] for q in 1:nqubits(device))
end

function propagator(::Type{Locality.Local}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis},
)
    H = hamiltonian(Locality.Local, Temporality.Driven, device, t)
    return LinearAlgebraTools.propagator(H, τ)
end
# SAME METHOD BUT FOR DIFFERENT BASIS TYPES
function propagator(::Type{Locality.Local}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    u = localpropagators(Temporality.Driven, device, t, τ, basis)
    return LinearAlgebraTools.kron(u[q] for q in 1:nqubits(device))
end

function propagator(::Type{Locality.Mixed}, ::Type{Temporality.Static},
    device::Device,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    H = hamiltonian(Locality.Mixed, Temporality.Static, device, basis)
    return LinearAlgebraTools.propagator(H, τ)
end

function propagator(::Type{Locality.Mixed}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    H = hamiltonian(Locality.Mixed, Temporality.Driven, device, t)
    return LinearAlgebraTools.propagator(H, τ, basis)
end






function localpropagators(
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    H = localhamiltonians(device, t, basis)
    return [LinearAlgebraTools.propagator(H[q], τ) for q in 1:nqubits(device)]
end

function localpropagators(::Type{Temporality.Static},
    device::Device,
    τ::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    h = localhamiltonians(Temporality.Static, device, basis)
    return [LinearAlgebraTools.propagator(h[q], τ) for q in 1:nqubits(device)]
end

function localpropagators(::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    v = localhamiltonians(Temporality.Driven, device, t, basis)
    return [LinearAlgebraTools.propagator(v[q], τ) for q in 1:nqubits(device)]
end






#= `propagate!` METHODS =#

function propagate!(
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = propagator(device, t, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(::Type{Temporality.Static},
    device::Device,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = propagator(Temporality.Static, device, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = propagator(Temporality.Driven, device, t, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(::Type{Locality.Local},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis},
)
    U = propagator(Locality.Local, device, t, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end
# SAME METHOD BUT FOR DIFFERENT BASIS TYPES
function propagate!(::Type{Locality.Local},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    u = localpropagators(device, t, τ, basis)
    return LinearAlgebraTools.rotate!(u, ψ)
end

function propagate!(::Type{Locality.Mixed},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = propagator(Locality.Mixed, device, t, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(::Type{Locality.Local}, ::Type{Temporality.Static},
    device::Device,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis},
)
    U = propagator(Locality.Local, Temporality.Static, device, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end
# SAME METHOD BUT FOR DIFFERENT BASIS TYPES
function propagate!(::Type{Locality.Local}, ::Type{Temporality.Static},
    device::Device,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    u = localpropagators(Temporality.Static, device, τ, basis)
    return LinearAlgebraTools.rotate!(u, ψ)
end

function propagate!(::Type{Locality.Local}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis},
)
    U = propagator(Locality.Local, Temporality.Driven, device, t, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end
# SAME METHOD BUT FOR DIFFERENT BASIS TYPES
function propagate!(::Type{Locality.Local}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:LocalBasis}=Basis.Occupation,
)
    u = localpropagators(Temporality.Driven, device, t, τ, basis)
    return LinearAlgebraTools.rotate!(u, ψ)
end

function propagate!(::Type{Locality.Mixed}, ::Type{Temporality.Static},
    device::Device,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = propagator(Locality.Mixed, Temporality.Static, device, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(::Type{Locality.Mixed}, ::Type{Temporality.Driven},
    device::Device,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    U = propagator(Locality.Mixed, Temporality.Driven, device, t, τ, basis)
    return LinearAlgebraTools.rotate!(U, ψ)
end









using .Devices: DeviceType
using .Devices: nlevels, nqubits, noperators, localalgebra

import ..CtrlVQE: LAT
import ..CtrlVQE: Bases, Operators

import TemporaryArrays: @temparray

import Memoization: @memoize

import LinearAlgebra: I, Eigen

Base.eltype(::DeviceType{F}) where {F} = F

#= Some comments:

This concrete interface mostly relates to interacting
    with the full Hilbert space of the device.
By default, the space of the device is the tensor product of all qubits,
    a very *dense* system.
Most of the machinery in this `Devices` module is to handle this density
    as efficiently as possible.
But that means you'll have a lot of muck to sort through
    if you decide to make things sparser!

If you would, for example, like to work in a truncated space where,
    low-likelihood states like |22⟩ are omitted while allowing for |20⟩,
    you would need to override these methods somehow.

This is also where most of the "magic" would happen if we were to have hybrid devices
    with multiple different types of qubit
    (or, say, spin qubits coupled via an optical cavity with its own distinct algebra).
Note that, to achieve this, the type returned by `localalgebra` (and `globalalgebra`)
    can no longer be a 4d array, since such a construct assumes equal-sized qubits,
    and that REALLY mucks things up.
It should be doable, but it's beyond my foresight.

=#

"""
    nstates(device::DeviceType)

The total number of states in the physical Hilbert space of the device.

(This is as opposed to `nlevels(device)`,
    the number of states in the physical Hilbert space of a single independent qubit.)

"""
function nstates(device::DeviceType)
    return nlevels(device) ^ nqubits(device)
end

"""
    globalalgebra(device::DeviceType[, basis::Bases.BasisType])

A globalized version of the matrix list defined by `localalgebra`.

That is, each operator `ā[:,:,q,σ]` is an N⨯N matrix acting on the whole Hilbert space,
    rather than an m⨯m matrix acting on the space of a single qubit.

(N is used throughout the docs for `nstates(device)` and m for `nlevels(device)`.)

The array is stored in `result` or, if not provided, returned from a cache.

"""
@memoize Dict function globalalgebra(
    device::DeviceType,
    basis::Bases.BasisType=Bases.BARE;
    result=nothing,
)
    isnothing(result) && return _globalalgebra(device, basis)

    ā0 = localalgebra(device)
    N = nstates(device)
    n = nqubits(device)
    o = noperators(device)

    U = basisrotation(basis, Bases.BARE, device)

    for q in 1:n
        for σ in 1:o
            result[:,:,σ,q] .= LAT.globalize(@view(ā0[:,:,σ,q]), n, q)
            LAT.rotate!(U, @view(result[:,:,σ,q]))
        end
    end
    return result
end

@memoize Dict function _globalalgebra(device::DeviceType, basis::Bases.BasisType)
    N = nstates(device)
    n = nqubits(device)
    o = noperators(device)
    result = Array{Complex{eltype(device)}}(undef, N, N, o, n)
    return globalalgebra(device, basis; result=result)
end




"""
    dress(device::DeviceType)

Diagonalize the static Hamiltonian and apply some post-processing
    to define the so-called `DRESSED` basis.

Compute the vector of eigenvalues `Λ` and the rotation matrix `U` for a given basis.

`U` is an operator acting on the global Hilbert space of the device.

The result is packed into a `LinearAlgebra.Eigen` object.
It may be unpacked directly into a vector eigenvalues Λ and a matrix of eigenvectors U by

    Λ, U = dress(device)

Alternatively:

    ΛU = dress(device)
    Λ = ΛU.values
    U = ΛU.vectors

"""
@memoize Dict function dress(device::DeviceType)
    H0 = operator(Operators.STATIC, device)
    Λ = Vector{real(eltype(H0))}(undef, size(H0,1))
    U = similar(H0)
    LAT.eigen!(Λ, U, H0)

    _dress_permutation!(Λ, U)
    _dress_phase!(Λ, U)
    _dress_zeros!(Λ, U)

    return Eigen(Λ,U)
end

function _dress_permutation!(Λ, U)
    σ = Vector{Int}(undef,length(Λ))
    for i in 1:length(Λ)
        perm = sortperm(abs.(U[i,:]), rev=true) # STABLE SORT BY "ith" COMPONENT
        perm_ix = 1                             # CAREFULLY HANDLE TIES
        while perm[perm_ix] ∈ @view(σ[1:i-1])
            perm_ix += 1
        end
        σ[i] = perm[perm_ix]
    end
    Λ .= Λ[  σ]
    U .= U[:,σ]
end

function _dress_phase!(Λ, U::Matrix{<:Real})
    for i in 1:length(Λ)
        U[:,i] .*= U[i,i] < 0 ? -1 : 1
    end
end

function _dress_phase!(Λ, U::Matrix{<:Complex})
    for i in 1:length(Λ)
        U[:,i] .*= exp(-im*angle(U[i,i]))
    end
end

function _dress_zeros!(Λ, U)
    Λ[abs.(Λ) .< eps(real(eltype(Λ)))] .= zero(eltype(Λ))
    U[abs.(U) .< eps(real(eltype(U)))] .= zero(eltype(U))
end




"""
    basisrotation(tgt::Bases.BasisType, src::Bases.BasisType, device::DeviceType)

Calculate the basis rotation `U` which transforms ``|ψ_{src}⟩ → |ψ_{tgt}⟩ = U|ψ_{src}⟩``.

"""
@memoize Dict function basisrotation(
    tgt::Bases.BasisType,
    src::Bases.BasisType,
    device::DeviceType,
)
    N = nstates(device)
    identity = Matrix{Complex{eltype(device)}}(I, N, N)
    U0 = (src == Bases.DRESSED) ? dress(device).vectors : identity
    U1 = (tgt == Bases.DRESSED) ? dress(device).vectors : identity
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` Bases.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` Bases.
    return U1' * U0
end




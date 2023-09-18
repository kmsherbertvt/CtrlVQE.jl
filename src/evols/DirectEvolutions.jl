import ..Evolutions
export Direct

import ..LinearAlgebraTools
import ..Devices
import ..Bases

import ..Bases: DRESSED
import ..Operators: STATIC, Drive

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using LinearAlgebra: norm

"""
    Direct(r)

A Trotterization method (using `r` steps) calculating drive terms in the rotation-frame.

The work basis for this algorithm is `Bases.DRESSED`,
    since the rotating-frame evolution ``U_t ≡ exp(-itH_0)`` happens at each step.

This algorithm exponentiates the matrix ``U_t' V(t) U_t`` at each time step,
    so it is not terribly efficient.

"""
struct Direct <: Evolutions.TrotterEvolution
    r::Int
end

Evolutions.workbasis(::Direct) = DRESSED
Evolutions.nsteps(evolution::Direct) = evolution.r

function Evolutions.evolve!(
    evolution::Direct,
    device::Devices.DeviceType,
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing,
)
    r = Evolutions.nsteps(evolution)
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(T, r)

    # REMEMBER NORM FOR NORM-PRESERVING STEP
    A = norm(ψ)

    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    N = Devices.nstates(device)
    U_TYPE = LinearAlgebraTools.cis_type(eltype(STATIC, device, DRESSED))
    V_TYPE = LinearAlgebraTools.cis_type(eltype(Drive(0), device, DRESSED))
    U = array(U_TYPE, (N,N), (LABEL, :intermediate))
    V = array(V_TYPE, (N,N), LABEL)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U = Devices.evolver(STATIC, device, DRESSED, t̄[i]; result=U)
        V = Devices.operator(Drive(t̄[i]), device, DRESSED; result=V)
        V = LinearAlgebraTools.rotate!(U', V)
        V = LinearAlgebraTools.cis!(V, -τ̄[i])
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    # ROTATE OUT OF INTERACTION PICTURE
    ψ = Devices.evolve!(STATIC, device, DRESSED, T, ψ)

    # ENFORCE NORM-PRESERVING TIME EVOLUTION
    ψ .*= A / norm(ψ)

    return ψ
end
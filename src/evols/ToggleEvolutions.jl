import ..Evolutions
export Toggle

import ..LinearAlgebraTools
import ..Devices
import ..Bases

import ..Bases: OCCUPATION
import ..Operators: STATIC, Drive

using LinearAlgebra: norm

"""
    Toggle(r)

A Trotterization method (using `r` steps) alternately propagating static and drive terms.

The work basis for this algorithm is `Bases.OCCUPATION`,
    since the time-dependent "Drive" operator at each step is usually qubit-local.

"""
struct Toggle <: Evolutions.TrotterEvolution
    r::Int
end

Evolutions.workbasis(::Toggle) = OCCUPATION
Evolutions.nsteps(evolution::Toggle) = evolution.r

function Evolutions.evolve!(
    evolution::Toggle,
    device::Devices.DeviceType,
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing,
)
    r = Evolutions.nsteps(evolution)
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(T, r)

    # REMEMBER NORM FOR NORM-PRESERVING STEP
    A = norm(ψ)

    # FIRST STEP: NO NEED TO APPLY STATIC OPERATOR
    callback !== nothing && callback(1, t̄[1], ψ)
    ψ = Devices.propagate!(Drive(t̄[1]),  device, OCCUPATION, τ̄[1], ψ)

    # RUN EVOLUTION
    for i in 2:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(STATIC, device, OCCUPATION, τ, ψ)
        ψ = Devices.propagate!(Drive(t̄[i]),  device, OCCUPATION, τ̄[i], ψ)
    end

    # ENFORCE NORM-PRESERVING TIME EVOLUTION
    ψ .*= A / norm(ψ)

    return ψ
end
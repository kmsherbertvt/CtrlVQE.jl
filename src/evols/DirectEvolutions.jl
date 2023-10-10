import ..Evolutions
export DIRECT

import ..LinearAlgebraTools
import ..Integrations, ..Devices
import ..Bases

import ..Bases: DRESSED
import ..Operators: STATIC, Drive

import ..TrapezoidalIntegrations: TrapezoidalIntegration

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using LinearAlgebra: norm

"""
    Direct

A Trotterization method (using `r` steps) calculating drive terms in the rotation-frame.

The work basis for this algorithm is `Bases.DRESSED`,
    since the rotating-frame evolution ``U_t ≡ exp(-itH_0)`` happens at each step.

This algorithm exponentiates the matrix ``U_t' V(t) U_t`` at each time step,
    so it is not terribly efficient.

NOTE: Currently, this method assumes a trapezoidal rule,
    so only `TrapezoidalIntegration` grids are allowed.
TODO (mid): Actually I'm pretty sure this one doesn't have any reason to...

"""
struct Direct <: Evolutions.EvolutionType end; DIRECT = Direct()

Evolutions.workbasis(::Direct) = DRESSED

function Evolutions.evolve!(::Direct,
    device::Devices.DeviceType,
    grid::TrapezoidalIntegration,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing,
)
    # PREPARE TEMPORAL LATTICE
    r = Integrations.nsteps(grid)
    t̄ = Integrations.lattice(grid)

    # REMEMBER NORM FOR NORM-PRESERVING STEP
    A = norm(ψ)

    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    N = Devices.nstates(device)
    U_TYPE = LinearAlgebraTools.cis_type(eltype(STATIC, device, DRESSED))
    V_TYPE = LinearAlgebraTools.cis_type(eltype(Drive(0), device, DRESSED))
    U = array(U_TYPE, (N,N), (LABEL, :intermediate))
    V = array(V_TYPE, (N,N), LABEL)

    # ROTATE INTO INTERACTION PICTURE
    ψ = Devices.evolve!(STATIC, device, DRESSED, -t̄[begin], ψ)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U = Devices.evolver(STATIC, device, DRESSED, t̄[i]; result=U)
        V = Devices.operator(Drive(t̄[i]), device, DRESSED; result=V)
        V = LinearAlgebraTools.rotate!(U', V)
        V = LinearAlgebraTools.cis!(V, -Integrations.stepat(grid, i-1))
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    # ROTATE OUT OF INTERACTION PICTURE
    ψ = Devices.evolve!(STATIC, device, DRESSED, t̄[end], ψ)

    # ENFORCE NORM-PRESERVING TIME EVOLUTION
    ψ .*= A / norm(ψ)

    return ψ
end
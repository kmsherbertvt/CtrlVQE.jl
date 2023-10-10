import ..Evolutions
export TOGGLE

import ..LinearAlgebraTools
import ..Integrations, ..Devices
import ..Bases

import ..Bases: OCCUPATION
import ..Operators: STATIC, Drive

import ..TrapezoidalIntegrations: TrapezoidalIntegration

using LinearAlgebra: norm

"""
    Toggle

A Trotterization method (using `r` steps) alternately propagating static and drive terms.

The work basis for this algorithm is `Bases.OCCUPATION`,
    since the time-dependent "Drive" operator at each step is usually qubit-local.

NOTE: This method assumes a trapezoidal rule,
    so only `TrapezoidalIntegration` grids are allowed.

"""
struct Toggle <: Evolutions.EvolutionType end; TOGGLE = Toggle()

Evolutions.workbasis(::Toggle) = OCCUPATION

function Evolutions.evolve!(::Toggle,
    device::Devices.DeviceType,
    grid::TrapezoidalIntegration,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing,
)
    # PREPARE TEMPORAL LATTICE
    r = Integrations.nsteps(grid)
    τ = Integrations.stepsize(grid)
    t̄ = Integrations.lattice(grid)

    # REMEMBER NORM FOR NORM-PRESERVING STEP
    A = norm(ψ)

    # RUN EVOLUTION
    for i in 1:r
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(Drive(t̄[i]),  device, OCCUPATION, τ/2, ψ)
        ψ = Devices.propagate!(STATIC, device, OCCUPATION, τ, ψ)
        ψ = Devices.propagate!(Drive(t̄[i+1]),  device, OCCUPATION, τ/2, ψ)
    end
    callback !== nothing && callback(r+1, t̄[r+1], ψ)

    #= NOTE:

    This implementation applies the drive about twice as many times as strictly necessary,
        since the latter propagation of step i can be combined with the first of i+1.
    But this symmetric form gives access to a "truer" intermediate state ψ(t).
    This doesn't matter for pure evolution, but it is meaningful for the callback,
        and more importantly to me it matches the `gradientsignals` method,
        which *needs* the true intermediate state to evaluate the gradient signal.
    For locally driven devices (which is what this evolution algorithm is designed for)
        there is no major cost to the drive propagations,
        so we can afford to favor parllel code structures.

    =#

    # ENFORCE NORM-PRESERVING TIME EVOLUTION
    ψ .*= A / norm(ψ)

    return ψ
end
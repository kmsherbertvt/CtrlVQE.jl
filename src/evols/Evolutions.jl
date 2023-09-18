export trapezoidaltimegrid
export EvolutionType, TrotterEvolution
export evolve, evolve!, workbasis, nsteps, gradientsignals

import ..LinearAlgebraTools
import ..Devices
import ..Bases

import ..Bases: OCCUPATION
import ..Operators: STATIC, Drive, Gradient

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using LinearAlgebra: norm
using Memoization: @memoize


"""
    trapezoidaltimegrid(T::Real, r::Int)

All the tools needed to integrate over time using a simple trapezoidal rule.

# Arguments
- `T`: the upper bound of the integral (0 is implicitly the lower bound).
        If `T` is negative, `|T|` is used as the lower bound and 0 as the upper bound.

- `r`: the number of time steps.
        Note this is the number of STEPS.
        The number of time POINTS is `r+1`, since they include t=0.

# Returns
- `τ`: the length of each time step, simply ``T/r``.
- `τ̄`: vector of `r+1` time spacings to use as `dt` in integration.
- `t̄`: vector of `r+1` time points

The integral ``∫_0^T f(t)⋅dt`` is evaluated with `sum(f.(t̄) .* τ̄)`.

# Explanation

Intuitively, `τ̄` is a vector giving `τ` for each time point.
But careful! The sum of all `τ̄` must match the length of the integral, ie. `T`.
But there are `r+1` points, and `(r+1)⋅τ > T`. How do we reconcile this?
A "Left Hand Sum" would omit `t=T` from the time points;
    a "Right Hand Sum" would omit `t=0`.
The trapezoidal rule omits half a point from each.

That sounds awfully strange, but it's mathematically sound!
We only integrate through *half* of each boundary time point `t=0` and `t=T`.
Thus, those points, and only those points, have a spacing of `τ/2`.

In principle, a different grid could be adopted giving more sophisticated quadrature.

"""
@memoize Dict function trapezoidaltimegrid(T::Real, r::Int)
    # NOTE: Negative values of T give reversed time grid.
    τ = T / r
    τ̄ = fill(τ, r+1); τ̄[[begin, end]] ./= 2
    t̄ = abs(τ) * (T ≥ 0 ? (0:r) : reverse(0:r))
    return τ, τ̄, t̄
end


"""
    EvolutionType

Super-type for all evolution algorithms.

# Implementation

Any concrete sub-type `A` must implement the following methods:
- `workbasis(::A)`: which Bases.BasisType the evolution algorithm uses
- `evolve!(::A, device, T, ψ; callback=nothing)`: evolve ψ (in-place) from time 0 to T
                                    (you may assume the basis of ψ is the work basis)

If your evolution algorithm breaks time up into equally-spaced discrete time steps,
    you should implement a `TrotterEvolution`, which has a couple extra requirements.

"""
abstract type EvolutionType end

"""
    workbasis(t::Real)

Which basis the evolution algorithm works in.

Also defines the default basis to interpret ψ as, in evolution methods.

"""
function workbasis(::EvolutionType)
    error("Not Implemented")
    return OCCUPATION
end

"""
    evolve!(evolution, device, [basis, ]T, ψ; basis=nothing, callback=nothing)

Evolve a state `ψ` by time `T` under a `device` Hamiltonian.

This method both mutates and returns `ψ`.

# Arguments
- `evolution::EvolutionType`: which evolution algorithm to use.
- `device::Devices.DeviceType`: specifies which Hamiltonian to evolve under.
- `basis::Bases.BasisType`: which basis `ψ` is represented in.
        Implicitly defaults to `workbasis(evolution)`.
- `T::Real`: the total amount of time to evolve by.
        The evolution is implicitly assumed to start at time `t=0`.
- `ψ`: the initial statevector, defined on the full Hilbert space of the device.

# Keyword Arguments
- `callback`: a function which is called at each iteration of the time evolution.
        The function is passed three arguments:
        - `i`: indexes the iteration
        - `t`: the current time point
        - `ψ`: the current statevector, in the work basis
        The function is called after having evolved ψ into |ψ(t)⟩.

"""
function evolve! end

function evolve!(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    T::Real,
    ψ0::AbstractVector;
    callback=nothing,
)
    error("Not Implemented")
    return ψ0
end

function evolve!(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector;
    kwargs...
)
    basis == workbasis(evolution) && return evolve!(evolution, device, T, ψ0; kwargs...)

    U = Devices.basisrotation(workbasis(evolution), basis, device)
    ψ0 = LinearAlgebraTools.rotate!(U, ψ0)      # ROTATE INTO WORK BASIS
    ψ0 = evolve!(evolution, device, T, ψ0; kwargs...)
    ψ0 = LinearAlgebraTools.rotate!(U', ψ0)     # ROTATE BACK INTO GIVEN BASIS
    return ψ0
end

"""
    evolve(evolution, device, [basis, ]T, ψ0; result=nothing, kwargs...)

Evolve a state `ψ0` by time `T` under a `device` Hamiltonian, without mutating `ψ0`.

This method simply copies `ψ0` (to `result` if provided, or else to a new array),
    then calls the mutating function `evolve!` on the copy.
Please see `evolve!` for detailed documentation.

"""
function evolve end

function evolve(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    T::Real,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    F = LinearAlgebraTools.cis_type(ψ0)
    result === nothing && (result = Vector{F}(undef, length(ψ0)))
    result .= ψ0
    return evolve!(evolution, device, T, result; kwargs...)
end

function evolve(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    F = LinearAlgebraTools.cis_type(ψ0)
    result === nothing && (result = Vector{F}(undef, length(ψ0)))
    result .= ψ0
    return evolve!(evolution, device, basis, T, result; kwargs...)
end



"""
    TrotterEvolution

Super-type for evolution algorithms which divide time into equally-spaced chunks.

This sub-typing facilitates easy comparison between different Trotter algorithms,
    and lets us enforce a consistent time grid
    in the implicilty-Trotterized `gradientsignals` method.

# Implementation

Any concrete sub-type `A` must implement
    *everything* required in the `EvolutionType` interface,
    so consult the documentation for `DeviceType` carefully.

In addition, the following method must be implemented:
- `nsteps(::A)`: the number of Trotter steps

The number of steps will usually be a simple integer field in the implementing struct,
    but this is left as an implementation detail.

"""
abstract type TrotterEvolution <: EvolutionType end

"""
    nsteps(device::DeviceType)

The number of Trotter steps.

"""
function nsteps(::TrotterEvolution)
    error("Not Implemented")
    return 0
end



"""
    gradientsignals(device[, basis], T, ψ0, r, O; kwargs...)

The gradient signals associated with a given `device` Hamiltonian, and an observable `O`.

Gradient signals are used to calculate analytical derivatives of a control pulse.

# Arguments
- `evolution::TrotterEvolution` how to initialize the co-state `|λ⟩`
        Also determines the number of Trotter steps `r` to evaluate ``ϕ_j(t)`` for.
        A standard choice would be `ToggleEvolutions.Toggle(r)`.

- `device::Devices.DeviceType`: specifies which Hamiltonian to evolve under.
        Also identifies each of the gradient operators used to calculate gradient signals.

- `basis::Bases.BasisType`: which basis `ψ` is represented in.
        ALSO determines the basis in which calculations are carried out.
        Defaults to `Bases.OCCUPATION`.

- `T::Real`: the total duration of the pulse.

- `ψ0`: the initial statevector, defined on the full Hilbert space of the device.

- `O`: a Hermitian observable, represented as a matrix.
    Gradients are calculated with respect to the expectation `⟨O⟩` at time `T`.

# Keyword Arguments
- `result`: an (optional) pre-allocated array to store gradient signals

- `callback`: a function called at each iteration of the gradient signal calculation.
        The function is passed three arguments:
        - `i`: indexes the iteration
        - `t`: the current time point
        - `ψ`: the current statevector, in the OCCUPATION basis
        The function is called after having evolved ψ into |ψ(t)⟩,
            but before calculating ϕ̄[i,:]. Evolution here runs backwards.

# Returns
A vector list `ϕ̄`, where each `ϕ̄[:,j]` is the gradient signal ``ϕ_j(t)``
    evaluated on a time grid given by `trapezoidaltimegrid(T,r)`.


# Explanation
A gradient signal ``ϕ_j(t)`` is defined with respect to a gradient operator ``Â_j``,
    an observable ``Ô``, a time-dependent state `|ψ(t)⟩`, and total pulse duration `T`.

Let us define the expectation value ``E(T) ≡ ⟨ψ(T)|Ô|ψ(T)⟩``.

Define the co-state ``|λ(t)⟩`` as the (un-normalized) statevector
    which satisfies ``E(T)=⟨λ(t)|ψ(t)⟩`` for any time `t∊[0,T]`.
The gradient signal is defined as ``ϕ_j(t) ≡ ⟨λ(t)|(iÂ_j)|ψ(t)⟩ + h.t.``.


    gradientsignals(device[, basis], T, ψ0, r, Ō; kwargs...)

When the matrix argument `O` is replaced by a matrix list `Ō`,
    each `Ō[:,:,k]` represents a different Hermitian observable ``Ô_k``.
In this case, a different set of gradient signals is computed for *each* ``Ô_k``.

# Returns
A 3d array `ϕ̄`, where each `ϕ̄[:,j,k]` is the gradient signal ``ϕ_j(t)``
    defined with respect to the observable ``Ô_k``.

# Explanation
Multiple sets of gradient signals may be useful
    if you want to compute gradients with respect to multiple observables.
For example, gradients with respect to a normalized molecular energy
    include contributions from both a molecular Hamiltonian and a leakage operator.
This method enables such calculations using only a single "pass" through time.

"""
function gradientsignals(
    evolution::TrotterEvolution,
    device::Devices.DeviceType,
    args...;
    kwargs...
)
    return gradientsignals(evolution, device, workbasis(evolution), args...; kwargs...)
end

function gradientsignals(
    evolution::TrotterEvolution,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector,
    O::AbstractMatrix;
    result=nothing,
    kwargs...
)
    # `O` AND `result` GIVEN AS 2D ARRAYS BUT MUST BE 3D FOR DELEGATION
    result !== nothing && (result = reshape(result, size(result)..., 1))
    Ō = reshape(O, size(O)..., 1)

    # PERFORM THE DELEGATION
    result = gradientsignals(evolution, device, basis, T, ψ0, Ō; result=result, kwargs...)

    # NOW RESHAPE `result` BACK TO 2D ARRAY
    result = reshape(result, size(result, 1), size(result, 2))
    return result
end

function gradientsignals(
    evolution::TrotterEvolution,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector,
    Ō::LinearAlgebraTools.MatrixList;
    result=nothing,
    callback=nothing,
)
    r = nsteps(evolution)
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # PREPARE SIGNAL ARRAYS ϕ̄[i,j,k]
    if result === nothing
        F = real(LinearAlgebraTools.cis_type(ψ0))
        result = Array{F}(undef, r+1, Devices.ngrades(device), size(Ō,3))
    end

    # PREPARE STATE AND CO-STATES
    ψTYPE = LinearAlgebraTools.cis_type(ψ0)
    ψ = array(ψTYPE, size(ψ0), LABEL); ψ .= ψ0
    ψ = evolve!(evolution, device, basis, T, ψ)

    λ̄ = array(ψTYPE, (size(ψ0,1), size(Ō,3)), LABEL)
    for k in axes(Ō,3)
        λ̄[:,k] .= ψ
        LinearAlgebraTools.rotate!(@view(Ō[:,:,k]), @view(λ̄[:,k]))
    end

    # ROTATE INTO OCCUPATION BASIS FOR THE REST OF THIS METHOD
    if basis != OCCUPATION
        U = Devices.basisrotation(OCCUPATION, basis, device)
        ψ = LinearAlgebraTools.rotate!(U, ψ)
        for k in axes(Ō,3)
            LinearAlgebraTools.rotate!(U, @view(λ̄[:,k]))
        end
    end

    # LAST GRADIENT SIGNALS
    callback !== nothing && callback(r+1, t̄[r+1], ψ)
    for k in axes(Ō,3)
        λ = @view(λ̄[:,k])
        for j in 1:Devices.ngrades(device)
            z = Devices.braket(Gradient(j, t̄[end]), device, OCCUPATION, λ, ψ)
            result[r+1,j,k] = 2 * imag(z)   # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
        end
    end

    # ITERATE OVER TIME
    for i in reverse(1:r)
        # COMPLETE THE PREVIOUS TIME-STEP AND START THE NEXT
        ψ = Devices.propagate!(Drive(t̄[i+1]), device, OCCUPATION, -τ/2, ψ)
        ψ = Devices.propagate!(STATIC, device, OCCUPATION, -τ, ψ)
        ψ = Devices.propagate!(Drive(t̄[i]),   device, OCCUPATION, -τ/2, ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            Devices.propagate!(Drive(t̄[i+1]), device, OCCUPATION, -τ/2, λ)
            Devices.propagate!(STATIC, device, OCCUPATION, -τ, λ)
            Devices.propagate!(Drive(t̄[i]),   device, OCCUPATION, -τ/2, λ)
        end

        # CALCULATE GRADIENT SIGNAL BRAKETS
        callback !== nothing && callback(i, t̄[i], ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Gradient(j, t̄[i]), device, OCCUPATION, λ, ψ)
                result[i,j,k] = 2 * imag(z) # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
            end
        end
    end

    return result
end


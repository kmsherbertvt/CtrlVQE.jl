"""
    EvolutionType

Defines a particular algorithm for performing time evolution.

# Implementation

Any concrete sub-type `A` must implement the following methods:
- `workbasis(::A)`: which Bases.BasisType the evolution algorithm uses

- `evolve!(::A, device, grid, ψ; callback=nothing)`:
    evolve a state ψ in-place on a time grid

If possible, it should also implement:
- `gradientsignals(::A, device, basis, grid, ψ0, r, Ō; kwargs...):
    compute the gradient signals of a device corresponding to multiple observables

You are allowed to implement these for restricted types of `grid`
    (eg. require it to be a `TrapezoidalIntegration`),
    so long as you are clear in your documentation.

"""
abstract type EvolutionType end

"""
    workbasis(evolution::EvolutionType)::Bases.BasisType

Which basis the evolution algorithm works in.

Also defines the default basis to interpret ψ as, in evolution methods.

"""
function workbasis end

"""
    evolve!(evolution, device, grid, ψ; callback=nothing)
    evolve!(evolution, device, basis, grid, ψ; callback=nothing)

Evolve a quantum computational state in-place under a device Hamiltonian.

This method both mutates and returns `ψ`.

# Arguments
- `evolution::EvolutionType`: which evolution algorithm to use.
- `device::Devices.DeviceType`: specifies which Hamiltonian to evolve under.
- `basis::Bases.BasisType`: which basis `ψ` is represented in.
    Assumed to be the workbasis of `evolution` when omitted.
- `grid::Integrations.IntegrationType`:
    defines the time integration bounds (eg. from 0 to ``T``)
- `ψ::AbstractVector`:
    the initial statevector, defined on the full Hilbert space of the device.

# Keyword Arguments
- `callback`: a function which is called at each iteration of the time evolution.
        The function is passed three arguments:
        - `i`: indexes the iteration
        - `t`: the current time point
        - `ψ`: the current statevector, in the work basis
        The function is called after having evolved ψ into |ψ(t)⟩.

# Implementation

Only the signature omitting `basis` need be implemented,
    so you can assume `ψ` is represented in whatever basis you return from `workbasis`.

The signature including `basis` will automatically rotate `ψ` into the `workbasis`,
    call the method you implement, and then rotate back.

"""
function evolve! end

"""
    gradientsignals(evolution, device, grid, ψ, Ō; kwargs...)
    gradientsignals(evolution, device, basis, grid, ψ, Ō; kwargs...)

The gradient signals associated with a given `device` Hamiltonian, and an observable `O`.

Gradient signals are used to calculate analytical derivatives of a control pulse.

# Arguments
- evolution::EvolutionType: which evolution algorithm to use.
- device::Devices.DeviceType: specifies which Hamiltonian to evolve under.
        Also identifies each of the gradient operators used to calculate gradient signals.
- `basis::Bases.BasisType`: which basis `ψ` is represented in.
    Assumed to be the workbasis of `evolution` when omitted.
- `grid::Integrations.IntegrationType`:
    defines the time integration bounds (eg. from 0 to ``T``)
- `ψ::AbstractVector`:
    the initial statevector, defined on the full Hilbert space of the device.
- `Ō::Union{LAT.MatrixList,AbstractMatrix}`:
    a list of Hermitian observables, represented as matrices, or a single such matrix.
    Gradients are calculated with respect to the expectation `⟨O⟩` at time ``T``.

This method signature assumes `ψ`, `Ō` are represented in the workbasis of `evolution`.

# Keyword Arguments
- `result`: an (optional) pre-allocated array to store gradient signals
- `callback`: a function called at each iteration of the gradient signal calculation.
        The function is passed three arguments:
        - `i`: indexes the iteration
        - `t`: the current time point
        - `ψ`: the current statevector, in the BARE basis
        The function is called after having evolved ψ into |ψ(t)⟩,
            but before calculating ϕ̄[i,:]. Evolution here runs backwards.

# Returns
A 3d array `ϕ̄`, where each `ϕ̄[:,j,k]` is the gradient signal ``ϕ_j(t)``
    defined with respect to the observable ``Ô_k``,
    or a 2d array when `Ō` is just a single matrix rather than a matrix list.

# Explanation
A gradient signal ``ϕ_j(t)`` is defined with respect to a gradient operator ``Â_j``,
    an observable ``Ô``, a time-dependent state `|ψ(t)⟩`, and total pulse duration `T`.

Let us define the expectation value ``E(T) ≡ ⟨ψ(T)|Ô|ψ(T)⟩``.

Define the co-state ``|λ(t)⟩`` as the (un-normalized) statevector
    which satisfies ``E(T)=⟨λ(t)|ψ(t)⟩`` for any time `t∊[0,T]`.
The gradient signal is defined as ``ϕ_j(t) ≡ ⟨λ(t)|(iÂ_j)|ψ(t)⟩ + h.t.``.

# Implementation

Only the signature omitting `basis` need be implemented,
    so can assume `ψ`, `Ō` are represented in whatever basis you return from `workbasis`.

The signature including `basis` will automatically rotate `ψ`, `Ō` into the `workbasis`
    before calling the method you implement.

Similiarly, you should only implement the method where `Ō` is a MatrixList.
When a single matrix, the (already-implemented) method will reshaped it into a MatrixList,
    call the method you implement, and then reshape the resulting array.

"""
function gradientsignals end
using .CostFunctions: CostFunctionType

"""
    EnergyFunction{F}

Super-type for cost functions calculating the expectation value of a time-evolved state.

# Implementation

Any concrete sub-type `CF` must implement
    *everything* required in the `CostFunctionType` interface,
    so consult the documentation for `CostFunctionType` carefully.

In additon, the following method must be implemented:
- `trajectory_callback(::CF, E::AbstractVector; callback=nothing)`
        returns a `Function` compatible with Evolutions.evolve callback
            ie. a callable expression (i::Int, t::Real, ψ::Vector) -> Nothing
        which sets E[i] to the energy of a partially evolved wavefunction ψ.
        If `callback` is provided, the function calls that `callback` afterwards.

Finally, the following methods must now accept a keyword argument:
- `cost_function(::CF; callback=nothing)`:
        When `callback` is provided, the time evolution must call it at each timestep.

- `grad!function(::CF; ϕ=nothing)`:
        When `ϕ` is provided, write the gradient signals to it.

"""
abstract type EnergyFunction{F} <: CostFunctionType{F} end

"""
    nobservables(fn::EnergyFunction)::Int

Identify the number of Hermitian observables needed for this energy function.

For example, to measure the normalized energy,
    separate observables are needed for both energy and normalization,
    and the results are combined in a non-linear way to produce the final outcome.

"""
function nobservables end

"""
    trajectory_callback(fn::EnergyFunction, E::AbstractVector; callback=nothing)::Function

Make a callback to write the energy at each step of a time evolution.

The callback function should be compatible with `Evolutions.evolve`
    (ie. a callable expression `(i::Int, t::Real, ψ::Vector) -> Nothing`),
    which sets `E[1+i]` to the energy of a partially evolved wavefunction ψ.
Note the `1+`, due to `i` indexing a time integration, which starts from 0.

If `callback` is provided, the function calls that `callback` afterwards
    (i.e. callback chaining).

"""
function trajectory_callback end

"""
    cost_function(fn::EnergyFunction; callback=nothing)

Same as for `CostFunctionType` except that whenever the function is called,
    the time evolution calls `callback` (if provided) in each time step.

"""
function cost_function(fn::EnergyFunction; callback=nothing)
    error("Not Implemented")
end

"""
    grad!function(fn::EnergyFunction; ϕ=nothing)

Same as for `CostFunctionType` except that whenever the function is called,
    ϕ (if provided) is updated to contain the gradient signals.
The array ϕ should be a 3d array with shape (:,nG,nK),
    where nK is the number of observables in the energy function,
    nG is the number of gradient operators in the underlying device,
    and the remaining dimension is the size of the time grid.

"""
function grad!function(fn::EnergyFunction; ϕ=nothing)
    error("Not Implemented")
end